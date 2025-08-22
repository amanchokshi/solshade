import json
from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from solshade.irradiance import compute_energy_metrics, compute_flux_timeseries

# -------------------------------
# Helpers
# -------------------------------


def _sun_vectors_from_altaz(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    """Unit ENU vectors from alt/az (deg); shape (T,3), float32."""
    alt_r = np.deg2rad(alt_deg)
    az_r = np.deg2rad(az_deg % 360.0)
    e = np.cos(alt_r) * np.sin(az_r)
    n = np.cos(alt_r) * np.cos(az_r)
    u = np.sin(alt_r)
    return np.stack([e, n, u], axis=1).astype(np.float32)


def _make_times(start: str, hours: int, step_h: float = 1.0) -> np.ndarray:
    """Uniform datetime64[s] vector of length `hours` with step `step_h` (hours)."""
    start64 = np.datetime64(start, "s")
    step_s = int(round(step_h * 3600.0))
    return start64 + np.arange(hours) * np.timedelta64(step_s, "s")


def _horizon(az: int, y: int, x: int, *, az_vals: np.ndarray | None = None) -> xr.DataArray:
    """Valid horizon with azimuth band and required attrs."""
    if az_vals is None:
        az_vals = np.linspace(0.0, 360.0, az, endpoint=False).astype(np.float32)
    data = np.zeros((az, y, x), dtype=np.float32)
    return xr.DataArray(
        data,
        dims=("azimuth", "y", "x"),
        coords={"azimuth": np.arange(az), "y": np.arange(y), "x": np.arange(x)},
        attrs={"azimuths_deg": json.dumps(az_vals.tolist())},
        name="horizon",
    )


def _normals_up(y: int, x: int) -> xr.DataArray:
    """Flat surface: east=0, north=0, up=1."""
    east = np.zeros((y, x), dtype=np.float32)
    north = np.zeros((y, x), dtype=np.float32)
    up = np.ones((y, x), dtype=np.float32)
    data = np.stack([east, north, up], axis=0)
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": ["east", "north", "up"], "y": np.arange(y), "x": np.arange(x)},
        name="normal_enu",
    )


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def tiny_scene() -> Tuple[xr.DataArray, xr.DataArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A small, valid setup you can mutate in tests."""
    y, x, az = 3, 4, 8
    horizon = _horizon(az, y, x)
    normals = _normals_up(y, x)
    sun_alt = np.array([10.0, 20.0], dtype=np.float32)
    sun_az = np.array([0.0, 0.0], dtype=np.float32)
    sun_enu = _sun_vectors_from_altaz(sun_alt, sun_az)
    sun_au = np.ones_like(sun_alt, dtype=np.float32)
    times = _make_times("2025-01-01T00:00:00", hours=sun_alt.size)
    return horizon, normals, sun_alt, sun_az, sun_au, sun_enu, times


# -------------------------------
# Core behavior
# -------------------------------


def test_flux_basic_shadow_lambertian_nan_and_au_scaling():
    # Grid/horizon with a shadow pixel and a NaN pixel
    y, x = 4, 5
    az = 8
    horizon = _horizon(az, y, x)
    # inject special cases
    horizon.values[:, 2, 3] = 20.0  # horizon=20° at (2,3)
    horizon.values[:, 1, 1] = np.nan  # NaN propagation at (1,1)

    normals = _normals_up(y, x)

    sun_alt = np.array([-5.0, 10.0, 45.0], dtype=np.float32)  # below horizon, mid, high
    sun_az = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sun_enu = _sun_vectors_from_altaz(sun_alt, sun_az)
    sun_au = np.array([1.0, 1.0, 2.0], dtype=np.float32)  # 1/r^2 scaling -> last is /4
    times = _make_times("2025-01-01T00:00:00", hours=3)

    toa = 1000.0
    flux = compute_flux_timeseries(
        horizon=horizon,
        sun_alt_deg=sun_alt,
        sun_az_deg=sun_az,
        sun_au=sun_au,
        sun_enu=sun_enu,
        normal_enu=normals,
        times_utc=times,
        toa=toa,
        dtype=np.float32,
        n_jobs=1,
        batch_size=3,
        backend="threading",
    )

    assert flux.shape == (3, y, x)
    assert flux.dtype == np.float32
    np.testing.assert_array_equal(flux.time.values.astype("datetime64[s]"), times)
    assert "toa_W_m2" in flux.attrs and flux.attrs["units"] == "W m^-2"

    s_up = np.sin(np.deg2rad(sun_alt)).astype(np.float32)
    scale = toa / (sun_au**2)

    # t0: zeros except NaN-propagated
    exp0 = np.zeros((y, x), dtype=np.float32)
    exp0[1, 1] = np.nan
    # t1: flux unless shadow at (2,3) and NaN at (1,1)
    exp1 = np.full((y, x), scale[1] * s_up[1], dtype=np.float32)
    exp1[2, 3] = 0.0
    exp1[1, 1] = np.nan
    # t2: flux with 1/4 scaling; NaN at (1,1)
    exp2 = np.full((y, x), scale[2] * s_up[2], dtype=np.float32)
    exp2[1, 1] = np.nan

    for t, exp in enumerate([exp0, exp1, exp2]):
        got = flux.isel(time=t).values
        m = np.isfinite(exp)
        np.testing.assert_allclose(got[m], exp[m], rtol=1e-6, atol=1e-6)
        assert np.isnan(got[1, 1])


# -------------------------------
# Validation / error paths
# -------------------------------


def test_missing_azimuths_attr_raises(tiny_scene):
    horizon, normals, sun_alt, sun_az, sun_au, sun_enu, times = tiny_scene
    horizon = horizon.drop_vars(["azimuth"])  # keep dims; just remove attr later
    # rebuild without attrs
    horizon = xr.DataArray(horizon.values, dims=horizon.dims, coords=horizon.coords, attrs={})
    with pytest.raises(ValueError, match="azimuths_deg"):
        compute_flux_timeseries(horizon, sun_alt, sun_az, sun_au, sun_enu, normals, times)


def test_bad_shapes_raise(tiny_scene):
    horizon, normals, sun_alt, _sun_az, sun_au, _sun_enu, times = tiny_scene
    sun_az_bad = np.array([0.0], dtype=np.float32)  # wrong length
    sun_enu_ok = _sun_vectors_from_altaz(sun_alt, np.zeros_like(sun_alt))
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_flux_timeseries(horizon, sun_alt, sun_az_bad, sun_au, sun_enu_ok, normals, times)


def test_bad_sun_au_raises(tiny_scene):
    horizon, normals, sun_alt, sun_az, _sun_au, sun_enu, times = tiny_scene
    with pytest.raises(ValueError, match="shape"):
        compute_flux_timeseries(horizon, sun_alt, sun_az, np.array([[1.0]], dtype=np.float32), sun_enu, normals, times)
    with pytest.raises(ValueError, match="non-finite"):
        compute_flux_timeseries(horizon, sun_alt, sun_az, np.array([np.nan, 1.0], dtype=np.float32), sun_enu, normals, times)


@pytest.mark.parametrize(
    "make_horizon,errmsg",
    [
        (
            lambda h: xr.DataArray(  # missing 'azimuth'/'band' dim
                np.zeros((h.sizes["y"], h.sizes["x"]), dtype=np.float32),
                dims=("y", "x"),
                coords={"y": h["y"], "x": h["x"]},
                attrs=h.attrs,
            ),
            "leading 'azimuth' or 'band'",
        ),
        (
            lambda h: xr.DataArray(  # wrong spatial dim names: ('row','col')
                np.zeros((h.sizes["azimuth"], h.sizes["y"], h.sizes["x"]), dtype=np.float32),
                dims=("azimuth", "row", "col"),
                coords={
                    "azimuth": np.arange(h.sizes["azimuth"]),
                    "row": np.arange(h.sizes["y"]),
                    "col": np.arange(h.sizes["x"]),
                },
                attrs=h.attrs,
            ),
            "spatial dims \\('y','x'\\)",
        ),
    ],
)
def test_horizon_dim_validation(make_horizon, errmsg, tiny_scene):
    horizon, normals, sun_alt, sun_az, sun_au, sun_enu, times = tiny_scene
    horizon_bad = make_horizon(horizon)
    with pytest.raises(ValueError, match=errmsg):
        compute_flux_timeseries(horizon_bad, sun_alt, sun_az, sun_au, sun_enu, normals, times)


def test_normals_dim_validation_and_labels(tiny_scene):
    horizon, _normals, sun_alt, sun_az, sun_au, sun_enu, times = tiny_scene
    y, x = horizon.sizes["y"], horizon.sizes["x"]

    # Missing band dim
    normals_bad_dims = xr.DataArray(
        np.ones((y, x), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": np.arange(y), "x": np.arange(x)},
    )
    with pytest.raises(ValueError, match="must have dims \\('band','y','x'\\)"):
        compute_flux_timeseries(horizon, sun_alt, sun_az, sun_au, sun_enu, normals_bad_dims, times)

    # Wrong band labels -> KeyError in selection
    normals_wrong_labels = xr.DataArray(
        np.zeros((3, y, x), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={"band": ["a", "b", "c"], "y": np.arange(y), "x": np.arange(x)},
    )
    with pytest.raises(KeyError, match="must contain bands \\['east','north','up'\\]"):
        compute_flux_timeseries(horizon, sun_alt, sun_az, sun_au, sun_enu, normals_wrong_labels, times)


# -------------------------------
# Time/attrs wiring
# -------------------------------


def test_time_coords_and_attrs_strings_present(tiny_scene):
    horizon, normals, sun_alt, sun_az, sun_au, sun_enu, times = tiny_scene
    flux = compute_flux_timeseries(
        horizon, sun_alt, sun_az, sun_au, sun_enu, normals, times, n_jobs=1, batch_size=2, backend="threading"
    )
    np.testing.assert_array_equal(flux.time.values.astype("datetime64[s]"), times)
    iso = json.loads(flux.attrs["time_utc_iso"])
    assert isinstance(iso, list) and len(iso) == times.size


def test_time_coord_without_datetime64_skips_iso_attr(tiny_scene):
    horizon, normals, sun_alt, sun_az, sun_au, sun_enu, _times = tiny_scene
    # pass non-datetime64 times
    times = [0, 1]
    flux = compute_flux_timeseries(
        horizon, sun_alt[:2], sun_az[:2], sun_au[:2], sun_enu[:2], normals, times, n_jobs=1, batch_size=2, backend="threading"
    )
    np.testing.assert_array_equal(flux.time.values, np.array([0, 1]))
    assert "time_utc_iso" not in flux.attrs


# -------------------------------
# Energy metrics
# -------------------------------


def test_compute_energy_metrics_shapes_and_peak_day():
    y, x = 3, 4
    start = np.datetime64("2025-06-10T00:00:00", "s")
    hours = 72  # 3 days hourly
    times = start + np.arange(hours) * np.timedelta64(3600, "s")

    # Day1=1, Day2=3 (peak), Day3=2
    day_vals = np.concatenate(
        [np.full(24, 1.0, dtype=np.float32), np.full(24, 3.0, dtype=np.float32), np.full(24, 2.0, dtype=np.float32)]
    )
    flux_data = day_vals[:, None, None] * np.ones((hours, y, x), dtype=np.float32)

    flux = xr.DataArray(
        flux_data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(y), "x": np.arange(x)},
        name="irradiance",
        attrs={"units": "W m^-2"},
    )

    daily_energy, total_energy, peak_energy, day_of_peak = compute_energy_metrics(flux)

    assert daily_energy.sizes["time"] == 3
    assert total_energy.shape == (y, x)
    assert peak_energy.shape == (y, x)
    assert day_of_peak.shape == (y, x)

    peak_dates = daily_energy.astype("float64").idxmax(dim="time")
    expected_doy = peak_dates.dt.dayofyear.values
    np.testing.assert_array_equal(day_of_peak.values, expected_doy)
