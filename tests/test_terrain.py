import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

from solshade.terrain import (
    compute_hillshade,
    compute_horizon_map,
    compute_slope_aspect_normals,
    horizon_interp,
    load_dem,
)

# -------------------------
# DEM creation helpers
# -------------------------


def flat_elevation(y, x):
    return np.zeros_like(x)


def constant_elevation(y, x):
    return np.ones_like(x) * 100


def slope_east(y, x):
    return x.astype(np.float32)


def slope_north(y, x):
    return y.astype(np.float32)


def create_mock_dem(shape=(32, 32), elevation_func=flat_elevation):
    y, x = np.indices(shape)
    data = elevation_func(y, x).astype(np.float32)
    dem = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(shape[0]) * 10, "x": np.arange(shape[1]) * 10},
        name="elevation",
    )
    dem.rio.write_crs(CRS.from_epsg(32633), inplace=True)
    dem.rio.write_transform(Affine.translation(0, 0) * Affine.scale(10, -10), inplace=True)
    return dem


@pytest.fixture
def synthetic_dem():
    return create_mock_dem((16, 16), elevation_func=flat_elevation)


# -------------------------
# DEM loading tests
# -------------------------


def test_load_valid_dem():
    path = Path("tests/data/MARS.tif")
    dem = load_dem(path)
    assert isinstance(dem, xr.DataArray)
    assert dem.ndim == 2


def test_load_dem_raises_type_error():
    with patch("rioxarray.open_rasterio") as mock_open:
        mock_open.return_value.squeeze.return_value = xr.Dataset()
        with pytest.raises(TypeError, match="DEM is not a single-band raster"):
            load_dem("dummy/path.tif")


# ------------------------------
# Slope, aspect & normals tests
# ------------------------------


def _assert_unit_normals(normal_enu: xr.DataArray, atol=1e-6):
    vec = normal_enu.values
    norms = np.linalg.norm(vec, axis=0)
    assert np.allclose(norms, 1.0, atol=atol)


@pytest.mark.parametrize(
    "elevation_func, expected_aspect",
    [
        (flat_elevation, 0),
        (slope_east, 270),
        (slope_north, 0),
    ],
)
def test_slope_and_aspect(elevation_func, expected_aspect):
    dem = create_mock_dem(elevation_func=elevation_func)
    slope, aspect, _ = compute_slope_aspect_normals(dem)
    assert np.allclose(slope.values, slope.values[0, 0])
    if elevation_func is flat_elevation:
        assert np.isnan(aspect.values).all() or np.allclose(aspect.values, 0)
    else:
        assert np.allclose(aspect.values, expected_aspect)


def test_normals_flat_surface_upward():
    dem = create_mock_dem((12, 12), flat_elevation)
    _, _, normal = compute_slope_aspect_normals(dem)
    assert normal.dims == ("band", "y", "x")
    assert list(normal.coords["band"].values) == ["east", "north", "up"]
    e, n, u = (normal.sel(band=b).values for b in ["east", "north", "up"])
    assert np.allclose([e, n], 0.0, atol=1e-12)
    assert np.allclose(u, 1.0, atol=1e-12)
    _assert_unit_normals(normal)


@pytest.mark.parametrize("elevation_func", [slope_east, slope_north])
def test_normals_match_formula(elevation_func):
    dem = create_mock_dem((20, 20), elevation_func=elevation_func)
    slope, aspect, normal = compute_slope_aspect_normals(dem)
    s_r = np.deg2rad(slope.values)
    a_r = np.deg2rad(aspect.values)
    e_expect = np.sin(s_r) * np.sin(a_r)
    n_expect = np.sin(s_r) * np.cos(a_r)
    u_expect = np.cos(s_r)
    np.testing.assert_allclose(normal.sel(band="east").values, e_expect, atol=1e-10)
    np.testing.assert_allclose(normal.sel(band="north").values, n_expect, atol=1e-10)
    np.testing.assert_allclose(normal.sel(band="up").values, u_expect, atol=1e-10)
    _assert_unit_normals(normal)


def test_random_surface_normals_are_valid():
    rng = np.random.default_rng(0)
    dem = create_mock_dem((24, 24), elevation_func=lambda y, x: rng.normal(0, 1, size=x.shape))
    _, _, normal = compute_slope_aspect_normals(dem)
    v = normal.values
    assert np.isfinite(v).all()
    assert (v >= -1.0 - 1e-9).all() and (v <= 1.0 + 1e-9).all()
    _assert_unit_normals(normal)


# -------------------------
# Hillshade tests
# -------------------------


def test_hillshade_basic_properties():
    slope = xr.DataArray([[30, 45], [60, 75]], dims=["y", "x"])
    aspect = xr.DataArray([[0, 90], [180, 270]], dims=["y", "x"])
    hs = compute_hillshade(slope, aspect)
    assert hs.shape == slope.shape
    assert np.all((hs >= 0) & (hs <= 1))


def test_flat_surface_hillshade():
    slope = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    hs = compute_hillshade(slope, aspect, azimuth_deg=0, altitude_deg=60)
    expected = np.cos(np.radians(30))
    assert np.allclose(hs.values, expected)


def test_hillshade_decreases_with_steeper_slope():
    slope_low = xr.DataArray([[30]], dims=["y", "x"])
    slope_high = xr.DataArray([[80]], dims=["y", "x"])
    aspect = xr.DataArray([[0]], dims=["y", "x"])
    h_low = compute_hillshade(slope_low, aspect)
    h_high = compute_hillshade(slope_high, aspect)
    assert h_low.values > h_high.values


def test_mismatched_shape_raises_alignment_error():
    slope = xr.DataArray(np.zeros((3, 3)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    with pytest.raises(xr.AlignmentError, match="conflicting dimension sizes"):
        compute_hillshade(slope, aspect)


# -------------------------
# Horizon map tests
# -------------------------


def test_horizon_output_dimensions():
    dem = create_mock_dem((16, 16))
    result = compute_horizon_map(dem, n_directions=8, max_distance=100, step=20, chunk_size=8, progress=False)
    assert result.dims == ("azimuth", "y", "x")
    assert result.shape == (8, 16, 16)


def test_flat_constant_dem_returns_zero_horizon():
    dem = create_mock_dem((8, 8), elevation_func=constant_elevation)
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert np.allclose(result.values[~np.isnan(result.values)], 0, atol=1e-3)


def test_nans_in_dem_propagate_to_horizon():
    dem = create_mock_dem((8, 8))
    dem.values[2:4, 2:4] = np.nan
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert np.all(np.isnan(result.values[:, 2:4, 2:4]))


def test_directional_slope_gives_nonzero_horizon():
    dem = create_mock_dem((8, 8), elevation_func=slope_east)
    result = compute_horizon_map(dem, n_directions=8, max_distance=100, step=20, chunk_size=4, progress=False)
    east_idx = np.argmin(np.abs(result.azimuth.values - 90))
    assert np.nanmean(result.values[east_idx]) > 0


def test_crs_and_metadata_are_preserved():
    dem = create_mock_dem((16, 16))
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert result.rio.crs == dem.rio.crs
    assert result.attrs["units"] == "degrees"
    assert "max_distance_m" in result.attrs
    assert len(json.loads(result.attrs["azimuths_deg"])) == 4


def test_serial_and_parallel_execution():
    dem = create_mock_dem((32, 32))
    serial = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, n_jobs=1, progress=False)
    parallel = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=8, n_jobs=2, progress=False)
    assert serial.shape == parallel.shape


def test_missing_crs_or_transform_raises():
    dem = xr.DataArray(np.zeros((8, 8)), dims=("y", "x"))
    with pytest.raises(ValueError, match="CRS and affine transform"):
        compute_horizon_map(dem)


@patch("solshade.terrain.Progress")
def test_progress_bar_called(mock_progress, synthetic_dem):
    result = compute_horizon_map(synthetic_dem, n_directions=8, max_distance=200, step=50, chunk_size=2, progress=True)
    assert result.ndim == 3


# -------------------------
# Horizon interpolation tests
# -------------------------


def test_cubic_and_fallback_interpolators():
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 10 * np.sin(np.radians(az))
    assert np.isclose(horizon_interp(az, vals)(0), horizon_interp(az, vals)(360), atol=1e-6)
    vals[10:50] = np.nan
    assert np.isclose(horizon_interp(az, vals)(0), horizon_interp(az, vals)(360), atol=1e-6)
    vals[:] = np.nan
    assert horizon_interp(az, vals) is None
    vals = 10 * np.sin(np.radians(az))
    vals[5] = 200
    assert np.isclose(horizon_interp(az, vals)(0), horizon_interp(az, vals)(360), atol=1e-6)
    vals = 5 * np.cos(np.radians(az))
    interp = horizon_interp(az, vals)
    assert np.isclose(interp(-360), interp(0), atol=1e-6)
    assert np.isclose(interp(720), interp(0), atol=1e-6)
