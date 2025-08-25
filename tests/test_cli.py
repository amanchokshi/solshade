# tests/test_cli.py

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from typer.testing import CliRunner

import solshade.cli as cli_mod
import solshade.viz as viz_mod
from solshade.cli import app

runner = CliRunner()

# Dummy path for test data -- replace with actual test DEM path
TEST_DATA = Path("tests/data/MARS.tif")


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def assert_no_unexpected_stderr(stderr: str):
    """Fail only if stderr contains unexpected lines (kept for potential future use)."""
    allowed_substrings = ["libpng warning: iCCP: known incorrect sRGB profile"]
    for line in stderr.splitlines():
        if any(allowed in line for allowed in allowed_substrings):
            continue
        assert False, f"Unexpected stderr: {line}"


@pytest.fixture
def mock_horizon_file(tmp_path):
    """Create a small synthetic horizon GeoTIFF with azimuth metadata."""
    ny, nx = 50, 50
    n_directions = 360
    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    crs = CRS.from_epsg(3413)
    data = np.random.uniform(0, 20, size=(n_directions, ny, nx)).astype(np.float32)
    azimuths = np.linspace(0, 360, n_directions, endpoint=False)
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, n_directions + 1),
            "x": np.arange(nx) * transform.a + transform.c,  # type: ignore[attr-defined]
            "y": np.arange(ny) * transform.e + transform.f,  # type: ignore[attr-defined]
        },
        attrs={"azimuths_deg": json.dumps(azimuths.tolist())},
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    tif_path = tmp_path / "HORIZON_TEST.tif"
    da.rio.to_raster(tif_path)
    return tif_path


# ---------------------------------------------------------------------------
# Basic plot/compute subcommands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subcommand, expected_suffix",
    [
        ("slope", "_SLOPE.png"),
        ("aspect", "_ASPECT.png"),
        ("hillshade", "_HILLSHADE_315_45.png"),
        ("dem", "_DEM.png"),
    ],
)
def test_plot_subcommands(subcommand, expected_suffix):
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            ["solshade", "plot", subcommand, str(TEST_DATA), "--output-dir", str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed {subcommand} with stderr: {result.stderr}"
        expected_file = outdir / (TEST_DATA.stem + expected_suffix)
        assert expected_file.exists(), f"{expected_file} was not created"


@pytest.mark.parametrize(
    "subcommand, expected_file",
    [
        ("slope", "_SLOPE.tif"),
        ("aspect", "_ASPECT.tif"),
        ("hillshade", "_HILLSHADE_315_45.tif"),
    ],
)
def test_compute_subcommands(subcommand, expected_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            ["solshade", "compute", subcommand, str(TEST_DATA), "--output-dir", str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{subcommand} failed with stderr: {result.stderr}"
        expected = outdir / (TEST_DATA.stem + expected_file)
        assert expected.exists(), f"{expected} was not created"


@pytest.mark.parametrize("subcommand", ["slope", "aspect", "hillshade", "dem"])
def test_plot_subcommands_show(subcommand):
    """Run plot subcommands in show-mode using the SOLSHADE_TEST_MODE guard."""
    env = os.environ.copy()
    env["SOLSHADE_TEST_MODE"] = "1"
    result = subprocess.run(
        ["solshade", "plot", subcommand, str(TEST_DATA)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"{subcommand} failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Meta command
# ---------------------------------------------------------------------------


def test_meta_subcommand_output():
    """Check human-readable metadata fields are printed by `meta`."""
    result = subprocess.run(["solshade", "meta", str(TEST_DATA)], capture_output=True, text=True)
    assert result.returncode == 0, f"`meta` failed: {result.stderr}"
    out = result.stdout
    for token in [
        "METADATA:",
        "CRS:",
        "SHAPE:",
        "RESOLUTION:",
        "TRANSFORM:",
        "BOUNDS:",
        "COORDS:",
        "DTYPE:",
        "ATTRIBUTES:",
    ]:
        assert token in out
    assert "EPSG:3413" in out
    assert "1250, 1250" in out
    assert "2.00" in out


def test_attribute_truncation():
    """Long attribute values are truncated in `meta` output."""
    mock_dem = MagicMock()
    mock_transform = Affine(2.0, 0.0, -824430.0, 0.0, -2.0, -801210.0)
    mock_dem.rio.transform.return_value = mock_transform
    mock_dem.rio.bounds.return_value = (-1, -1, 1, 1)
    mock_dem.shape = (100, 100)
    mock_dem.coords = {"x": None, "y": None}
    mock_dem.dtype = "float32"
    mock_dem.attrs = {"azimuths_deg": "[" + "1234567890" * 10 + "]"}
    mock_dem.rio.crs = CRS.from_epsg(4326)

    with patch("solshade.cli.load_dem", return_value=mock_dem):
        result = runner.invoke(app, ["meta", "dummy.tif"])
        assert result.exit_code == 0
        assert "AZIMUTHS_DEG" in result.stdout
        assert "(truncated)" in result.stdout


# ---------------------------------------------------------------------------
# Horizon plotting
# ---------------------------------------------------------------------------


def test_plot_horizon_cmd_within_bounds(mock_horizon_file, tmp_path):
    """Plot horizon for a lat/lon inside raster bounds; PNG should be saved."""
    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)  # type: ignore
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)
    result = runner.invoke(
        app,
        ["plot", "horizon", "--lat", str(lat), "--lon", str(lon), str(mock_horizon_file), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Saved horizon polar plot" in result.stdout


def test_plot_horizon_cmd_out_of_bounds(mock_horizon_file):
    """Out-of-bounds lat/lon should cause an error exit."""
    result = runner.invoke(app, ["plot", "horizon", "--lat", "20.0", "--lon", "20.0", str(mock_horizon_file)])
    assert result.exit_code != 0


def test_plot_horizon_cmd_show(monkeypatch, mock_horizon_file):
    """When no output-dir and guard unset, CLI should call plt.show()."""
    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)  # type: ignore
    reverse_transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    lon, lat = reverse_transformer.transform(center_x, center_y)
    monkeypatch.delenv("SOLSHADE_TEST_MODE", raising=False)
    with patch("matplotlib.pyplot.show") as mock_show:
        result = runner.invoke(
            app,
            ["plot", "horizon", "--lat", str(lat), "--lon", str(lon), str(mock_horizon_file)],
        )
        assert result.exit_code == 0, result.stderr
        mock_show.assert_called_once()


def test_plot_horizon_cmd_with_solar_overlay(monkeypatch, mock_horizon_file, tmp_path):
    """Overlay solar envelope; verify a SOLAR PNG is saved."""
    monkeypatch.setenv("SOLSHADE_TEST_MODE", "1")

    # Stub compute_solar_ephem
    def _fake_compute(lat, lon, startutc=None, stoputc=None, timestep=3600, cache_dir=None):
        n = 24
        times = np.array([np.datetime64("2025-01-01T00:00") + np.timedelta64(i, "h") for i in range(n)])
        alt = np.linspace(5.0, 25.0, n)
        az = (np.linspace(0, 360, n, endpoint=False) + 7.0) % 360.0
        dist_au = np.zeros(az.size)
        enu_unit = np.zeros((az.size, 3))
        assert startutc is None or startutc.tzinfo is not None
        assert stoputc is None or stoputc.tzinfo is not None
        return times, alt, az, dist_au, enu_unit

    # Stub envelope
    def _fake_envelope(times_utc, alt_deg, az_deg, smooth_n=360):
        az_plot = np.array([0, 90, 180, 270, 360], dtype=float)
        min_plot = np.array([0, 5, 10, 5, 0], dtype=float)
        max_plot = np.array([10, 15, 20, 15, 10], dtype=float)
        return az_plot, min_plot, max_plot

    import solshade.solar as solar_mod

    monkeypatch.setattr(cli_mod, "compute_solar_ephem", _fake_compute, raising=True)
    monkeypatch.setattr(solar_mod, "solar_envelope_by_folding", _fake_envelope, raising=True)

    # In-bounds lat/lon (center of the raster)
    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)
    reverse_transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    lon, lat = reverse_transformer.transform(center_x, center_y)

    outdir = tmp_path / "plots"
    args = [
        "plot",
        "horizon",
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        str(mock_horizon_file),
        "--solar",
        "--startutc",
        "2025-01-01T00:00:00Z",
        "--stoputc",
        "2025-01-02T00:00:00Z",
        "--timestep",
        "3600",
        "--output-dir",
        str(outdir),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = list(outdir.glob("*_SOLAR_*.png"))
    assert len(saved) == 1


def test_compute_horizon_cmd_runner_covers_cli(tmp_path, monkeypatch):
    """Cover compute_horizon_cmd via Typer runner (no subprocess) and assert output file."""
    # Minimal DEM (shape/metadata only; values don't matter)
    dem_da = (
        xr.DataArray(
            np.ones((10, 10), dtype=np.float32),
            dims=("y", "x"),
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        .rio.write_crs("EPSG:3413")
        .rio.write_transform(from_origin(-1_000_000, 1_000_000, 2000, 2000))
    )

    # Stub load_dem to avoid IO
    monkeypatch.setattr(cli_mod, "load_dem", lambda p: dem_da, raising=True)

    # Stub compute_horizon_map to validate flags and return a small DataArray
    def _fake_horizon(
        dem,
        n_directions,
        max_distance,
        step,
        chunk_size,
        n_jobs,
        progress,
    ):
        # Ensure --no-progress flipped progress to False
        assert progress is False
        # Return a (band, y, x) array with rio metadata so .rio.to_raster works
        data = np.zeros((n_directions, dem.sizes["y"], dem.sizes["x"]), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=("band", "y", "x"),
            coords={"band": np.arange(1, n_directions + 1), "y": dem.y, "x": dem.x},
        )
        return da.rio.write_crs(dem.rio.crs).rio.write_transform(dem.rio.transform())

    monkeypatch.setattr(cli_mod, "compute_horizon_map", _fake_horizon, raising=True)

    outdir = tmp_path / "horizon_out"
    dem_path = tmp_path / "DEM_ANY.tif"  # path value is irrelevant; load_dem is stubbed
    dem_path.write_text("stub")

    # Invoke inside the process so coverage sees it
    result = runner.invoke(
        app,
        [
            "compute",
            "horizon",
            str(dem_path),
            "--n-directions",
            "4",
            "--max-distance",
            "100",
            "--step",
            "25",
            "--chunk-size",
            "64",
            "--no-progress",
            "--output-dir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.output
    expected = outdir / f"{dem_path.stem}_HORIZON_4.tif"
    assert expected.exists(), f"{expected} was not created"
    assert "Saved horizon map" in result.stdout


def test_plot_horizon_cmd_with_custom_filename(mock_horizon_file, tmp_path):
    """Covers the --filename branch of plot_horizon_cmd."""
    from solshade.cli import app

    outdir = tmp_path / "plots"
    custom_name = "custom_horizon.png"

    # A valid lat/lon inside the mock_horizon_file (its center pixel)
    from pyproj import Transformer
    from rasterio.transform import from_origin

    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)  # center of 50x50
    reverse_transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    lon, lat = reverse_transformer.transform(center_x, center_y)

    result = runner.invoke(
        app,
        [
            "plot",
            "horizon",
            "--lat",
            str(lat),
            "--lon",
            str(lon),
            str(mock_horizon_file),
            "--output-dir",
            str(outdir),
            "--filename",
            custom_name,
        ],
    )
    assert result.exit_code == 0, result.stdout
    out_file = outdir / custom_name
    assert out_file.exists(), f"{out_file} was not created"


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


def test_compute_normals_subcommand():
    """`solshade compute normals` writes a 3-band GeoTIFF of ENU normals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            ["solshade", "compute", "normals", str(TEST_DATA), "--output-dir", str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"compute normals failed: {result.stderr}"
        expected = outdir / (TEST_DATA.stem + "_NORMALS.tif")
        assert expected.exists(), f"{expected} was not created"


def test_plot_normals_subcommand_saves_png():
    """`solshade plot normals --output-dir` saves an RGB visual of ENU normals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            ["solshade", "plot", "normals", str(TEST_DATA), "--output-dir", str(outdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"plot normals failed: {result.stderr}"
        expected = outdir / (TEST_DATA.stem + "_NORMALS.png")
        assert expected.exists(), f"{expected} was not created"


def test_plot_normals_subcommand_show_mode():
    """`solshade plot normals` without --output-dir runs in show-mode (test guard set)."""
    env = os.environ.copy()
    env["SOLSHADE_TEST_MODE"] = "1"  # prevent plt.show()
    result = subprocess.run(
        ["solshade", "plot", "normals", str(TEST_DATA)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"plot normals show-mode failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Fluxseries compute (smoke) and EPSG:4326 branch
# ---------------------------------------------------------------------------


def test_compute_fluxseries_and_metrics_cli_smoke(tmp_path, monkeypatch, mock_horizon_file):
    """
    Smoke-test for `solshade compute fluxseries` that avoids expensive numerics.
    Verifies it exits cleanly and writes the expected set of files.
    """
    # Create a simple 2D DEM GeoTIFF (50x50) separate from the horizon file.
    ny, nx = 50, 50
    dem_da = xr.DataArray(
        np.ones((ny, nx), dtype=np.float32), dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)}, attrs={}
    )
    dem_da = dem_da.rio.write_crs("EPSG:3413")
    dem_da = dem_da.rio.write_transform(from_origin(-1_000_000, 1_000_000, 2000, 2000))
    dem_path = tmp_path / "DEM_TEST.tif"
    dem_da.rio.to_raster(dem_path)

    # Stubs for heavy functions
    def _fake_normals(dem):
        slope = xr.zeros_like(dem)
        aspect = xr.zeros_like(dem)
        normal_enu = xr.DataArray(
            np.zeros((3, dem.sizes["y"], dem.sizes["x"]), dtype=np.float32),
            dims=("band", "y", "x"),
            coords={"band": [0, 1, 2], "y": dem.y, "x": dem.x},
        )
        return slope, aspect, normal_enu

    def _fake_ephem(lat, lon, startutc=None, stoputc=None, timestep=3600, cache_dir=None):
        n = 6
        times = np.array([np.datetime64("2025-01-01T00:00") + i * np.timedelta64(1, "h") for i in range(n)])
        alt = np.linspace(5.0, 10.0, n)
        az = np.linspace(0.0, 360.0, n, endpoint=False)
        au = np.ones(n, dtype=np.float32)
        enu = np.zeros((n, 3), dtype=np.float32)
        return times, alt, az, au, enu

    def _fake_flux(horizon, sun_alt_deg, sun_az_deg, sun_au, sun_enu, normal_enu, times_utc, batch_size, n_jobs):
        data = np.random.rand(times_utc.size, ny, nx).astype(np.float32)
        return xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={"time": times_utc, "y": np.arange(ny), "x": np.arange(nx)},
        )

    def _fake_metrics(flux_da):
        daily_energy = xr.DataArray(
            flux_da.values, dims=("time", "y", "x"), coords={"time": flux_da.time, "y": flux_da.y, "x": flux_da.x}
        )
        total_energy = daily_energy.sum("time")
        peak_energy = daily_energy.max("time")
        day_of_peak = xr.DataArray(
            np.argmax(daily_energy.values, axis=0).astype(np.float32),
            dims=("y", "x"),
            coords={"y": daily_energy.y, "x": daily_energy.x},
        )
        return daily_energy, total_energy, peak_energy, day_of_peak

    def _fake_transfer(arr, ref, extra_dim=None, attrs=None):
        out = arr.copy()
        if attrs:
            out.attrs.update(attrs)
        try:
            out = out.rio.write_crs(ref.rio.crs)
            out = out.rio.write_transform(ref.rio.transform())
        except Exception:
            pass
        return out

    def _fake_write(arr, path_str):
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        a = arr
        if "time" in a.dims:
            a = a.isel(time=0, drop=True)
        if "band" in a.dims and a.ndim == 3:
            pass
        try:
            a.rio.crs
        except Exception:
            a = a.rio.write_crs("EPSG:3413").rio.write_transform(from_origin(-1_000_000, 1_000_000, 2000, 2000))
        a.rio.to_raster(path_str)

    monkeypatch.setattr(cli_mod, "compute_slope_aspect_normals", _fake_normals, raising=True)
    monkeypatch.setattr(cli_mod, "compute_solar_ephem", _fake_ephem, raising=True)
    monkeypatch.setattr(cli_mod, "compute_flux_timeseries", _fake_flux, raising=True)
    monkeypatch.setattr(cli_mod, "compute_energy_metrics", _fake_metrics, raising=True)
    monkeypatch.setattr(cli_mod, "transfer_spatial_metadata", _fake_transfer, raising=True)
    monkeypatch.setattr(cli_mod, "write_geotiff", _fake_write, raising=True)

    outdir = tmp_path / "flux_out"
    result = runner.invoke(
        app,
        [
            "compute",
            "fluxseries",
            str(dem_path),
            str(mock_horizon_file),
            "--start-utc",
            "2025-01-01T00:00:00Z",
            "--stop-utc",
            "2025-01-01T06:00:00Z",
            "--output-dir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.stdout + "\n" + result.stderr

    stem = dem_path.stem
    for suffix in ["_DAILY_ENERGY.tif", "_TOTAL_ENERGY.tif", "_PEAK_ENERGY.tif", "_DAY_OF_PEAK.tif"]:
        f = outdir / f"{stem}{suffix}"
        assert f.exists(), f"{f} was not created"


def test_compute_fluxseries_uses_xy_when_dem_is_epsg4326(tmp_path, monkeypatch):
    """
    Cover the branch where DEM is EPSG:4326 so CLI uses x/y directly as lon/lat.
    """
    ny, nx = 20, 20
    dem_da = (
        xr.DataArray(np.ones((ny, nx), dtype=np.float32), dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)})
        .rio.write_crs("EPSG:4326")
        .rio.write_transform(from_origin(-120.0, 35.0, 0.01, 0.01))
    )

    # load_dem returns the same EPSG:4326 DEM for both inputs
    monkeypatch.setattr(cli_mod, "load_dem", lambda p: dem_da, raising=True)

    def _fake_normals(dem):
        slope = xr.zeros_like(dem)
        aspect = xr.zeros_like(dem)
        normal_enu = xr.DataArray(
            np.zeros((3, dem.sizes["y"], dem.sizes["x"]), dtype=np.float32),
            dims=("band", "y", "x"),
            coords={"band": [0, 1, 2], "y": dem.y, "x": dem.x},
        )
        return slope, aspect, normal_enu

    def _fake_ephem(lat, lon, startutc=None, stoputc=None, timestep=3600, cache_dir=None):
        n = 3
        times = np.array([np.datetime64("2025-01-01T00:00") + i * np.timedelta64(1, "h") for i in range(n)])
        alt = np.linspace(5.0, 7.0, n)
        az = np.linspace(0.0, 180.0, n, endpoint=False)
        au = np.ones(n, dtype=np.float32)
        enu = np.zeros((n, 3), dtype=np.float32)
        return times, alt, az, au, enu

    def _fake_flux(horizon, sun_alt_deg, sun_az_deg, sun_au, sun_enu, normal_enu, times_utc, batch_size, n_jobs):
        data = np.random.rand(times_utc.size, ny, nx).astype(np.float32)
        return xr.DataArray(data, dims=("time", "y", "x"), coords={"time": times_utc, "y": np.arange(ny), "x": np.arange(nx)})

    def _fake_metrics(flux_da):
        daily_energy = xr.DataArray(flux_da.values, dims=("time", "y", "x"), coords=flux_da.coords)
        total_energy = daily_energy.sum("time")
        peak_energy = daily_energy.max("time")
        day_of_peak = xr.DataArray(
            np.argmax(daily_energy.values, axis=0).astype(np.float32),
            dims=("y", "x"),
            coords={"y": daily_energy.y, "x": daily_energy.x},
        )
        return daily_energy, total_energy, peak_energy, day_of_peak

    def _fake_transfer(arr, ref, extra_dim=None, attrs=None):
        out = arr.copy()
        if attrs:
            out.attrs.update(attrs)
        try:
            out = out.rio.write_crs(ref.rio.crs).rio.write_transform(ref.rio.transform())
        except Exception:
            pass
        return out

    def _fake_write(arr, path_str):
        a = arr
        if "time" in a.dims:
            a = a.isel(time=0, drop=True)
        try:
            a.rio.crs
        except Exception:
            a = a.rio.write_crs("EPSG:4326").rio.write_transform(from_origin(-120.0, 35.0, 0.01, 0.01))
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        a.rio.to_raster(path_str)

    monkeypatch.setattr(cli_mod, "compute_slope_aspect_normals", _fake_normals, raising=True)
    monkeypatch.setattr(cli_mod, "compute_solar_ephem", _fake_ephem, raising=True)
    monkeypatch.setattr(cli_mod, "compute_flux_timeseries", _fake_flux, raising=True)
    monkeypatch.setattr(cli_mod, "compute_energy_metrics", _fake_metrics, raising=True)
    monkeypatch.setattr(cli_mod, "transfer_spatial_metadata", _fake_transfer, raising=True)
    monkeypatch.setattr(cli_mod, "write_geotiff", _fake_write, raising=True)

    outdir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "compute",
            "fluxseries",
            str(tmp_path / "dem.tif"),
            str(tmp_path / "horizon.tif"),
            "--start-utc",
            "2025-01-01T00:00:00Z",
            "--stop-utc",
            "2025-01-01T03:00:00Z",
            "--output-dir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, result.stdout + "\n" + result.stderr
    stem = "dem"
    for suffix in ["_DAILY_ENERGY.tif", "_TOTAL_ENERGY.tif", "_PEAK_ENERGY.tif", "_DAY_OF_PEAK.tif"]:
        assert (outdir / f"{stem}{suffix}").exists()


# ---------------------------------------------------------------------------
# Energy metrics plotting (save to PNG)
# ---------------------------------------------------------------------------


def test_plot_total_energy_cmd_saves_png(tmp_path, monkeypatch):
    """Plot total-energy and save PNG."""
    te_da = xr.DataArray(np.random.rand(10, 10).astype(np.float32), dims=("y", "x"))
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: te_da, raising=True)
    monkeypatch.setattr(viz_mod, "plot_total_energy", lambda arr, ax=None: ax.imshow(arr.values), raising=True)

    outdir = tmp_path / "plots"
    tif_path = tmp_path / "metric_total.tif"
    tif_path.write_text("stub")

    result = runner.invoke(app, ["plot", "total-energy", str(tif_path), "--output-dir", str(outdir)])
    assert result.exit_code == 0, result.output
    expected_png = outdir / (tif_path.stem + "_TOTAL_ENERGY.png")
    assert expected_png.exists(), f"{expected_png} was not created"


def test_plot_peak_energy_cmd_saves_png(tmp_path, monkeypatch):
    """Plot peak-energy and save PNG."""
    pe_da = xr.DataArray(np.random.rand(12, 8).astype(np.float32), dims=("y", "x"))
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: pe_da, raising=True)
    monkeypatch.setattr(viz_mod, "plot_peak_energy", lambda arr, ax=None: ax.imshow(arr.values), raising=True)

    outdir = tmp_path / "plots"
    tif_path = tmp_path / "metric_peak.tif"
    tif_path.write_text("stub")

    result = runner.invoke(app, ["plot", "peak-energy", str(tif_path), "--output-dir", str(outdir)])
    assert result.exit_code == 0, result.output
    expected_png = outdir / (tif_path.stem + "_PEAK_ENERGY.png")
    assert expected_png.exists(), f"{expected_png} was not created"


def test_plot_day_of_peak_cmd_saves_png(tmp_path, monkeypatch):
    """Plot day-of-peak using parsed daily_iso_times and save PNG."""
    dop_da = xr.DataArray(np.random.rand(8, 8).astype(np.float32), dims=("y", "x"))
    daily_iso = [f"2025-01-01T0{i}:00:00Z" for i in range(6)]
    dop_da.attrs["daily_iso_times"] = json.dumps(daily_iso)
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: dop_da, raising=True)

    def _fake_plot(arr, times_utc, sigma=9, ax=None):
        assert str(getattr(times_utc, "dtype", "")) in ("datetime64[s]", "datetime64[us]", "datetime64[ns]")
        ax.imshow(arr.values)

    monkeypatch.setattr(cli_mod, "plot_day_of_peak", _fake_plot, raising=True)

    outdir = tmp_path / "plots"
    tif_path = tmp_path / "metric_dop.tif"
    tif_path.write_text("stub")

    result = runner.invoke(app, ["plot", "day-of-peak", str(tif_path), "--output-dir", str(outdir)])
    assert result.exit_code == 0, result.output
    expected_png = outdir / (tif_path.stem + "_DAY_OF_PEAK.png")
    assert expected_png.exists(), f"{expected_png} was not created"


# ---------------------------------------------------------------------------
# Energy metrics plotting (show-mode)
# ---------------------------------------------------------------------------


def test_plot_total_energy_cmd_show(monkeypatch):
    """When no output-dir, total-energy should call plt.show()."""
    monkeypatch.delenv("SOLSHADE_TEST_MODE", raising=False)

    te_da = xr.DataArray(np.random.rand(5, 5).astype(np.float32), dims=("y", "x"))
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: te_da, raising=True)
    monkeypatch.setattr(viz_mod, "plot_total_energy", lambda arr, ax=None: ax.imshow(arr.values), raising=True)

    with patch("matplotlib.pyplot.show") as mock_show:
        result = runner.invoke(app, ["plot", "total-energy", "dummy.tif"])
        assert result.exit_code == 0, result.output
        mock_show.assert_called_once()


def test_plot_peak_energy_cmd_show(monkeypatch):
    """When no output-dir, peak-energy should call plt.show()."""
    monkeypatch.delenv("SOLSHADE_TEST_MODE", raising=False)

    pe_da = xr.DataArray(np.random.rand(6, 4).astype(np.float32), dims=("y", "x"))
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: pe_da, raising=True)
    monkeypatch.setattr(viz_mod, "plot_peak_energy", lambda arr, ax=None: ax.imshow(arr.values), raising=True)

    with patch("matplotlib.pyplot.show") as mock_show:
        result = runner.invoke(app, ["plot", "peak-energy", "dummy.tif"])
        assert result.exit_code == 0, result.output
        mock_show.assert_called_once()


def test_plot_day_of_peak_cmd_show(monkeypatch):
    """When no output-dir, day-of-peak should call plt.show()."""
    monkeypatch.delenv("SOLSHADE_TEST_MODE", raising=False)

    dop_da = xr.DataArray(np.random.rand(8, 8).astype(np.float32), dims=("y", "x"))
    daily_iso = [f"2025-01-01T0{i}:00:00Z" for i in range(6)]
    dop_da.attrs["daily_iso_times"] = json.dumps(daily_iso)
    monkeypatch.setattr(cli_mod, "read_geotiff", lambda p: dop_da, raising=True)

    def _fake_plot(arr, times_utc, sigma=9, ax=None):
        ax.imshow(arr.values)

    monkeypatch.setattr(cli_mod, "plot_day_of_peak", _fake_plot, raising=True)

    with patch("matplotlib.pyplot.show") as mock_show:
        result = runner.invoke(app, ["plot", "day-of-peak", "dummy.tif"])
        assert result.exit_code == 0, result.output
        mock_show.assert_called_once()
