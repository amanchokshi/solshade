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

from solshade.cli import app

runner = CliRunner()

# Dummy path for test data -- replace with actual test DEM path
TEST_DATA = Path("tests/data/MARS.tif")


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


def test_compute_horizon_subcommand():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            [
                "solshade",
                "compute",
                "horizon",
                str(TEST_DATA),
                "--output-dir",
                str(outdir),
                "--n-directions",
                "4",
                "--max-distance",
                "100",
                "--step",
                "25",
                "--chunk-size",
                "64",
                "--no-progress",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Horizon map compute failed with stderr: {result.stderr}"
        expected_file = outdir / f"{TEST_DATA.stem}_HORIZON_4.tif"
        assert expected_file.exists(), f"{expected_file} was not created"


@pytest.mark.parametrize("subcommand", ["slope", "aspect", "hillshade", "dem"])
def test_plot_subcommands_show(subcommand):
    env = os.environ.copy()
    env["SOLSHADE_TEST_MODE"] = "1"
    result = subprocess.run(
        ["solshade", "plot", subcommand, str(TEST_DATA)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"{subcommand} failed: {result.stderr}"


def test_meta_subcommand_output():
    result = subprocess.run(["solshade", "meta", str(TEST_DATA)], capture_output=True, text=True)
    assert result.returncode == 0, f"`meta` failed: {result.stderr}"
    out = result.stdout
    assert "METADATA:" in out
    assert "CRS:" in out
    assert "SHAPE:" in out
    assert "RESOLUTION:" in out
    assert "TRANSFORM:" in out
    assert "BOUNDS:" in out
    assert "COORDS:" in out
    assert "DTYPE:" in out
    assert "ATTRIBUTES:" in out
    assert "EPSG:3413" in out
    assert "1250, 1250" in out
    assert "2.00" in out


def test_attribute_truncation():
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


@pytest.fixture
def mock_horizon_file(tmp_path):
    ny, nx = 50, 50
    n_directions = 360
    transform = from_origin(-1000000, 1000000, 2000, 2000)
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


def test_plot_horizon_cmd_within_bounds(mock_horizon_file, tmp_path):
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
    result = runner.invoke(app, ["plot", "horizon", "--lat", "20.0", "--lon", "20.0", str(mock_horizon_file)])
    assert result.exit_code != 0


def test_plot_horizon_cmd_show(monkeypatch, mock_horizon_file):
    from unittest.mock import patch

    from rasterio.transform import from_origin

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
    monkeypatch.setenv("SOLSHADE_TEST_MODE", "1")

    # stub compute_solar_altaz
    def _fake_compute(lat, lon, startutc=None, stoputc=None, timestep=3600, cache_dir=None):
        n = 24
        times = np.array([np.datetime64("2025-01-01T00:00") + np.timedelta64(i, "h") for i in range(n)])
        alt = np.linspace(5.0, 25.0, n)
        az = (np.linspace(0, 360, n, endpoint=False) + 7.0) % 360.0
        enu_unit = np.zeros((az.size, 3))
        assert startutc is None or startutc.tzinfo is not None
        assert stoputc is None or stoputc.tzinfo is not None
        return times, alt, az, enu_unit

    def _fake_envelope(times_utc, alt_deg, az_deg, smooth_n=360):
        az_plot = np.array([0, 90, 180, 270, 360], dtype=float)
        min_plot = np.array([0, 5, 10, 5, 0], dtype=float)
        max_plot = np.array([10, 15, 20, 15, 10], dtype=float)
        return az_plot, min_plot, max_plot

    from rasterio.transform import from_origin

    import solshade.cli as cli_mod
    import solshade.solar as solar_mod

    monkeypatch.setattr(cli_mod, "compute_solar_altaz", _fake_compute, raising=True)
    monkeypatch.setattr(solar_mod, "solar_envelope_by_folding", _fake_envelope, raising=True)

    # Compute an in-bounds lat/lon at the raster center
    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)  # center of 50x50
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


def test_plot_horizon_cmd_with_naive_startutc(monkeypatch, mock_horizon_file, tmp_path):
    # Stub out heavy functions
    monkeypatch.setenv("SOLSHADE_TEST_MODE", "1")
    import solshade.cli as cli_mod
    import solshade.solar as solar_mod

    monkeypatch.setattr(
        cli_mod, "compute_solar_altaz", lambda *a, **k: (np.array([]), np.array([]), np.array([]), np.array([]))
    )
    monkeypatch.setattr(solar_mod, "solar_envelope_by_folding", lambda *a, **k: (np.array([]), np.array([]), np.array([])))

    # In-bounds lat/lon
    from pyproj import Transformer
    from rasterio.transform import from_origin

    transform = from_origin(-1_000_000, 1_000_000, 2000, 2000)
    center_x, center_y = transform * (25, 25)
    lon, lat = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True).transform(center_x, center_y)

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
        "2025-01-01T00:00:00",  # no timezone -> triggers tz attach
        "--stoputc",
        "2025-01-01T01:00:00",  # no timezone
        "--timestep",
        "3600",
        "--output-dir",
        str(outdir),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0


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
    """`solshade plot normals` without --output-dir runs in show mode; use test flag to avoid GUI."""
    env = os.environ.copy()
    env["SOLSHADE_TEST_MODE"] = "1"  # prevent plt.show()
    result = subprocess.run(
        ["solshade", "plot", "normals", str(TEST_DATA)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"plot normals show-mode failed: {result.stderr}"
