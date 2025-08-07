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
