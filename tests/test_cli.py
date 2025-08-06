import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from affine import Affine
from pyproj import CRS
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


def test_compute_subcommand_creates_expected_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            [
                "solshade",
                "compute",
                str(TEST_DATA),
                "--output-dir",
                str(outdir),
                "--slope",
                "--aspect",
                "--hillshade",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Compute failed with stderr: {result.stderr}"
        expected_files = [
            outdir / f"{TEST_DATA.stem}_SLOPE.tif",
            outdir / f"{TEST_DATA.stem}_ASPECT.tif",
            outdir / f"{TEST_DATA.stem}_HILLSHADE_315_45.tif",
        ]
        for file in expected_files:
            assert file.exists(), f"{file} was not created"


def test_compute_horizon_map_subcommand():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        result = subprocess.run(
            [
                "solshade",
                "compute",
                str(TEST_DATA),
                "--output-dir",
                str(outdir),
                "--horizon-map",
                "--n-directions",
                "4",  # reduce load for test
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
        expected_file = outdir / f"{TEST_DATA.stem}_HORIZON.tif"
        assert expected_file.exists(), f"{expected_file} was not created"


@pytest.mark.parametrize("subcommand", ["slope", "aspect", "hillshade", "dem"])
def test_plot_subcommands_show(subcommand):
    """Simulate interactive mode (no output-dir) and suppress plt.show() with env var."""
    env = os.environ.copy()
    env["SOLSHADE_TEST_MODE"] = "1"  # This disables plt.show() inside the CLI

    result = subprocess.run(
        ["solshade", "plot", subcommand, str(TEST_DATA)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"{subcommand} failed: {result.stderr}"


def test_meta_subcommand_output():
    """Test that the `meta` command runs and outputs expected metadata summary."""
    result = subprocess.run(
        ["solshade", "meta", str(TEST_DATA)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"`meta` failed: {result.stderr}"

    out = result.stdout

    # Check for key metadata fields
    assert "METADATA:" in out
    assert "CRS:" in out
    assert "SHAPE:" in out
    assert "RESOLUTION:" in out
    assert "TRANSFORM:" in out
    assert "BOUNDS:" in out
    assert "COORDS:" in out
    assert "DTYPE:" in out
    assert "ATTRIBUTES:" in out

    # Check for some known values from the test DEM
    assert "EPSG:3413" in out
    assert "1250 x 1250" in out
    assert "2.00" in out  # Resolution


def test_attribute_truncation():
    """Ensure long string-like attribute values are truncated and tagged."""
    mock_dem = MagicMock()
    mock_transform = Affine(2.0, 0.0, -824430.0, 0.0, -2.0, -801210.0)
    mock_dem.rio.transform.return_value = mock_transform
    mock_dem.rio.bounds.return_value = (-1, -1, 1, 1)
    mock_dem.shape = (100, 100)
    mock_dem.coords = {"x": None, "y": None}
    mock_dem.dtype = "float32"
    mock_dem.attrs = {"azimuths_deg": "[" + "1234567890" * 10 + "]"}

    # Use a real CRS (not a MagicMock)
    mock_dem.rio.crs = CRS.from_epsg(4326)

    with patch("solshade.cli.load_dem", return_value=mock_dem):
        result = runner.invoke(app, ["meta", "dummy.tif"])
        assert result.exit_code == 0
        assert "AZIMUTHS_DEG" in result.stdout
        assert "(truncated)" in result.stdout
