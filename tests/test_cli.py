import os
import subprocess
import tempfile
from pathlib import Path

import pytest

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


# @pytest.mark.parametrize("subcommand", ["slope", "aspect", "hillshade", "dem"])
# def test_plot_subcommands_show(subcommand):
#     """Ensure CLI runs interactively when no output-dir is specified (simulated via env)."""
#     result = subprocess.run(
#         ["solshade", "plot", subcommand, str(TEST_DATA)],
#         capture_output=True,
#         text=True,
#         env={**os.environ, "SOLSHADE_TEST_MODE": "1"},
#     )
#     assert result.returncode == 0, f"{subcommand} failed: {result.stderr}"


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
