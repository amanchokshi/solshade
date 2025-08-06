"""
Unit tests for the `terrain` module in the `solshade` package.

These tests verify:
- DEM loading and error handling
- Slope, aspect, and hillshade computation on synthetic terrain
- Horizon map generation including edge cases (NaNs, flatness, slope)
- Metadata preservation and parallelization behavior

Synthetic DEMs are used throughout to ensure test coverage and speed.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

from solshade.terrain import compute_hillshade, compute_horizon_map, compute_slope_aspect, load_dem

# -------------------------
# Elevation functions
# -------------------------


def flat_elevation(y, x):
    return np.zeros_like(x)


def constant_elevation(y, x):
    return np.ones_like(x) * 100


def slope_east(y, x):
    return x.astype(np.float32)


def slope_north(y, x):
    return y.astype(np.float32)


# -------------------------
# DEM creation helpers
# -------------------------


def create_mock_dem(shape=(32, 32), elevation_func=flat_elevation):
    """Create a synthetic DEM with CRS and transform."""
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
    """Fixture for a small flat DEM with CRS and transform."""
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


# -------------------------
# Slope & aspect tests
# -------------------------


def test_flat_dem():
    dem = create_mock_dem()
    slope, aspect = compute_slope_aspect(dem)
    assert np.allclose(slope.values, 0)
    assert np.isnan(aspect.values).all() or np.allclose(aspect.values, 0)


def test_linear_slope_east():
    dem = create_mock_dem(elevation_func=slope_east)
    slope, aspect = compute_slope_aspect(dem)
    assert np.allclose(slope.values, slope.values[0, 0])
    assert np.allclose(aspect.values, 270)


def test_linear_slope_north():
    dem = create_mock_dem(elevation_func=slope_north)
    slope, aspect = compute_slope_aspect(dem)
    assert np.allclose(slope.values, slope.values[0, 0])
    assert np.allclose(aspect.values, 0)


# -------------------------
# Hillshade tests
# -------------------------


def test_hillshade_shape_and_range():
    slope = xr.DataArray([[30, 45], [60, 75]], dims=["y", "x"])
    aspect = xr.DataArray([[0, 90], [180, 270]], dims=["y", "x"])
    hillshade = compute_hillshade(slope, aspect)
    assert hillshade.shape == slope.shape
    assert np.all((hillshade >= 0) & (hillshade <= 1))


def test_flat_slope_hillshade():
    slope = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    hillshade = compute_hillshade(slope, aspect, azimuth_deg=0, altitude_deg=60)
    expected = np.cos(np.radians(30))
    assert np.allclose(hillshade.values, expected)


def test_slope_only_hillshade_decreases():
    slope_low = xr.DataArray([[30]], dims=["y", "x"])
    slope_high = xr.DataArray([[80]], dims=["y", "x"])
    aspect = xr.DataArray([[0]], dims=["y", "x"])
    h_low = compute_hillshade(slope_low, aspect)
    h_high = compute_hillshade(slope_high, aspect)
    assert h_low.values > h_high.values


def test_invalid_dimensions_raise():
    slope = xr.DataArray(np.zeros((3, 3)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    with pytest.raises(xr.AlignmentError, match="conflicting dimension sizes"):
        compute_hillshade(slope, aspect)


# -------------------------
# Horizon map tests
# -------------------------


def test_output_shape_and_dimensions():
    dem = create_mock_dem((16, 16))
    result = compute_horizon_map(dem, n_directions=8, max_distance=100, step=20, chunk_size=8, progress=False)
    assert result.dims == ("y", "x", "azimuth")
    assert result.shape == (16, 16, 8)
    assert np.allclose(result.azimuth.values, np.linspace(0, 360, 8, endpoint=False))


def test_flat_dem_returns_zero_horizon():
    dem = create_mock_dem((8, 8), elevation_func=constant_elevation)
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert np.allclose(result.values[~np.isnan(result.values)], 0, atol=1e-3)


def test_nan_handling_in_dem():
    dem = create_mock_dem((8, 8))
    dem.values[2:4, 2:4] = np.nan
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert np.all(np.isnan(result.values[2:4, 2:4]))


def test_slope_generates_nonzero_horizon():
    dem = create_mock_dem((8, 8), elevation_func=slope_east)
    result = compute_horizon_map(dem, n_directions=8, max_distance=100, step=20, chunk_size=4, progress=False)
    east_idx = np.argmin(np.abs(result.azimuth.values - 90))
    assert np.nanmean(result.values[:, :, east_idx]) > 0


def test_crs_and_metadata_retained():
    dem = create_mock_dem((16, 16))
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, progress=False)
    assert result.rio.crs == dem.rio.crs
    assert result.attrs["units"] == "degrees"
    assert "max_distance_m" in result.attrs
    assert len(result.attrs["azimuths_deg"]) == 4


def test_serial_execution_with_small_chunks():
    dem = create_mock_dem((16, 16))
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=20, chunk_size=4, n_jobs=1, progress=False)
    assert isinstance(result, xr.DataArray)


def test_parallel_execution_with_large_chunks():
    dem = create_mock_dem((32, 32))
    result = compute_horizon_map(dem, n_directions=16, max_distance=200, step=20, chunk_size=16, n_jobs=2, progress=False)
    assert isinstance(result, xr.DataArray)


def test_missing_crs_or_transform_raises():
    dem = xr.DataArray(np.zeros((8, 8)), dims=("y", "x"))
    with pytest.raises(ValueError, match="CRS and affine transform"):
        compute_horizon_map(dem)


def test_compute_horizon_map_with_progress(synthetic_dem):
    result = compute_horizon_map(synthetic_dem, n_directions=8, max_distance=200, step=50, chunk_size=2, progress=True)
    assert isinstance(result, xr.DataArray)
    assert "azimuth" in result.dims


@patch("solshade.terrain.Progress")
def test_with_progress_bar(mock_progress, synthetic_dem):
    result = compute_horizon_map(synthetic_dem, n_directions=8, max_distance=200, step=50, chunk_size=2, progress=True)
    assert result.ndim == 3


def test_output_metadata_preserved(synthetic_dem):
    result = compute_horizon_map(synthetic_dem, n_directions=4, max_distance=100, step=50, chunk_size=2, progress=False)
    assert result.rio.crs == synthetic_dem.rio.crs
    assert result.attrs["n_directions"] == 4
    assert result.attrs["units"] == "degrees"
    assert "azimuths_deg" in result.attrs


def test_nan_propagation():
    dem = xr.DataArray(
        np.array([[10, 20], [np.nan, 40]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    dem.rio.write_crs("EPSG:32633", inplace=True)
    dem.rio.write_transform(Affine(1, 0, 0, 0, -1, 0), inplace=True)
    result = compute_horizon_map(dem, n_directions=4, max_distance=100, step=50, chunk_size=1, progress=False)
    assert np.isnan(result.sel(x=0, y=1, azimuth=0, method="nearest"))
