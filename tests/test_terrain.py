"""
Unit tests for the `terrain` module in the `solshade` package.

These tests cover:
- Loading DEMs from disk or mocks
- Validating slope and aspect calculation on synthetic terrain
- Handling of erroneous inputs
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine

from solshade.terrain import compute_hillshade, compute_slope_aspect, load_dem


def test_load_valid_dem():
    """
    Test loading a valid single-band DEM.

    Asserts
    -------
    - Returned object is a 2D xarray.DataArray.
    """
    path = Path("tests/data/mars_dem.tif")
    dem = load_dem(path)

    assert isinstance(dem, xr.DataArray)
    assert dem.ndim == 2


def test_load_dem_raises_type_error():
    """
    Test that loading a multi-band raster raises a TypeError.

    Uses mocking to simulate rioxarray returning an xarray.Dataset after squeezing.

    Asserts
    -------
    - A TypeError is raised with an informative message.
    """
    with patch("rioxarray.open_rasterio") as mock_open:
        mock_open.return_value.squeeze.return_value = xr.Dataset()

        with pytest.raises(TypeError, match="DEM is not a single-band raster"):
            load_dem("dummy/path.tif")


def create_synthetic_dem(shape=(5, 5), dx=1.0, dy=1.0, values=None) -> xr.DataArray:
    """
    Helper to create synthetic DEMs for testing.

    Parameters
    ----------
    shape : tuple of int
        Size of the DEM grid in (rows, columns).
    dx : float
        Spatial resolution in the x direction (e.g. meters per pixel).
    dy : float
        Spatial resolution in the y direction (e.g. meters per pixel).
    values : ndarray, optional
        2D array of elevation values. Defaults to all zeros.

    Returns
    -------
    dem : xarray.DataArray
        Synthetic DEM with spatial metadata (CRS and transform).
    """
    if values is None:
        values = np.zeros(shape)

    dem = xr.DataArray(values, dims=["y", "x"])
    dem.rio.write_crs("EPSG:32633", inplace=True)
    dem.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    transform = Affine.translation(0, 0) * Affine.scale(dx, -dy)
    dem.rio.write_transform(transform, inplace=True)

    return dem


def test_flat_dem():
    """
    Test slope/aspect on a completely flat terrain.

    Asserts
    -------
    - Slope is zero everywhere.
    - Aspect is either undefined (NaN) or zero.
    """
    dem = create_synthetic_dem()
    slope, aspect = compute_slope_aspect(dem)

    assert np.allclose(slope.values, 0), "Slope should be 0 for flat DEM"
    assert np.isnan(aspect.values).all() or np.allclose(aspect.values, 0), (
        "Aspect should be undefined or 0 for flat terrain"
    )


def test_linear_slope_east():
    """
    Test slope/aspect on a DEM that increases eastward (x direction).

    Asserts
    -------
    - Slope is constant across the DEM.
    - Aspect is 270° (downslope direction: west).
    """
    values = np.tile(np.arange(5), (5, 1))  # Increasing along x-axis
    dem = create_synthetic_dem(values=values)
    slope, aspect = compute_slope_aspect(dem)

    assert np.allclose(slope.values, slope.values[0, 0]), "Slope should be constant"
    assert np.allclose(aspect.values, 270), "Aspect should be 270° (downslope west)"


def test_linear_slope_north():
    """
    Test slope/aspect on a DEM that increases northward (y direction).

    Asserts
    -------
    - Slope is constant across the DEM.
    - Aspect is 0° (downslope direction: north).
    """
    values = np.tile(np.arange(5), (5, 1)).T  # Increasing along y-axis
    dem = create_synthetic_dem(values=values)
    slope, aspect = compute_slope_aspect(dem)

    assert np.allclose(slope.values, slope.values[0, 0]), "Slope should be constant"
    assert np.allclose(aspect.values, 0), "Aspect should be 0° (downslope north)"


def test_hillshade_shape_and_range():
    """Hillshade output should match input shape and values should be between 0 and 1."""
    slope = xr.DataArray([[30, 45], [60, 75]], dims=["y", "x"])
    aspect = xr.DataArray([[0, 90], [180, 270]], dims=["y", "x"])
    hillshade = compute_hillshade(slope, aspect)

    assert hillshade.shape == slope.shape
    assert np.all((hillshade >= 0) & (hillshade <= 1))


def test_flat_slope_hillshade():
    """Flat terrain should result in constant hillshade equal to cos(altitude)."""
    slope = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    hillshade = compute_hillshade(slope, aspect, azimuth_deg=0, altitude_deg=60)

    expected = np.cos(np.radians(30))  # 60° altitude → zenith angle = 30°
    assert np.allclose(hillshade.values, expected)


def test_slope_only_hillshade_decreases():
    """Steeper slopes should result in lower hillshade for fixed lighting."""
    slope_low = xr.DataArray([[30]], dims=["y", "x"])
    slope_high = xr.DataArray([[80]], dims=["y", "x"])
    aspect = xr.DataArray([[0]], dims=["y", "x"])
    h_low = compute_hillshade(slope_low, aspect)
    h_high = compute_hillshade(slope_high, aspect)

    assert h_low.values > h_high.values


def test_invalid_dimensions_raise():
    """Raise error if slope/aspect inputs have mismatched shapes."""
    slope = xr.DataArray(np.zeros((3, 3)), dims=["y", "x"])
    aspect = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    with pytest.raises(xr.AlignmentError, match="conflicting dimension sizes"):
        compute_hillshade(slope, aspect)
