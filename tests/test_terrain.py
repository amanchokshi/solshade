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

from solshade.terrain import compute_slope_aspect, load_dem


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
