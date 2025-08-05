import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from solshade.terrain import compute_slope_aspect
from solshade.viz import plot_aspect, plot_dem, plot_hillshade, plot_slope


def create_test_dem(shape=(10, 10), dx=1.0, dy=1.0, values=None) -> xr.DataArray:
    """
    Create a synthetic test DEM for plotting purposes.

    Parameters
    ----------
    shape : tuple of int, optional
        The shape of the DEM grid as (rows, columns). Default is (10, 10).
    dx : float, optional
        Spatial resolution in the x-direction (e.g., meters/pixel). Default is 1.0.
    dy : float, optional
        Spatial resolution in the y-direction (e.g., meters/pixel). Default is 1.0.
    values : np.ndarray or None, optional
        Optional array of elevation values. If None, random values are generated.

    Returns
    -------
    xr.DataArray
        A 2D xarray DataArray with 'x' and 'y' coordinates and synthetic elevation data.
    """
    if values is None:
        values = np.random.rand(*shape)

    x = np.arange(shape[1]) * dx
    y = np.arange(shape[0]) * dy
    return xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])


def test_plot_dem():
    """
    Test plotting of a synthetic DEM.

    Asserts that the resulting axis is a matplotlib Figure
    and has the correct title.
    """
    dem = create_test_dem()
    ax = plot_dem(dem)
    assert isinstance(ax.figure, Figure)
    assert ax.get_title() == "Digital Elevation Model"


def test_plot_slope():
    """
    Test plotting of slope from a synthetic DEM.

    Uses a linear DEM increasing eastward and checks
    that the returned axis has the expected plot title.
    """
    dem = create_test_dem(values=np.tile(np.arange(10), (10, 1)))
    slope, _ = compute_slope_aspect(dem)
    ax = plot_slope(slope)
    assert isinstance(ax.figure, Figure)
    assert ax.get_title() == "Slope"


def test_plot_aspect():
    """
    Test plotting of aspect from a synthetic DEM.

    Uses a linear DEM increasing eastward and verifies
    the resulting axis has the correct title.
    """
    dem = create_test_dem(values=np.tile(np.arange(10), (10, 1)))
    _, aspect = compute_slope_aspect(dem)
    ax = plot_aspect(aspect)
    assert isinstance(ax.figure, Figure)
    assert ax.get_title() == "Aspect"


def test_plot_hillshade():
    """
    Test hillshade plotting using slope and aspect from a synthetic DEM.

    Uses a DEM increasing eastward, computes slope and aspect,
    and ensures that hillshade is plotted with the expected title.
    """
    dem = create_test_dem(values=np.tile(np.arange(10), (10, 1)))
    slope, aspect = compute_slope_aspect(dem)
    ax = plot_hillshade(slope, aspect, azimuth_deg=270, altitude_deg=30)
    assert isinstance(ax.figure, Figure)
    assert "Hillshade" in ax.get_title()
