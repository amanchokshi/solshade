import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from solshade.terrain import compute_slope_aspect
from solshade.viz import plot_aspect, plot_dem, plot_hillshade, plot_horizon_polar, plot_slope


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


def test_plot_horizon_polar_basic():
    """Test that the function returns a PolarAxes object and does not error."""
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 10 + 5 * np.sin(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)
    assert isinstance(ax, PolarAxes)


def test_plot_horizon_polar_closed_curve():
    """Test that the horizon curve is closed (wraps back to starting point)."""
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 15 + 10 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)

    # Find the line with the most points (should be the horizon line)
    horizon_line = max(ax.lines, key=lambda line: len(line.get_xdata()))
    az_data = horizon_line.get_xdata()
    r_data = horizon_line.get_ydata()

    # Should have one more point than input
    assert len(az_data) == len(az) + 1
    assert len(r_data) == len(vals) + 1

    # First and last points match (closed curve)
    assert np.isclose(az_data[0], az_data[-1])
    assert np.isclose(r_data[0], r_data[-1], rtol=1e-5, atol=1e-8)


def test_plot_horizon_polar_passes_external_axis():
    """Ensure function uses provided axis rather than making a new one."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    az = np.linspace(0, 360, 32, endpoint=False)
    vals = np.full_like(az, 5)
    ax_out = plot_horizon_polar(az, vals, ax=ax)
    assert ax_out is ax


def test_plot_horizon_polar_handles_negative_vals():
    """Ensure that negative horizon values do not break the plot."""
    az = np.linspace(0, 360, 128, endpoint=False)
    vals = -10 + 20 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)

    # Ensure the radial limits span the negative values
    rmin, rmax = ax.get_rmin(), ax.get_rmax()  # type: ignore[attr-defined]
    assert rmin < 0
    assert rmax > 0
