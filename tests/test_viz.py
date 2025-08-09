import os

import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from solshade.terrain import compute_slope_aspect
from solshade.viz import (
    plot_aspect,
    plot_dem,
    plot_hillshade,
    plot_horizon_polar,
    plot_slope,
)

# -----------------------------------------------------------------------------
# Test helpers / fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _non_interactive_backend():
    # Ensure tests don't try to pop up a window anywhere
    os.environ.setdefault("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=True)


def _make_dem(values: np.ndarray | None = None, shape=(10, 10), dx=1.0, dy=1.0) -> xr.DataArray:
    """
    Build a small synthetic DEM as an xarray.DataArray with x/y coords.
    """
    if values is None:
        values = np.random.rand(*shape)
    y = np.arange(values.shape[0]) * dy
    x = np.arange(values.shape[1]) * dx
    return xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])


def _simple_horizon(n: int = 36):
    az = np.linspace(0, 360, n, endpoint=False)  # 0..350
    hr = 5.0 + 2.0 * np.sin(np.deg2rad(az))
    return az, hr


# -----------------------------------------------------------------------------
# DEM / slope / aspect / hillshade map plotting
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, title_match",
    [
        (lambda: ("dem", plot_dem, _make_dem()), "Digital Elevation Model"),
        (
            lambda: (
                "slope",
                plot_slope,
                compute_slope_aspect(_make_dem(values=np.tile(np.arange(10), (10, 1))))[0],
            ),
            "Slope",
        ),
        (
            lambda: (
                "aspect",
                plot_aspect,
                compute_slope_aspect(_make_dem(values=np.tile(np.arange(10), (10, 1))))[1],
            ),
            "Aspect",
        ),
    ],
)
def test_basic_map_plots(builder, title_match):
    _name, fn, da = builder()
    ax = fn(da)
    assert isinstance(ax.figure, Figure)
    assert ax.get_title() == title_match


def test_plot_hillshade():
    dem = _make_dem(values=np.tile(np.arange(10), (10, 1)))
    slope, aspect = compute_slope_aspect(dem)
    ax = plot_hillshade(slope, aspect, azimuth_deg=270, altitude_deg=30)
    assert isinstance(ax.figure, Figure)
    assert "Hillshade" in ax.get_title()


# -----------------------------------------------------------------------------
# Polar horizon plotting (including solar overlay branches)
# -----------------------------------------------------------------------------


def test_plot_horizon_polar_basic():
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 10 + 5 * np.sin(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)
    assert isinstance(ax, PolarAxes)


def test_plot_horizon_polar_closed_curve():
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 15 + 10 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)

    # Grab the line with the most points (the horizon curve)
    horizon_line = max(ax.lines, key=lambda line: len(line.get_xdata()))
    az_data = horizon_line.get_xdata()
    r_data = horizon_line.get_ydata()

    # Closed curve: 1 extra point and first==last
    assert len(az_data) == len(az) + 1
    assert len(r_data) == len(vals) + 1
    assert np.isclose(az_data[0], az_data[-1])
    assert np.isclose(r_data[0], r_data[-1], rtol=1e-5, atol=1e-8)


def test_plot_horizon_polar_passes_external_axis():
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    az = np.linspace(0, 360, 32, endpoint=False)
    vals = np.full_like(az, 5.0)
    out = plot_horizon_polar(az, vals, ax=ax)
    assert out is ax


def test_plot_horizon_polar_handles_negative_vals():
    az = np.linspace(0, 360, 128, endpoint=False)
    vals = -10 + 20 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)
    rmin, rmax = ax.get_rmin(), ax.get_rmax()  # type: ignore[attr-defined]
    assert rmin < 0
    assert rmax > 0


def test_polar_adds_cardinals_and_hides_ang_labels():
    az, hr = _simple_horizon()
    ax = plot_horizon_polar(az, hr)

    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)
    assert all(t.get_text() == "" for t in ax.get_xticklabels())

    fig, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2 = plot_horizon_polar(az, hr, ax=ax2)
    labels2 = {t.get_text() for t in ax2.texts}
    assert {"N", "E", "S", "W"}.issubset(labels2)


def test_polar_with_solar_envelope_full_band():
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 72, endpoint=False)
    sunaltmin = 10.0 + 5.0 * np.sin(np.deg2rad(sunaz))
    sunaltmax = sunaltmin + 10.0

    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=sunaltmin,
        sunaltmax=sunaltmax,
    )
    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)
    assert all(t.get_text() == "" for t in ax.get_xticklabels())


def test_polar_with_only_sunaltmin_hits_lo_branch():
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 36, endpoint=False)
    sunaltmin = 10.0 + np.cos(np.deg2rad(sunaz))

    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=sunaltmin,
        sunaltmax=None,  # Force lo-only branch
    )

    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)


def test_polar_with_only_sunaltmax_hits_hi_branch():
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 36, endpoint=False)
    sunaltmax = 20.0 + np.sin(np.deg2rad(sunaz))

    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=None,
        sunaltmax=sunaltmax,  # Force hi-only branch
    )

    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)
