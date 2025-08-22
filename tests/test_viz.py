import os

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from solshade.terrain import compute_slope_aspect_normals
from solshade.viz import (
    mirrored_discrete_doy_cmap,
    plot_aspect,
    plot_day_of_peak,
    plot_dem,
    plot_hillshade,
    plot_horizon_polar,
    plot_normals,
    plot_peak_energy,
    plot_slope,
    plot_total_energy,
    truncate_colormap,
)

# -----------------------------------------------------------------------------
# Global test config / fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _non_interactive_backend():
    """Force a non-interactive backend for all tests in this module."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Close all figures after each test to avoid max_open_warning."""
    yield
    plt.close("all")


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _rng():
    return np.random.default_rng(42)


def _make_dem(values: np.ndarray | None = None, shape=(10, 10), dx=1.0, dy=1.0) -> xr.DataArray:
    """Build a small synthetic DEM as an xarray.DataArray with x/y coords."""
    if values is None:
        values = _rng().random(shape)
    y = np.arange(values.shape[0]) * dy
    x = np.arange(values.shape[1]) * dx
    return xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])


def _simple_horizon(n: int = 36):
    az = np.linspace(0, 360, n, endpoint=False)  # 0..350
    hr = 5.0 + 2.0 * np.sin(np.deg2rad(az))
    return az, hr


def _fake_energy_map(shape=(10, 10), base=100.0):
    """Simple synthetic 2D map for energy values."""
    values = base + _rng().random(shape) * 10.0
    return xr.DataArray(values, coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])}, dims=["y", "x"])


# -----------------------------------------------------------------------------
# DEM / slope / aspect / normals / hillshade map plotting
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, title_match",
    [
        (lambda: ("dem", plot_dem, _make_dem()), "Digital Elevation Model"),
        (
            lambda: (
                "slope",
                plot_slope,
                compute_slope_aspect_normals(_make_dem(values=np.tile(np.arange(10), (10, 1))))[0],
            ),
            "Slope",
        ),
        (
            lambda: (
                "aspect",
                plot_aspect,
                compute_slope_aspect_normals(_make_dem(values=np.tile(np.arange(10), (10, 1))))[1],
            ),
            "Aspect",
        ),
        (
            lambda: (
                "normal",
                plot_normals,
                # Use the actual normals (index 2), not aspect.
                compute_slope_aspect_normals(_make_dem(values=np.tile(np.arange(10), (10, 1))))[2],
            ),
            "Normals: R->E, G->N, B->U",
        ),
    ],
)
def test_basic_map_plots(builder, title_match):
    """Basic sanity checks for DEM/slope/aspect/normals plotting APIs."""
    _name, fn, da = builder()
    ax = fn(da)
    assert isinstance(ax.figure, Figure)
    assert ax.get_title() == title_match


def test_plot_hillshade():
    """Hillshade title and figure creation."""
    dem = _make_dem(values=np.tile(np.arange(10), (10, 1)))
    slope, aspect, _ = compute_slope_aspect_normals(dem)
    ax = plot_hillshade(slope, aspect, azimuth_deg=270, altitude_deg=30)
    assert isinstance(ax.figure, Figure)
    assert "Hillshade" in ax.get_title()


# -----------------------------------------------------------------------------
# Polar horizon plotting (including solar overlay branches)
# -----------------------------------------------------------------------------


def test_plot_horizon_polar_basic():
    """Default call creates a polar axes and draws a closed curve."""
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 10 + 5 * np.sin(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)
    assert isinstance(ax, PolarAxes)


def test_plot_horizon_polar_closed_curve():
    """Horizon curve is closed (first/last sample equal)."""
    az = np.linspace(0, 360, 64, endpoint=False)
    vals = 15 + 10 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals)

    # Grab the line with the most points (the horizon curve)
    horizon_line = max(ax.lines, key=lambda line: len(line.get_xdata()))
    az_data = horizon_line.get_xdata()
    r_data = horizon_line.get_ydata()

    assert len(az_data) == len(az) + 1
    assert len(r_data) == len(vals) + 1
    assert np.isclose(az_data[0], az_data[-1])
    assert np.isclose(r_data[0], r_data[-1], rtol=1e-5, atol=1e-8)


def test_plot_horizon_polar_passes_external_axis():
    """Supplied axis is respected and returned."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    az = np.linspace(0, 360, 32, endpoint=False)
    vals = np.full_like(az, 5.0)
    out = plot_horizon_polar(az, vals, ax=ax)
    assert out is ax


def test_plot_horizon_polar_handles_negative_vals():
    """Negative/positive horizon values set r-limits on both sides of zero."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    az = np.linspace(0, 360, 128, endpoint=False)
    vals = -10 + 20 * np.cos(np.deg2rad(az))
    ax = plot_horizon_polar(az, vals, ax=ax)
    rmin, rmax = ax.get_rmin(), ax.get_rmax()  # type: ignore[attr-defined]
    assert rmin < 0 and rmax > 0


def test_polar_adds_cardinals_and_hides_ang_labels():
    """Cardinal labels present; angular tick labels hidden."""
    az, hr = _simple_horizon()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax = plot_horizon_polar(az, hr, ax=ax)

    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)
    assert all(t.get_text() == "" for t in ax.get_xticklabels())

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2 = plot_horizon_polar(az, hr, ax=ax2)
    labels2 = {t.get_text() for t in ax2.texts}
    assert {"N", "E", "S", "W"}.issubset(labels2)


def test_polar_with_solar_envelope_full_band():
    """Overlay with both min/max envelopes."""
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 72, endpoint=False)
    sunaltmin = 10.0 + 5.0 * np.sin(np.deg2rad(sunaz))
    sunaltmax = sunaltmin + 10.0

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=sunaltmin,
        sunaltmax=sunaltmax,
        ax=ax,
    )
    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)
    assert all(t.get_text() == "" for t in ax.get_xticklabels())


def test_polar_with_only_sunaltmin_hits_lo_branch():
    """Envelope with only `sunaltmin` (lo-only branch)."""
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 36, endpoint=False)
    sunaltmin = 10.0 + np.cos(np.deg2rad(sunaz))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=sunaltmin,
        sunaltmax=None,
        ax=ax,
    )
    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)


def test_polar_with_only_sunaltmax_hits_hi_branch():
    """Envelope with only `sunaltmax` (hi-only branch)."""
    az, hr = _simple_horizon()
    sunaz = np.linspace(0, 360, 36, endpoint=False)
    sunaltmax = 20.0 + np.sin(np.deg2rad(sunaz))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax = plot_horizon_polar(
        azimuths=az,
        horizon_vals=hr,
        sunaz=sunaz,
        sunaltmin=None,
        sunaltmax=sunaltmax,
        ax=ax,
    )
    labels = {t.get_text() for t in ax.texts}
    assert {"N", "E", "S", "W"}.issubset(labels)


# -----------------------------------------------------------------------------
# truncate_colormap
# -----------------------------------------------------------------------------


def test_truncate_colormap_happy_path_from_name():
    cmap = truncate_colormap("viridis", vmin=0.2, vmax=0.7, n=32)
    assert isinstance(cmap, mcolors.Colormap)
    samples = cmap(np.linspace(0, 1, 32))
    assert samples.shape == (32, 4)


def test_truncate_colormap_happy_path_from_instance_and_preserve_specials():
    base = plt.get_cmap("plasma").copy()
    base.set_bad((0, 0, 0, 0.0))
    base.set_under("cyan")
    base.set_over("magenta")

    cmap = truncate_colormap(base, vmin=0.1, vmax=0.9, n=16)

    out = cmap(np.ma.masked_invalid([np.nan]))[0]
    assert np.allclose(out, (0, 0, 0, 0.0))

    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    rgba = sm.to_rgba([-0.1, 0.5, 1.1])
    assert np.allclose(rgba[0], mcolors.to_rgba("cyan"))
    assert np.allclose(rgba[2], mcolors.to_rgba("magenta"))


@pytest.mark.parametrize(
    "vmin,vmax",
    [
        (-0.1, 0.8),
        (0.8, 0.8),
        (0.9, 0.1),
        (1.0, 1.1),
    ],
)
def test_truncate_colormap_invalid_ranges_raise(vmin, vmax):
    with pytest.raises(ValueError):
        truncate_colormap("viridis", vmin=vmin, vmax=vmax, n=16)


def test_truncate_colormap_invalid_n_raises():
    with pytest.raises(ValueError):
        truncate_colormap("viridis", vmin=0.2, vmax=0.8, n=0)


# -----------------------------------------------------------------------------
# mirrored_discrete_doy_cmap
# -----------------------------------------------------------------------------


def test_mirrored_discrete_basic_properties_and_center_color():
    rng = _rng()
    data = rng.normal(loc=200, scale=20, size=10_000).round()
    data = np.clip(data, 1, 366)

    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        sigma=3.0,
        cmap="viridis",
        center_color="whitesmoke",
        clip_to_data=True,
    )

    assert np.allclose(boundaries % 1.0, 0.5)
    n_bins = len(boundaries) - 1
    assert n_bins % 2 == 1
    assert lo < med < hi

    center_idx = n_bins // 2
    center_rgba_expected = mcolors.to_rgba("whitesmoke")
    assert np.allclose(listed.colors[center_idx], center_rgba_expected, atol=1e-6)

    mid_color = listed(listed.N // 2)
    mapped_color = listed(norm(med))
    assert np.allclose(mapped_color, mid_color)


def test_mirrored_discrete_lohi_override():
    data = np.array([150, 200, 250], dtype=float)
    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        cmap="plasma",
        lo_hi=(120, 280),
    )
    assert (lo, hi) == (120, 280)
    assert (len(boundaries) - 1) % 2 == 1
    assert abs(med - 200) <= 1


def test_mirrored_discrete_single_side_bin_edgecases():
    data = np.array([200, 201, 202, 203], dtype=float)
    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        sigma=0.1,
        cmap="viridis",
    )
    n_bins = len(boundaries) - 1
    assert n_bins % 2 == 1
    assert listed.N == n_bins


def test_mirrored_discrete_under_over_colors_set():
    data = np.linspace(100, 300, 100)
    listed, norm, boundaries, _ = mirrored_discrete_doy_cmap(
        data,
        sigma=2.0,
        cmap="viridis",
        under_color="black",
        over_color="white",
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=listed)
    vals = np.array([boundaries[0] - 10, (boundaries[0] + boundaries[-1]) / 2.0, boundaries[-1] + 10])
    rgba = sm.to_rgba(vals)
    assert np.allclose(rgba[0], mcolors.to_rgba("black"))
    assert np.allclose(rgba[2], mcolors.to_rgba("white"))


def test_mirrored_discrete_sigma_must_be_positive():
    data = np.array([100, 150, 200], dtype=float)
    with pytest.raises(ValueError):
        mirrored_discrete_doy_cmap(data, sigma=0.0)


def test_mirrored_discrete_no_finite_values_raises():
    a = np.array([np.nan, np.inf, -np.inf])
    with pytest.raises(ValueError, match="No finite values"):
        mirrored_discrete_doy_cmap(a, sigma=2.0)


def test_mirrored_discrete_accepts_colormap_instance_branch_else():
    """Pass a Colormap instance to cover the `else: base = cmap` branch."""
    cmap_instance = plt.get_cmap("viridis")
    data = np.arange(100, 110)

    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        sigma=2.0,
        cmap=cmap_instance,
        clip_to_data=True,
    )

    assert isinstance(listed, ListedColormap)
    assert boundaries.ndim == 1
    assert lo < med < hi


def test_mirrored_discrete_zero_std_forces_range_expansion():
    """
    std == 0 path + guard adjustments so we end with at least three bins.
    """
    data = np.full((4, 5), 200, dtype=float)
    cmap_instance = plt.get_cmap("plasma")

    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        sigma=1e-6,
        cmap=cmap_instance,
        clip_to_data=True,
    )

    assert (lo, med, hi) == (199, 200, 201)
    assert boundaries.size == (hi - lo + 1) + 1


def test_mirrored_discrete_even_bins_trigger_parity_fix():
    """
    Even requested bins -> function expands to make it odd.
    lo_hi=(9,12) -> 4 bins becomes 5 bins.
    """
    data = np.array([10.0, 12.0], dtype=float)

    listed, norm, boundaries, (lo, med, hi) = mirrored_discrete_doy_cmap(
        data,
        sigma=1.0,
        cmap="viridis",
        clip_to_data=False,
        lo_hi=(9, 12),
    )

    assert lo == 9 and med == int(np.rint(np.nanmedian(data))) and hi == 13
    assert (hi - lo + 1) % 2 == 1
    assert boundaries.size == (hi - lo + 1) + 1


# -----------------------------------------------------------------------------
# Energy/metrics plotting
# -----------------------------------------------------------------------------
# Use external axes to avoid creating extra figures.


def test_plot_total_energy_returns_ax():
    total_energy = _fake_energy_map()
    fig, ax = plt.subplots()
    ax = plot_total_energy(total_energy, ax=ax)
    assert isinstance(ax, Axes)
    assert "Total Energy" in ax.get_title()


def test_plot_peak_energy_returns_ax():
    peak_energy = _fake_energy_map(base=150.0)
    fig, ax = plt.subplots()
    ax = plot_peak_energy(peak_energy, ax=ax)
    assert isinstance(ax, Axes)
    assert "Peak Daily Energy" in ax.get_title()


def test_plot_day_of_peak_labels_and_colormap():
    """Colorbar labels are calendar-formatted and title is correct."""
    shape = (10, 10)
    doy = _rng().normal(loc=180, scale=20, size=shape).clip(1, 366).astype(int)
    day_of_peak = xr.DataArray(doy, coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])}, dims=["y", "x"])

    times = np.array(["2025-01-01T00:00:00"], dtype="datetime64[s]")

    fig, ax = plt.subplots()
    ax = plot_day_of_peak(day_of_peak, times, ax=ax)
    assert isinstance(ax, Axes)
    assert "Day of Peak Energy" in ax.get_title()

    # Ensure colorbar tick labels are calendar-formatted (e.g., "JUL 01")
    cbar = ax.figure.axes[-1]  # last axis is usually the colorbar
    tick_labels = [tick.get_text() for tick in cbar.get_yticklabels()]
    assert all(isinstance(label, str) and len(label.split()) == 2 for label in tick_labels)
    assert any(label.isupper() for label in tick_labels)


def test_plot_total_energy_creates_ax_when_none():
    total_energy = _fake_energy_map()
    ax = plot_total_energy(total_energy)  # no ax passed
    assert isinstance(ax, Axes)
    assert "Total Energy" in ax.get_title()


def test_plot_peak_energy_creates_ax_when_none():
    peak_energy = _fake_energy_map(base=150.0)
    ax = plot_peak_energy(peak_energy)  # no ax passed
    assert isinstance(ax, Axes)
    assert "Peak Daily Energy" in ax.get_title()


def test_plot_day_of_peak_creates_ax_when_none():
    doy = np.full((5, 5), 200, dtype=int)
    day_of_peak = xr.DataArray(doy, coords={"y": np.arange(5), "x": np.arange(5)}, dims=["y", "x"])
    times = np.array(["2025-01-01T00:00:00"], dtype="datetime64[s]")
    ax = plot_day_of_peak(day_of_peak, times)  # no ax passed
    assert isinstance(ax, Axes)
    assert "Day of Peak Energy" in ax.get_title()
