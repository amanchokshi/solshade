from typing import Optional, Tuple

import cmasher as cmr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, to_rgba

from solshade.terrain import compute_hillshade
from solshade.utils import get_logger

log = get_logger(__name__)


def truncate_colormap(
    cmap: str | Colormap,
    vmin: float = 0.0,
    vmax: float = 1.0,
    n: int = 256,
) -> Colormap:
    """
    Create a new colormap from a subrange of an existing colormap.

    Parameters
    ----------
    cmap : str or Colormap
        The original colormap (name or instance).
    vmin, vmax : float
        Fractional range of the original colormap to use (0.0–1.0, with vmin < vmax).
    n : int
        Number of discrete colors in the new map.

    Returns
    -------
    new_cmap : Colormap
        The truncated colormap.

    Notes
    -----
    - Preserves the base colormap's special colors: 'bad', 'under', and 'over'.
    """
    log.debug("truncate_colormap(cmap=%r, vmin=%s, vmax=%s, n=%s)", cmap, vmin, vmax, n)

    # Resolve cmap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        log.debug("Resolved string cmap to Colormap: %s", cmap.name)

    # STRICT validation (no pre-clipping)
    if not (0.0 <= vmin < vmax <= 1.0):
        raise ValueError("vmin and vmax must satisfy 0.0 <= vmin < vmax <= 1.0")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Build truncated colors and new map
    new_colors = cmap(np.linspace(vmin, vmax, n))
    new = mcolors.LinearSegmentedColormap.from_list(f"{cmap.name}_{vmin:.3f}_{vmax:.3f}", new_colors)

    # Preserve special colors if present on the base cmap
    if hasattr(cmap, "_rgba_bad") and getattr(cmap, "_rgba_bad") is not None:
        new.set_bad(getattr(cmap, "_rgba_bad"))
    if hasattr(cmap, "_rgba_under") and getattr(cmap, "_rgba_under") is not None:
        new.set_under(getattr(cmap, "_rgba_under"))
    if hasattr(cmap, "_rgba_over") and getattr(cmap, "_rgba_over") is not None:
        new.set_over(getattr(cmap, "_rgba_over"))

    log.debug("Created truncated colormap: %s", new.name)
    return new


def mirrored_discrete_doy_cmap(
    doy_values: np.ndarray,
    *,
    sigma: float = 5.0,
    cmap: str | Colormap = "viridis",
    center_color: str = "whitesmoke",
    clip_to_data: bool = True,
    under_color: str | None = None,
    over_color: str | None = None,
    lo_hi: Tuple[int, int] | None = None,
) -> tuple[ListedColormap, BoundaryNorm, np.ndarray, tuple[int, int, int]]:
    """
    Build a *mirrored*, discrete DoY colormap from a **single** base colormap.

    Steps
    -----
    1) Compute median (int) and std over finite values in `doy_values` (or use `lo_hi` if provided).
    2) Define integer range [lo, hi] around the median with an **odd** number of bins so the
       median has its own center bin (assigned `center_color`).
    3) Sample the *single* input colormap on [0, 0.5) and [0.5, 1.0) and reverse each half,
       so colors near the center are the darkest/saturated.
    4) Concatenate: [low_half_reversed] + [center] + [high_half_reversed].
    5) Return a `ListedColormap`, a `BoundaryNorm` with half-integer edges, and `(lo, med, hi)`.

    Parameters
    ----------
    doy_values : ndarray
        DoY integer array (any shape). Only finite values are used to compute stats.
    sigma : float, default 5.0
        Range half-width in units of standard deviations from the median.
    cmap : str or Colormap, default "viridis"
        Base continuous colormap to split and invert around its midpoint.
    center_color : str, default "whitesmoke"
        Color assigned to the exact median bin.
    clip_to_data : bool, default True
        Clip [lo, hi] to the finite min/max of `doy_values`.
    under_color, over_color : str or None
        Optional colors for values < lo and > hi.
    lo_hi : (int, int) or None
        If provided, use this explicit (lo, hi) instead of median±sigma*std.

    Returns
    -------
    listed : ListedColormap
    norm : BoundaryNorm
    boundaries : np.ndarray
        Half-integer boundaries of length (number_of_bins + 1).
    lo_med_hi : tuple[int, int, int]
        The (lo, median, hi) integers used to build the bins.
    """
    log.debug(
        "mirrored_discrete_doy_cmap(sigma=%s, cmap=%r, center_color=%r, clip_to_data=%s, lo_hi=%r)",
        sigma,
        cmap,
        center_color,
        clip_to_data,
        lo_hi,
    )

    a = np.asarray(doy_values)
    finite = np.isfinite(a)
    if not np.any(finite):
        raise ValueError("No finite values in `doy_values`.")

    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    med = int(np.rint(np.nanmedian(a[finite])))
    log.debug("Computed median DoY: %d", med)

    if lo_hi is None:
        std = float(np.nanstd(a[finite]))
        if std == 0:
            std = 0.5  # avoid zero range collapse
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        lo = int(np.floor(med - sigma * std))
        hi = int(np.ceil(med + sigma * std))
        if clip_to_data:
            dmin = int(np.floor(np.nanmin(a[finite])))
            dmax = int(np.ceil(np.nanmax(a[finite])))
            lo = max(lo, dmin)
            hi = min(hi, dmax)
        log.debug("Derived lo/hi from stats: lo=%d, hi=%d (std=%.3f)", lo, hi, std)
    else:
        lo, hi = map(int, lo_hi)
        log.debug("Using explicit lo_hi: lo=%d, hi=%d", lo, hi)

    # Ensure [lo, hi] brackets the median and yields an odd bin count
    if hi <= med:
        hi = med + 1
    if lo >= med:
        lo = med - 1
    if ((hi - lo + 1) % 2) == 0:
        hi += 1
    n_low = max(med - lo, 1)
    n_high = max(hi - med, 1)

    # Sample halves safely
    if n_low == 1:
        low_grid = np.array([0.25])
    else:
        low_grid = np.linspace(0.0, 0.5, n_low, endpoint=False)
    low_half = base(low_grid)[::-1]

    if n_high == 1:
        high_grid = np.array([0.75])
    else:
        high_grid = np.linspace(0.5, 1.0, n_high + 1, endpoint=True)[1:]
    high_half = base(high_grid)[::-1]

    center_rgba = to_rgba(center_color)
    colors = np.vstack([low_half, center_rgba, high_half])

    listed = ListedColormap(colors, name=f"{base.name}_mirrored_discrete")
    listed.set_bad(alpha=0.0)
    if under_color is not None:
        listed.set_under(under_color)
    if over_color is not None:
        listed.set_over(over_color)

    boundaries = np.arange(lo - 0.5, hi + 1.5, 1.0, dtype=float)
    norm = BoundaryNorm(boundaries, ncolors=listed.N, clip=False)

    log.debug(
        "Built mirrored discrete cmap with %d bins, boundaries [%s..%s]",
        listed.N,
        f"{boundaries[0]:.1f}",
        f"{boundaries[-1]:.1f}",
    )
    return listed, norm, boundaries, (lo, med, hi)


def _get_extent(data: xr.DataArray) -> tuple[float, float, float, float]:
    """
    Get the spatial extent of an xarray DataArray for plotting.

    Parameters
    ----------
    data : xr.DataArray
        The 2D data array with 'x' and 'y' coordinates.

    Returns
    -------
    extent : tuple[float, float, float, float]
        The extent in the format (xmin, xmax, ymin, ymax), suitable for imshow().
    """
    extent = (
        float(data.x.min()),
        float(data.x.max()),
        float(data.y.min()),
        float(data.y.max()),
    )
    log.debug("Computed extent: %s", extent)
    return extent


def plot_dem(dem: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot a digital elevation model (DEM) with contours and physical coordinates.

    Parameters
    ----------
    dem : xr.DataArray
        A 2D array representing elevation in meters, with spatial coordinates.
    ax : matplotlib.axes.Axes, optional
        Optional Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the DEM plot.
    """
    log.debug("plot_dem called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(dem)

    truncated = truncate_colormap(cmr.pride, vmin=0.05, vmax=0.55)
    img = ax.imshow(dem.values, cmap=truncated, extent=extent, origin="upper")
    ax.contour(dem.x, dem.y, dem.values, levels=9, colors="whitesmoke", linewidths=0.7, alpha=0.9, linestyles="dotted")
    ax.set_title("Digital Elevation Model")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(img, ax=ax, label="Elevation (m)")
    return ax


def plot_slope(slope: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot a slope map in degrees.

    Parameters
    ----------
    slope : xr.DataArray
        A 2D array of terrain slope in degrees, with spatial coordinates.
    ax : matplotlib.axes.Axes, optional
        Optional Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the slope plot.
    """
    log.debug("plot_slope called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(slope)
    img = ax.imshow(slope.values, cmap=cmr.pride, extent=extent, origin="upper")
    ax.set_title("Slope")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(img, ax=ax, label="Slope (°)")
    return ax


def plot_aspect(aspect: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot an aspect map in degrees clockwise from north.

    Parameters
    ----------
    aspect : xr.DataArray
        A 2D array of terrain aspect in degrees clockwise from north, with spatial coordinates.
    ax : matplotlib.axes.Axes, optional
        Optional Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the aspect plot.
    """
    log.debug("plot_aspect called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(aspect)
    img = ax.imshow(aspect.values, cmap=cmr.pride, extent=extent, origin="upper")
    ax.set_title("Aspect")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(img, ax=ax, label="Aspect (°)")
    return ax


def plot_normals(normals_enu: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot rgb normals unit vector map.

    Parameters
    ----------
    normal_enu : xarray.DataArray (3, y, x)
        ENU unit normal vectors. Bands: [east, north, up].
    ax : matplotlib.axes.Axes, optional
        Optional Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the aspect plot.
    """
    log.debug("plot_normals called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(normals_enu)
    rgb = (normals_enu.values + 1) / 2
    rgb = np.moveaxis(rgb, 0, -1)
    ax.imshow(rgb, extent=extent, origin="upper")
    ax.set_title("Normals: R->E, G->N, B->U")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return ax


def plot_hillshade(
    slope: xr.DataArray,
    aspect: xr.DataArray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a hillshade map based on terrain slope and aspect using Lambertian illumination.

    Parameters
    ----------
    slope : xr.DataArray
        A 2D array of terrain slope in degrees.
    aspect : xr.DataArray
        A 2D array of terrain aspect in degrees clockwise from north.
    azimuth_deg : float, default=315.0
        Solar azimuth angle in degrees (0° = north, 90° = east).
    altitude_deg : float, default=45.0
        Solar altitude angle above the horizon in degrees.
    ax : matplotlib.axes.Axes, optional
        Optional Matplotlib axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the hillshade plot.
    """
    log.debug(
        "plot_hillshade called; ax is None=%s, azimuth_deg=%.2f, altitude_deg=%.2f",
        ax is None,
        azimuth_deg,
        altitude_deg,
    )
    if ax is None:
        _, ax = plt.subplots()

    hillshade = compute_hillshade(slope, aspect, azimuth_deg, altitude_deg)
    extent = _get_extent(slope)
    truncated = truncate_colormap(cmr.pride_r, vmin=0.05, vmax=0.55)
    ax.imshow(hillshade.values, cmap=truncated, extent=extent, origin="upper")
    ax.set_title(f"Hillshade (Azimuth: {azimuth_deg}°, Altitude: {altitude_deg}°)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    return ax


def plot_horizon_polar(
    azimuths: np.ndarray,
    horizon_vals: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    sunaz: Optional[np.ndarray] = None,
    sunaltmin: Optional[np.ndarray] = None,
    sunaltmax: Optional[np.ndarray] = None,
) -> "Axes":
    """
    Plot a stylized polar horizon profile with compass-style ticks and (optionally)
    a solar envelope (min/max altitude versus azimuth).

    Parameters
    ----------
    azimuths : array-like
        Azimuth angles in degrees (clockwise from North).
    horizon_vals : array-like
        Horizon elevation values in degrees.
    ax : matplotlib.axes.PolarAxes, optional
        Polar axis to plot on. If None, a new one is created.
    sunaz : array-like, optional
        Solar azimuth samples in degrees (same length as sunaltmin/sunaltmax).
    sunaltmin : array-like, optional
        Minimum solar altitude per azimuth (degrees).
    sunaltmax : array-like, optional
        Maximum solar altitude per azimuth (degrees).

    Returns
    -------
    ax : matplotlib.axes.PolarAxes
        The axis with the plotted horizon profile (and optional solar envelope).
    """
    log.debug(
        "plot_horizon_polar called; ax is None=%s, have_solar=%s",
        ax is None,
        (sunaz is not None and (sunaltmin is not None or sunaltmax is not None)),
    )

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # ---- Horizon curve (closed) ----
    az = np.asarray(azimuths, float)
    hr = np.asarray(horizon_vals, float)

    az_rad = np.deg2rad(np.append(az, az[0]))
    hr_closed = np.append(hr, hr[0])

    horizon_edge = cmr.pride([0.21])
    horizon_fill = cmr.pride([0.30])

    ax.fill(
        az_rad,
        hr_closed,
        facecolor=horizon_fill,
        alpha=0.4,
        hatch="/////",
        edgecolor=horizon_edge,
        linewidth=0,
        zorder=1,
    )
    ax.plot(az_rad, hr_closed, lw=2.1, color=horizon_edge, alpha=0.9, zorder=2)

    # ---- Optional solar envelope ----
    have_sun = sunaz is not None and (sunaltmin is not None or sunaltmax is not None)

    solar_lo = None
    solar_hi = None
    solar_th = None

    if have_sun:
        saz = np.mod(np.asarray(sunaz, float), 360.0)

        lo = np.asarray(sunaltmin, float) if sunaltmin is not None else None
        hi = np.asarray(sunaltmax, float) if sunaltmax is not None else None

        # Mask NaNs consistently
        mask = np.isfinite(saz)
        if lo is not None:
            mask &= np.isfinite(lo)
        if hi is not None:
            mask &= np.isfinite(hi)

        saz = saz[mask]
        if lo is not None:
            lo = lo[mask]
        if hi is not None:
            hi = hi[mask]

        if saz.size:
            # Sort by azimuth to keep the band well-behaved
            order = np.argsort(saz)
            saz = saz[order]
            if lo is not None:
                lo = lo[order]
            if hi is not None:
                hi = hi[order]

            solar_th = np.deg2rad(saz)

            solar_color = cmr.pride([0.68])

            if lo is not None and hi is not None:
                # Fill envelope between min and max
                # Build a closed polygon in theta–r space
                th_poly = np.concatenate([solar_th, solar_th[::-1]])
                r_poly = np.concatenate([lo, hi[::-1]])
                ax.fill(
                    th_poly,
                    r_poly,
                    facecolor=cmr.pride([0.6]),
                    alpha=0.1,
                    hatch="\\\\\\\\\\",
                    edgecolor=cmr.pride([0.6]),
                    linewidth=0,
                    zorder=0,
                )
                # Outline top/bottom
                ax.plot(solar_th, lo, color=solar_color, lw=2.1, ls=":", alpha=0.9, zorder=0)
                ax.plot(solar_th, hi, color=solar_color, lw=2.1, alpha=0.9, zorder=0)
                solar_lo, solar_hi = lo, hi
            elif lo is not None:
                ax.plot(solar_th, lo, color=solar_color, lw=1.5, alpha=0.9, zorder=0)
                solar_lo = lo
            elif hi is not None:
                ax.plot(solar_th, hi, color=solar_color, lw=1.5, alpha=0.9, zorder=0)
                solar_hi = hi

    # ---- Orientation ----
    ax.set_theta_zero_location("N")  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]

    # ---- Radial limits: include horizon and any solar curves ----
    pad = 5.0
    r_candidates = [np.nanmin(hr_closed), np.nanmax(hr_closed)]
    if have_sun and solar_th is not None:
        if solar_lo is not None:
            r_candidates.extend([np.nanmin(solar_lo), np.nanmax(solar_lo)])
        if solar_hi is not None:
            r_candidates.extend([np.nanmin(solar_hi), np.nanmax(solar_hi)])

    rmin = float(np.nanmin(r_candidates))
    rmax = float(np.nanmax(r_candidates))
    ax.set_rlim(rmin - 10.0, rmax + pad)  # type: ignore[attr-defined]
    ax.set_rlabel_position(150)  # type: ignore[attr-defined]
    ax.tick_params(axis="y", labelsize=10)

    # ---- Rim ticks ----
    major_deg = np.arange(0, 360, 30)
    minor_deg = np.arange(0, 360, 2)

    for deg in minor_deg:
        th = np.deg2rad(deg)
        ax.plot([th, th], [rmax + pad - 1.0, rmax + pad], color="gray", lw=1.2, alpha=0.6, solid_capstyle="butt", zorder=2)
    for deg in major_deg:
        th = np.deg2rad(deg)
        ax.plot(
            [th, th],
            [rmax + pad - 2.0, rmax + pad],
            color=cmr.pride([0.79]),
            lw=2.5,
            alpha=0.8,
            solid_capstyle="butt",
            zorder=3,
        )

    # ---- Cardinal labels at fixed fraction of radius (no jitter) ----
    rmin_f = ax.get_rmin()  # type: ignore[attr-defined]
    rmax_f = ax.get_rmax()  # type: ignore[attr-defined]
    label_r = rmin_f + 0.90 * (rmax_f - rmin_f)

    for deg, label in {0: "N", 90: "E", 180: "S", 270: "W"}.items():
        ax.text(np.deg2rad(deg), label_r, label, ha="center", va="center", fontsize=15, fontstyle="italic")

    ax.set_xticklabels([])  # hide default angle numbers
    ax.grid(ls=":", lw=0.7, zorder=7)

    log.debug("Finished plot_horizon_polar with rlim=(%.2f, %.2f)", rmin - 10.0, rmax + pad)
    return ax


def plot_total_energy(total_energy: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot total integrated energy (J/m²) map.

    Parameters
    ----------
    total_energy : xr.DataArray
        2D array of total energy in J/m².
    ax : matplotlib.axes.Axes, optional
        Optional axis to draw the plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    log.debug("plot_total_energy called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(total_energy)
    truncated = truncate_colormap(cmr.pride_r, vmin=0.05, vmax=0.55)
    im = ax.imshow(
        total_energy.values,
        cmap=truncated,
        origin="upper",
        extent=extent,
    )
    ax.set_title("Total Energy")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(im, ax=ax, label="Flux [J/m²]")
    return ax


def plot_peak_energy(peak_energy: xr.DataArray, ax: Axes | None = None) -> Axes:
    """
    Plot peak daily energy (J/m²) map.

    Parameters
    ----------
    peak_energy : xr.DataArray
        2D array of maximum daily energy.
    ax : matplotlib.axes.Axes, optional
        Optional axis to draw the plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    log.debug("plot_peak_energy called; ax is None=%s", ax is None)
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(peak_energy)
    truncated = truncate_colormap(cmr.pride_r, vmin=0.05, vmax=0.55)
    im = ax.imshow(
        peak_energy.values,
        cmap=truncated,
        origin="upper",
        extent=extent,
    )
    ax.set_title("Peak Daily Energy")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(im, ax=ax, label="Flux [J/m²]")
    return ax


def plot_day_of_peak(
    day_of_peak: xr.DataArray,
    times_utc: np.ndarray,
    sigma: float = 9,
    cmap_base: str | Colormap = cmr.seasons,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot the day-of-peak energy using a mirrored discrete colormap.

    Parameters
    ----------
    day_of_peak : xr.DataArray
        2D array of DoY (1–366) when peak energy occurred.
    times_utc : np.ndarray
        Time array used to resolve calendar year for tick labeling.
    sigma : float, default 9.0
        Std-deviation half-width used in colormap.
    cmap_base : Colormap or str, default cmr.seasons
        Base continuous colormap to mirror and discretize.
    ax : matplotlib.axes.Axes, optional
        Optional axis to draw the plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    log.debug(
        "plot_day_of_peak called; ax is None=%s, sigma=%s, cmap_base=%r",
        ax is None,
        sigma,
        getattr(cmap_base, "name", cmap_base),
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    dop_vals = day_of_peak.values.astype(float)

    truncated = truncate_colormap(cmap_base, vmin=0.2, vmax=0.8)
    cmap, norm, boundaries, (lo, _, hi) = mirrored_discrete_doy_cmap(dop_vals, sigma=sigma, cmap=truncated)

    extent = _get_extent(day_of_peak)
    im = ax.imshow(dop_vals, cmap=cmap, norm=norm, origin="upper", extent=extent)

    ticks = np.arange(lo, hi + 1, 2)
    year = pd.to_datetime(times_utc[0]).year
    dates = pd.to_datetime([f"{year}-{doy:03d}" for doy in ticks], format="%Y-%j")
    labels = [d.upper() for d in dates.strftime("%b %d")]

    cbar = plt.colorbar(im, ax=ax, boundaries=boundaries, ticks=ticks, extend="both")
    cbar.ax.set_yticklabels(labels)

    ax.set_title("Day of Peak Energy")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return ax
