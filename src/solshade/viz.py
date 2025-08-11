"""
viz.py — Plotting utilities for DEM-based terrain analysis.

This module provides functions to visualize:
- Digital Elevation Models (DEMs)
- Slope maps
- Aspect maps
- Hillshades

All plots use physical coordinates (easting/northing in meters) and meaningful colorbars.
"""

from typing import Optional

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from solshade.terrain import compute_hillshade


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
    return (
        float(data.x.min()),
        float(data.x.max()),
        float(data.y.min()),
        float(data.y.max()),
    )


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
    if ax is None:
        _, ax = plt.subplots()

    extent = _get_extent(dem)
    img = ax.imshow(dem.values, cmap=cmr.savanna, extent=extent, origin="upper")
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
    if ax is None:
        _, ax = plt.subplots()

    hillshade = compute_hillshade(slope, aspect, azimuth_deg, altitude_deg)
    extent = _get_extent(slope)
    ax.imshow(hillshade.values, cmap=cmr.savanna, extent=extent, origin="upper")
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
    import cmasher as cmr
    import matplotlib.pyplot as plt
    import numpy as np

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

    return ax
