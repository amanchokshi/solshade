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
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a stylized polar horizon profile with compass-style ticks and shading.

    Parameters
    ----------
    azimuths : array-like
        Azimuth angles in degrees (clockwise from North).
    horizon_vals : array-like
        Horizon elevation values (in degrees).
    ax : matplotlib.axes.PolarAxes, optional
        Polar axis to plot on. If None, a new one is created.

    Returns
    -------
    ax : matplotlib.axes.PolarAxes
        The axis with the plotted horizon profile.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Close the curve
    az_rad = np.deg2rad(np.append(azimuths, azimuths[0]))
    horizon_vals = np.append(horizon_vals, horizon_vals[0])

    rmax = np.nanmax(horizon_vals)
    rmin = np.nanmin(horizon_vals)
    pad = 5

    # Fill under the curve (with color and hatch)
    ax.fill(
        az_rad,
        horizon_vals,
        facecolor=cmr.pride([0.3]),
        edgecolor=cmr.pride([0.21]),
        hatch="/////",
        linewidth=0,
        alpha=0.21,
        zorder=1,
    )

    # Plot main line
    ax.plot(az_rad, horizon_vals, lw=2.1, color=cmr.pride([0.21]), alpha=0.9, zorder=2)

    # Configure polar orientation
    ax.set_theta_zero_location("N")  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]

    # Draw compass-style radial ticks
    major_deg = np.arange(0, 360, 30)
    minor_deg = np.arange(0, 360, 2)

    for deg in minor_deg:
        theta = np.deg2rad(deg)
        ax.plot(
            [theta, theta],
            [rmax + pad - 1, rmax + pad],
            color="gray",
            lw=1.2,
            alpha=0.6,
            solid_capstyle="butt",
            zorder=2,
        )

    for deg in major_deg:
        theta = np.deg2rad(deg)
        ax.plot(
            [theta, theta],
            [rmax + pad - 2, rmax + pad],
            color=cmr.pride([0.79]),
            lw=2.5,
            alpha=0.8,
            solid_capstyle="butt",
            zorder=3,
        )

    # Set limits first
    ax.set_rlim(rmin - 10, rmax + pad)  # type: ignore[attr-defined]
    ax.set_rlabel_position(150)  # type: ignore[attr-defined]
    ax.tick_params(axis="y", labelsize=10)

    # Now re-query rmin/rmax to get the final values
    rmin = ax.get_rmin()  # type: ignore[attr-defined]
    rmax = ax.get_rmax()  # type: ignore[attr-defined]
    label_radius = rmin + 0.9 * (rmax - rmin)

    # Then place the cardinal direction labels
    for deg, label in {0: "N", 90: "E", 180: "S", 270: "W"}.items():
        ax.text(
            np.deg2rad(deg),
            label_radius,
            label,
            ha="center",
            va="center",
            fontsize=15,
            fontstyle="italic",
        )

    # Set limits, label settings
    ax.set_rlabel_position(150)  # type: ignore[attr-defined]
    ax.tick_params(axis="y", labelsize=10)

    # Hide default angular labels
    ax.set_xticklabels([])

    # Grid aesthetics
    ax.grid(ls=":", lw=0.5)

    return ax
