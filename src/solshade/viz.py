"""
viz.py — Plotting utilities for DEM-based terrain analysis.

This module provides functions to visualize:
- Digital Elevation Models (DEMs)
- Slope maps
- Aspect maps
- Hillshades

All plots use physical coordinates (easting/northing in meters) and meaningful colorbars.
"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes


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
    img = ax.imshow(dem.values, cmap=cmr.ocean, extent=extent, origin="upper")
    ax.contour(dem.x, dem.y, dem.values, levels=10, colors="k", linewidths=0.5)
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

    # Convert angles to radians
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    az_rad = np.radians(azimuth_deg)
    alt_rad = np.radians(altitude_deg)

    # Lambertian hillshade model
    shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    hillshade = np.clip(shaded, 0, 1)

    extent = _get_extent(slope)
    ax.imshow(hillshade, cmap=cmr.swamp, extent=extent, origin="upper")
    ax.set_title(f"Hillshade (Azimuth: {azimuth_deg}°, Altitude: {altitude_deg}°)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    return ax
