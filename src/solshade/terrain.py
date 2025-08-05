from pathlib import Path
from typing import cast

import numpy as np
import rioxarray as rxr
import xarray as xr


def load_dem(path: str | Path) -> xr.DataArray:
    """
    Load a single-band Digital Elevation Model (DEM) from a GeoTIFF file.

    Uses rioxarray to preserve CRS and coordinate metadata.

    Parameters
    ----------
    path : str or Path
        Path to a single-band GeoTIFF file containing elevation data.
        The file must contain exactly one raster band.

    Returns
    -------
    dem : xarray.DataArray
        A 2D array of elevation values with dimensions (y, x),
        including CRS, transform, and coordinate metadata.

    Raises
    ------
    TypeError
        If the input file contains more than one band or does not reduce to a DataArray.

    Notes
    -----
    - Elevation units are inherited from the input file (typically meters).
    - The spatial reference (CRS) must be projected (not geographic/latlon).
    - Squeezes band dimension if present.
    """
    raw = cast(xr.Dataset, rxr.open_rasterio(path, masked=True))
    squeezed = raw.squeeze()

    if not isinstance(squeezed, xr.DataArray):
        raise TypeError("DEM is not a single-band raster.")

    return squeezed


def compute_slope_aspect(dem: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute terrain slope and aspect from a 2D DEM.

    Uses the central difference method via numpy.gradient.
    Assumes a projected CRS (e.g., UTM, polar stereographic) where
    spatial resolution is in meters or equivalent linear units.

    Parameters
    ----------
    dem : xarray.DataArray
        2D elevation array with dimensions (y, x), coordinates, and valid CRS.

    Returns
    -------
    slope : xarray.DataArray
        Slope in degrees, where 0° is flat and 90° is vertical.
        Same shape and coordinates as input DEM.

    aspect : xarray.DataArray
        Aspect in degrees clockwise from North.
        0° = North, 90° = East, 180° = South, 270° = West.
        Flat regions may contain undefined or noisy aspect values.

    Raises
    ------
    ValueError
        If spatial resolution cannot be determined from the DEM.

    Notes
    -----
    - Aspect is computed using arctangent of partial derivatives:
        arctan2(-dz/dx, dz/dy)
    - The output arrays include metadata: "units" and "long_name".
    - Edge pixels may be less accurate due to gradient estimation.
    """
    z = dem.values
    dy, dx = dem.rio.resolution()
    dx = abs(dx)
    dy = abs(dy)

    dzdy, dzdx = np.gradient(z, dy, dx)

    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    slope = xr.DataArray(slope_deg, coords=dem.coords, dims=dem.dims, attrs={"units": "degrees", "long_name": "slope"})

    aspect = xr.DataArray(
        aspect_deg, coords=dem.coords, dims=dem.dims, attrs={"units": "degrees", "long_name": "aspect"}
    )

    return slope, aspect


def compute_hillshade(
    slope: xr.DataArray,
    aspect: xr.DataArray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> xr.DataArray:
    """
    Compute a hillshade array using slope and aspect with Lambertian illumination.

    Parameters
    ----------
    slope : xarray.DataArray
        Slope in degrees.
    aspect : xarray.DataArray
        Aspect in degrees clockwise from north.
    azimuth_deg : float, optional
        Azimuth angle of the sun (0° = north, 90° = east). Default is 315°.
    altitude_deg : float, optional
        Altitude angle of the sun above the horizon. Default is 45°.

    Returns
    -------
    hillshade : xarray.DataArray
        Normalized hillshade values from 0 (dark) to 1 (bright),
        preserving coordinates and CRS metadata.

    Notes
    -----
    - Uses a Lambertian reflection model.
    - All input and output arrays are 2D.
    """
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    az_rad = np.radians(azimuth_deg)
    alt_rad = np.radians(altitude_deg)

    shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    hillshade = np.clip(shaded, 0, 1)

    return xr.DataArray(
        hillshade,
        coords=slope.coords,
        dims=slope.dims,
        attrs={"units": "unitless", "long_name": "hillshade"},
    )
