import json
import warnings
from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
import rioxarray as rxr
import xarray as xr
from joblib import Parallel, delayed
from numba import njit, prange
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import map_coordinates


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

    aspect = xr.DataArray(aspect_deg, coords=dem.coords, dims=dem.dims, attrs={"units": "degrees", "long_name": "aspect"})

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


def compute_horizon_map(
    dem: xr.DataArray,
    n_directions: int = 64,
    max_distance: float = 5000,
    step: float = 20,
    chunk_size: int = 32,
    n_jobs: int = -1,
    progress: bool = True,
) -> xr.DataArray:
    """
    Compute a per-pixel horizon angle map from a digital elevation model (DEM)
    using chunked ray tracing.

    For each pixel, rays are cast in `n_directions` azimuthal directions
    and sampled up to `max_distance` away. The maximum elevation angle
    encountered along each ray is recorded as the local horizon.

    Parallel processing is done with Joblib, and a rich progress bar can
    optionally be shown.

    Parameters
    ----------
    dem : xr.DataArray
        Input digital elevation model with a defined CRS and affine transform.
        Must be a 2D array with shape (y, x) and spatial coordinates.
    n_directions : int, optional
        Number of azimuthal directions to trace (default is 64).
    max_distance : float, optional
        Maximum distance (in meters) to trace each ray from each pixel
        (default is 5000 m).
    step : float, optional
        Distance step (in meters) between ray samples (default is 20 m).
    chunk_size : int, optional
        Size (in pixels) of square chunks to process independently
        (default is 32).
    n_jobs : int, optional
        Number of parallel jobs to run. Use -1 to use all available cores
        (default is -1).
    progress : bool, optional
        If True, display a rich progress bar (default is True).

    Returns
    -------
    xr.DataArray
        A 3D xarray DataArray with dimensions (azimuth, y, x), representing
        the local horizon angle (in degrees) in each direction for each pixel.
        The azimuthal directions are stored in the `azimuth` coordinate.
    """
    if dem.rio.crs is None or dem.rio.transform() is None:
        raise ValueError("DEM must have CRS and affine transform defined.")

    transform = dem.rio.transform()
    res_x, res_y = transform.a, -transform.e
    ny, nx = dem.shape

    azimuths = np.linspace(0, 360, n_directions, endpoint=False)
    distances = np.arange(0, max_distance + step, step)
    ns = len(distances)

    dx_pix = np.cos(np.deg2rad(azimuths))[:, np.newaxis] * distances / res_x
    dy_pix = -np.sin(np.deg2rad(azimuths))[:, np.newaxis] * distances / res_y
    dx_pix = dx_pix[np.newaxis, np.newaxis, :, :]
    dy_pix = dy_pix[np.newaxis, np.newaxis, :, :]

    @njit(parallel=True)  # pragma: no cover
    def _compute_horizon(elev_prof, distances, out_arr):
        nyc, nxc, nd, ns = elev_prof.shape
        for iy in prange(nyc):
            for ix in prange(nxc):
                for idir in range(nd):
                    elev0 = elev_prof[iy, ix, idir, 0]
                    if np.isnan(elev0):
                        out_arr[iy, ix, idir] = np.nan
                        continue
                    max_ang = -np.inf
                    for idist in range(1, ns):
                        elev_sample = elev_prof[iy, ix, idir, idist]
                        if np.isnan(elev_sample):
                            continue
                        dz = elev_sample - elev0
                        angle = np.arctan2(dz, distances[idist])
                        max_ang = max(max_ang, angle)
                    out_arr[iy, ix, idir] = np.nan if max_ang == -np.inf else np.rad2deg(max_ang)

    def process_chunk(x0, y0):
        y1 = min(y0 + chunk_size, ny)
        x1 = min(x0 + chunk_size, nx)

        iy, ix = np.mgrid[y0:y1, x0:x1]
        iy = iy[..., np.newaxis, np.newaxis]
        ix = ix[..., np.newaxis, np.newaxis]

        sample_y = iy + dy_pix
        sample_x = ix + dx_pix
        sample_coords = np.stack([sample_y, sample_x], axis=0)

        elev_profiles = map_coordinates(
            dem.values,
            sample_coords.reshape(2, -1),
            order=1,
            mode="constant",
            cval=np.nan,
        ).reshape(y1 - y0, x1 - x0, n_directions, ns)

        chunk_out = np.full((y1 - y0, x1 - x0, n_directions), np.nan, dtype=np.float32)
        _compute_horizon(elev_profiles, distances, chunk_out)

        return y0, y1, x0, x1, chunk_out

    tasks = [(x0, y0) for y0 in range(0, ny, chunk_size) for x0 in range(0, nx, chunk_size)]
    horizon_data = np.full((n_directions, ny, nx), np.nan, dtype=np.float32)

    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*worker stopped while some jobs were given to the executor.*"
    )

    def run_all():
        return Parallel(n_jobs=n_jobs, return_as="generator")(delayed(process_chunk)(x0, y0) for x0, y0 in tasks)

    if progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Computing horizon map"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task_id = bar.add_task("computing", total=len(tasks))
            results = run_all()
            for result in results:
                if result is not None:
                    y0, y1, x0, x1, chunk = result
                    horizon_data[:, y0:y1, x0:x1] = np.moveaxis(chunk, -1, 0)
                bar.advance(task_id)
    else:
        for result in run_all():
            if result is not None:
                y0, y1, x0, x1, chunk = result
                horizon_data[:, y0:y1, x0:x1] = np.moveaxis(chunk, -1, 0)

    horizon_da = xr.DataArray(
        horizon_data,
        dims=("azimuth", "y", "x"),
        coords={"azimuth": azimuths, "y": dem.y, "x": dem.x},
        name="horizon_angle",
        attrs={"units": "degrees"},
    )
    horizon_da.rio.write_crs(dem.rio.crs, inplace=True)
    horizon_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    horizon_da.rio.write_transform(dem.rio.transform(), inplace=True)
    horizon_da.attrs.update(
        {
            "max_distance_m": max_distance,
            "step_m": step,
            "n_directions": n_directions,
            "azimuth_meaning": "Azimuthal directions clockwise from North",
            "azimuths_deg": json.dumps(azimuths.tolist()),
        }
    )

    return horizon_da


def horizon_interp(
    azimuths: np.ndarray,
    horizon_vals: np.ndarray,
    fallback_frac: float = 0.5,
) -> Optional[Callable]:
    """
    Interpolate horizon angles as a function of azimuth using cubic interpolation,
    ensuring periodic continuity. Falls back to linear if data is sparse or if
    cubic interpolation yields unphysical results.

    Parameters
    ----------
    azimuths : np.ndarray
        1D array of azimuth angles in degrees [0, 360).
    horizon_vals : np.ndarray
        1D array of horizon angles (in degrees), same shape as azimuths.
        May contain NaNs.
    fallback_frac : float, optional
        Threshold fraction of NaNs above which to fall back to linear interpolation.

    Returns
    -------
    interp_func : callable or None
        A function that takes azimuth(s) in degrees and returns interpolated
        horizon angle(s). Returns None if no valid data.
    """
    az_valid = azimuths[~np.isnan(horizon_vals)]
    vals_valid = horizon_vals[~np.isnan(horizon_vals)]

    if len(vals_valid) == 0:
        print("No valid horizon data — returning None.")
        return None

    nan_frac = np.isnan(horizon_vals).sum() / len(horizon_vals)
    print(f"Valid points: {len(vals_valid)} / {len(horizon_vals)} ({(1 - nan_frac) * 100:.1f}% valid)")
    print(f"Azimuth domain: {az_valid.min():.2f} to {az_valid.max():.2f}")
    print(f"Value domain: {vals_valid.min():.2f} to {vals_valid.max():.2f}")

    # Enforce periodicity only if we already have a valid value at 0.0 deg
    if az_valid[0] == 0.0:
        print("Copying value at 0.0 deg to enforce periodicity at 360.0 deg.")
        az_valid = np.append(az_valid, 360.0)
        vals_valid = np.append(vals_valid, vals_valid[0])

    # Attempt cubic interpolation if data is sufficiently complete
    if nan_frac <= fallback_frac:
        try:
            interp_func = CubicSpline(az_valid, vals_valid, bc_type="periodic")
            test_vals = interp_func(np.arange(361))
            min_val, max_val = np.nanmin(test_vals), np.nanmax(test_vals)

            if min_val < -90 or max_val > 90:
                print(f"CubicSpline produced unphysical output: min={min_val:.2f}, max={max_val:.2f}")
                raise ValueError("Unphysical cubic spline result")

            print(f"Using CubicSpline interpolation. Output range: [{min_val:.2f}, {max_val:.2f}]")
        except Exception as e:
            print(f"CubicSpline interpolation failed: {e}")
            print("Falling back to interp1d.")
            interp_func = interp1d(
                az_valid,
                vals_valid,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",  # type: ignore[arg-type]
            )
    else:
        print("Too many NaNs — using interp1d.")
        interp_func = interp1d(
            az_valid,
            vals_valid,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore[arg-type]
        )

    def wrapped_interp(query_az):
        return interp_func(np.mod(query_az, 360))

    return wrapped_interp
