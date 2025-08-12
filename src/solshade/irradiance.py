import json

import numpy as np
import xarray as xr
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from solshade.solar import nearest_horizon_indices


def compute_flux_timeseries(
    horizon: xr.DataArray,
    horizon_az_deg: np.ndarray,
    sun_alt_deg: np.ndarray,
    sun_az_deg: np.ndarray,
    sun_enu: np.ndarray,
    normal_enu: xr.DataArray,
    times_utc: np.ndarray,
    *,
    toa: float = 1361.0,
    dtype=np.float32,
) -> xr.DataArray:
    """
    Compute horizon-masked, Lambertian irradiance time series over a DEM.

    Overview
    --------
    For each time step:
      1) The Sun direction is provided as a unit ENU vector `sun_enu[t]`.
      2) Terrain unit normals per pixel come from `normal_enu` bands (east,north,up).
      3) A horizon map gives the elevation angle of the skyline per azimuth
         (uniform samples over [0, 360)). We choose the nearest horizon azimuth
         to the Sun azimuth and get the horizon elevation for each pixel.
      4) If the Sun altitude < horizon at a pixel → shadow (irradiance = 0).
      5) Otherwise irradiance ∝ cos(incidence) = max(0, n · s_enu), scaled by `toa`.

    TOA and units
    -------------
    `toa` is the top-of-atmosphere solar irradiance in W·m⁻². The default
    (1361 W·m⁻²) is a standard solar constant. The output irradiance is
    `toa * max(0, n · s_enu)` and is **not** atmosphere-corrected. You may
    apply atmospheric/clear-sky/terrain albedo corrections in a later stage.

    Normals
    -------
    `normal_enu` contains per-pixel unit normals in ENU coordinates:
      - band "east"  → n_E
      - band "north" → n_N
      - band "up"    → n_U
    `sun_enu[t] = (s_E, s_N, s_U)` is the unit Sun direction in ENU.

    NaN propagation
    ---------------
    Horizon NaNs indicate unknown horizon values (e.g., border pixels). For
    those pixels at a time step, the output flux is set to NaN (propagated),
    regardless of Sun altitude.

    Parameters
    ----------
    horizon : xarray.DataArray
        Horizon elevation angles (degrees), shape (azimuth|band, y, x).
        The azimuth dimension is either named "azimuth" or "band".
    horizon_az_deg : np.ndarray, shape (N,)
        Uniform azimuth sample centers for `horizon`, degrees in [0, 360).
        Typically from `np.linspace(0, 360, N, endpoint=False)`.
    sun_alt_deg : np.ndarray, shape (T,)
        Sun altitude in degrees per time step.
    sun_az_deg : np.ndarray, shape (T,)
        Sun azimuth in degrees per time step (CW from North).
    sun_enu : np.ndarray, shape (T, 3)
        Unit Sun direction vectors (E, N, U) per time step.
    normal_enu : xarray.DataArray, shape (band=3, y, x)
        Terrain unit normal components. Must have band labels ["east","north","up"].
    times_utc : np.ndarray or sequence, optional
        Optional time axis to store and/or use as coordinates. If provided as a
        numpy datetime64 array, it will also become the DataArray time coordinate.
        ISO-8601 UTC strings are additionally stored in attrs for round-tripping.
    toa : float, default 1361.0
        Top-of-atmosphere irradiance in W·m⁻² used as a scalar multiplier.
    dtype : dtype, default np.float32
        Output array dtype.

    Returns
    -------
    flux : xarray.DataArray (time, y, x)
        Irradiance (W·m⁻²) per time and pixel, Lambertian and horizon-masked.
        Attributes include:
          - "units": "W m^-2"
          - "note": description string
          - "time_utc_iso": JSON list of ISO UTC strings for each time step.

    Raises
    ------
    KeyError
        If `normal_enu` lacks required bands "east", "north", "up".
    ValueError
        On mismatched shapes among inputs.
    """
    az_dim = "azimuth" if "azimuth" in horizon.dims else ("band" if "band" in horizon.dims else None)
    if az_dim is None:
        raise ValueError("`horizon` must have a leading 'azimuth' or 'band' dimension.")

    if set(["y", "x"]) - set(map(str, horizon.dims)):
        raise ValueError("`horizon` must have spatial dims ('y','x').")

    if set(["y", "x"]) - set(map(str, normal_enu.dims)) or "band" not in map(str, normal_enu.dims):
        raise ValueError("`normal_enu` must have dims ('band','y','x').")

    try:
        e = normal_enu.sel(band="east").values
        n = normal_enu.sel(band="north").values
        u = normal_enu.sel(band="up").values
    except Exception as exc:  # noqa: BLE001
        raise KeyError("`normal_enu` must contain bands ['east','north','up'].") from exc

    n_sun_times = int(sun_alt_deg.size)
    if not (sun_az_deg.size == n_sun_times and sun_enu.shape == (n_sun_times, 3)):
        raise ValueError("Length mismatch among sun_alt_deg, sun_az_deg and sun_enu.")

    n_horiz_dir = horizon.sizes[az_dim]
    if horizon_az_deg.size != n_horiz_dir:
        raise ValueError("`horizon_az_deg` length must match the horizon azimuth dimension.")

    y_dim, x_dim = horizon.sizes["y"], horizon.sizes["x"]
    flux_data = np.empty((n_sun_times, y_dim, x_dim), dtype=dtype)

    # Pre-select nearest horizon band per step
    idx = nearest_horizon_indices(sun_az_deg, horizon_az_deg)  # (n_sun_times,)

    # Prepare (optional) time coordinate + attrs strings
    time_coord = None
    if isinstance(times_utc, np.ndarray) and np.issubdtype(times_utc.dtype, np.datetime64):
        time_coord = times_utc.astype("datetime64[s]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Computing irradiance"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as bar:
        task = bar.add_task("irradiance", total=n_sun_times)

        for t in range(n_sun_times):
            # Fast-path: Sun below astronomical horizon → zero flux
            if sun_alt_deg[t] < 0.0:
                # start with zeros, then propagate NaNs (from horizon) below
                slice_out = np.zeros((y_dim, x_dim), dtype=dtype)
                h_t = horizon.isel({az_dim: int(idx[t])}).values
                # NaNs in horizon -> NaNs in output
                nan_mask = np.isnan(h_t)
                if nan_mask.any():
                    slice_out[nan_mask] = np.nan  # type: ignore[assignment]
                flux_data[t, :, :] = slice_out
                bar.update(task, advance=1)
                continue

            # Horizon at nearest azimuth and NaN propagation mask
            h_t = horizon.isel({az_dim: int(idx[t])}).values  # (y_dim, x_dim)
            nan_mask = np.isnan(h_t)

            # Shadow mask: 1 if sun altitude >= horizon; else 0
            # (treat NaNs as False here; we overwrite them to NaN below)
            mask = float(sun_alt_deg[t]) >= h_t
            mask = np.where(np.isnan(h_t), False, mask).astype(dtype, copy=False)

            # Lambertian cosine from normals & Sun ENU
            s_east, s_north, s_up = sun_enu[t, 0], sun_enu[t, 1], sun_enu[t, 2]
            dot = e * s_east + n * s_north + u * s_up
            flux2d = np.where(dot > 0.0, dot * toa, 0.0).astype(dtype, copy=False)

            # Apply horizon shadow
            slice_out = flux2d * mask

            # Propagate horizon NaNs
            if nan_mask.any():
                slice_out[nan_mask] = np.nan  # type: ignore[assignment]

            flux_data[t, :, :] = slice_out
            bar.update(task, advance=1)

    flux = xr.DataArray(
        flux_data,
        dims=("time", "y", "x"),
        coords={
            "time": (time_coord if time_coord is not None else np.arange(n_sun_times)),
            "y": horizon["y"],
            "x": horizon["x"],
        },
        name="irradiance",
        attrs={
            "units": "W m^-2",
            "note": "Lambertian, horizon-masked; TOA scaling only (no atmosphere).",
            "time_utc_iso": json.dumps(
                np.datetime_as_string(times_utc.astype("datetime64[s]"), unit="s", timezone="UTC").tolist()
            ),
            "toa_W_m2": float(toa),
        },
    )
    return flux
