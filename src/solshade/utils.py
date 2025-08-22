import json
from datetime import datetime, timezone
from typing import Any, Optional, cast

import numpy as np
import rioxarray as rxr
import xarray as xr


def parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp string (optionally with 'Z' or timezone) into a UTC datetime.

    Parameters
    ----------
    s : str or None
        ISO timestamp string or None.

    Returns
    -------
    datetime or None
        UTC-normalized datetime object or None if input was None.
    """
    if s is None:
        return None

    iso = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def transfer_spatial_metadata(
    dst: xr.DataArray,
    ref: xr.DataArray,
    *,
    extra_dim: Optional[tuple[str, Any]] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Transfer CRS, transform, and spatial dimension metadata from a reference xarray to another.
    Optionally annotate an extra leading dimension (e.g. 'time', 'azimuth') for GeoTIFF compatibility.

    Parameters
    ----------
    dst : xr.DataArray
        The destination array to receive metadata.
    ref : xr.DataArray
        The reference array with the correct CRS and transform.
    extra_dim : tuple[str, array-like], optional
        Name and values for an added leading dimension (e.g. ("azimuth", azimuths)).
        This will be stored in attrs to recover from GeoTIFF band flattening.
    attrs : dict, optional
        Any additional metadata to include.

    Returns
    -------
    xr.DataArray
        Destination array with spatial metadata and optional extra dimension info.
    """
    dst.rio.write_crs(ref.rio.crs, inplace=True)
    dst.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    dst.rio.write_transform(ref.rio.transform(), inplace=True)

    if extra_dim is not None:
        name, values = extra_dim
        values = np.asarray(values)

        if np.issubdtype(values.dtype, np.datetime64):
            iso = np.datetime_as_string(values.astype("datetime64[s]"), unit="s", timezone="UTC").tolist()  # type: ignore
            dst.attrs["extra_dim_values"] = json.dumps(iso)
            dst.attrs["extra_dim_type"] = "datetime64[ns]"
        else:
            dst.attrs["extra_dim_values"] = json.dumps(values.tolist())
            dst.attrs["extra_dim_type"] = str(values.dtype)

        dst.attrs["extra_dim_name"] = name

    if attrs:
        dst.attrs.update(attrs)

    return dst


def write_geotiff(da: xr.DataArray, path: str, **kwargs) -> None:
    """
    Write an xarray DataArray to GeoTIFF, ensuring CRS and transform are preserved.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with spatial metadata.
    path : str
        Output file path.
    **kwargs :
        Additional keyword args passed to `rio.to_raster()`
    """
    da.rio.to_raster(path, **kwargs)


def read_geotiff(path: str) -> xr.DataArray:
    """
    Read a GeoTIFF file into an xarray DataArray using rioxarray, with CRS and transform retained.

    If the file was saved with `extra_dim_name` and `extra_dim_values` in its attrs,
    it will rename the 'band' dimension accordingly.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.

    Returns
    -------
    xr.DataArray
        The loaded DataArray, with restored spatial metadata and optional extra dimension.
    """
    # da = rxr.open_rasterio(path, masked=True).squeeze()
    raw = cast(xr.DataArray, rxr.open_rasterio(path, masked=True))
    da = raw.squeeze()

    name = da.attrs.get("extra_dim_name", None)
    values = da.attrs.get("extra_dim_values", None)
    dtype = da.attrs.get("extra_dim_type")

    if name and values:
        decoded = json.loads(values)

        if dtype == "datetime64[ns]":
            decoded = [s.rstrip("Z") for s in decoded]
            coords = np.array(decoded, dtype="datetime64[s]")
        else:
            coords = np.array(decoded)

        if "band" in da.dims and da.sizes.get("band", 1) == len(coords):
            da = da.rename({"band": name})
            da = da.assign_coords({name: coords})

    return da
