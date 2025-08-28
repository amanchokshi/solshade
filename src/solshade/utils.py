import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import rioxarray as rxr
import xarray as xr

_LOGGER_ROOT_NAME = "solshade"

# Module-level logger (child of the solshade root); safe to import in libraries.
_log = logging.getLogger(__name__)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger under the `solshade` namespace.

    Typical usage in library modules:

        >>> from solshade.utils import get_logger
        >>> log = get_logger(__name__)
        >>> log.debug("hello")

    Parameters
    ----------
    name : str | None
        Logger name. Use `__name__` in modules. If None, returns the `solshade` root.

    Returns
    -------
    logging.Logger
        A child logger (or the solshade root if `name` is None).
    """
    return logging.getLogger(name or _LOGGER_ROOT_NAME)


class LevelColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        orig = record.levelname
        try:
            padded = f"{orig:<8}"  # keep your alignment
            color = self.COLORS.get(orig, "")
            record.levelname = f"{color}{padded}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = orig


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Path] = None,
    *,
    fmt: str = "[%(asctime)s %(levelname)-8s ] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> logging.Logger:
    """
    Configure the `solshade` root logger exactly once (idempotent).

    Safe to call many times across your app; on subsequent calls (without `force=True`)
    we simply update the level and keep existing handlers.

    Parameters
    ----------
    level : int | str, default logging.INFO
        Logging level (e.g., 10/20 or "DEBUG"/"INFO").
    log_file : Path | None
        Optional path to a file that also receives logs.
    fmt : str
        Formatter string applied to installed handlers.
    datefmt : str
        Date format string.
    force : bool
        If True, remove existing handlers and reconfigure from scratch.

    Returns
    -------
    logging.Logger
        The configured `solshade` root logger.
    """
    logger = logging.getLogger(_LOGGER_ROOT_NAME)

    # Normalize string level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if getattr(logger, "_configured", False) and not force:
        logger.setLevel(level)
        _log.debug("Logging already configured; updated level to %s", level)
        return logger

    # Fresh (or forced) setup
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicating to the global root

    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter_file = logging.Formatter(fmt, datefmt)
    formatter = LevelColorFormatter(fmt, datefmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    logger._configured = True  # type: ignore[attr-defined]
    logger.debug(
        "Configured logging: level=%s, handlers=%d, file=%s",
        level,
        len(logger.handlers),
        str(log_file) if log_file else "None",
    )
    return logger


def parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO-8601 timestamp into a timezone-aware UTC `datetime`.

    Accepts Z-suffix or offset (e.g., `...Z`, `+00:00`). Naive inputs are
    assumed to be UTC. Returns None if input is None.

    Parameters
    ----------
    s : str | None
        ISO-8601 timestamp string, or None.

    Returns
    -------
    datetime | None
        UTC-normalized datetime, or None.
    """
    if s is None:
        _log.debug("parse_iso_utc: input is None -> returning None")
        return None

    iso = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)

    if dt.tzinfo is None:
        out = dt.replace(tzinfo=timezone.utc)
        _log.debug("parse_iso_utc: parsed naive -> assumed UTC: %s", out.isoformat())
        return out

    out = dt.astimezone(timezone.utc)
    _log.debug("parse_iso_utc: parsed aware -> normalized to UTC: %s", out.isoformat())
    return out


def transfer_spatial_metadata(
    dst: xr.DataArray,
    ref: xr.DataArray,
    *,
    extra_dim: Optional[tuple[str, Any]] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Copy spatial metadata (CRS, transform, spatial dims) from `ref` to `dst`,
    and optionally record a logical leading dimension for GeoTIFF round-trips.

    This is useful when writing 3D arrays to GeoTIFF where the leading dimension
    (e.g., "time" or "azimuth") will be flattened to `band`. We persist enough
    info in attrs to restore that logical dimension on read.

    Parameters
    ----------
    dst : xr.DataArray
        Destination array to annotate with spatial metadata.
    ref : xr.DataArray
        Reference array providing CRS/transform and spatial dims.
    extra_dim : (str, array-like) | None
        If provided, stores:
          - `extra_dim_name`: the logical leading dimension name,
          - `extra_dim_values`: JSON-encoded values
          - `extra_dim_type`: dtype (or "datetime64[ns]").
        Datetime values are stored as ISO strings (UTC, second resolution).
    attrs : dict | None
        Additional attrs to merge into `dst.attrs`.

    Returns
    -------
    xr.DataArray
        Updated `dst` with spatial metadata and optional extra-dim hints.
    """
    _log.debug("Transferring spatial metadata: dst.shape=%s, ref.shape=%s", dst.shape, ref.shape)

    dst.rio.write_crs(ref.rio.crs, inplace=True)
    dst.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    dst.rio.write_transform(ref.rio.transform(), inplace=True)

    if extra_dim is not None:
        name, values = extra_dim
        values = np.asarray(values)
        _log.debug("Annotating extra_dim: name=%s, dtype=%s, size=%d", name, values.dtype, values.size)

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
        _log.debug("Merged additional attrs: %s", list(attrs.keys()))

    return dst


def write_geotiff(da: xr.DataArray, path: str, **kwargs) -> None:
    """
    Write an `xarray.DataArray` to GeoTIFF via rioxarray, preserving spatial metadata.

    Parameters
    ----------
    da : xr.DataArray
        Array with CRS/transform set (`.rio` accessors).
    path : str
        Output file path.
    **kwargs :
        Extra args passed through to `DataArray.rio.to_raster`.

    Notes
    -----
    This function does not alter the array; ensure CRS + transform are already set.
    """
    _log.info("Writing GeoTIFF to %s (shape=%s, dtype=%s)", path, da.shape, da.dtype)
    da.rio.to_raster(path, **kwargs)


def read_geotiff(path: str) -> xr.DataArray:
    """
    Read a GeoTIFF into an `xarray.DataArray` with CRS/transform retained.

    If the file was previously saved with `transfer_spatial_metadata` and contains
    `extra_dim_*` attrs, the function will:
      - rename the `band` dimension to the recorded logical name, and
      - attach the recorded coordinate values (re-decoded from JSON, including
        ISO datetimes restored to `datetime64[s]`).

    Parameters
    ----------
    path : str
        Path to the GeoTIFF.

    Returns
    -------
    xr.DataArray
        The loaded array with spatial metadata and any restored logical dimension.
    """
    _log.info("Reading GeoTIFF from %s", path)
    raw = cast(xr.DataArray, rxr.open_rasterio(path, masked=True))
    da = raw.squeeze()

    name = da.attrs.get("extra_dim_name")
    values = da.attrs.get("extra_dim_values")
    dtype = da.attrs.get("extra_dim_type")

    if name and values:
        _log.debug("Restoring extra dimension: name=%s, dtype=%s", name, dtype)
        decoded = json.loads(values)

        if dtype == "datetime64[ns]":
            decoded = [s.rstrip("Z") for s in decoded]
            coords = np.array(decoded, dtype="datetime64[s]")
        else:
            coords = np.array(decoded)

        if "band" in da.dims and da.sizes.get("band", 1) == len(coords):
            da = da.rename({"band": name})
            da = da.assign_coords({name: coords})
            _log.debug("Renamed 'band' to '%s' with %d coords", name, len(coords))
        else:
            _log.warning(
                "Extra-dim hints present but band length (%s) != values length (%s); leaving dims unchanged.",
                da.sizes.get("band"),
                len(coords),
            )

    return da
