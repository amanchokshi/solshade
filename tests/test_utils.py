import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin

from solshade.utils import (
    _LOGGER_ROOT_NAME,
    LevelColorFormatter,
    configure_logging,
    get_logger,
    parse_iso_utc,
    read_geotiff,
    transfer_spatial_metadata,
    write_geotiff,
)


@pytest.fixture
def dummy_dataarrays():
    data = np.ones((10, 10), dtype=np.float32)
    ref = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "x": np.linspace(0, 9, 10),
            "y": np.linspace(0, 9, 10),
        },
    )
    ref.rio.write_crs("EPSG:32633", inplace=True)
    ref.rio.write_transform(Affine.translation(100, 200) * Affine.scale(10, -10), inplace=True)
    ref.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    dst = xr.DataArray(np.zeros_like(data), dims=("y", "x"))
    return dst, ref


# ------------------------------------------------------------------------------
# Tests: logging helpers in utils.py
# ------------------------------------------------------------------------------


def test_get_logger_names():
    """get_logger(None) returns the solshade root; get_logger(child) returns a namespaced logger."""
    root = get_logger()
    assert isinstance(root, logging.Logger)
    assert root.name == _LOGGER_ROOT_NAME

    child = get_logger("solshade.sub.module")
    assert isinstance(child, logging.Logger)
    assert child.name == "solshade.sub.module"


def test_level_color_formatter_only_colors_levelname():
    """
    LevelColorFormatter should colorize the level *only* and leave the message/body uncolored.
    We format a synthetic record and check where ANSI codes appear.
    """
    fmt = "[%(asctime)s %(levelname)-8s ] %(name)s: %(message)s"
    formatter = LevelColorFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    record = logging.LogRecord(
        name="solshade.test",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=123,
        msg="hello 123",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)

    # Has one or more ANSI sequences
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    assert ansi_re.search(rendered), f"Expected ANSI color codes in: {rendered!r}"

    # Split after the bracketed prefix; right-hand side should be free of ANSI
    # e.g. "[2025-... DEBUG   ] solshade.test: hello 123"
    rhs = rendered.split("] ", 1)[-1]
    assert "solshade.test: hello 123" in rhs
    assert ansi_re.search(rhs) is None, f"Message/body should not contain ANSI: {rhs!r}"

    # The level field is padded to width >= 8 (we used %-8s in fmt), so a closing bracket aligns.
    # We can check the exact token appears (with any color codes stripped).
    # Remove ANSI then verify the bracketed prefix contains 'DEBUG   '.
    no_ansi = ansi_re.sub("", rendered)
    prefix = no_ansi.split("]")[0]  # "[YYYY-mm-dd ... DEBUG   "
    assert "DEBUG   " in prefix  # DEBUG padded to width 8


def test_configure_logging_idempotent_and_force(tmp_path):
    """
    - First call installs handlers.
    - Second call (no force) updates level but doesn't duplicate handlers.
    - Force=True removes existing handlers (including a sentinel) and re-adds fresh ones.
    """
    # Fresh config
    logger = configure_logging(level=logging.INFO, force=True)
    hcount1 = len(logger.handlers)

    # Add a sentinel handler to ensure force clears it
    sentinel = logging.NullHandler()
    logger.addHandler(sentinel)
    assert sentinel in logger.handlers

    # Reconfigure without force: level updated, handlers unchanged (sentinel still there)
    logger2 = configure_logging(level=logging.DEBUG)
    assert logger2 is logger
    assert logger.level == logging.DEBUG
    assert sentinel in logger.handlers
    assert len(logger.handlers) == hcount1 + 1

    # Reconfigure with force: sentinel removed; handler count back to baseline (stream + maybe file)
    logger3 = configure_logging(level=logging.INFO, force=True)
    assert logger3 is logger
    assert sentinel not in logger.handlers
    assert len(logger.handlers) == hcount1  # stream handler restored (no logfile yet)


def test_configure_logging_writes_to_file(tmp_path):
    """
    When log_file is provided, logs should be written to that file with our bracketed timestamp format.
    """
    log_path = tmp_path / "solshade.log"
    _ = configure_logging(level="INFO", log_file=log_path, force=True)

    test_log = get_logger("solshade.cli.test")
    test_log.info("File write check")

    data = log_path.read_text()
    # Contains message
    assert "File write check" in data
    # Contains bracketed timestamp and level column
    # Example: "[2025-08-28 05:26:03 INFO     ] solshade.cli.test: File write check"
    assert data.startswith("[")
    assert "]" in data and " solshade.cli.test: " in data


# ------------------------------------------------------------------------------
# Tests: transfer_spatial_metadata
# ------------------------------------------------------------------------------


def test_transfer_spatial_metadata_basic(dummy_dataarrays):
    dst, ref = dummy_dataarrays
    result = transfer_spatial_metadata(dst, ref)

    assert result.rio.crs == ref.rio.crs
    assert result.rio.transform() == ref.rio.transform()
    assert set(result.dims) == {"y", "x"}


def test_transfer_spatial_metadata_with_extra_dim(dummy_dataarrays):
    dst, ref = dummy_dataarrays
    azimuths = np.linspace(0, 360, 4)
    result = transfer_spatial_metadata(dst, ref, extra_dim=("azimuth", azimuths))

    assert result.attrs["extra_dim_name"] == "azimuth"
    assert json.loads(result.attrs["extra_dim_values"]) == azimuths.tolist()
    assert result.attrs["extra_dim_type"].startswith("float")


def test_transfer_spatial_metadata_with_datetime(dummy_dataarrays):
    dst, ref = dummy_dataarrays
    times = np.array(["2025-01-01T00:00", "2025-01-01T01:00"], dtype="datetime64[s]")
    result = transfer_spatial_metadata(dst, ref, extra_dim=("time", times))

    assert result.attrs["extra_dim_name"] == "time"
    assert result.attrs["extra_dim_type"] == "datetime64[ns]"
    assert json.loads(result.attrs["extra_dim_values"]) == [
        "2025-01-01T00:00:00Z",
        "2025-01-01T01:00:00Z",
    ]


def test_transfer_spatial_metadata_with_attrs(dummy_dataarrays):
    dst, ref = dummy_dataarrays
    custom_attrs = {"units": "W/m^2", "long_name": "test variable"}
    result = transfer_spatial_metadata(dst, ref, attrs=custom_attrs)

    for k, v in custom_attrs.items():
        assert result.attrs[k] == v


# ------------------------------------------------------------------------------
# Tests: GeoTIFF roundtrip
# ------------------------------------------------------------------------------


def test_write_and_read_geotiff(tmp_path, dummy_dataarrays):
    dst, ref = dummy_dataarrays
    times = np.array(["2025-01-01T00:00", "2025-01-01T01:00"], dtype="datetime64[s]")

    stacked = xr.concat([dst, dst], dim="time")
    stacked = transfer_spatial_metadata(stacked, ref, extra_dim=("time", times))

    path = tmp_path / "test.tif"
    write_geotiff(stacked, path)

    reloaded = read_geotiff(path)

    assert reloaded.sizes["time"] == 2
    assert reloaded.rio.crs == ref.rio.crs
    assert "extra_dim_name" in reloaded.attrs
    assert reloaded.attrs["extra_dim_name"] == "time"
    assert np.issubdtype(reloaded.coords["time"].dtype, np.datetime64)


def test_read_geotiff_with_non_datetime_extra_dim(tmp_path, dummy_dataarrays):
    dst, ref = dummy_dataarrays
    levels = np.array([1, 2], dtype="int16")

    stacked = xr.concat([dst, dst], dim="level")
    stacked = transfer_spatial_metadata(stacked, ref, extra_dim=("level", levels))

    path = tmp_path / "int_band.tif"
    write_geotiff(stacked, path)

    reloaded = read_geotiff(path)

    assert "level" in reloaded.dims
    assert np.array_equal(reloaded["level"].values, levels)


def test_read_geotiff_mismatched_extra_dim_warns(tmp_path, caplog):
    """If attrs specify extra_dim_name/values but the band size doesn't match,
    read_geotiff should warn and leave 'band' dimension unchanged."""

    # Build a small (band, y, x) array with band=3, but 4 "azimuth" values to force mismatch
    bands, ny, nx = 3, 5, 4
    data = np.zeros((bands, ny, nx), dtype=np.float32)
    da = (
        xr.DataArray(
            data,
            dims=("band", "y", "x"),
            coords={"band": np.arange(1, bands + 1), "y": np.arange(ny), "x": np.arange(nx)},
            attrs={
                "extra_dim_name": "azimuth",
                "extra_dim_values": json.dumps([0, 90, 180, 270]),  # mismatch (4) vs band (3)
                "extra_dim_type": "int64",
            },
        )
        .rio.write_crs("EPSG:3413")
        .rio.write_transform(from_origin(0, 0, 1, 1))
    )

    path = tmp_path / "mismatch.tif"
    da.rio.to_raster(path)

    # Temporarily allow pytest to capture our log via root handlers
    solshade_logger = logging.getLogger("solshade")
    old_handlers = list(solshade_logger.handlers)
    old_propagate = solshade_logger.propagate
    try:
        solshade_logger.handlers[:] = []  # detach custom handlers
        solshade_logger.propagate = True  # let records reach root (caplog)
        with caplog.at_level(logging.WARNING):
            re = read_geotiff(str(path))
        # Dimension should remain 'band' (no rename)
        assert "band" in re.dims and "azimuth" not in re.dims
        # Warning message captured
        assert any("band length" in rec.getMessage() and "leaving dims unchanged" in rec.getMessage() for rec in caplog.records)
    finally:
        solshade_logger.handlers[:] = old_handlers
        solshade_logger.propagate = old_propagate


# ------------------------------------------------------------------------------
# Tests: parse_iso_utc
# ------------------------------------------------------------------------------


def test_parse_iso_utc_with_z_suffix():
    result = parse_iso_utc("2025-08-17T12:34:56Z")
    expected = datetime(2025, 8, 17, 12, 34, 56, tzinfo=timezone.utc)
    assert result == expected


def test_parse_iso_utc_with_offset():
    result = parse_iso_utc("2025-08-17T14:34:56+02:00")
    expected = datetime(2025, 8, 17, 12, 34, 56, tzinfo=timezone.utc)
    assert result == expected


def test_parse_iso_utc_naive_datetime():
    result = parse_iso_utc("2025-08-17T12:34:56")
    expected = datetime(2025, 8, 17, 12, 34, 56, tzinfo=timezone.utc)
    assert result == expected


def test_parse_iso_utc_none_returns_none():
    assert parse_iso_utc(None) is None
