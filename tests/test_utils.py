import json
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from affine import Affine

from solshade.utils import (
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
