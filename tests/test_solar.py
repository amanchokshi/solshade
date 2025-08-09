# tests/test_solar.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import solshade.solar as solar

# -------------------------------
# Test fakes / helpers
# -------------------------------


class FakeTimescale:
    def from_datetime(self, dt: datetime) -> object:  # noqa: D401 - simple fake
        # We don't use the returned value directly in our fakes.
        return object()

    def from_datetimes(self, dt_list: list[datetime]) -> object:  # noqa: D401
        return object()


class _AltAzCarrier:
    """Carries fixed alt/az arrays and returns them from .altaz()."""

    def __init__(self, alt_arr: np.ndarray, az_arr: np.ndarray) -> None:
        self._alt = alt_arr
        self._az = az_arr

    def altaz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Return arrays that look like Skyfield Angle objects w/ .degrees
        class _Angle:
            def __init__(self, degrees: np.ndarray) -> None:
                self.degrees = degrees

        return _Angle(self._alt), _Angle(self._az), _Angle(np.zeros_like(self._alt))


class FakeObserver:
    """
    Chainable object for:
       (earth + wgs84.latlon(...)).at(t).observe(sun).apparent().altaz()
    """

    def __init__(self, alt_arr: np.ndarray, az_arr: np.ndarray) -> None:
        self._alt = alt_arr
        self._az = az_arr

    # Support `earth + FakeObserver` by returning self from __radd__
    def __radd__(self, _other: Any) -> "FakeObserver":
        return self

    def at(self, _t: object) -> "FakeObserver":
        return self

    def observe(self, _sun: object) -> "FakeObserver":
        return self

    def apparent(self) -> _AltAzCarrier:
        return _AltAzCarrier(self._alt, self._az)


def _make_fake_wgs84(alt_arr: np.ndarray, az_arr: np.ndarray):
    """Return a tiny module-like object with .latlon() → FakeObserver."""
    import types

    mod = types.SimpleNamespace()

    def _latlon(**_kwargs: Any) -> FakeObserver:
        return FakeObserver(alt_arr, az_arr)

    mod.latlon = _latlon  # type: ignore[attr-defined]
    return mod


def _stub_load_sun_ephemeris(_cache_dir: Path | None = None):
    """Return (sun, earth, ts) fakes for monkeypatch."""
    sun = object()
    earth = object()
    return sun, earth, FakeTimescale()


# -------------------------------
# compute_solar_altaz tests
# -------------------------------


def test_compute_solar_altaz_shapes_and_wrapping(monkeypatch: pytest.MonkeyPatch):
    """
    Validate:
      - time vector length and dtype
      - azimuth wrapped to [0, 360)
      - alt/az line up with samples
    """
    # 3 hours inclusive ⇒ 4 samples
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stop = start + timedelta(hours=3)
    timestep = 3600

    # Set alt arbitrary; set az to include negatives and >360 to validate wrapping.
    alt_arr = np.array([10.0, 5.0, 0.0, -5.0])
    az_arr = np.array([-10.0, 0.0, 10.0, 370.0])  # expect → [350, 0, 10, 10]

    # Patch Skyfield usage
    monkeypatch.setattr(solar, "load_sun_ephemeris", _stub_load_sun_ephemeris)
    monkeypatch.setattr(solar, "wgs84", _make_fake_wgs84(alt_arr, az_arr))

    times, alt, az = solar.compute_solar_altaz(10.0, 20.0, startutc=start, stoputc=stop, timestep=timestep)

    assert times.dtype.kind == "M" and str(times.dtype) == "datetime64[ns]"
    assert len(times) == 4
    np.testing.assert_allclose(alt, alt_arr)
    np.testing.assert_allclose(az, np.mod(az_arr, 360.0))


def test_compute_solar_altaz_naive_vs_aware(monkeypatch: pytest.MonkeyPatch):
    """Naive datetimes should be treated as UTC and match aware results."""
    start_naive = datetime(2025, 1, 1, 0, 0, 0)
    stop_naive = start_naive + timedelta(hours=2)

    start_aware = start_naive.replace(tzinfo=timezone.utc)
    stop_aware = stop_naive.replace(tzinfo=timezone.utc)

    alt_arr = np.array([0.0, 1.0, 2.0])
    az_arr = np.array([0.0, 90.0, 180.0])

    monkeypatch.setattr(solar, "load_sun_ephemeris", _stub_load_sun_ephemeris)
    monkeypatch.setattr(solar, "wgs84", _make_fake_wgs84(alt_arr, az_arr))

    t_n, alt_n, az_n = solar.compute_solar_altaz(10.0, 20.0, start_naive, stop_naive, 3600)
    t_a, alt_a, az_a = solar.compute_solar_altaz(10.0, 20.0, start_aware, stop_aware, 3600)

    np.testing.assert_array_equal(t_n, t_a)
    np.testing.assert_allclose(alt_n, alt_a)
    np.testing.assert_allclose(az_n, az_a)


def test_compute_solar_altaz_invalid_timestep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(solar, "load_sun_ephemeris", _stub_load_sun_ephemeris)
    monkeypatch.setattr(solar, "wgs84", _make_fake_wgs84(np.array([0.0]), np.array([0.0])))

    with pytest.raises(ValueError):
        solar.compute_solar_altaz(0.0, 0.0, timestep=0)


def test_compute_solar_altaz_stop_before_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(solar, "load_sun_ephemeris", _stub_load_sun_ephemeris)
    monkeypatch.setattr(solar, "wgs84", _make_fake_wgs84(np.array([0.0]), np.array([0.0])))

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    stop = start - timedelta(hours=1)
    with pytest.raises(ValueError):
        solar.compute_solar_altaz(0.0, 0.0, startutc=start, stoputc=stop, timestep=3600)


# -------------------------------
# load_sun_ephemeris tests
# -------------------------------


def test_load_sun_ephemeris_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Happy path: Loader provides timescale and ephemeris without raising."""

    class FakeLoader:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def timescale(self) -> FakeTimescale:
            return FakeTimescale()

        def __call__(self, _name: str) -> dict[str, object]:
            # Return mapping like a BSP kernel
            return {"sun": object(), "earth": object()}

    monkeypatch.setattr(solar, "Loader", FakeLoader)  # type: ignore[assignment]
    sun, earth, ts = solar.load_sun_ephemeris(cache_dir=tmp_path)
    assert ts is not None
    assert sun is not None
    assert earth is not None


def test_load_sun_ephemeris_missing_timescale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If Loader.timescale() fails, we raise FileNotFoundError."""

    class FakeLoader:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def timescale(self) -> FakeTimescale:
            raise RuntimeError("no timescale file")

        def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(solar, "Loader", FakeLoader)  # type: ignore[assignment]
    with pytest.raises(FileNotFoundError) as ei:
        solar.load_sun_ephemeris(cache_dir=tmp_path)
    assert "Timescale data not found" in str(ei.value)


def test_load_sun_ephemeris_missing_ephemeris(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If Loader('de440s.bsp') fails, we raise FileNotFoundError."""

    class FakeLoader:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def timescale(self) -> FakeTimescale:
            return FakeTimescale()

        def __call__(self, _name: str) -> dict[str, object]:
            raise RuntimeError("no de440s.bsp in cache")

    monkeypatch.setattr(solar, "Loader", FakeLoader)  # type: ignore[assignment]
    with pytest.raises(FileNotFoundError) as ei:
        solar.load_sun_ephemeris(cache_dir=tmp_path)
    assert "Ephemeris 'de440s.bsp' not found" in str(ei.value)
