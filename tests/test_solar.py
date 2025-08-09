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


def _make_times(start="2025-01-01T00:00:00", hours=48, step_h=1):
    """Utility: build a uniformly spaced datetime64 array.

    Supports fractional hours by converting to whole seconds.
    """
    start = np.datetime64(start, "s")
    # Convert fractional hours to whole seconds for timedelta64
    step_s = int(round(float(step_h) * 3600.0))
    if step_s <= 0:
        raise ValueError("step_h must be positive")
    step = np.timedelta64(step_s, "s")
    return start + np.arange(hours * 3600 // step_s) * step


def test_envelope_shape_and_closure_hourly_two_days():
    """Hourly samples over 2 days; check output shapes, closure, and monotonic az grid."""
    # 48 hourly samples, 24 per day
    times = _make_times(hours=48, step_h=1)

    # Build a simple diurnal pattern:
    # Day 1: sin over 24 slots
    # Day 2: sin + 1 (so per-slot maxima should come from day 2)
    slots = 24
    t0 = np.arange(slots)
    alt_day1 = 10.0 * np.sin(2 * np.pi * t0 / slots)
    alt_day2 = alt_day1 + 1.0

    alt = np.concatenate([alt_day1, alt_day2])

    # Azimuth: hourly steps of 15° (covers 0..345 each day)
    az_one_day = (t0 * 15.0) % 360.0
    az = np.concatenate([az_one_day, az_one_day])

    # Compute envelope with smoothing to 360 points
    az_plot, alt_min_plot, alt_max_plot = solar.solar_envelope_by_folding(times, alt, az, smooth_n=360)

    # Shapes: smooth_n + 1 (closed curve)
    assert az_plot.shape == (361,)
    assert alt_min_plot.shape == (361,)
    assert alt_max_plot.shape == (361,)

    # Closed curves: last equals first
    assert np.isclose(az_plot[-1], 360.0)
    assert np.isclose(alt_min_plot[-1], alt_min_plot[0])
    assert np.isclose(alt_max_plot[-1], alt_max_plot[0])

    # az_plot should be non-decreasing and start at 0
    assert np.isclose(az_plot[0], 0.0)
    assert np.all(np.diff(az_plot) >= 0.0)

    # The max envelope should be generally >= min envelope
    assert np.all(alt_max_plot >= alt_min_plot - 1e-6)


def test_envelope_works_with_subhourly_and_truncation():
    """15-minute cadence (>1 full day, not exact days). Function should truncate to whole days."""
    # 15-minute sampling for 1 day + 3 extra hours => 24*4 + 12 = 108 samples
    # Truncated to 96 (exactly one day with 15-min slots)
    times = _make_times(hours=27, step_h=0.25)  # 27 hours at 15-min step

    # Simple pattern: alt increases linearly during the day across slots, duplicated across the 27h
    n = times.shape[0]
    slots_per_day = int(24 / 0.25)  # 96
    t = np.arange(n)
    alt = (t % slots_per_day).astype(float)
    az = (t % slots_per_day) * (360.0 / slots_per_day)

    az_plot, alt_min_plot, alt_max_plot = solar.solar_envelope_by_folding(times, alt, az, smooth_n=180)

    # Shapes as expected
    assert az_plot.shape == (181,)
    assert alt_min_plot.shape == (181,)
    assert alt_max_plot.shape == (181,)
    # Reasonable bounds
    assert np.isfinite(alt_min_plot).all()
    assert np.isfinite(alt_max_plot).all()


def test_error_non_uniform_cadence():
    """Times not uniformly spaced should raise ValueError."""
    times = _make_times(hours=25, step_h=1)
    # Make one gap 2 hours
    times = times.copy()
    times[10:] = times[10:] + np.timedelta64(3600, "s")  # shift everything after idx 9 by 1 hour, breaking uniformity

    alt = np.zeros(times.shape[0])
    az = np.zeros(times.shape[0])

    with pytest.raises(ValueError, match="uniformly sampled"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=360)


def test_error_cadence_not_dividing_day():
    """Cadence that does not evenly divide 86400s should raise."""
    # 3000-second step (2.5h) → 86400/3000 = 28.8 not integer
    start = np.datetime64("2025-01-01T00:00:00", "s")
    step = np.timedelta64(3000, "s")
    times = start + np.arange(30) * step  # >1 day
    alt = np.linspace(-5, 5, times.size)
    az = np.linspace(0, 359, times.size) % 360

    with pytest.raises(ValueError, match="does not divide 86400"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=180)


def test_error_not_enough_for_one_day():
    """Less than one complete day of samples should raise."""
    times = _make_times(hours=12, step_h=1)  # only half a day
    alt = np.zeros(times.shape[0])
    az = np.zeros(times.shape[0])

    with pytest.raises(ValueError, match="Not enough samples for a single complete day"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=90)


def test_error_too_few_unique_az_for_cubic():
    """If slot azimuths collapse to <4 unique values, spline fit should raise."""
    times = _make_times(hours=48, step_h=1)  # 2 days, hourly
    # alt arbitrary
    alt = np.sin(np.linspace(0, 4 * np.pi, times.size))
    # Force az to a constant, so per-slot unique az stays < 4
    az = np.zeros(times.size)

    with pytest.raises(ValueError, match="unique azimuth samples"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=360)


def test_value_errors_on_bad_shapes_and_lengths():
    """Mismatched lengths or wrong dimensions should raise useful errors."""
    # Case A: times length does not match alt/az -> "lengths must match"
    times = _make_times(hours=24, step_h=1)
    alt = np.zeros(times.shape[0])
    az = np.zeros(times.shape[0])
    times_short = times[:-1]

    with pytest.raises(ValueError, match="lengths must match"):
        solar.solar_envelope_by_folding(times_short, alt, az, smooth_n=120)

    # Case B: alt/az shape mismatch -> "same shape"
    alt_mismatch = np.zeros(times.shape[0] + 1)
    with pytest.raises(ValueError, match="same shape"):
        solar.solar_envelope_by_folding(times, alt_mismatch, az, smooth_n=120)

    # Case C: wrong dimension -> "1-D arrays"
    alt_2d = np.zeros((times.shape[0], 1))
    az_2d = np.zeros((times.shape[0], 1))  # make shape match so we hit the ndim check
    with pytest.raises(ValueError, match="1-D arrays"):
        solar.solar_envelope_by_folding(times, alt_2d, az_2d, smooth_n=120)


def test_envelope_raises_when_cadence_not_dividing_day():
    """Cadence of 7 minutes (420s) does NOT divide 86400s -> ValueError."""
    # 7 minutes = 7/60 hours
    step_h = 7.0 / 60.0
    # Build > 1 day so we also satisfy the “at least one full day” requirement
    times = _make_times(hours=25, step_h=step_h)
    n = times.shape[0]
    alt = np.linspace(-5, 5, n)
    az = np.linspace(0, 359, n) % 360.0

    with pytest.raises(ValueError, match=r"does not divide 86400"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=120)


def test_envelope_raises_with_insufficient_unique_az_for_spline():
    """If the per-slot extrema produce <4 unique azimuths, smoothing must fail."""
    # Hourly for 2 full days -> clean daily folding
    times = _make_times(hours=48, step_h=1)
    n = times.shape[0]

    # Force azimuth to be constant so unique az = 1
    az = np.zeros(n, dtype=float)

    # Altitude can vary; constant works fine for provoking the az uniqueness failure
    alt = np.sin(np.linspace(0, 10, n))

    with pytest.raises(ValueError, match=r"unique azimuth samples"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=120)


def test_envelope_minimal_smoothing_path():
    """Use the minimum allowed smoothing (smooth_n=4)."""
    times = _make_times(hours=24, step_h=1)
    n = times.shape[0]

    # Non-trivial but simple signals
    alt = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    az = (np.linspace(0, 360, n, endpoint=False) + 5) % 360

    az_plot, alt_min_plot, alt_max_plot = solar.solar_envelope_by_folding(times, alt, az, smooth_n=4)
    # smooth_n=4 => az_plot length is 5 (closed curve)
    assert az_plot.shape == (5,)
    assert alt_min_plot.shape == (5,)
    assert alt_max_plot.shape == (5,)


def test_envelope_raises_when_smoothn_too_small():
    """smooth_n < 4 is invalid for cubic spline fitting."""
    times = _make_times(hours=24, step_h=1)
    n = times.shape[0]
    alt = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    az = (np.linspace(0, 360, n, endpoint=False) + 5) % 360

    with pytest.raises(ValueError, match=r"smooth_n must be >= 4"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=1)


def test_envelope_rejects_nonuniform_sampling():
    times = _make_times(hours=24, step_h=1)
    times_bad = times.copy()
    times_bad[5] = times_bad[5] + np.timedelta64(30, "s")  # break uniformity
    n = times.shape[0]
    alt = np.zeros(n)
    az = np.zeros(n)
    with pytest.raises(ValueError, match="uniformly sampled"):
        solar.solar_envelope_by_folding(times_bad, alt, az, smooth_n=120)


def test_envelope_requires_one_full_day():
    times = _make_times(hours=10, step_h=1)  # < 24h total
    n = times.shape[0]
    alt = np.zeros(n)
    az = np.zeros(n)
    with pytest.raises(ValueError, match="single complete day"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=120)


def test_envelope_requires_at_least_two_samples():
    """Covers: 'Need at least two samples to infer cadence.'"""
    times = np.array([np.datetime64("2025-01-01T00:00:00")], dtype="datetime64[s]")
    alt = np.array([10.0])
    az = np.array([180.0])
    with pytest.raises(ValueError, match="at least two samples"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=120)


def test_envelope_rejects_non_positive_timestep():
    """Covers: 'Non-positive timestep inferred from times_utc.' (zero step)."""
    t0 = np.datetime64("2025-01-01T00:00:00", "s")
    times = np.array([t0, t0], dtype="datetime64[s]")  # zero delta -> step_s == 0
    alt = np.array([10.0, 12.0])
    az = np.array([0.0, 10.0])
    with pytest.raises(ValueError, match="Non-positive timestep"):
        solar.solar_envelope_by_folding(times, alt, az, smooth_n=120)
