from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skyfield.api import Loader, wgs84
from skyfield.timelib import Timescale

# Skyfield object types are a bit loose; use "object" for now.
SunSegment = object
EarthSegment = object


def load_sun_ephemeris(
    cache_dir: Optional[Path] = None,
) -> Tuple[SunSegment, EarthSegment, Timescale]:
    """
    Load the Sun and Earth ephemeris segments from the DE440s file and a Timescale.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for Skyfield's cache. If None, uses Skyfield’s default (~/.skyfield).

    Returns
    -------
    sun : object
        Skyfield segment for the Sun (use with `.observe()`).
    earth : object
        Skyfield segment for the Earth (use with `earth + topos`).
    ts : skyfield.timelib.Timescale
        Timescale object for constructing Skyfield times.

    Raises
    ------
    FileNotFoundError
        If required ephemeris or timescale files are missing from the cache.

    Notes
    -----
    - Uses the compact DE440s kernel (sufficient for Sun/Earth work).
    - You must run once with internet access to populate the Skyfield cache,
      or manually place 'de440s.bsp' and timescale data in the cache_dir.
    """
    eph_name = "de440s.bsp"
    loader = Loader(str(cache_dir)) if cache_dir else Loader(None)

    # Load timescale
    try:
        ts = loader.timescale()
    except Exception as exc:
        where = f"cache_dir={cache_dir}" if cache_dir else "~/.skyfield"
        raise FileNotFoundError(
            f"Timescale data not found in Skyfield cache ({where}).\n"
            f"Run once with internet or manually copy the required files."
        ) from exc

    # Load ephemeris
    try:
        ephem = loader(eph_name)
    except Exception as exc:
        where = f"cache_dir={cache_dir}" if cache_dir else "~/.skyfield"
        raise FileNotFoundError(
            f"Ephemeris '{eph_name}' not found in Skyfield cache ({where}).\n"
            f"Run once with internet or manually copy the BSP file."
        ) from exc

    sun = ephem["sun"]
    earth = ephem["earth"]
    return sun, earth, ts


def compute_solar_altaz(
    lat: float,
    lon: float,
    startutc: Optional[datetime] = None,
    stoputc: Optional[datetime] = None,
    timestep: int = 3600,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Sun’s altitude and azimuth over a specified time range at a given location.

    This function uses Skyfield to compute the apparent altitude and azimuth of the Sun
    at regular time intervals between ``startutc`` and ``stoputc`` for a given geographic
    latitude and longitude.

    Parameters
    ----------
    lat : float
        Geographic latitude in degrees (positive north).
    lon : float
        Geographic longitude in degrees (positive east).
    startutc : datetime, optional
        UTC start time for calculations. If ``None``, defaults to current UTC time.
        If naive (no timezone), assumed to be UTC.
    stoputc : datetime, optional
        UTC stop time for calculations. If ``None``, defaults to one year after ``startutc``.
        If naive (no timezone), assumed to be UTC.
    timestep : int, default=3600
        Time step between calculations, in seconds. Must be positive.
    cache_dir : Path, optional
        Directory for Skyfield’s cache. If ``None``, Skyfield’s default cache directory
        (``~/.skyfield``) is used.

    Returns
    -------
    times_utc : numpy.ndarray of datetime64[ns]
        Array of UTC timestamps corresponding to each calculation time step.
    alt_deg : numpy.ndarray of float
        Apparent altitude of the Sun in degrees, same length as ``times_utc``.
        Positive values indicate the Sun is above the horizon.
    az_deg : numpy.ndarray of float
        Apparent azimuth of the Sun in degrees, same length as ``times_utc``.
        Measured clockwise from true north (0° = North, 90° = East).

    Raises
    ------
    ValueError
        If ``timestep`` is not positive or if ``stoputc`` is before or equal to ``startutc``.
    FileNotFoundError
        If required Skyfield ephemeris or timescale files are missing from the cache.

    Notes
    -----
    - Uses the compact DE440s kernel.
    - Alt/Az are computed from apparent positions (light-time delay, aberration) with
      **no** atmospheric refraction applied.
    """

    def ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    now_utc = datetime.now(timezone.utc)
    start_dt = ensure_utc(startutc) if startutc else now_utc
    stop_dt = ensure_utc(stoputc) if stoputc else (now_utc + timedelta(days=365))

    if timestep <= 0:
        raise ValueError("timestep must be a positive number of seconds")
    if stop_dt <= start_dt:
        raise ValueError("stoputc must be after startutc")

    # Build time vector (tz-aware list for Skyfield; tz-naive numpy array for return)
    total_seconds = int((stop_dt - start_dt).total_seconds())
    steps = total_seconds // timestep
    offsets = np.arange(0, steps * timestep + 1, timestep, dtype="int64")

    times_py = [start_dt + timedelta(seconds=int(s)) for s in offsets]  # tz-aware UTC
    times_utc = np.array([t.replace(tzinfo=None) for t in times_py], dtype="datetime64[ns]")

    # Skyfield computation (match frames: Earth + topocentric observer, observing Sun)
    sun, earth, ts = load_sun_ephemeris(cache_dir)
    t = ts.from_datetimes(times_py)  # expects tz-aware datetimes

    observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    astrometric = (earth + observer).at(t).observe(sun).apparent()  # type: ignore
    alt, az, _ = astrometric.altaz()

    alt_deg = np.asarray(alt.degrees)
    az_deg = np.asarray(np.mod(az.degrees, 360.0))

    return times_utc, alt_deg, az_deg
