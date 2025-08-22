from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline
from skyfield.api import Loader, wgs84
from skyfield.timelib import Timescale

# Skyfield object types are a bit loose; use "object" for now.
SunSegment = object
EarthSegment = object


def load_sun_ephemeris(
    cache_dir: Optional[Union[str, Path]] = "data/skyfield",
) -> Tuple[SunSegment, EarthSegment, Timescale]:
    """
    Load the Sun and Earth ephemeris segments from the DE440s file and a Timescale.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for ephemeris and timescale cache. Defaults to ./data/skyfield.

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
    cache_root = Path(cache_dir) if cache_dir is not None else Path("data/skyfield")
    cache_root.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(cache_root))

    # Load timescale
    try:
        ts = loader.timescale()
    except Exception as exc:
        raise FileNotFoundError(
            f"Timescale data not found in Skyfield cache ({cache_root}).\n"
            f"Run once with internet or manually copy the required files."
        ) from exc

    # Load ephemeris
    try:
        ephem = loader(eph_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Ephemeris '{eph_name}' not found in Skyfield cache ({cache_root}).\n"
            f"Run once with internet or manually copy the BSP file."
        ) from exc

    sun = ephem["sun"]
    earth = ephem["earth"]
    return sun, earth, ts


def compute_solar_ephem(
    lat: float,
    lon: float,
    startutc: Optional[datetime] = None,
    stoputc: Optional[datetime] = None,
    timestep: int = 3600,
    cache_dir: Optional[Union[str, Path]] = "data/skyfield",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Sun’s apparent altitude/azimuth and the **unit ENU direction vector**
    over a time range at a given site.

    The ENU vector is derived *directly* from the local topocentric (alt, az) returned
    by Skyfield—avoiding intermediate ECEF rotations and their numerical pitfalls.

    Parameters
    ----------
    lat : float
        Geographic latitude in degrees (+N).
    lon : float
        Geographic longitude in degrees (+E).
    startutc : datetime, optional
        UTC start time. If None, uses start of current UTC year. If naive, assumed UTC.
    stoputc : datetime, optional
        UTC stop time. If None, defaults to one year after ``startutc``.
        Must be strictly after ``startutc``. If naive, assumed UTC.
    timestep : int, default=3600
        Sampling step in seconds. Must be > 0.
    cache_dir : str or Path, optional
        Skyfield cache directory (timescale + DE440s kernel). Default: ``./data/skyfield``.

    Returns
    -------
    times_utc : ndarray of datetime64[ns], shape (N,)
        UTC timestamps for each sample.
    alt_deg : ndarray of float, shape (N,)
        Apparent altitude (degrees). No atmospheric refraction applied.
    az_deg : ndarray of float, shape (N,)
        Apparent azimuth (degrees), clockwise from true north, wrapped to [0, 360).
    dist_au : ndarray of fload, shape (N,)
        Apparent distance between observer and Sun in astronomical units
    enu_unit : ndarray of float, shape (N, 3)
        Unit vectors pointing from the site toward the Sun in local ENU coordinates
        (columns: E, N, U). Each row has norm ~= 1.

    Raises
    ------
    ValueError
        If ``timestep`` <= 0 or if ``stoputc`` <= ``startutc``.
    FileNotFoundError
        If Skyfield cache files (timescale or ephemeris) are missing.

    Notes
    -----
    - Uses the compact DE440s kernel.
    - Apparent alt/az include light-time/aberration, **no refraction**.
    - ENU unit vector is computed via:

          E = cos(alt) * sin(az)
          N = cos(alt) * cos(az)
          U = sin(alt)

      where azimuth is CW from north (Skyfield’s default for topocentric alt/az).
    """

    def ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # ---- Time range & validation ----
    start_of_year_utc = datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc)
    start_dt = ensure_utc(startutc) if startutc else start_of_year_utc
    stop_dt = ensure_utc(stoputc) if stoputc else (start_of_year_utc + timedelta(days=365))

    if timestep <= 0:
        raise ValueError("timestep must be a positive number of seconds")
    if stop_dt <= start_dt:
        raise ValueError("stoputc must be after startutc")

    # Build time vector for Skyfield (aware) and for return (datetime64[ns])
    total_seconds = int((stop_dt - start_dt).total_seconds())
    steps = total_seconds // timestep
    offsets = np.arange(0, steps * timestep + 1, timestep, dtype="int64")

    times_py = [start_dt + timedelta(seconds=int(s)) for s in offsets]  # tz-aware
    times_utc = np.array([t.replace(tzinfo=None) for t in times_py], dtype="datetime64[ns]")

    # ---- Skyfield apparent topocentric alt/az ----
    sun, earth, ts = load_sun_ephemeris(cache_dir)
    t = ts.from_datetimes(times_py)

    observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    apparent = (earth + observer).at(t).observe(sun).apparent()  # type: ignore
    dist_au = apparent.distance().au
    alt, az, _ = apparent.altaz()

    alt_deg = np.asarray(alt.degrees, dtype=float)
    az_deg = np.asarray(np.mod(az.degrees, 360.0), dtype=float)

    # ---- Direct ENU unit vector from (alt, az) ----
    alt_rad = np.deg2rad(alt_deg)
    az_rad = np.deg2rad(az_deg)

    c_alt = np.cos(alt_rad)
    enu_unit = np.stack(
        (
            c_alt * np.sin(az_rad),  # E
            c_alt * np.cos(az_rad),  # N
            np.sin(alt_rad),  # U
        ),
        axis=1,
    ).astype(float)

    return times_utc, alt_deg, az_deg, dist_au, enu_unit


def solar_envelope_by_folding(
    times_utc: np.ndarray,
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    smooth_n: int = 360,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build daily solar envelopes (min/max altitude vs azimuth) by folding a
    uniformly sampled time series into UTC slots and taking per-slot extrema.

    Parameters
    ----------
    times_utc : np.ndarray of datetime64[*]
        UTC timestamps at uniform cadence covering >= 1 full day. Any datetime64
        unit is accepted and internally normalized to seconds.
    alt_deg : np.ndarray, shape (N,)
        Solar altitude samples in degrees.
    az_deg : np.ndarray, shape (N,)
        Solar azimuth samples in degrees. Values are wrapped to [0, 360).
    smooth_n : int, default=360
        Number of equally spaced azimuth points (degrees) for the smoothed
        envelope. Must be >= 4.

    Returns
    -------
    az_plot : np.ndarray, shape (smooth_n+1,)
        Azimuth grid in degrees, wrapped to include 360 for closed plotting.
    alt_min_plot : np.ndarray, shape (smooth_n+1,)
        Smoothed minimum-altitude envelope at az_plot.
    alt_max_plot : np.ndarray, shape (smooth_n+1,)
        Smoothed maximum-altitude envelope at az_plot.

    Raises
    ------
    ValueError
        - Non-1D or mismatched input lengths
        - Non-uniform sampling cadence, or cadence that doesn't divide 86400 s
        - Not enough samples for one full day
        - <4 unique azimuth samples for cubic spline
    """
    # ---- basic checks ----
    if alt_deg.shape != az_deg.shape:
        raise ValueError("alt_deg and az_deg must have the same shape.")
    if alt_deg.ndim != 1:
        raise ValueError("alt_deg and az_deg must be 1-D arrays.")
    if times_utc.shape[0] != alt_deg.shape[0]:
        raise ValueError("times_utc, alt_deg, az_deg lengths must match.")

    n = alt_deg.size
    if n < 2:
        raise ValueError("Need at least two samples to infer cadence.")

    # ---- infer cadence from times_utc ----
    t_sec = times_utc.astype("datetime64[s]")
    diffs = np.diff(t_sec.astype("int64"))  # seconds between samples
    step_s = int(diffs[0])
    if step_s <= 0:
        raise ValueError("Non-positive timestep inferred from times_utc.")
    if not np.all(diffs == step_s):
        raise ValueError("times_utc must be uniformly sampled.")
    if 86400 % step_s != 0:
        raise ValueError(f"Inferred cadence {step_s}s does not divide 86400s.")

    slots_per_day = 86400 // step_s

    # ---- truncate to whole days and reshape ----
    n_complete = (n // slots_per_day) * slots_per_day
    if n_complete == 0:
        raise ValueError("Not enough samples for a single complete day.")

    alt = np.asarray(alt_deg[:n_complete], dtype=float)
    az = np.mod(np.asarray(az_deg[:n_complete], dtype=float), 360.0)

    n_days = n_complete // slots_per_day
    alt_2d = alt.reshape(n_days, slots_per_day)
    az_2d = az.reshape(n_days, slots_per_day)

    # ---- per-slot extrema (vectorized) ----
    cols = np.arange(slots_per_day)
    idx_min = np.argmin(alt_2d, axis=0)
    idx_max = np.argmax(alt_2d, axis=0)

    slot_min_alt = alt_2d[idx_min, cols]
    slot_min_az = az_2d[idx_min, cols]
    slot_max_alt = alt_2d[idx_max, cols]
    slot_max_az = az_2d[idx_max, cols]

    # ---- periodic cubic spline fit ----
    if smooth_n < 4:
        raise ValueError("smooth_n must be >= 4 for cubic spline fitting.")

    az_smooth = np.linspace(0.0, 360.0, smooth_n, endpoint=False)

    def _periodic_cubic(az_in: np.ndarray, alt_in: np.ndarray) -> np.ndarray:
        order = np.argsort(az_in)
        x = az_in[order]
        y = alt_in[order]

        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]
        if xu.size < 4:
            raise ValueError("Not enough unique azimuth samples for cubic spline.")

        # wrap endpoint to enforce periodic continuity
        xw = np.concatenate([xu, [xu[0] + 360.0]])
        yw = np.concatenate([yu, [yu[0]]])

        cs = CubicSpline(xw, yw, bc_type="periodic")
        return cs(az_smooth)

    min_alt_smooth = _periodic_cubic(slot_min_az, slot_min_alt)
    max_alt_smooth = _periodic_cubic(slot_max_az, slot_max_alt)

    # ---- close the curves for continuous plotting ----
    az_plot = np.append(az_smooth, 360.0)
    alt_min_plot = np.append(min_alt_smooth, min_alt_smooth[0])
    alt_max_plot = np.append(max_alt_smooth, max_alt_smooth[0])

    return az_plot, alt_min_plot, alt_max_plot


def nearest_horizon_indices(
    sun_az_deg: np.ndarray,
    horizon_az_deg: np.ndarray,
) -> np.ndarray:
    """
    Map each solar azimuth to the nearest horizon azimuth index.

    Parameters
    ----------
    sun_az_deg : np.ndarray, shape (M,)
        Solar azimuths in degrees (any real values). Wrapped to [0, 360).
    horizon_az_deg : np.ndarray, shape (N,)
        Uniformly spaced horizon azimuths in degrees over [0, 360).
        Typically from ``np.linspace(0, 360, n_directions, endpoint=False)``.

    Returns
    -------
    idx : np.ndarray, shape (M,)
        Indices into ``horizon_az_deg`` of the nearest azimuth for each sun azimuth.
    """
    n = horizon_az_deg.size
    step = 360.0 / n
    sun_wrapped = np.mod(sun_az_deg, 360.0)
    idx = np.rint(sun_wrapped / step).astype(int) % n
    return idx
