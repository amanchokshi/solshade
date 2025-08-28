import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from matplotlib import pyplot as plt
from pyproj import Transformer
from rasterio.transform import Affine, rowcol
from rich.console import Console
from rich.markup import escape

from solshade.irradiance import compute_energy_metrics, compute_flux_timeseries
from solshade.solar import compute_solar_ephem, solar_envelope_by_folding
from solshade.terrain import (
    compute_hillshade,
    compute_horizon_map,
    compute_slope_aspect_normals,
    load_dem,
)
from solshade.utils import (
    configure_logging,
    get_logger,
    parse_iso_utc,
    read_geotiff,
    transfer_spatial_metadata,
    write_geotiff,
)
from solshade.viz import (
    plot_aspect,
    plot_day_of_peak,
    plot_dem,
    plot_hillshade,
    plot_horizon_polar,
    plot_normals,
    plot_peak_energy,
    plot_slope,
    plot_total_energy,
)

console = Console()
app = typer.Typer(help="Terrain-aware solar illumination modeling using DEMs and orbital solar geometry.")
log = get_logger(__name__)


# =========================
# Global logging options
# =========================
@app.callback(invoke_without_command=False)
def _configure_cli_logging(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Increase verbosity (DEBUG). Default without this flag is INFO.",
    ),
    logfile: Optional[Path] = typer.Option(
        None,
        "--logfile",
        help="If provided, also write logs to this file.",
    ),
):
    """
    Configure solshade logging for the CLI process via utils.configure_logging.
    Default level is INFO; -v/--verbose switches to DEBUG. If --logfile is set,
    logs are also written to that file.
    """
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level, log_file=logfile)
    log.debug("CLI logging configured (level=%s, logfile=%s)", logging.getLevelName(level), str(logfile) if logfile else "None")


# =========================
# META
# =========================
@app.command()
def meta(dem_path: Path = typer.Argument(..., help="Path to the input DEM GeoTIFF.")):
    """A neat metadata summary from a DEM file."""
    dem = load_dem(dem_path)
    transform: Affine = dem.rio.transform()
    bounds = dem.rio.bounds()

    console.print("\n--------------------------------------------------------")
    console.print(f"  [cyan]METADATA:[/cyan] [ {dem_path.name} ]")
    console.print("--------------------------------------------------------\n")

    def field(label: str, value: str):
        console.print(f"  [green]{label:<12}[/green] [white][ {value} ][/white]")

    def format_decimal(val: float, int_width: int = 8, frac_width: int = 2) -> str:
        d = Decimal(val).quantize(Decimal(f"1.{'0' * frac_width}"))
        int_part, frac_part = str(d).split(".")
        return f"{int_part.rjust(int_width)}.{frac_part}"

    def print_transform(t: Affine):
        rows = [[t.a, t.b, t.c], [t.d, t.e, t.f], [0.0, 0.0, 1.0]]
        label = "[green]TRANSFORM:[/green]"
        indent = " " * 14

        line = ", ".join(format_decimal(v) for v in rows[0])
        console.print(f"  {label:<14}   [white]| {line} |[/white]")
        for row in rows[1:]:
            line = ", ".join(format_decimal(v) for v in row)
            console.print(f" {indent}[white]| {line} |[/white]")

    if dem.rio.crs and dem.rio.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(bounds[0], bounds[1])
        lon_max, lat_max = transformer.transform(bounds[2], bounds[3])
    else:
        lon_min, lat_min, lon_max, lat_max = bounds

    field("CRS:", dem.rio.crs.to_string() if dem.rio.crs else "None")
    field("SHAPE:", f"{str(dem.shape)[1:-1]}")
    field("RESOLUTION:", f"{abs(transform.a)} x {abs(transform.e)}")
    print_transform(transform)

    field("BOUNDS:", ", ".join(f"{v:.1f}" for v in bounds))
    field("LATITUDE:", f"{lat_min:.6f} to {lat_max:.6f}")
    field("LONGITUDE:", f"{lon_min:.6f} to {lon_max:.6f}")
    field("COORDS:", ", ".join(str(c).upper() for c in dem.coords))
    field("DTYPE:", str(dem.dtype).upper())

    console.print("  [green]ATTRIBUTES:[/green]")
    for k, v in dem.attrs.items():
        pretty = v
        if isinstance(v, str) and v.startswith("[") and len(v) > 60:
            pretty = v[:60] + "... (truncated)"
        console.print(f"\t[white]{str(k).upper()}: {escape(str(pretty))}[/white]")


# =========================
# Compute sub-commands
# =========================
compute_app = typer.Typer(help="Compute slope, aspect, normals, hillshade, horizon or flux timeseries maps from DEMs.")
app.add_typer(compute_app, name="compute")


@compute_app.command("slope")
def compute_slope_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """Compute slope from DEM and save GeoTIFF."""
    dem = load_dem(dem_path)
    slope_da, _, _ = compute_slope_aspect_normals(dem)

    outdir = output_dir or dem_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (filename if filename else f"{dem_path.stem}_SLOPE.tif")

    slope_da.rio.to_raster(out_path)
    log.info(f"Saved slope to {out_path}")
    typer.echo(" ")


@compute_app.command("aspect")
def compute_aspect_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """Compute aspect from DEM and save GeoTIFF."""
    dem = load_dem(dem_path)
    _, aspect_da, _ = compute_slope_aspect_normals(dem)

    outdir = output_dir or dem_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (filename if filename else f"{dem_path.stem}_ASPECT.tif")

    aspect_da.rio.to_raster(out_path)
    log.info(f"Saved aspect to {out_path}")
    typer.echo(" ")


@compute_app.command("normals")
def compute_normal_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """Compute ENU surface normals from DEM and save 3-band GeoTIFF."""
    dem = load_dem(dem_path)
    _, _, normal = compute_slope_aspect_normals(dem)

    outdir = output_dir or dem_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (filename if filename else f"{dem_path.stem}_NORMALS.tif")

    normal.rio.to_raster(out_path)
    log.info(f"Saved normals to {out_path}")
    typer.echo(" ")


@compute_app.command("hillshade")
def compute_hillshade_cmd(
    dem_path: Path,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """Compute hillshade from DEM and save GeoTIFF."""
    dem = load_dem(dem_path)
    slope, aspect, _ = compute_slope_aspect_normals(dem)
    hillshade_da = compute_hillshade(slope, aspect, azimuth, altitude)

    outdir = output_dir or dem_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    default_name = f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.tif"
    out_path = outdir / (filename if filename else default_name)

    hillshade_da.rio.to_raster(out_path)
    log.info(f"Saved hillshade to {out_path}")
    typer.echo(" ")


@compute_app.command("horizon")
def compute_horizon_cmd(
    dem_path: Path,
    n_directions: int = 360,
    max_distance: float = 2000,
    step: float = 20,
    chunk_size: int = 32,
    n_jobs: int = -1,
    no_progress: bool = False,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """Compute horizon map from DEM and save GeoTIFF."""
    dem = load_dem(dem_path)
    result = compute_horizon_map(
        dem,
        n_directions=n_directions,
        max_distance=max_distance,
        step=step,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        progress=not no_progress,
    )

    outdir = output_dir or dem_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    default_name = f"{dem_path.stem}_HORIZON_{int(n_directions)}.tif"
    out_path = outdir / (filename if filename else default_name)

    result.rio.to_raster(out_path)
    log.info(f"Saved horizon map to {out_path}")
    typer.echo(" ")


@compute_app.command("fluxseries")
def compute_fluxseries_cmd(
    dem_path: Path = typer.Argument(..., help="Path to the input DEM GeoTIFF."),
    horizon_path: Path = typer.Argument(..., help="Path to the HORIZON GeoTIFF."),
    lat: Optional[float] = typer.Option(None, help="Latitude in degrees (optional, defaults to DEM center)."),
    lon: Optional[float] = typer.Option(None, help="Longitude in degrees (optional, defaults to DEM center)."),
    start_utc: Optional[str] = typer.Option(None, help="ISO UTC start time, e.g. '2025-01-01T00:00:00Z'."),
    stop_utc: Optional[str] = typer.Option(None, help="ISO UTC stop time, e.g. '2026-01-01T00:00:00Z'."),
    batch_size: int = typer.Option(512, help="Number of time steps per parallel batch."),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (default: all cores)."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save output GeoTIFFs"),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        help="Custom output tif/png filename for compute/plot commands respectively.",
    ),
):
    """
    Compute per-pixel solar flux timeseries using DEM and horizon maps, accounting for terrain shading and geometry.

    Produces daily energy maps, total integrated energy, day of peak, and peak energy GeoTIFFs.

    Note: --filename is treated as a base stem; metric-specific suffixes are still appended.
    """
    output_dir = output_dir or dem_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Computing flux timeseries from {dem_path.name} and {horizon_path.name}...")

    dem = load_dem(dem_path)
    horizon = load_dem(horizon_path)
    slope, aspect, normal_enu = compute_slope_aspect_normals(dem)

    # Derive lat/lon from DEM center if not provided
    if lat is None or lon is None:
        ny, nx = dem.shape[-2:]
        cy, cx = ny // 2, nx // 2
        x, y = dem.rio.transform() * (cx + 0.5, cy + 0.5)
        if dem.rio.crs and dem.rio.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
            lon_derived, lat_derived = transformer.transform(x, y)
        else:
            lon_derived, lat_derived = x, y
        lat_val: float = float(lat_derived)
        lon_val: float = float(lon_derived)
    else:
        # lat/lon are already floats (but Optional in the signature), so narrow them
        lat_val = float(lat)
        lon_val = float(lon)

    times_utc, solar_alt, solar_az, sun_au, solar_enu = compute_solar_ephem(
        lat=lat_val, lon=lon_val, startutc=parse_iso_utc(start_utc), stoputc=parse_iso_utc(stop_utc)
    )

    flux = compute_flux_timeseries(
        horizon=horizon,
        sun_alt_deg=solar_alt,
        sun_az_deg=solar_az,
        sun_au=sun_au,
        sun_enu=solar_enu,
        normal_enu=normal_enu,
        times_utc=times_utc,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    log.info("Computing flux metrics")
    daily_energy, total_energy, peak_energy, day_of_peak = compute_energy_metrics(flux)
    daily_iso = np.datetime_as_string(daily_energy.time.astype("datetime64[s]"), unit="s", timezone="UTC").tolist()  # type: ignore

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "daily_energy": daily_energy,
        "total_energy": total_energy,
        "peak_energy": peak_energy,
        "day_of_peak": day_of_peak,
    }

    # If --filename provided, treat it as the base stem for all outputs
    base_stem = Path(filename).stem if filename else dem_path.stem

    for xr_name, xr_array in data.items():
        out_path = (output_dir or horizon_path.parent) / f"{base_stem}_{xr_name.upper()}.tif"
        xr_array = transfer_spatial_metadata(xr_array, dem, attrs={"daily_iso_times": json.dumps(daily_iso)})
        write_geotiff(xr_array, str(out_path))
        log.info(f"Saved {xr_name.upper()} map to {out_path}")

    typer.echo(" ")


# =========================
# Plot sub-commands
# =========================
plot_app = typer.Typer(help="Plot dem, aspect, slope or hillshade maps from a DEM and display or save them to pngs.")
app.add_typer(plot_app, name="plot")
plt.rcParams["font.family"] = "serif"


@plot_app.command("dem")
def plot_dem_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot the DEM with contours."""
    dem = load_dem(dem_path)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_dem(dem, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{dem_path.stem}_DEM.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved DEM plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("slope")
def plot_slope_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot the slope derived from a DEM."""
    dem = load_dem(dem_path)
    slope, _, _ = compute_slope_aspect_normals(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_slope(slope, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{dem_path.stem}_SLOPE.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved slope plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("aspect")
def plot_aspect_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot the aspect derived from a DEM."""
    dem = load_dem(dem_path)
    _, aspect, _ = compute_slope_aspect_normals(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_aspect(aspect, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{dem_path.stem}_ASPECT.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved aspect plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("normals")
def plot_normals_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot RGB map of ENU normals derived from a DEM."""
    dem = load_dem(dem_path)
    _, _, normals = compute_slope_aspect_normals(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_normals(normals, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{dem_path.stem}_NORMALS.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved normals plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("hillshade")
def plot_hillshade_cmd(
    dem_path: Path,
    azimuth: float = typer.Option(315.0, help="Sun azimuth in degrees."),
    altitude: float = typer.Option(45.0, help="Sun altitude in degrees."),
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot hillshade from a DEM using specified illumination angles."""
    dem = load_dem(dem_path)
    slope, aspect, _ = compute_slope_aspect_normals(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_hillshade(slope, aspect, azimuth, altitude, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.png"
        out_path = output_dir / (filename if filename else default_name)
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved hillshade plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("horizon")
def plot_horizon_cmd(
    horizon_path: Path = typer.Argument(..., help="Path to HORIZON_*.tif GeoTIFF."),
    lat: float = typer.Option(..., help="Latitude of point of interest (degrees, +N)."),
    lon: float = typer.Option(..., help="Longitude of point of interest (degrees, +E)."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save polar plot."),
    solar: bool = typer.Option(False, help="Overlay solar envelope (min/max altitude)."),
    startutc: Optional[str] = typer.Option(None, help="ISO UTC start time, e.g. '2025-01-01T00:00:00Z'."),
    stoputc: Optional[str] = typer.Option(None, help="ISO UTC stop time, e.g. '2026-01-01T00:00:00Z'."),
    timestep: int = typer.Option(3600, help="Sampling step (seconds) for solar calculation."),
    cache_dir: Optional[Path] = typer.Option(None, help="Skyfield cache directory (defaults to ./data/skyfield)."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot polar horizon profile at specified lat/lon from a HORIZON_*.tif.
    Optionally overlay a solar altitude envelope computed from Skyfield.
    """
    horizon_da = load_dem(horizon_path)

    # Reproject the query lon/lat into the raster CRS and find its pixel
    transformer = Transformer.from_crs("EPSG:4326", horizon_da.rio.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    transform = horizon_da.rio.transform()
    row, col = rowcol(transform, x, y)

    ny, nx = horizon_da.shape[1], horizon_da.shape[2]
    if not (0 <= row < ny and 0 <= col < nx):
        # Build friendly bounds in geographic coords
        left, bottom, right, top = horizon_da.rio.bounds()
        reverse_transformer = Transformer.from_crs(horizon_da.rio.crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = reverse_transformer.transform(left, bottom)
        lon_max, lat_max = reverse_transformer.transform(right, top)
        raise typer.BadParameter(
            f"LAT/LON ({lat:.6f}, {lon:.6f}) falls outside the raster bounds.\n"
            f"Valid LAT range: [{lat_min:.6f}, {lat_max:.6f}]\n"
            f"Valid LON range: [{lon_min:.6f}, {lon_max:.6f}]"
        )

    # Read azimuth axis and the horizon profile at the target pixel
    azimuths = np.asarray(json.loads(horizon_da.attrs["azimuths_deg"]))
    profile = horizon_da[:, row, col].values

    # Make the plot
    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 5))

    # Optional solar overlay (envelope)
    sun_kwargs = {}
    if solar:
        times_utc, alt_deg, az_deg, _, _ = compute_solar_ephem(
            lat=lat,
            lon=lon,
            startutc=parse_iso_utc(startutc),
            stoputc=parse_iso_utc(stoputc),
            timestep=timestep,
            cache_dir=(cache_dir or "data/skyfield"),
        )
        az_smooth, min_alt_smooth, max_alt_smooth = solar_envelope_by_folding(times_utc, alt_deg, az_deg, smooth_n=360)
        sun_kwargs = {"sunaz": az_smooth, "sunaltmin": min_alt_smooth, "sunaltmax": max_alt_smooth}

    plot_horizon_polar(azimuths, profile, ax, **sun_kwargs)
    ax.set_title(f"Horizon Map: [Lat: {lat:.6f}°, Lon: {lon:.6f}°]", va="bottom")

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            out_path = output_dir / filename
        else:
            if solar:
                out_path = output_dir / f"{horizon_path.stem}_SOLAR_{lat:.8f}_{lon:.8f}.png"
            else:
                out_path = output_dir / f"{horizon_path.stem}_{lat:.8f}_{lon:.8f}.png"
        plt.savefig(out_path)
        log.info(f"Saved horizon polar plot to {out_path}")
        typer.echo(" ")
    else:
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("total-energy")
def plot_total_energy_cmd(
    total_energy_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot total energy map."""
    total_energy = read_geotiff(str(total_energy_path))
    _, ax = plt.subplots(figsize=(7, 5))
    plot_total_energy(total_energy, ax=ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{total_energy_path.stem}_TOTAL_ENERGY.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved TOTAL ENERGY plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("peak-energy")
def plot_peak_energy_cmd(
    peak_energy_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot peak energy map."""
    peak_energy = read_geotiff(str(peak_energy_path))
    _, ax = plt.subplots(figsize=(7, 5))
    plot_peak_energy(peak_energy, ax=ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{peak_energy_path.stem}_PEAK_ENERGY.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved PEAK ENERGY plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("day-of-peak")
def plot_dop_cmd(
    day_of_peak_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
    filename: Optional[str] = typer.Option(
        None, "--filename", help="Custom output tif/png filename for compute/plot commands respectively."
    ),
):
    """Plot day of peak map."""
    dop = read_geotiff(str(day_of_peak_path))
    raw = dop.attrs.get("daily_iso_times", "[]")
    daily_iso = np.array([s.rstrip("Z") for s in json.loads(raw)], dtype="datetime64[s]")
    _, ax = plt.subplots(figsize=(7, 5))
    plot_day_of_peak(dop, daily_iso, sigma=9, ax=ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (filename if filename else f"{day_of_peak_path.stem}_DAY_OF_PEAK.png")
        plt.tight_layout()
        plt.savefig(out_path)
        log.info(f"Saved DAY OF PEAK plot to {out_path}")
        typer.echo(" ")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


def main():
    app()
