import json
import os
from datetime import datetime, timezone
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

from solshade.solar import compute_solar_altaz
from solshade.terrain import compute_hillshade, compute_horizon_map, compute_slope_aspect, load_dem
from solshade.viz import plot_aspect, plot_dem, plot_hillshade, plot_horizon_polar, plot_slope

console = Console()
app = typer.Typer(help="Terrain-aware solar illumination modeling using DEMs and orbital solar geometry.")


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


compute_app = typer.Typer(help="Compute slope, aspect, hillshade or horizon maps from DEMs.")
app.add_typer(compute_app, name="compute")


@compute_app.command("slope")
def compute_slope_cmd(dem_path: Path, output_dir: Optional[Path] = None):
    dem = load_dem(dem_path)
    slope_da, _ = compute_slope_aspect(dem)
    out_path = (output_dir or dem_path.parent) / f"{dem_path.stem}_SLOPE.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slope_da.rio.to_raster(out_path)
    typer.echo(f"Saved slope to {out_path}")


@compute_app.command("aspect")
def compute_aspect_cmd(dem_path: Path, output_dir: Optional[Path] = None):
    dem = load_dem(dem_path)
    _, aspect_da = compute_slope_aspect(dem)
    out_path = (output_dir or dem_path.parent) / f"{dem_path.stem}_ASPECT.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aspect_da.rio.to_raster(out_path)
    typer.echo(f"Saved aspect to {out_path}")


@compute_app.command("hillshade")
def compute_hillshade_cmd(
    dem_path: Path,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    output_dir: Optional[Path] = None,
):
    dem = load_dem(dem_path)
    slope, aspect = compute_slope_aspect(dem)
    hillshade_da = compute_hillshade(slope, aspect, azimuth, altitude)
    out_path = (output_dir or dem_path.parent) / f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hillshade_da.rio.to_raster(out_path)
    typer.echo(f"Saved hillshade to {out_path}")


@compute_app.command("horizon")
def compute_horizon_cmd(
    dem_path: Path,
    n_directions: int = 64,
    max_distance: float = 5000,
    step: float = 20,
    chunk_size: int = 32,
    n_jobs: int = -1,
    no_progress: bool = False,
    output_dir: Optional[Path] = None,
):
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
    out_path = (output_dir or dem_path.parent) / f"{dem_path.stem}_HORIZON_{int(n_directions)}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.rio.to_raster(out_path)
    typer.echo(f"Saved horizon map to {out_path}")


plot_app = typer.Typer(help="Plot dem, aspect, slope or hillshade maps from a DEM and display or save them to pngs.")
app.add_typer(plot_app, name="plot")
plt.rcParams["font.family"] = "serif"


@plot_app.command("dem")
def plot_dem_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
):
    """Plot the DEM with contours."""
    dem = load_dem(dem_path)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_dem(dem, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (dem_path.stem + "_DEM.png")
        plt.tight_layout()
        plt.savefig(out_path)
        typer.echo(f"Saved DEM plot to {out_path}")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("slope")
def plot_slope_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
):
    """Plot the slope derived from a DEM."""
    dem = load_dem(dem_path)
    slope, _ = compute_slope_aspect(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_slope(slope, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (dem_path.stem + "_SLOPE.png")
        plt.tight_layout()
        plt.savefig(out_path)
        typer.echo(f"Saved slope plot to {out_path}")
    else:
        plt.tight_layout()
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


@plot_app.command("aspect")
def plot_aspect_cmd(
    dem_path: Path,
    output_dir: Optional[Path] = typer.Option(None, help="Save plot to this directory."),
):
    """Plot the aspect derived from a DEM."""
    dem = load_dem(dem_path)
    _, aspect = compute_slope_aspect(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_aspect(aspect, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (dem_path.stem + "_ASPECT.png")
        plt.tight_layout()
        plt.savefig(out_path)
        typer.echo(f"Saved aspect plot to {out_path}")
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
):
    """Plot hillshade from a DEM using specified illumination angles."""
    dem = load_dem(dem_path)
    slope, aspect = compute_slope_aspect(dem)
    _, ax = plt.subplots(figsize=(7, 5))
    ax = plot_hillshade(slope, aspect, azimuth, altitude, ax)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        typer.echo(f"Saved hillshade plot to {out_path}")
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
    # New options:
    solar: bool = typer.Option(False, help="Overlay solar envelope (min/max altitude)."),
    startutc: Optional[str] = typer.Option(None, help="ISO UTC start time, e.g. '2025-01-01T00:00:00Z'."),
    stoputc: Optional[str] = typer.Option(None, help="ISO UTC stop time, e.g. '2026-01-01T00:00:00Z'."),
    timestep: int = typer.Option(3600, help="Sampling step (seconds) for solar calculation."),
    cache_dir: Optional[Path] = typer.Option(None, help="Skyfield cache directory (defaults to ./data/skyfield)."),
):
    """Plot polar horizon profile at specified lat/lon from a HORIZON_*.tif.
    Optionally overlay a solar altitude envelope computed from Skyfield.
    """

    from solshade.solar import solar_envelope_by_folding  # assumes you've added this
    from solshade.terrain import load_dem

    def _parse_iso_utc(s: str) -> datetime:
        # Accept '...Z' or timezone-aware/naive; normalize to UTC
        iso = s.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

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
        # Times: use defaults if not supplied (your compute_solar_altaz handles defaults)
        start_dt = _parse_iso_utc(startutc) if startutc else None
        stop_dt = _parse_iso_utc(stoputc) if stoputc else None

        times_utc, alt_deg, az_deg = compute_solar_altaz(
            lat=lat, lon=lon, startutc=start_dt, stoputc=stop_dt, timestep=timestep, cache_dir=(cache_dir or "data/skyfield")
        )

        # Build smooth solar envelope (min/max altitude vs azimuth)
        az_smooth, min_alt_smooth, max_alt_smooth = solar_envelope_by_folding(times_utc, alt_deg, az_deg, smooth_n=360)

        sun_kwargs = {
            "sunaz": az_smooth,
            "sunaltmin": min_alt_smooth,
            "sunaltmax": max_alt_smooth,
        }

    plot_horizon_polar(azimuths, profile, ax, **sun_kwargs)
    ax.set_title(f"Horizon Map: [Lat: {lat:.6f}°, Lon: {lon:.6f}°]", va="bottom")

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if solar:
            out_path = output_dir / f"{horizon_path.stem}_SOLAR_{lat:.8f}_{lon:.8f}.png"
        else:
            out_path = output_dir / f"{horizon_path.stem}_{lat:.8f}_{lon:.8f}.png"
        plt.savefig(out_path)
        typer.echo(f"Saved horizon polar plot to {out_path}")
    else:
        if not os.getenv("SOLSHADE_TEST_MODE"):
            plt.show()  # pragma: no cover


def main():
    app()
