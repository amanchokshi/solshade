import os
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from matplotlib import pyplot as plt
from pyproj import Transformer
from rasterio.transform import Affine
from rich.console import Console
from rich.markup import escape

from solshade.terrain import compute_hillshade, compute_horizon_map, compute_slope_aspect, load_dem
from solshade.viz import plot_aspect, plot_dem, plot_hillshade, plot_slope

console = Console()
app = typer.Typer(help="Terrain-aware solar illumination modeling using DEMs and orbital solar geometry.")


@app.command()
def meta(dem_path: Path = typer.Argument(..., help="Path to the input DEM GeoTIFF.")):
    """Print a neat summary of metadata from a DEM file."""
    dem = load_dem(dem_path)
    transform: Affine = dem.rio.transform()
    bounds = dem.rio.bounds()

    console.print("\n--------------------------------------------------------")
    console.print(f"  [cyan]METADATA:[/cyan] [ {dem_path.name} ]")
    console.print("--------------------------------------------------------\n")

    def field(label: str, value: str):
        console.print(f"  [green]{label:<12}[/green] [white][ {value} ][/white]")

    def format_decimal(val: float, int_width: int = 8, frac_width: int = 2) -> str:
        """
        Format float so decimal points align by padding integer part to `int_width`.
        """
        d = Decimal(val).quantize(Decimal(f"1.{'0' * frac_width}"))
        int_part, frac_part = str(d).split(".")
        return f"{int_part.rjust(int_width)}.{frac_part}"

    def print_transform(t: Affine):
        rows = [
            [t.a, t.b, t.c],
            [t.d, t.e, t.f],
            [0.0, 0.0, 1.0],
        ]
        label = "[green]TRANSFORM:[/green]"
        indent = " " * 14

        # First row
        line = ", ".join(format_decimal(v) for v in rows[0])
        console.print(f"  {label:<14}   [white]| {line} |[/white]")

        # Next rows
        for row in rows[1:]:
            line = ", ".join(format_decimal(v) for v in row)
            console.print(f" {indent}[white]| {line} |[/white]")

    # Geographic bounds (WGS84)
    if dem.rio.crs and dem.rio.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(bounds[0], bounds[1])
        lon_max, lat_max = transformer.transform(bounds[2], bounds[3])
    else:
        lon_min, lat_min, lon_max, lat_max = bounds

    field("CRS:", dem.rio.crs.to_string() if dem.rio.crs else "None")
    field("SHAPE:", f"{dem.shape[0]} x {dem.shape[1]}")
    field("RESOLUTION:", f"{abs(transform.a)} x {abs(transform.e)}")
    print_transform(transform)

    b = ", ".join(f"{v:.1f}" for v in bounds)
    field("BOUNDS:", b)

    field("LATITUDE:", f"{lat_min:.6f} to {lat_max:.6f}")
    field("LONGITUDE:", f"{lon_min:.6f} to {lon_max:.6f}")

    coords = ", ".join(str(c).upper() for c in dem.coords)
    field("COORDS:", coords)

    field("DTYPE:", str(dem.dtype).upper())

    console.print("  [green]ATTRIBUTES:[/green]")
    for k, v in dem.attrs.items():
        pretty = v
        if isinstance(v, str) and v.startswith("[") and len(v) > 60:
            pretty = v[:60] + "... (truncated)"
        console.print(f"\t[white]{str(k).upper()}: {escape(str(pretty))}[/white]")


@app.command()
def compute(
    dem_path: Path = typer.Argument(..., help="Path to the input DEM GeoTIFF."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save output GeoTIFFs."),
    slope: bool = typer.Option(False, help="Compute and save slope."),
    aspect: bool = typer.Option(False, help="Compute and save aspect."),
    hillshade: bool = typer.Option(False, help="Compute and save hillshade."),
    azimuth: float = typer.Option(315.0, help="Sun azimuth for hillshade."),
    altitude: float = typer.Option(45.0, help="Sun altitude for hillshade."),
    horizon_map: bool = typer.Option(False, help="Compute and save per-pixel horizon map."),
    n_directions: int = typer.Option(64, help="Number of azimuthal directions for horizon map."),
    max_distance: float = typer.Option(5000, help="Maximum ray length in meters for horizon map."),
    step: float = typer.Option(20, help="Step size in meters for horizon map."),
    chunk_size: int = typer.Option(32, help="Chunk size in pixels for horizon map."),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all cores)."),
    no_progress: bool = typer.Option(False, help="Disable progress bar during horizon computation."),
):
    """Compute slope, aspect, hillshade, and/or horizon map from a DEM and save them as GeoTIFFs."""
    dem = load_dem(dem_path)
    save_dir = output_dir or dem_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if slope:
        slope_da, _ = compute_slope_aspect(dem)
        out_path = save_dir / f"{dem_path.stem}_SLOPE.tif"
        slope_da.rio.to_raster(out_path)
        typer.echo(f"Saved slope to {out_path}")

    if aspect:
        _, aspect_da = compute_slope_aspect(dem)
        out_path = save_dir / f"{dem_path.stem}_ASPECT.tif"
        aspect_da.rio.to_raster(out_path)
        typer.echo(f"Saved aspect to {out_path}")

    if hillshade:
        slope_da, aspect_da = compute_slope_aspect(dem)
        hillshade_da = compute_hillshade(slope_da, aspect_da, azimuth_deg=azimuth, altitude_deg=altitude)
        out_path = save_dir / f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.tif"
        hillshade_da.rio.to_raster(out_path)
        typer.echo(f"Saved hillshade to {out_path}")

    if horizon_map:
        result = compute_horizon_map(
            dem,
            n_directions=n_directions,
            max_distance=max_distance,
            step=step,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            progress=not no_progress,
        )
        out_path = save_dir / f"{dem_path.stem}_HORIZON.tif"
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


def main():
    app()
