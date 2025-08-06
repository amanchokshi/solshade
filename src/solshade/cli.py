import os
from pathlib import Path
from typing import Optional

import typer
from matplotlib import pyplot as plt

from solshade.terrain import compute_hillshade, compute_slope_aspect, load_dem
from solshade.viz import plot_aspect, plot_dem, plot_hillshade, plot_slope

app = typer.Typer(help="Terrain-aware solar illumination modeling using DEMs and orbital solar geometry.")


@app.command()
def compute(
    dem_path: Path = typer.Argument(..., help="Path to the input DEM GeoTIFF."),
    output_dir: Optional[Path] = typer.Option(None, help="Directory to save output GeoTIFFs."),
    slope: bool = typer.Option(False, help="Compute and save slope."),
    aspect: bool = typer.Option(False, help="Compute and save aspect."),
    hillshade: bool = typer.Option(False, help="Compute and save hillshade."),
    azimuth: float = typer.Option(315.0, help="Sun azimuth for hillshade."),
    altitude: float = typer.Option(45.0, help="Sun altitude for hillshade."),
):
    """Compute slope, aspect, and/or hillshade from a DEM and save them as GeoTIFFs."""
    dem = load_dem(dem_path)
    save_dir = output_dir or dem_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    slope_da, aspect_da = compute_slope_aspect(dem)

    if slope:
        out_path = save_dir / (dem_path.stem + "_SLOPE.tif")
        slope_da.rio.to_raster(out_path)
        typer.echo(f"Saved slope to {out_path}")

    if aspect:
        out_path = save_dir / (dem_path.stem + "_ASPECT.tif")
        aspect_da.rio.to_raster(out_path)
        typer.echo(f"Saved aspect to {out_path}")

    if hillshade:
        hillshade_da = compute_hillshade(slope_da, aspect_da, azimuth_deg=azimuth, altitude_deg=altitude)
        out_path = save_dir / f"{dem_path.stem}_HILLSHADE_{int(azimuth)}_{int(altitude)}.tif"
        hillshade_da.rio.to_raster(out_path)
        typer.echo(f"Saved hillshade to {out_path}")


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
