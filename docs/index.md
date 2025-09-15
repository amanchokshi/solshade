<p align="center">
  <img src="https://github.com/amanchokshi/solshade/raw/main/docs/imgs/solshade.gif" alt="Solshade Banner" width="100%">
</p>

**Solshade** is a Python toolkit for **terrain-aware solar illumination modeling**. It combines high-resolution **digital elevation models** with accurate **orbital solar geometry** to provide detailed estimates of how sunlight interacts with complex terrain.

---

## Features

- **Terrain analysis** - Compute slope, aspect, surface normals, hillshade maps, and horizon profiles directly from DEMs.

- **Solar modeling** - Generate per-pixel solar altitude/azimuth tracks and compute terrain-aware solar flux time series.

- **Energy metrics** - Derive integrated daily energy, total energy, peak energy, and day-of-peak maps for any region.

- **Outputs & visualization** - Export results as GeoTIFFs for GIS workflows, or as publication-ready plots (PNG/SVG).  
  Interactive plotting commands help visualize DEMs, slopes, aspects, horizons, and flux maps.

---

## Potential Use Cases

- **Environmental and climate research**  
  Quantify how shading and illumination patterns affect snow/ice melt, permafrost stability, or vegetation growth. Could be of particular interest to studing and identifying microhabitats in extreme environments

- **Astronomy & observatories**  
  Understand site illumination patterns for telescope siting in remote mountainous regions.

- **Archaeology & cultural heritage**  
  Reconstruct ancient sunlight exposure of landscapes and structures.

- **Renewable energy siting**  
  Assess solar resource potential for photovoltaic installations, particularly in rugged or polar terrain.

- **Education & outreach**  
  Provide a clear, hands-on way to explore the interaction of topography and solar geometry.

---

## Getting Started

Check out the [Installation Guide](guides/install.md) and the [Getting Started Guide](guides/index.md) for installation and basic usage examples. You can also dive into the [API Reference](api/index.md) to explore documentation straight from the code.

---

## Project Goals

Solshade is designed with a few key principles in mind:

- **Accuracy** — Leverages high-quality DEMs and precise orbital ephemerides.  
- **Reproducibility** — Provides tested command-line tools and Python APIs.  
- **Flexibility** — Works for small test DEMs or large-area regional modeling.  
- **Open science** — 100% open-source, designed to integrate with the scientific Python ecosystem.

---

## Contributing

Contributions are always welcome! If you’d like to add features, improve performance, or help with documentation, please check out the repository on GitHub:

✨ [GitHub: amanchokshi/solshade](https://github.com/amanchokshi/solshade)
