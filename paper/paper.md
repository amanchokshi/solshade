---
title: "Solshade: Terrain-aware Solar Illumination Modeling using DEMs and Orbital Geometry"
tags:
  - Python
  - Solar Radiation
  - Terrain Analysis
  - Geospatial Modeling
  - Digital Elevation Models
authors:
  - name: Aman Chokshi
    orcid: 0000-0003-1130-6390
    affiliation: "1, 2"
affiliations:
  - name: Department of Physics, McGill University, Montréal, Québec H3A 2T8, Canada
    index: 1
  - name: Trottier Space Institute, McGill University, Montréal, Québec H3A 2T8, Canada
    index: 2
date: 21 September 2025
bibliography: paper.bib
---
  
# Introduction

*Solshade* is a Python library for modeling solar illumination over complex
terrain using *Digital Elevation Models (DEMs)* and precise *orbital geometry*.
It bridges geospatial analysis and astronomical modeling, enabling researchers
to quantify how sunlight interacts with landscapes over time. Applications
include studying permafrost thaw, plant life cycles, snowmelt dynamics, and
other solar-driven processes in diverse environments.

Solshade provides both a *command-line interface (CLI)* and a *Python API*.
Outputs are written as *GeoTIFFs* for geospatial compatibility, and built-in
visualization tools allow rapid inspection of terrain attributes and solar flux
maps.

# Statement of Need

Understanding the spatial and temporal variability of solar illumination over
terrain is essential for environmental science, ecology, and energy modeling.
Existing GIS tools focus on the geospatial and terrestrial aspects of this
problem but offer no capability for integrating astronomical modeling of solar
orbits or bridging these domains.

*Solshade* fills this gap by combining terrain analysis, solar orbit
forecasting, and ray-traced shading into a single, reproducible Python package
It enables studies requiring both astronomical accuracy and geospatial
flexibility.

# Software Design and Theory

Solshade computes solar flux using four main components: terrain modeling,
horizon mapping, orbital modeling, and flux computation.

## Terrain Modelling

DEMs encode elevation values on a geographic grid. From this data, Solshade
computes terrain slope, aspect, and surface normals using Numpy [@numpy]. These
normals form the basis for Lambertian solar flux calculations.

## Horizon Mapping

Shadows depend on local topography. For each pixel, Solshade samples discrete
azimuthal rays, tracing elevations outward from the pixel center. The peak
elevations along each ray define the local horizon profile, enabling shadow
masking at arbitrary solar positions.

## Solar Orbital Modelling

Using high-precision ephemerides from NASA's Jet Propulsion Laboratory [@de440]
via Skyfield [@skyfield], Solshade computes solar position vectors at
user-defined times and locations on Earth. This provides accurate solar
geometry for any observing period.

## Solar Flux Time Series

For each time step, Solshade computes the dot product between terrain normals
and solar position vectors, masking periods when the Sun is below the horizon.
The result is a per-pixel time series of incident solar radiation, accounting
for both terrain slope and topographic shading.

# Demonstration

To illustrate *Solshade*'s capabilities, we analyze a Digital Elevation Model
(DEM) of an Arctic landscape and compute solar illumination metrics over an
entire year. Figure 1 shows three geospatial layers produced by
Solshade:
(i) the input DEM,
(ii) the total accumulated solar energy, and
(iii) the day of peak solar energy for each pixel.

The bottom panels show solar irradiance time series for eight selected
locations, chosen to span the full range of total energy values. These
light-curve panels highlight how topography strongly modulates solar exposure:
valley pixels receive sunlight for only brief intervals, while ridgeline pixels
remain illuminated nearly all day. The analysis demonstrates how *Solshade*
integrates terrain geometry and solar orbital modeling to produce both spatial
and temporal diagnostics of solar radiation.

![Top row: (i) Digital Elevation Model, (ii) Total solar energy over the study
period, and (iii) Day of peak solar energy. Bottom panels: Solar irradiance
time series for eight selected locations, illustrating differences in diurnal
illumination across terrain features.](imgs/solshade.pdf){ width=100% }

# Acknowledgements

*Solshade* was inspired by many interesting conversations with Anna O'Flynn,
Anthony Zerafa & Chris Omelon at the McGill Arctic Research Station (MARS)
on Axel Heiberg Island, 2025.

# References
