---
title: "Solshade: Terrain-aware Solar Illumination Modeling using Digital Elevation Models and Orbital Geometry"
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
  - name: Trottier Space Institute, McGill University, Montréal, Québec H3A 2A7, Canada
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
terrain is essential for environmental science, ecology, hydrology, and energy
modeling. Widely used GIS tools such as *GRASS GIS* and *SAGA GIS* provide
terrain analysis and solar radiation estimates, but their focus remains
primarily on geospatial processing. These packages often assume simplified
astronomical inputs or fixed time intervals, offering limited control over
orbital precision or temporal resolution.

Similarly, solar energy modeling tools like *pvlib* or *Solar Analyst*
(ArcGIS) provide detailed solar irradiance calculations but lack functionality
for high-resolution terrain shading analysis or integration with custom
topographic datasets.

*Solshade* bridges these critical gap by combining:

1. *High-precision solar orbit modeling* using NASA ephemerides via *Skyfield*,
2. *Terrain-aware ray-traced shading* over arbitrary DEMs,
3. *Flexible Python API and CLI workflows* for reproducible analysis.

This integration enables studies requiring both astronomical accuracy and
geospatial flexibility, supporting applications from permafrost melt modeling
to ecological microhabitat analysis.

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

I would like to thank Anna O'Flynn, Anthony Zerafa & Chris Omelon, for the many
fascinating conversations at the McGill Arctic Research Station (MARS) on Axel
Heiberg Island, 2025. *Solshade* is the first of many ideas which were sparked
by these discussions.

A. C. acknowledges support from the Trottier Space Institute Fellowship program,
which enabled parts of this research.

# References
