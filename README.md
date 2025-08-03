# *solshade*

**Terrain-aware solar illumination modeling from digital elevation data**

`solshade` is a Python toolkit for simulating solar radiation across landscapes, accounting for terrain shadows, solar angles, and orbital geometry. It’s designed for interdisciplinary research at the intersection of astronomy, glaciology, botany, and geology.

---

## What does `solshade` do?

- Computes per-pixel solar exposure over time from a DEM
- Generates terrain-aware **horizon maps** to determine shadowing
- Uses precise **solar ephemerides** (via Skyfield)
- Calculates:
  - Total annual insolation
  - Date of maximum sunlight
  - Mean solar incidence angle
  - Terrain shading based on real orbital paths

---

## Example Applications

- Modeling plant growing seasons across topography
- Studying microhabitats in extreme environments
- Predicting snowmelt timing in complex terrain
- Understanding glacial melt and shadowed regions

---

## Project Status

This project is in early development.  
Expect breaking changes, experiments, and rapid iteration.

---

## License

MIT License — see `LICENSE` file.

---

## Acknowledgments

Inspired by many interesting conversations with Anna O'Flynn, Anthony Zerafa & Chris Omelon at the McGill Arctic Research Station (MARS) on Axel Heiberg Island, 2025.
