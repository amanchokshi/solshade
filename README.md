![](https://github.com/user-attachments/assets/69bef1b9-10b1-44d9-9bf0-46eac1fda3ce)

[![PyPI](https://img.shields.io/pypi/v/solshade.svg?logo=pypi&logoColor=white&label=PyPI&color=2a3573)](https://pypi.org/project/solshade/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-34777e?logo=python&logoColor=white)](https://pypi.org/project/solshade/)
[![CI](https://img.shields.io/github/actions/workflow/status/amanchokshi/solshade/ci.yml?branch=main&logo=github&color=82af74)](https://github.com/amanchokshi/solshade/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/amanchokshi/solshade?color=f9dc28)](https://codecov.io/gh/amanchokshi/solshade)
[![Docs](https://img.shields.io/readthedocs/solshade/latest?logo=Read%20The%20Docs&logoColor=white&color=ea7b17)](https://solshade.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/amanchokshi/EMBERS?label-License&color=b72c40&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAdCAYAAADLnm6HAAAACXBIWXMAAB2HAAAdhwGP5fFlAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAflJREFUSIntljFrVUEQhb95CViYCGItGLSIIIJiISadYiOIAS0stFAEK0G0DQRs8wOs7PQvqCC2kjQJaiEICoJYaZMYzCP6WWSfLs97c/deQtJ4YODu7syZs7N7l4GOUCfVj8kmu/L0ugYCM8ChZDO7IWBPzfeOCdg5qD31urqs9q1HX11Sr6nbszl1Wl3cImkdFtXpNonm1fUOidpiXZ2vErDaELimPlDHk/9ctjaX5saTz1oD1+ogb35OD4F+RXG+Ao+AoxExGxErdVWMiJWImAUmU8y3Crd+ylV5DD31fab0gzpS4/tPBbrwDd/US8CRbDwBXKxUW4ZGvmEB9ypIquZK0cj3R4B6GjiThuvJAKbSWiuU8uUVuJ99Pwae1KkuRDmfOqFuZJfluHpM/ZXGG+rhoZjaS9iGb1CBu8Dgdj6NiNcR8RZ4nuZGgDstdl/Op+5XVzK1Z7OdnMsfD/VAUwXa8vWA28BY8nkDvBwERMQLYDkN9wK3Cnbfiq8H3MiCPwEn1H3JTgKfs/WbBQJa8Y3y9/cAuJCsDj8KBLTi6wGXgYUC4gXgSoFfK77RiHinTgHngavAKeAgMAp8AV6x+Q8/i4ifTazbzVeLrd6BNtj1nvC/gEbYvVes7v2GEAUCVtl8tbrge0SMbeVQcgR1vWIT6nu/DL8BDHb5/EeYsAMAAAAASUVORK5CYII=)](https://github.com/amanchokshi/solshade/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/Code%20Style-Black-222222.svg?logo=powershell&logoColor=white)](https://black.readthedocs.io)
[![JOSS](https://img.shields.io/badge/JOSS-paper-222222.svg?&color=82af74)](https://doi.org/10.21105/joss.09944)
---
`solshade` is a Python toolkit for simulating solar radiation across
landscapes, accounting for terrain shadows, solar angles, and orbital geometry.
It’s designed for interdisciplinary research at the intersection of astronomy,
glaciology, botany, and geology. Capturing the dance of *sol* (Sun) and *shade*
(Shadow) over landscapes.

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

`solshade` is designed to install cleanly with a single invocation of the standard Python package tool:

```
pip install solshade
```

Here are some essential project links:

- [Home page and documentation](https://solshade.readthedocs.io/en/latest)
- [Installing solshade](https://solshade.readthedocs.io/en/latest/guides/install)
- [Getting started with solshade](https://solshade.readthedocs.io/en/latest/guides)
- [Solshade package on PyPI](https://pypi.org/project/solshade)
- [Issue tracker on github](https://github.com/amanchokshi/solshade/issues)
- [Contributing to solshade](CONTRIBUTING.md)

## Example Applications

- Studying microhabitats in extreme environments
- Predicting snowmelt timing in complex terrain
- Understanding glacial melt and shadowed regions
- Modeling plant growing seasons across topography

---

## License

MIT License — see `LICENSE` file.

---

## Acknowledgments

Inspired by many interesting conversations with Anna O'Flynn, Anthony Zerafa
& Chris Omelon at the McGill Arctic Research Station (MARS) on Axel Heiberg
Island, 2025.
