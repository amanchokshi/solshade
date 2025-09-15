# Installation

Solshade targets Python **3.13+** on Linux, macOS, untested on windows, but should support WSL.  
We recommend installing inside a **virtual environment** so your setup stays isolated from system Python.

---

## 1. Virtual environment

You can use either `venv` (built into Python) **or** [uv](https://github.com/astral-sh/uv) (a modern fast manager).

### A) `venv` (built-in)

```bash
# Create and activate a venv in .venv/
python -m venv .venv
source .venv/bin/activate
```

### B) – uv (fast & modern)

```bash
# If you don’t have uv yet:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a venv in .venv/
uv venv
source .venv/bin/activate
```

---

## 2. Install Solshade

### A) From PyPI

```bash
pip install solshade
# or with uv:
uv pip install solshade
```

### B) From source

```bash
git clone https://github.com/amanchokshi/solshade.git
cd solshade

# Regular install
pip install .

# Editable install (recommended for development)
pip install -e .
```

Using uv:

```bash
uv pip install .
uv pip install -e .
```

---

## 3. Development install

If you want to run the test suite, build the docs, or contribute code, install with the dev extra defined in `pyproject.toml`:

```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

This installs additional dependencies for:

- **Testing** (`pytest`, `pytest-cov`, etc.)
- **Docs** (`mkdocs`, `mkdocs-material`, `mkdocstrings`, etc.)
- **Linting & formatting** (`ruff`, `black`, etc.)
- **Test data**: the repo includes small DEMs and horizon maps under `tests/data/` for running examples and tests.

---

## 4. Verify the installation

```bash
solshade --help
```

You should see the CLI help. Try a quick command (with your own DEM):

```bash
solshade compute slope path/to/DEM.tif --output-dir out/
```

---

## Notes & troubleshooting

- **Binary deps**: Solshade depends on packages like `rasterio`, `pyproj`, `xarray`, and `rioxarray`.  
  On most platforms, prebuilt wheels are available. If you encounter compiler or GDAL errors, ensure you’re using a recent Python and pip, or try installing via `uv` which handles wheels efficiently.

- **Conda users**: You can also install dependencies like `rasterio` and `gdal` via conda-forge if needed, then `pip install solshade` into that environment.

---

You’re now ready to start computing solar illumination maps with Solshade ☼
