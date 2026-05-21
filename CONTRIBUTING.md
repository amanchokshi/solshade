# Contributing to Solshade

Thank you for your interest in contributing to Solshade!

Contributions of all kinds are welcome, including:

- Bug reports
- Feature requests
- Documentation improvements
- Tests
- New terrain or illumination analysis functionality

---

## Reporting issues

If you encounter a bug, unexpected behaviour, or would like to request a feature, please open an issue on GitHub.

When reporting a bug, it is helpful to include:

- Your operating system
- Python version
- The Solshade version
- A minimal reproducible example
- Any relevant error messages or traceback

---

## Development setup

We recommend working inside a virtual environment.

### A) `venv` (built-in)

```bash
python -m venv .venv
source .venv/bin/activate
```

### B) `uv` (optional)

```bash
uv venv
source .venv/bin/activate
```

Clone the repository:

```bash
git clone https://github.com/amanchokshi/solshade.git
cd solshade
```

Install Solshade in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Or using `uv`:

```bash
uv pip install -e ".[dev]"
```

---

## Running tests

Run the test suite with:

```bash
pytest
```

Coverage can be checked with:

```bash
pytest --cov=solshade
```

---

## Formatting and linting

Solshade uses `ruff` for linting and formatting.

Check linting:

```bash
ruff check .
```

Format the code:

```bash
ruff format .
```

---

## Pull requests

Pull requests are welcome!

Before submitting a PR, please ensure that:

- Tests pass locally
- New functionality includes tests where appropriate
- Public functions/classes include docstrings
- Documentation is updated if needed

Small, focused pull requests are generally easier to review.

---

## Seeking support

For questions, usage help, or discussion, please open a GitHub issue.

---

Thanks for helping improve Solshade ☼
