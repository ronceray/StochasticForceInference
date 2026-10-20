# Contributing to SFI

Thank you for contributing!

## Quick start

```bash
git clone https://github.com/ronceray/StochasticForceInference.git
cd StochasticForceInference
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,io]"
pytest tests/ -v
```

Full environment and build details:
[`docs/source/dev_notes.rst`](docs/source/dev_notes.rst).

## Where things live

- `SFI/` — the package. See [`AGENTS.md`](AGENTS.md) §4 for the
  canonical imports table.
- `examples/gallery/` — Sphinx-gallery demos (follow
  [`GALLERY_STYLE_GUIDE.md`](GALLERY_STYLE_GUIDE.md)).
- `examples/benchmarks/` — validation / regression benchmarks (follow
  [`docs/source/dev_benchmark_patterns.md`](docs/source/dev_benchmark_patterns.md)).
- `tests/` — pytest suite. Files matching `benchmark_*.py`,
  `validate_*.py`, `audit_*.py` are **not** collected; they are
  manually-invoked scripts.
- `docs/source/` — Sphinx documentation source.

## Style

- Numpy-style docstrings (rendered by Sphinx autodoc).
- Plots must call `SFI.utils.plotting.apply_style()` and use
  `SFI_COLORS`; never pure black.
- New public symbols should be re-exported from the relevant
  `__init__.py`.

## Using AI coding assistants

If you use GitHub Copilot, Claude Code, Cursor, or similar agents,
they will pick up the canonical guide at [`AGENTS.md`](AGENTS.md).
Skim it so you and your assistant are on the same page.

## Pull requests

- Add or update tests for any behaviour change.
- Run `pytest tests/ -v` locally before pushing.
- If you add a user-visible feature, add or update a gallery demo and
  the corresponding `docs/source/` user guide.
- If you touch parametric-estimator internals, run
  `pytest tests/inference/ -v` and re-run the relevant benchmark in
  `tests/benchmark_*.py`.
