# Tests

Run the collected test suite with:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/ -x --ff -q \
    --ignore=tests/inference/test_parametric_loss_parity.py
```

`test_parametric_loss_parity.py` is excluded because it is a slow
parity-check that requires a specific reference run; run it manually
when needed.

---

## File inventory

### Collected by pytest (`test_*.py`)

| File | What it covers |
|------|---------------|
| `test_dual_mask.py` | Dual-mask trajectory handling |
| `test_multi_window_smoke.py` | Multi-particle window + legacy backend smoke |
| `test_parametric_mask.py` | Masked-trajectory parametric inference |
| `test_parametric_mask_abp.py` | ABP-specific masked parametric inference |
| `test_parametric_multi_particle.py` | Multi-particle parametric inference |
| `test_parametric_sfi_design.py` | Design-note unit checks for the parametric solver |
| `test_same_particle_jacobian.py` | Per-edge Jacobian correctness |
| `test_window_reprofile_e2e.py` | Window-based reprofile end-to-end |
| `bases/` | Basis-function correctness |
| `diagnostics/` | Diagnostic utilities |
| `inference/` | Overdamped and underdamped parametric inference |
| `integrate/` | Numerical integrator correctness |
| `langevin/` | Langevin simulator |
| `statefunc/` | State-function / structured-expression machinery |
| `trajectory/` | Trajectory / collection objects |
| `utils/` | Formatting and utility helpers |

### Not collected — smoke scripts (`smoke_*.py`)

Quick integration checks meant to be run locally or in CI pre-flight.

| File | Description |
|------|-------------|
| `smoke_abp_multi.py` | One cell of the multi-particle ABP smoke check |
| `smoke_dw.py` | Double-well only, with memory tracking |

### Not collected — validation scripts (`validate_*.py`)

Correctness checks comparing two implementations against each other.

| File | Description |
|------|-------------|
| `validate_od_window_gram.py` | Window Gram vs legacy `od_parametric_precompute` path |
| `validate_reprofile.py` | Window-based OD reprofile vs legacy implementation |

### Helpers

| File | Description |
|------|-------------|
| `conftest.py` | pytest fixtures shared across the suite |
| `_abp_helpers.py` | ABP simulation helpers used by ABP-related tests |
