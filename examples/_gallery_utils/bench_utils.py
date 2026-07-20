# TODO: review this file
"""Minimal benchmark utilities (v2.0 port).

The v1.97 benchmark suite carried a large ``bench_utils`` (grid execution,
presets, plotting); the v2.0 sweep scripts (``scripts/rk4_decision_sweep.py``,
``scripts/nonregression_10x.py``, and the ``lorenz_method_evolution`` /
``vdp_uli_10x`` runners) need only the environment side effects and the
cache clearer, so only those are ported.
"""

from __future__ import annotations

import gc
import os

# Must run BEFORE any JAX/SFI import (the runners import this module first):
# float64 for the parametric solvers, CPU-only for benchmark stability.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _clear_jax_caches():
    """Drop compiled-program caches between cells to bound memory."""
    import jax

    jax.clear_caches()
    gc.collect()
