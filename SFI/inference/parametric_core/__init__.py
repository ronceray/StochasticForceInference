# SFI/inference/parametric_core/ — the exact parametric estimator.
"""
The parametric force/diffusion estimator: RK4-flow residuals, block-banded
residual covariance whitened *exactly* by a reverse-time block-LDLᵀ
innovations recursion (:mod:`banded` / :mod:`runner`), errors-in-variables
instrument, and native ``(D, Λ)`` profiling on cached fixed-θ tensors.
Per-interval (non-uniform) dt is supported end-to-end.

A single, paper-ready objective with a handful of parameters, integrating
like ``infer_*_linear`` (Basis/PSF in → ``InferenceResultSF`` out,
multi-particle aware).

Docs: ``docs/source/inference/parametric_concept.rst`` (foundations) and
``docs/source/inference/parametric_algorithm.rst`` (implementation).
"""

from __future__ import annotations

from .solve import (
    DiffusionSolveResult,
    ForceSolveResult,
    solve_diffusion_od,
    solve_diffusion_ud,
    solve_force_od,
    solve_force_ud,
)

__all__ = [
    "solve_force_od", "solve_force_ud",
    "solve_diffusion_od", "solve_diffusion_ud",
    "ForceSolveResult", "DiffusionSolveResult",
]
