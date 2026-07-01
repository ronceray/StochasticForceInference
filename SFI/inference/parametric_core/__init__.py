# SFI/inference/parametric_core/ — stripped-to-the-bone parametric estimator.
"""
Parallel, minimal reimplementation of the parametric force/diffusion
estimator (RK4-flow residuals + banded residual covariance + windowed
precision + AD-minimised windowed NLL).

Goal: a single, paper-ready, one-size-fits-all objective with a handful
of parameters, integrating like ``infer_*_linear`` (Basis/PSF in →
``InferenceResultSF`` out, multi-particle aware) and reusing the
``SFI.integrate`` engine for chunking/masking/JIT.

This package is intentionally independent of ``SFI.inference.parametric``
(the existing implementation, kept as the benchmark baseline).

Design note: ``docs/source/inference/parametric_core.rst`` (to be written);
working draft in the session plan.
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
