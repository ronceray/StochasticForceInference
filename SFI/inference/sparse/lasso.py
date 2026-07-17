"""
SFI.inference.sparse.lasso — ℓ₁-penalised regression (LASSO)
=============================================================

Solve the :math:`\\ell_1`-regularised normal-equations problem:

.. math::

   \\hat C = \\arg\\min_C \\;
      \\tfrac{1}{2}\\,C^\\top G\\,C \\;-\\; M^\\top C
      \\;+\\; \\alpha\\,\\|C\\|_1

using proximal coordinate descent.  No access to the raw design matrix
:math:`\\Phi` is needed — only ``(M, G)`` — preserving the clean
decoupling from the data pipeline.

Sweeping over a regularisation path of :math:`\\alpha` values produces
supports at varying sparsity levels, from which a Pareto front is
assembled.
"""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np

from .base import SparsityStrategy
from .result import SparsityResult
from .scorer import SparseScorer

logger = logging.getLogger(__name__)


def _soft_threshold(x: float, lam: float) -> float:
    """Scalar soft-thresholding operator."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


class LassoStrategy(SparsityStrategy):
    r"""Coordinate-descent LASSO on the normal equations.

    Parameters
    ----------
    alpha : float or None
        Fixed regularisation strength.  If *None* (default), an
        automatic log-spaced path from :math:`\alpha_{\max}` (where the
        solution is entirely zero) down to :math:`10^{-4}\,\alpha_{\max}`
        is constructed.
    n_alphas : int, default 50
        Number of :math:`\alpha` values in the automatic path.
    max_iter : int, default 1000
        Maximum coordinate-descent iterations per :math:`\alpha`.
    tol : float, default 1e-7
        Convergence tolerance (max absolute change in any coefficient).
    report_time : bool, default False
        Log elapsed wall-clock time when done.
    """

    name = "lasso"

    def __init__(
        self,
        *,
        alpha: float | None = None,
        n_alphas: int = 50,
        max_iter: int = 1000,
        tol: float = 1e-7,
        report_time: bool = False,
    ):
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.tol = tol
        self.report_time = report_time

    # -----------------------------------------------------------------
    def _coordinate_descent(self, G: np.ndarray, M: np.ndarray, alpha: float, C_init: np.ndarray) -> np.ndarray:
        """Run coordinate descent for one alpha.  Pure numpy, no JAX."""
        p = len(M)
        C = C_init.copy()
        for _ in range(self.max_iter):
            max_delta = 0.0
            for j in range(p):
                # Partial residual for coordinate j
                r_j = float(M[j] - G[j] @ C + G[j, j] * C[j])
                G_jj = float(G[j, j])
                if G_jj < 1e-30:
                    new_val = 0.0
                else:
                    new_val = _soft_threshold(r_j, alpha) / G_jj
                delta = abs(new_val - C[j])
                if delta > max_delta:
                    max_delta = delta
                C[j] = new_val
            if max_delta < self.tol:
                break
        return C

    # -----------------------------------------------------------------
    def run(self, scorer: SparseScorer, *, max_k: int, **_kwargs) -> SparsityResult:
        t0 = time.perf_counter()
        p = scorer.p
        max_k = min(max_k, p)

        # Convert to numpy for the coordinate descent loop
        G_np = np.asarray(scorer.G)
        M_np = np.asarray(scorer.M)

        # alpha_max: smallest alpha that zeros everything out
        alpha_max = float(np.max(np.abs(M_np)))

        if self.alpha is not None:
            alphas = [self.alpha]
        else:
            alphas = np.logspace(
                np.log10(alpha_max),
                np.log10(alpha_max * 1e-4),
                self.n_alphas,
            ).tolist()

        best_info = [-np.inf] * (max_k + 1)
        best_support = [[] for _ in range(max_k + 1)]
        best_coeffs = [None] * (max_k + 1)

        # Null model
        best_info[0] = 0.0

        # Warm-start: start from zeros for the largest alpha
        C_warm = np.zeros(p)

        for alpha_val in alphas:
            C_warm = self._coordinate_descent(G_np, M_np, alpha_val, C_warm)

            # Identify nonzero support
            support = [j for j in range(p) if abs(C_warm[j]) > 1e-14]
            k = len(support)

            if k > max_k or k == 0:
                continue

            # Re-solve exactly on the LASSO support (de-biased LASSO)
            B = jnp.array(support, dtype=jnp.int32)
            info, coeffs = scorer.info_and_coeffs(B)

            if float(info) > best_info[k]:
                best_info[k] = float(info)
                best_support[k] = support
                best_coeffs[k] = coeffs

        # Also record the full model if within max_k
        if p <= max_k:
            full_info = float(scorer.total_info)
            if full_info > best_info[p]:
                best_info[p] = full_info
                best_support[p] = list(range(p))
                best_coeffs[p] = scorer.total_C

        if self.report_time:
            dt = time.perf_counter() - t0
            logger.info("LASSO done in %.2fs (%d alphas).", dt, len(alphas))

        return SparsityResult(
            p=scorer.p,
            total_info=float(scorer.total_info),
            method=self.name,
            best_info_by_k=best_info,
            best_support_by_k=best_support,
            best_coeffs_by_k=best_coeffs,
            second_info_by_k=[-np.inf] * (max_k + 1),
            second_support_by_k=[[] for _ in range(max_k + 1)],
        )
