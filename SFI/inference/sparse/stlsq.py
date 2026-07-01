"""
SFI.inference.sparse.stlsq — Sequential Thresholded Least Squares
=================================================================

The :class:`STLSQStrategy` implements the iterative hard-thresholding
algorithm popularised by SINDy (Brunton *et al.*, 2016):

1. Solve the full problem :math:`G\\,C = M`.
2. Zero out coefficients whose magnitude falls below a threshold.
3. Re-solve on the surviving support.
4. Repeat until convergence.

The threshold can be **absolute** (``mode="absolute"``) or
**relative** to the current maximum coefficient magnitude
(``mode="relative"``, default — matches the PySINDy convention).

Running over a sweep of thresholds produces a Pareto front.
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


class STLSQStrategy(SparsityStrategy):
    """Sequential Thresholded Least Squares (SINDy-style).

    Parameters
    ----------
    threshold : float or None
        Single threshold value.  If *None*, an automatic log-spaced
        sweep from a small fraction of the maximum coefficient up to
        the maximum is performed.
    mode : ``"relative"`` | ``"absolute"``, default ``"relative"``
        Whether ``threshold`` is interpreted as a fraction of
        :math:`\\max|C|` (relative) or a fixed value (absolute).
    max_iter : int, default 50
        Maximum STLSQ iterations per threshold.
    n_thresholds : int, default 30
        Number of thresholds in the automatic sweep (used only when
        ``threshold is None``).
    report_time : bool, default False
        Log elapsed wall-clock time when done.
    """

    name = "stlsq"

    def __init__(
        self,
        *,
        threshold: float | None = None,
        mode: str = "relative",
        max_iter: int = 50,
        n_thresholds: int = 30,
        report_time: bool = False,
    ):
        if mode not in ("relative", "absolute"):
            raise ValueError(f"mode must be 'relative' or 'absolute'; got {mode!r}")
        self.threshold = threshold
        self.mode = mode
        self.max_iter = max_iter
        self.n_thresholds = n_thresholds
        self.report_time = report_time

    # -----------------------------------------------------------------
    def _stlsq_once(
        self, scorer: SparseScorer, threshold: float, *, is_relative: bool
    ) -> tuple[list[int], float, jnp.ndarray]:
        """Run STLSQ for a single threshold.  Returns (support, info, coeffs)."""
        support = list(range(scorer.p))

        for _ in range(self.max_iter):
            B = jnp.array(support, dtype=jnp.int32)
            info, C = scorer.info_and_coeffs(B)

            abs_C = jnp.abs(C)
            cutoff = threshold * float(jnp.max(abs_C)) if is_relative else threshold
            mask = abs_C >= cutoff

            new_support = [support[i] for i in range(len(support)) if mask[i]]

            if new_support == support:
                break
            support = sorted(new_support)

            if len(support) == 0:
                return [], 0.0, jnp.zeros(0, dtype=scorer.M.dtype)

        # Final solve on converged support
        B = jnp.array(support, dtype=jnp.int32)
        info, C = scorer.info_and_coeffs(B)
        return support, float(info), C

    # -----------------------------------------------------------------
    def run(self, scorer: SparseScorer, *, max_k: int, **_kwargs) -> SparsityResult:
        t0 = time.perf_counter()
        max_k = min(max_k, scorer.p)

        best_info = [-np.inf] * (max_k + 1)
        best_support = [[] for _ in range(max_k + 1)]
        best_coeffs = [None] * (max_k + 1)

        # Null model
        best_info[0] = 0.0

        is_relative = self.mode == "relative"

        if self.threshold is not None:
            thresholds = [self.threshold]
        else:
            # Automatic sweep: from ~0.001 to ~1.0 (relative)
            # or from a small absolute value to max(|C_full|) (absolute)
            if is_relative:
                thresholds = np.logspace(-3, -0.05, self.n_thresholds).tolist()
            else:
                max_abs = float(jnp.max(jnp.abs(scorer.total_C)))
                if max_abs > 0:
                    thresholds = np.logspace(
                        np.log10(max_abs * 1e-4),
                        np.log10(max_abs * 0.9),
                        self.n_thresholds,
                    ).tolist()
                else:
                    thresholds = [0.0]

        for thr in thresholds:
            support, info, coeffs = self._stlsq_once(scorer, thr, is_relative=is_relative)
            k = len(support)
            if k <= max_k and info > best_info[k]:
                best_info[k] = info
                best_support[k] = support
                best_coeffs[k] = coeffs

        # Also record the full model if within max_k
        if scorer.p <= max_k:
            full_info = float(scorer.total_info)
            if full_info > best_info[scorer.p]:
                best_info[scorer.p] = full_info
                best_support[scorer.p] = list(range(scorer.p))
                best_coeffs[scorer.p] = scorer.total_C

        if self.report_time:
            dt = time.perf_counter() - t0
            logger.info("STLSQ done in %.2fs (%d thresholds).", dt, len(thresholds))

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
