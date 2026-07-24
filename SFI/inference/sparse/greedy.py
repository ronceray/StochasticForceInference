"""
SFI.inference.sparse.greedy — Stepwise greedy selection
=======================================================

Classic forward / backward / bidirectional stepwise search.

* **Forward**: at each step, add the feature that maximises the
  information gain.
* **Backward**: start from the full model and drop the least useful
  feature.
* **Bidirectional**: alternate one forward step then one backward step,
  keeping whichever direction improves the score.

The algorithm naturally produces exactly one support per cardinality
(forward path) or a monotonic path from full to empty (backward),
yielding a clean Pareto front.
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


class GreedyStepwiseStrategy(SparsityStrategy):
    """Forward / backward / bidirectional stepwise selection.

    Parameters
    ----------
    direction : ``"forward"`` | ``"backward"`` | ``"both"``, default ``"forward"``
        Which direction(s) to search.

        * ``"forward"``:  start empty, add one feature at a time.
        * ``"backward"``: start full, drop one feature at a time.
        * ``"both"``:     run both directions and merge the Pareto fronts.
    report_time : bool, default False
        Log elapsed wall-clock time when done.
    """

    name = "greedy"

    def __init__(
        self,
        *,
        direction: str = "forward",
        report_time: bool = False,
    ):
        if direction not in ("forward", "backward", "both"):
            raise ValueError(f"direction must be 'forward', 'backward', or 'both'; got {direction!r}")
        self.direction = direction
        self.report_time = report_time

    # -----------------------------------------------------------------
    def run(self, scorer: SparseScorer, *, max_k: int, **_kwargs) -> SparsityResult:
        t0 = time.perf_counter()
        p = scorer.p
        max_k = min(max_k, p)

        best_info = [-np.inf] * (max_k + 1)
        best_support = [[] for _ in range(max_k + 1)]
        best_coeffs = [None] * (max_k + 1)

        # Null model
        best_info[0] = 0.0

        def _record(k, info, support, coeffs):
            if info > best_info[k]:
                best_info[k] = info
                best_support[k] = list(map(int, support))
                best_coeffs[k] = coeffs

        # ----- Forward path -------------------------------------------
        if self.direction in ("forward", "both"):
            current = jnp.array([], dtype=jnp.int32)
            remaining = set(range(p))

            for step in range(1, max_k + 1):
                # Try adding each remaining feature
                candidates = sorted(remaining)
                children = []
                for j in candidates:
                    child = jnp.sort(jnp.concatenate([current, jnp.array([j], jnp.int32)]))
                    children.append(child)

                batch = jnp.stack(children)
                infos, coeffs = scorer.vmap_info(batch)

                best_idx = int(jnp.argmax(infos))
                best_j = candidates[best_idx]
                current = jnp.sort(jnp.concatenate([current, jnp.array([best_j], jnp.int32)]))
                remaining.remove(best_j)

                _record(step, float(infos[best_idx]), current, coeffs[best_idx])
                logger.debug(
                    "Forward step %d: added feature %d, info=%.4f",
                    step,
                    best_j,
                    float(infos[best_idx]),
                )

        # ----- Backward path ------------------------------------------
        if self.direction in ("backward", "both"):
            # Start from the full model; its info/coeffs are already cached
            # on the scorer, so no extra solve is needed.
            current = jnp.arange(p, dtype=jnp.int32)
            info_cur, coeffs_cur = float(scorer.total_info), scorer.total_C
            k = p

            while k > 0:
                if k <= max_k:
                    _record(k, info_cur, current, coeffs_cur)
                if k == 1:
                    break

                # Try dropping each feature and pick the drop with the
                # largest information gain.
                children = [jnp.delete(current, pos) for pos in range(k)]
                batch = jnp.stack(children)
                infos, coeffs = scorer.vmap_info(batch)

                best_pos = int(jnp.argmax(infos))
                current = children[best_pos]
                info_cur = float(infos[best_pos])
                coeffs_cur = coeffs[best_pos]
                k -= 1

                logger.debug("Backward step k=%d: info=%.4f", k, info_cur)

        if self.report_time:
            dt = time.perf_counter() - t0
            logger.info("Greedy (%s) done in %.2fs.", self.direction, dt)

        return SparsityResult(
            p=scorer.p,
            total_info=float(scorer.total_info),
            method=f"greedy-{self.direction}",
            best_info_by_k=best_info,
            best_support_by_k=best_support,
            best_coeffs_by_k=best_coeffs,
            second_info_by_k=[-np.inf] * (max_k + 1),
            second_support_by_k=[[] for _ in range(max_k + 1)],
        )
