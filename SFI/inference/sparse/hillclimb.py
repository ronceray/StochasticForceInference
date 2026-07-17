"""
SFI.inference.sparse.hillclimb — Stochastic hill-climbing selection
====================================================================

Implements the search strategy described in Gerardos & Ronceray (2025):
starting from an initial model, accept random single-parameter
additions or removals if they increase the chosen information
criterion, until convergence.  Multiple independent chains are run in
parallel, one per cardinality *k* ∈ {0, …, max_k} (including the null
and full models), each starting from a uniformly random support of
size *k*.

This is complementary to the deterministic strategies (greedy, beam):

* **Greedy / beam** explore a systematic path and may get trapped in a
  single monotonic trajectory through the lattice.
* **Hill-climbing** performs a stochastic local search at every
  complexity level, which can escape greedy traps and explore broader
  neighbourhoods of the combinatorial lattice.

References
----------
* Gerardos, A. & Ronceray, P. (2025).  "Principled model selection for
  stochastic dynamics."  (describes the algorithm in the main text.)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .base import SparsityStrategy
from .result import SparsityResult
from .scorer import SparseScorer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Inline IC penalties (no SparsityResult needed during the search)
# ------------------------------------------------------------------


def _ic_penalty(
    name: str,
    k: int,
    *,
    n0: int,
    p_param: float,
    tau: Optional[float],
    gamma: float,
    total_info: float,
) -> float:
    """Return the penalty term for a given information criterion.

    The score is ``info - penalty``, so a *larger* penalty means more
    pruning.
    """
    name = name.upper()
    if name == "AIC":
        return float(k)
    if name == "BIC":
        if tau is None:
            raise ValueError("BIC requires 'tau' (total trajectory time).")
        return 0.5 * k * math.log(tau)
    if name == "EBIC":
        if tau is None:
            raise ValueError("EBIC requires 'tau' (total trajectory time).")
        lc = math.lgamma(n0 + 1) - math.lgamma(k + 1) - math.lgamma(n0 - k + 1)
        return 0.5 * k * math.log(tau) + 2.0 * gamma * lc
    if name == "PASTIS":
        return k * math.log(n0 / p_param)
    if name == "SIC":
        if total_info <= 0:
            return float("inf")
        return k * math.log(total_info)
    raise ValueError(f"Unknown criterion {name!r}")


# ------------------------------------------------------------------
# Single chain
# ------------------------------------------------------------------


def _run_chain(
    scorer: SparseScorer,
    init_support: np.ndarray,
    *,
    ic_name: str,
    n0: int,
    max_k: int,
    p_param: float,
    tau: Optional[float],
    gamma: float,
    total_info: float,
    patience: int,
    rng: np.random.Generator,
) -> dict:
    """Run one stochastic hill-climbing chain.

    Returns
    -------
    trace : dict[int, (info, support, coeffs)]
        Best raw information gain seen per cardinality across every
        proposal evaluated by the chain (accepted *or* rejected).  This
        is the slice of the chain's work that contributes to the global
        Pareto front.
    """
    current = set(int(j) for j in init_support)
    all_indices = set(range(n0))
    trace: dict = {}

    def _evaluate(support_set: set) -> tuple[float, float, Optional[jnp.ndarray]]:
        """Return ``(ic_score, raw_info, coeffs)``."""
        k = len(support_set)
        penalty = _ic_penalty(
            ic_name, k,
            n0=n0, p_param=p_param, tau=tau, gamma=gamma, total_info=total_info,
        )
        if k == 0:
            return -penalty, 0.0, None
        B = jnp.array(sorted(support_set), dtype=jnp.int32)
        info, coeffs = scorer.info_and_coeffs(B)
        info_f = float(info)
        return info_f - penalty, info_f, coeffs

    def _record_trace(support_set: set, info: float, coeffs) -> None:
        k = len(support_set)
        entry = trace.get(k)
        if entry is None or info > entry[0]:
            trace[k] = (info, sorted(support_set), coeffs)

    best_ic, raw_info, coeffs = _evaluate(current)
    _record_trace(current, raw_info, coeffs)
    fails = 0

    while fails < patience:
        # Decide on a random move: add or remove
        can_add = len(current) < max_k and len(current) < n0
        can_remove = len(current) > 0
        if can_add and can_remove:
            do_add = bool(rng.integers(2))
        elif can_add:
            do_add = True
        elif can_remove:
            do_add = False
        else:
            break  # stuck at empty with max_k=0

        if do_add:
            candidates = list(all_indices - current)
            j = int(rng.choice(candidates))
            proposal = current | {j}
        else:
            j = int(rng.choice(list(current)))
            proposal = current - {j}

        ic_proposal, info_proposal, coeffs_proposal = _evaluate(proposal)
        _record_trace(proposal, info_proposal, coeffs_proposal)
        if ic_proposal > best_ic:
            current = proposal
            best_ic = ic_proposal
            fails = 0
        else:
            fails += 1

    return trace


# ------------------------------------------------------------------
# Strategy class
# ------------------------------------------------------------------


class HillClimbStrategy(SparsityStrategy):
    """Stochastic hill-climbing model selection.

    For each cardinality *k* ∈ {0, …, max_k}, a chain starts from a
    random support of size *k* and accepts random add/remove moves that
    improve the chosen information criterion, stopping after *patience*
    consecutive failures.  The best-per-*k* results form the Pareto
    front returned as a :class:`SparsityResult`.

    This is the search algorithm described in Gerardos & Ronceray (2025),
    §"Model selection", which recommends parallel searches from null,
    full, and random starting points.

    Parameters
    ----------
    ic : str, default ``"PASTIS"``
        Information criterion used as the acceptance objective.
        One of ``"AIC"``, ``"BIC"``, ``"EBIC"``, ``"PASTIS"``,
        ``"SIC"``.
    p_param : float, default 1e-3
        PASTIS significance level :math:`p_0`.
    tau : float or None
        Total trajectory time (required for BIC / EBIC).
    gamma : float, default 0.5
        EBIC tuning parameter (:math:`\\gamma \\in [0,1]`).
    patience : int, default 200
        Stop a chain after this many consecutive rejected moves.
    seed : int or None
        Random seed for reproducibility.
    report_time : bool, default False
        Log elapsed wall-clock time when done.
    """

    name = "hillclimb"

    def __init__(
        self,
        *,
        ic: str = "PASTIS",
        p_param: float = 1e-3,
        tau: Optional[float] = None,
        gamma: float = 0.5,
        patience: int = 200,
        seed: Optional[int] = None,
        report_time: bool = False,
    ):
        self.ic = ic.upper()
        self.p_param = p_param
        self.tau = tau
        self.gamma = gamma
        self.patience = patience
        self.seed = seed
        self.report_time = report_time

    # -----------------------------------------------------------------
    def run(self, scorer: SparseScorer, *, max_k: int, **_kwargs) -> SparsityResult:
        t0 = time.perf_counter()
        n0 = scorer.p
        max_k = min(max_k, n0)
        rng = np.random.default_rng(self.seed)
        total_info = float(scorer.total_info)

        best_info = [-np.inf] * (max_k + 1)
        best_support = [[] for _ in range(max_k + 1)]
        best_coeffs = [None] * (max_k + 1)

        # Null model (k=0) always has info=0
        best_info[0] = 0.0

        def _record(k: int, info: float, support: list, coeffs):
            if 0 <= k <= max_k and info > best_info[k]:
                best_info[k] = info
                best_support[k] = list(support)
                best_coeffs[k] = coeffs

        # ------ Launch one chain per cardinality k --------------------
        for k in range(0, max_k + 1):
            # Random starting support of size k
            if k == 0:
                init = np.array([], dtype=np.int32)
            elif k >= n0:
                init = np.arange(n0, dtype=np.int32)
            else:
                init = np.sort(rng.choice(n0, size=k, replace=False))

            trace = _run_chain(
                scorer,
                init,
                ic_name=self.ic,
                n0=n0,
                max_k=max_k,
                p_param=self.p_param,
                tau=self.tau,
                gamma=self.gamma,
                total_info=total_info,
                patience=self.patience,
                rng=rng,
            )
            # Fold every cardinality the chain explored into the global
            # Pareto front, not only its final landing point.
            for k_state, (info, support, coeffs) in trace.items():
                _record(k_state, info, support, coeffs)

            logger.debug(
                "Hill-climb chain k₀=%d → %d cardinalities recorded.",
                k,
                len(trace),
            )

        if self.report_time:
            dt = time.perf_counter() - t0
            logger.info(
                "Hill-climb (%s, patience=%d) done in %.2fs.",
                self.ic,
                self.patience,
                dt,
            )

        return SparsityResult(
            p=n0,
            total_info=total_info,
            method="hillclimb",
            best_info_by_k=best_info,
            best_support_by_k=best_support,
            best_coeffs_by_k=best_coeffs,
            second_info_by_k=[-np.inf] * (max_k + 1),
            second_support_by_k=[[] for _ in range(max_k + 1)],
        )
