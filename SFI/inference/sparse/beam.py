"""
SFI.inference.sparse.beam — Bidirectional beam search
=====================================================

The :class:`BeamSearchStrategy` explores the support lattice by
expanding every ±1 neighbour of every support in the current frontier
and retaining only the ``beam_width`` best models per cardinality *k*.

This is the original algorithm described in the PASTIS paper.

Performance notes
~~~~~~~~~~~~~~~~~
* Child generation uses pure Python integer lists — no per-child JAX
  array allocations.
* Each child is scored individually via ``info_and_coeffs`` (the
  single-support JIT kernel) rather than batched through
  ``scorer.vmap_info``.  On CPU this is the faster path here because
  beam children come in many distinct cardinalities, and the vmap
  cache would otherwise compile O(max_k × unique_batch_sizes) shapes;
  per-child scoring caps the cache at O(max_k) entries.
* Frontier capping uses ``heapq.nlargest`` instead of full sort.
* Skyline update only touches cardinalities modified in the current
  generation.
"""

from __future__ import annotations

import collections
import heapq
import itertools
import logging
import time

import jax.numpy as jnp
import numpy as np

from .base import SparsityStrategy
from .result import SparsityResult
from .scorer import SparseScorer

logger = logging.getLogger(__name__)


class BeamSearchStrategy(SparsityStrategy):
    """Bidirectional beam search over the support lattice.

    Parameters
    ----------
    beam_width : int, default 20
        Maximum number of candidate models retained per cardinality.
    aic_patience : int, default 2
        Stop early when AIC has strictly declined for this many
        consecutive *closed* cardinality levels.
    report_time : bool, default False
        If *True*, log elapsed time and number of explored supports.
    """

    name = "beam"

    def __init__(
        self,
        *,
        beam_width: int = 20,
        aic_patience: int = 2,
        report_time: bool = False,
    ):
        self.beam_width = beam_width
        self.aic_patience = aic_patience
        self.report_time = report_time

    # -----------------------------------------------------------------
    def run(
        self,
        scorer: SparseScorer,
        *,
        max_k: int,
        init_supports: list[tuple[int, ...]] | None = None,
        **_kwargs,
    ) -> SparsityResult:
        """Execute the beam search.

        Parameters
        ----------
        scorer : SparseScorer
        max_k : int
            Maximum model size to consider.
        init_supports : list of tuples of int, optional
            Seed supports to inject into the initial frontier.  Each
            entry is a tuple of basis-function indices.  Useful for
            seeding the search with a known good model (e.g. the true
            support) so that the Pareto front is guaranteed to include
            it.

        Returns
        -------
        SparsityResult
        """
        t0 = time.perf_counter()
        p = scorer.p
        beam_width = self.beam_width

        # ---- skyline arrays ------------------------------------------
        best_info = [-np.inf] * (max_k + 1)
        best_support = [[] for _ in range(max_k + 1)]
        best_coeffs = [None] * (max_k + 1)

        # ---- per-k heaps (min-heap keyed by info) --------------------
        beam = [[] for _ in range(max_k + 1)]
        uid = itertools.count()

        def heap_item(info, state):
            return (info, next(uid), state)

        # ---- visited set (tuples of Python ints) ---------------------
        visited: set[tuple[int, ...]] = set()

        def push_into_beam(state: dict) -> bool:
            k = len(state["B"])
            h = beam[k]
            if len(h) < beam_width:
                heapq.heappush(h, heap_item(state["info"], state))
                return True
            if state["info"] > h[0][0]:
                heapq.heappushpop(h, heap_item(state["info"], state))
                return True
            return False

        # ---- initialise with empty support ---------------------------
        null = dict(B=(), info=0.0, coeffs=None)
        heapq.heappush(beam[0], heap_item(0.0, null))
        visited.add(())
        best_info[0] = 0.0

        frontier = [collections.deque([null])] + [collections.deque() for _ in range(max_k)]

        # ---- seed with user-supplied supports ------------------------
        if init_supports is not None:
            n_seeded = 0
            for B_raw in init_supports:
                B = tuple(sorted(int(i) for i in B_raw))
                k = len(B)
                if k == 0 or k > max_k or B in visited:
                    continue
                visited.add(B)
                B_arr = jnp.array(B, dtype=jnp.int32)
                info_val, coeffs_val = scorer.info_and_coeffs(B_arr)
                info_f = float(info_val)
                st = dict(B=B, info=info_f, coeffs=coeffs_val)
                push_into_beam(st)
                frontier[k].append(st)
                if info_f > best_info[k]:
                    best_info[k] = info_f
                    best_support[k] = list(B)
                    best_coeffs[k] = coeffs_val
                n_seeded += 1
            logger.info("Seeded beam with %d user-supplied supports.", n_seeded)

        best_aic_by_k = [-np.inf] * (max_k + 1)
        all_indices = set(range(p))

        # ---- main loop -----------------------------------------------
        generation = 0
        logger.info(
            "Beam search: max_k=%d, beam_width=%d, p=%d.",
            max_k,
            beam_width,
            p,
        )

        while any(frontier[k] for k in range(max_k + 1)):
            new_frontier = [collections.deque() for _ in range(max_k + 1)]

            # Build child batches per target size j (pure Python ints)
            child_batches: list[list[tuple[int, ...]]] = [[] for _ in range(max_k + 1)]

            for k, parents in enumerate(frontier):
                if not parents:
                    continue
                for st in parents:
                    B_set = set(st["B"])

                    # --- ADD children (size k+1) -----------------------
                    if k < max_k and k < p:
                        remaining = all_indices - B_set
                        B_sorted = sorted(st["B"])
                        for j in remaining:
                            # sorted insert via bisect would be faster,
                            # but for small k (<100) sorted() is fine
                            child = tuple(sorted(B_sorted + [j]))
                            if child not in visited:
                                visited.add(child)
                                child_batches[k + 1].append(child)

                    # --- DROP children (size k-1) ----------------------
                    if k > 0:
                        B_list = sorted(st["B"])
                        for pos in range(k):
                            child = tuple(B_list[:pos] + B_list[pos + 1 :])
                            if child not in visited:
                                visited.add(child)
                                child_batches[k - 1].append(child)

            # Track which cardinalities were touched
            touched_ks: set[int] = set()

            # Score each child individually via the single-support JIT
            # kernel; see the module docstring for why this beats vmap
            # for beam search.
            for j in range(max_k + 1):
                batch = child_batches[j]
                if not batch:
                    continue
                touched_ks.add(j)
                for child in batch:
                    info_jax, coeff_jax = scorer.info_and_coeffs(
                        jnp.array(child, dtype=jnp.int32),
                    )
                    info_i = float(info_jax)
                    coeff_i = np.asarray(coeff_jax)
                    st = dict(B=child, info=info_i, coeffs=coeff_i)
                    inserted = push_into_beam(st)
                    if inserted:
                        new_frontier[j].append(st)

            # Cap new frontier to beam_width per k (heapq.nlargest)
            for j in range(max_k + 1):
                if len(new_frontier[j]) > beam_width:
                    top = heapq.nlargest(beam_width, new_frontier[j], key=lambda s: s["info"])
                    new_frontier[j] = collections.deque(top)

            # Update skyline — only touched cardinalities
            for kk in touched_ks:
                h = beam[kk]
                if not h:
                    continue
                best_info_val, _, best_state = max(h)
                if best_info_val > best_info[kk]:
                    best_info[kk] = best_info_val
                    best_support[kk] = list(best_state["B"])
                    best_coeffs[kk] = best_state["coeffs"]
                best_aic_by_k[kk] = max(best_aic_by_k[kk], best_info_val - kk)

            # AIC early-stop
            k_star = -1
            for i in range(max_k + 1):
                if new_frontier[i]:
                    break
                k_star = i

            if k_star >= self.aic_patience:
                window = range(k_star - self.aic_patience + 1, k_star + 1)
                aic_vals = [best_aic_by_k[i] for i in window]
                strictly_down = all(aic_vals[i] < aic_vals[i - 1] - 1e-9 for i in range(1, len(aic_vals)))
                if strictly_down:
                    logger.info("Early stop: AIC declined on closed window %s.", list(window))
                    break

            if any(len(q) > 0 for q in new_frontier):
                tot_front = sum(len(q) for q in new_frontier)
                knonzero = [i for i, q in enumerate(new_frontier) if q]
                kmin, kmax_val = knonzero[0], knonzero[-1]
                logger.info(
                    "Generation %d — %d visited — max info=%.4f [frontier %d, k=%d–%d]",
                    generation,
                    len(visited),
                    max(best_info),
                    tot_front,
                    kmin,
                    kmax_val,
                )

            frontier = new_frontier
            generation += 1

        # ---- second-best extraction ----------------------------------
        second_info = [-np.inf] * (max_k + 1)
        second_support = [[] for _ in range(max_k + 1)]

        for k, h in enumerate(beam):
            if len(h) >= 2:
                best_two = heapq.nlargest(2, h, key=lambda t: t[0])
                s_info, _, s_state = best_two[1]
                second_info[k] = s_info
                second_support[k] = list(s_state["B"])

        if self.report_time:
            dt = time.perf_counter() - t0
            logger.info(
                "Beam search done in %.2fs (%d supports explored).",
                dt,
                len(visited),
            )

        return SparsityResult(
            p=scorer.p,
            total_info=float(scorer.total_info),
            method=self.name,
            best_info_by_k=best_info,
            best_support_by_k=best_support,
            best_coeffs_by_k=best_coeffs,
            second_info_by_k=second_info,
            second_support_by_k=second_support,
        )
