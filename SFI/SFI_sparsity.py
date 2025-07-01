"""
===============================================================================
SFI_sparsity
===============================================================================
Likelihood-based sparse model selection module. Performs coefficient inference
 + Pareto-front sparsification for Stochastic-Force Inference (SFI).

What this module does
---------------------
The input are the following arrays, calculated elsewhere:

* **M** – a (p,) vector of cross-moments between data and basis functions;
* **G** – a (p × p) positive-definite matrix coming from the normal equations.
          **G is fully pre-normalised**: it may include noise matrix *A*
          (over-damped vs. under-damped cases), time-step factors, or
          trapezoidal-rule weights.  It is *not* assumed to equal ΦᵀΦ.

For any support *B* ⊆ {0,…, p-1} we must

    1. Solve   G_BB · C_B = M_B
    2. Score   I_B  (log-likelihood gain w.r.t. the null model).

Exhaustively scanning all 2ᵖ supports is intractable, so we:

1. **Construct a Pareto front** of sparse candidates with a *bidirectional
   beam search* that keeps only the best few supports per cardinality.

2. **Pick the final model** by maximising an information criterion
   (e.g. AIC, BIC, SIC/PASTIS) along that skyline.

Public entry point
------------------
`SparseModelSelector` – manages dense solves, caches scores, runs the beam
search, and exposes helpers for IC-based selection.

"""
import time
from functools import partial
from typing import Dict, List, Optional, Tuple
from sys import stdout
import jax
import jax.numpy as jnp
from SFI.SFI_utils import solve_or_pinv

# -----------------------------------------------------------------------------
# Type aliases – project-wide conventions
# -----------------------------------------------------------------------------
Array = jnp.ndarray
Key   = jax.random.PRNGKey

class SparseModelSelector:
    """
    Helper object that caches dense coefficient solves and navigates the
    sparsity lattice via a beam search.

    Parameters
    ----------
    M : (p,) Array
        Pre-computed moment vector.
    G : (p, p) Array
        Positive-definite “normal-equations” matrix.  Can already contain
        time-step or noise-matrix normalisation and is **not** assumed to be
        ΦᵀΦ.
    norm_X2 : float, default 0.
        Sum of squares of the observed trajectory.  Required only when
        `use_residuals=True`.
    n : int, default 1
        Number of samples (kept for API compatibility with other SFI modules).
    pinv_tol : float, default 1e-8
        Threshold below which `SFI_utils.solve_or_pinv` falls back to a
        pseudo-inverse.
    use_residuals : bool, default False
        If *True*, the information gain is computed via residual
        sum-of-squares instead of the explicit quadratic form.
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        M: Array,
        G: Array,
        norm_X2: float = 0.0,
        n: int = 1,
        pinv_tol: float = 1e-8,
        use_residuals: bool = False,
    ):
        # --- persistent inputs -----------------------------------------
        self.M, self.G = M, G
        self.p = int(M.shape[0])          # number of candidate basis functions
        self.pinv_tol = pinv_tol
        self.norm_X2 = norm_X2
        self.n = n
        self.use_residuals = use_residuals

        # --- skyline placeholders --------------------------------------
        self.best_info_by_k: Optional[List[float]]         = None  # best raw info
        self.best_support_by_k: Optional[List[List[int]]]  = None
        self.best_coeffs_by_k: Optional[List[Optional[Array]]] = None

        # cache of vmapped dense evaluators keyed by support size k
        self._vm_dense: dict[int, callable] = {}

        # Pre-compute the full dense solution for convenience
        self.total_info, self.total_C = self._info_and_coeffs(jnp.arange(self.p))

    # ------------------------------------------------------------------ #
    # Internal: score ONE support B                                      #
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=(0,))
    def _info_and_coeffs(
        self, B: Array
    ) -> Tuple[Array, Array]:
        """
        Compute log-likelihood gain and coefficients for a single support *B*.

        Returns
        -------
        info : scalar Array
            0.5 · C_Bᵀ M_B (quadratic form) by default, or the RSS-based
            expression when `use_residuals=True`.
        C_B : (|B|,) Array
            Maximum-likelihood coefficients for the restricted support.
        """

        # ────────────────────────────────────────────────────────────────
        # Trivial empty-support case
        # ────────────────────────────────────────────────────────────────
        if B.size == 0:
            return jnp.array(0.0), jnp.zeros(0, dtype=self.M.dtype)

        # 1 | gather sub-matrices / vectors
        B_idx = jnp.array(B, dtype=jnp.int32)
        M_B   = self.M[B_idx]                         # (k,)
        G_BB  = self.G[jnp.ix_(B_idx, B_idx)]         # (k, k)

        # 2 | solve – fall back to pinv if G_BB is ill-conditioned
        C_B = solve_or_pinv(G_BB, M_B, tol=self.pinv_tol)

        # 3 | quadratic term Q_B = C_Bᵀ M_B  (explained variance)
        Q_B = C_B @ M_B

        # 4 | convert to information gain
        if self.use_residuals:
            RSS_B = self.norm_X2 - Q_B
            info  = 0.5 * self.n * jnp.log(self.norm_X2 / RSS_B)
        else:
            info  = 0.5 * Q_B

        return info, C_B

    # ------------------------------------------------------------------ #
    # Vectorised dense evaluator                                         #
    # ------------------------------------------------------------------ #
    def _vmap_info(self, batch: Array):
        """
        Vectorised wrapper around `_info_and_coeffs`.

        Compiled & cached **per support size k** to avoid repeated JITs during
        the beam search.
        """
        k = int(batch.shape[1])
        if k not in self._vm_dense:
            self._vm_dense[k] = jax.jit(
                jax.vmap(self._info_and_coeffs, out_axes=(0, 0))
            )
        return self._vm_dense[k](batch)

    # ------------------------------------------------------------------ #
    # Bidirectional beam search  (add & drop, dense evaluator)           #
    # ------------------------------------------------------------------
    def build_pareto_front(
        self, *,
        max_k: int,
        beam_width: int = 20,
        verbosity: int = 0,
        aic_patience: int = 2,
        report_time: bool = False
    ):
        """
        Explore the lattice by expanding every ±1 neighbour of every support in
        the *current* frontier; keep ≤ beam_width best models per cardinality k.
        Stops when (i) frontiers are empty and (ii) AIC has declined for
        `aic_patience` consecutive eligible k-values (see below).
        """
        import collections
        import itertools
        import heapq

        t0 = time.perf_counter()
        p  = self.G.shape[0]

        # --------------- skyline arrays -----------------------------------
        self.best_info_by_k     = [-jnp.inf]*(max_k+1)
        self.best_support_by_k  = [[] for _ in range(max_k+1)]
        self.best_coeffs_by_k   = [None]*(max_k+1)

        # --------------- per-k beams (min-heap keyed by info) --------------
        self.beam = [ [] for _ in range(max_k+1) ]
        uid  = itertools.count()                    # tie-break counter

        def heap_item(info, state):
            return (info, next(uid), state)

        # --------------- visited set ---------------------------------------
        visited: set[tuple[int, ...]] = set()

        # --------------- helpers -------------------------------------------
        def push_into_beam(state: dict) -> bool:
            """Try putting state into its k-beam; return True if inserted."""
            k = len(state['B'])
            h = self.beam[k]
            if len(h) < beam_width:
                heapq.heappush(h, heap_item(state['info'], state))
                return True
            if state['info'] > h[0][0]:
                heapq.heappushpop(h, heap_item(state['info'], state))
                return True
            return False

        def _expand_batch(batch: Array, new_frontier_k: collections.deque):
            """
            Batch has homogeneous length (all add or all drop children).
            Filters against `visited`, scores unseen supports, pushes those that
            enter the beam, and enqueues each *inserted* state into the new
            frontier deque (so they will be expanded in the next outer loop).
            """
            # --- filter unseen supports first (cheap) ---------------------
            keep_rows, keep_tuples = [], []
            for row in batch:
                row_sorted = jnp.sort(row)                    # canonical order
                tpl = tuple(map(int, row_sorted))             # hash key
                if tpl not in visited:
                    visited.add(tpl)
                    keep_rows.append(row_sorted)
                    keep_tuples.append(tpl)

            if not keep_rows:
                return

            keep_arr = jnp.asarray(keep_rows)
            infos, coeffs = self._vmap_info(keep_arr)      # V-map once

            for i, tpl in enumerate(keep_tuples):
                st = dict(B=jnp.array(tpl, jnp.int32),
                          info=float(infos[i]),
                          coeffs=coeffs[i])
                inserted = push_into_beam(st)
                if inserted:
                    new_frontier_k.append(st)              # schedule expansion

        # --------------- initialise with empty support ---------------------
        null = dict(B=jnp.array([], jnp.int32), info=0.0,
                    coeffs=jnp.zeros(0, self.M.dtype))
        heapq.heappush(self.beam[0], heap_item(0.0, null))
        visited.add(tuple())
        self.best_info_by_k[0] = 0.0

        frontier = [collections.deque([null])] + [collections.deque()
                                                  for _ in range(max_k)]

        # --------------- AIC tracking --------------------------------------
        best_aic_by_k = [-jnp.inf]*(max_k+1)   # store best AIC per k

        # ------------------------------------------------------------
        # Main loop: one vmap per child‐size batch
        # ------------------------------------------------------------
        generation_counter = 0
        if verbosity >= 1:
            print(f"Starting beam search with kmax={max_k} and beam_width={beam_width} over a basis of size {self.p}.")
        while any(frontier[k] for k in range(max_k + 1)):

            # Prepare the next‐level frontier queues
            new_frontier = [collections.deque() for _ in range(max_k + 1)]

            # 2) Build child_batches[j] = list of new supports of size j (sorted, unvisited)
            child_batches: list[list[Array]] = [[] for _ in range(max_k + 1)]
            for k, parents in enumerate(frontier):
                if not parents:
                    continue

                for st in parents:
                    B = st['B']  # Array shape (k,)

                    # --- ADD to size (k+1) ---------------------------------
                    if k < max_k and k < p:
                        rem = jnp.setdiff1d(jnp.arange(p, dtype=jnp.int32), B,
                                             assume_unique=True)
                        if k == 0:
                            # B is empty; each child is [j]
                            for j in rem:
                                child = jnp.array([j], jnp.int32)
                                tpl = (int(j),)
                                if tpl not in visited:
                                    visited.add(tpl)
                                    child_batches[1].append(child)
                        else:
                            for j in rem:
                                candidate = jnp.concatenate([B, jnp.array([j], jnp.int32)])
                                child = jnp.sort(candidate)        # shape (k+1,)
                                tpl = tuple(map(int, child))
                                if tpl not in visited:
                                    visited.add(tpl)
                                    child_batches[k+1].append(child)

                    # --- DROP to size (k-1) --------------------------------
                    if k > 0:
                        for pos in range(k):
                            child = jnp.delete(B, pos)             # shape (k-1,)
                            tpl = tuple(map(int, child))
                            if tpl not in visited:
                                visited.add(tpl)
                                child_batches[k-1].append(child)

            # 3) Score each batch by its size j in one Vmap call
            for j in range(max_k + 1):
                batch = child_batches[j]
                if not batch:
                    continue

                batch_arr = jnp.stack(batch)  # shape (N_j, j) (all supports of size j)
                infos, coeffs = self._vmap_info(batch_arr)  # single vmap

                for idx in range(batch_arr.shape[0]):
                    B_child = tuple(map(int, batch_arr[idx]))  # sorted size-j support
                    info_i = float(infos[idx])
                    coeff_i = coeffs[idx]
                    st = dict(B=jnp.array(B_child, jnp.int32),
                              info=info_i,
                              coeffs=coeff_i)
                    inserted = push_into_beam(st)      # updates beam[j] 
                    if inserted:
                        new_frontier[j].append(st)

            # 4) Cap each new_frontier[j] to beam_width
            for j in range(max_k + 1):
                if len(new_frontier[j]) > beam_width:
                    top_j = sorted(new_frontier[j], key=lambda st: st['info'],
                                   reverse=True)[:beam_width]
                    new_frontier[j] = collections.deque(top_j)

            # 5) Update skyline & verbosity for each k
            for kk, h in enumerate(self.beam):
                if not h:
                    continue
                best_info, _, best_state = max(h)
                if best_info > self.best_info_by_k[kk]:
                    self.best_info_by_k[kk]    = best_info
                    self.best_support_by_k[kk] = list(map(int, best_state['B']))
                    self.best_coeffs_by_k[kk]  = best_state['coeffs']

                best_aic_by_k[kk] = max(best_aic_by_k[kk], best_info - kk)

            # 6) AIC early-stop criterion
            k_star = -1
            for i in range(max_k + 1):
                if new_frontier[i]:
                    break
                k_star = i

            if k_star >= aic_patience:
                window = range(k_star - aic_patience + 1, k_star + 1)
                aic_vals = [best_aic_by_k[i] for i in window]
                strictly_down = all(aic_vals[i] < aic_vals[i - 1] - 1e-9
                                    for i in range(1, len(aic_vals)))
                if strictly_down:
                    if verbosity >= 1:
                        print("   Early stop: AIC score declined on closed window "
                              f"{list(window)}.")
                    break
                
            if verbosity >= 1 and any(len(q)>0 for q in new_frontier):
                tot_front = sum(len(q) for q in new_frontier)
                knonzero = [ i for i,q in enumerate(new_frontier) if q ]
                kmin,kmax = knonzero[0],knonzero[-1]
                best_info = max(self.best_info_by_k)
                Nvisited = len(visited)
                msg = f"   Generation: {generation_counter} -- N visited models {Nvisited} -- max info={best_info:.4f} [beam frontier size {tot_front} with k range {kmin}-{kmax}]"
                if verbosity == 1:
                    # Don't flood the output: overwrite the output
                    stdout.write('\r' + msg.ljust(160))   
                    stdout.flush()
                else:
                    # Flood it!
                    print(msg)

                
                        
            # 7) Swap frontiers and continue
            frontier = new_frontier
            generation_counter += 1

        # --------------- second-best extraction --------------------------
        self.second_info_by_k   = [-jnp.inf]*(max_k+1)
        self.second_support_by_k= [[] for _ in range(max_k+1)]

        for k, h in enumerate(self.beam):
            if len(h) >= 2:
                # heap holds (info, uid, state) tuples; take the two largest infos
                best_two = heapq.nlargest(2, h, key=lambda t: t[0])
                second_info, _, second_state = best_two[1]
                self.second_info_by_k[k]    = second_info
                self.second_support_by_k[k] = list(map(int, second_state['B']))
        # --------------- timing -------------------------------------------
        if report_time and verbosity >= 1:
            dt = time.perf_counter() - t0
            nsupports = len(visited)
            print(f"Beam search completed in {dt:.2f}s. Number of supports explored: {nsupports}" )


    # ------------------------------------------------------------------
    # Information criteria & summary table(self, name: str, *, p_param: float = 1e-3) -> Tuple[int,List[int],float]:
    def select_by_ic(self, name: str, *, p_param: float = 1e-3, verbose = True) -> Tuple[int, List[int], float]:
        """Return (k*, support, IC score) for one criterion.

        Parameters
        ----------
        name : 'aic' | 'bic' | 'pastis' | 'sic'
        p_param : constant *p₀* for the PASTIS penalty.
        """
        if self.best_info_by_k is None:
            raise RuntimeError("Run build_pareto_front() first")
        name = name.upper()
        def score(k, info):
            if info==-jnp.inf:
                return -jnp.inf
            if name=='AIC': return info - k
            if name=='BIC': return info - 0.5*k*jnp.log(self.p)
            if name=='PASTIS': return info - k*jnp.log(self.p/p_param)
            if name=='SIC': return info - k*jnp.log(self.total_info) # Experimental.
            raise ValueError(name)
        scores = [score(k,info) for k,info in enumerate(self.best_info_by_k)]
        k_star = int(jnp.argmax(jnp.array(scores)))
        if verbose:
            print(f"Criterion {name} selected a model with {k_star} terms out of {self.p}.")
        return k_star, self.best_support_by_k[k_star], scores[k_star], self.best_coeffs_by_k[k_star]

    # ------------------------------------------------------------------
    # Convenience: compute all ICs at once (optional truth metrics)
    # ------------------------------------------------------------------
    def all_ic(self, *, p_param: float = 1e-3, true_support: Optional[List[int]] = None, true_coeffs: Optional[List[float]] = None, Phi_test: Optional[Array] = None, verbose: bool = True) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for name in ('aic', 'bic', 'pastis', 'sic'):
            k, support, score, coeffs = self.select_by_ic(name, p_param=p_param)
            entry = dict(k=k, support=support, score=float(score), coeffs=coeffs)
            if true_support is not None:
                entry.update(overlap_metrics(true_support, support))
                if Phi_test is not None:
                    entry['predictive_NMSE'] = predictive_nmse(Phi_test, true_support, true_coeffs, support, coeffs)
            summary[name.upper()] = entry
        if verbose:
            # pretty print
            print('=== Information-criterion summary ===')
            print(f"{'IC':<8}  {'k*':>3}  {'score':>10}  {'TP/FP/FN':>15}  {'exact':>10} {'predictive NMSE':>10} support")
            for ic, entry in summary.items():
                extra1 = f" {entry['TP']}/{entry['FP']}/{entry['FN']}   {entry['exact']}" if 'exact' in entry else ''
                extra2 = f" {entry['predictive_NMSE']:10.4f}" if 'predictive_NMSE' in entry else ''
                print(f"{ic:<8}  {entry['k']:>3}  {entry['score']:10.2f}  {extra1}  {extra2}  {entry['support']}")
        return summary



# =============================================================================
# 4  Benchmark helpers (omniscient code)
# =============================================================================

def overlap_metrics(true_support: List[int], pred_support: List[int]) -> Dict:
    t, p = set(true_support), set(pred_support)
    tp, fp, fn = len(t & p), len(p - t), len(t - p)
    return dict(TP=tp, FP=fp, FN=fn,
                prec=tp/(tp+fp) if tp+fp else 0.0,
                rec=tp/(tp+fn) if tp+fn else 0.0,
                exact=(fp == 0 and fn == 0))

def predictive_nmse(Phi_test: Array, true_support: List[int], true_coeffs: List[float], inferred_support: List[int], inferred_coeffs: List[float]) -> float:

    if len(inferred_support) == 0:
        return 1.
    true_signal = Phi_test[:,jnp.array(true_support)] @ jnp.array(true_coeffs)
    inferred_signal = Phi_test[:,jnp.array(inferred_support)] @ jnp.array(inferred_coeffs)

    residue = true_signal - inferred_signal
    return float(jnp.sum(residue**2)/jnp.sum(inferred_signal**2))



