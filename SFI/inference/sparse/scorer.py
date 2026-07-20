"""
SFI.inference.sparse.scorer — Normal-equations scorer
=====================================================

The :class:`SparseScorer` owns the pre-computed moment vector **M** and
Gram matrix **G** and provides efficient (JIT / vmap) evaluation of the
log-likelihood gain for any candidate support :math:`B`.

It is *stateless* with respect to the search: no Pareto-front data lives
here.  Every strategy receives a scorer and calls its methods.

Performance notes
~~~~~~~~~~~~~~~~~
* Symmetry of **G** is detected at construction time.  When symmetric
  PSD, the restricted solve uses Cholesky (``assume_a="pos"``),
  which is ~2× faster and more stable than the general LU path.
* The solve is fully JIT-compatible (no Python-level ``try``/``except``).
  Singular/rank-deficient cases are handled via ``jnp.linalg.lstsq``.
* ``vmap_info`` avoids double-JIT: the vmapped kernel is compiled
  once per support size *k* as a standalone pure function.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_la
import numpy as np

logger = logging.getLogger(__name__)

Array = jnp.ndarray


# =====================================================================
# Pure-function kernels (no class reference ⇒ clean JIT, clean vmap)
# =====================================================================


def _solve_sym(A_norm: Array, b_norm: Array) -> Array:
    """Cholesky-based solve for a symmetric PSD system (normalised)."""
    return jsp_la.solve(A_norm, b_norm, assume_a="pos")


def _solve_gen(A_norm: Array, b_norm: Array) -> Array:
    """General LU solve for a non-symmetric system (normalised)."""
    return jsp_la.solve(A_norm, b_norm, assume_a="gen")


def _solve_lstsq(A_norm: Array, b_norm: Array) -> Array:
    """Least-squares fallback (always JIT-safe)."""
    x, _, _, _ = jnp.linalg.lstsq(A_norm, b_norm, rcond=None)
    return x


def _info_kernel_sym(M: Array, G: Array, B: Array, tol: float) -> Tuple[Array, Array]:
    """Score one support — symmetric-PSD fast path."""
    B_idx = jnp.asarray(B, dtype=jnp.int32)
    M_B = M[B_idx]
    G_BB = G[jnp.ix_(B_idx, B_idx)]

    # Diagonal preconditioning
    diag_clipped = jnp.clip(jnp.diag(G_BB), min=tol)
    d = jnp.sqrt(diag_clipped)
    D_inv = 1.0 / d
    A_norm = (G_BB * D_inv[:, None]) * D_inv[None, :]
    b_norm = M_B / d

    # Cholesky-based solve; lstsq fallback via lax.cond on det
    # We use a simple heuristic: if reciprocal condition number is
    # reasonable, use Cholesky; otherwise lstsq.
    # For JIT compatibility we always compute both paths and select.
    x_chol = _solve_sym(A_norm, b_norm)
    # If Cholesky produced NaN/Inf (silent failure on non-PSD input),
    # fall back to least-squares.
    chol_ok = jnp.all(jnp.isfinite(x_chol))
    x_norm = jax.lax.cond(
        chol_ok,
        lambda _: x_chol,
        lambda _: _solve_lstsq(A_norm, b_norm),
        None,
    )

    C_B = x_norm / d
    Q_B = C_B @ M_B
    info = 0.5 * Q_B
    return info, C_B


def _info_kernel_gen(M: Array, G: Array, B: Array, tol: float) -> Tuple[Array, Array]:
    """Score one support — general (non-symmetric) G."""
    B_idx = jnp.asarray(B, dtype=jnp.int32)
    M_B = M[B_idx]
    G_BB = G[jnp.ix_(B_idx, B_idx)]

    diag_clipped = jnp.clip(jnp.diag(G_BB), min=tol)
    d = jnp.sqrt(diag_clipped)
    D_inv = 1.0 / d
    A_norm = (G_BB * D_inv[:, None]) * D_inv[None, :]
    b_norm = M_B / d

    x_gen = _solve_gen(A_norm, b_norm)
    gen_ok = jnp.all(jnp.isfinite(x_gen))
    x_norm = jax.lax.cond(
        gen_ok,
        lambda _: x_gen,
        lambda _: _solve_lstsq(A_norm, b_norm),
        None,
    )

    C_B = x_norm / d
    Q_B = C_B @ M_B
    info = 0.5 * Q_B
    return info, C_B


def _info_kernel_residual_sym(M: Array, G: Array, B: Array, tol: float, norm_X2: float, n: int) -> Tuple[Array, Array]:
    """Symmetric-PSD path with residual-based information gain."""
    info_raw, C_B = _info_kernel_sym(M, G, B, tol)
    Q_B = 2.0 * info_raw  # undo the 0.5 factor
    RSS_B = norm_X2 - Q_B
    info = 0.5 * n * jnp.log(norm_X2 / RSS_B)
    return info, C_B


def _info_kernel_residual_gen(M: Array, G: Array, B: Array, tol: float, norm_X2: float, n: int) -> Tuple[Array, Array]:
    """General path with residual-based information gain."""
    info_raw, C_B = _info_kernel_gen(M, G, B, tol)
    Q_B = 2.0 * info_raw
    RSS_B = norm_X2 - Q_B
    info = 0.5 * n * jnp.log(norm_X2 / RSS_B)
    return info, C_B


# =====================================================================
# Shared JIT / vmap caches   (keyed by scorer config, NOT instance)
# =====================================================================
# By keeping M and G as call-time arguments instead of partial-bound,
# all SparseScorer instances with the same (is_sym, use_residuals, tol,
# norm_X2, n) share compiled XLA code.  This avoids per-instance
# recompilation — a major speed-up when running many scorers with the
# same p (e.g. benchmarks, cross-validation, bootstrapping).

_single_jit_cache: dict[tuple, callable] = {}
_vmap_jit_cache: dict[tuple, callable] = {}


def _get_single_jit(is_sym: bool, use_residuals: bool, tol: float, norm_X2: float = 0.0, n: int = 1):
    key = (is_sym, use_residuals, tol, norm_X2, n)
    if key not in _single_jit_cache:
        kernel = _pick_kernel(is_sym, use_residuals, tol, norm_X2, n)
        _single_jit_cache[key] = jax.jit(kernel)
    return _single_jit_cache[key]


def _get_vmap_jit(k: int, is_sym: bool, use_residuals: bool, tol: float, norm_X2: float = 0.0, n: int = 1):
    key = (k, is_sym, use_residuals, tol, norm_X2, n)
    if key not in _vmap_jit_cache:
        kernel = _pick_kernel(is_sym, use_residuals, tol, norm_X2, n)
        _vmap_jit_cache[key] = jax.jit(jax.vmap(kernel, in_axes=(None, None, 0), out_axes=(0, 0)))
    return _vmap_jit_cache[key]


def _pick_kernel(is_sym, use_residuals, tol, norm_X2, n):
    """Return a partial-bound kernel (M, G, B) -> (info, C_B)."""
    if use_residuals:
        if is_sym:
            return partial(_info_kernel_residual_sym, tol=tol, norm_X2=norm_X2, n=n)
        return partial(_info_kernel_residual_gen, tol=tol, norm_X2=norm_X2, n=n)
    if is_sym:
        return partial(_info_kernel_sym, tol=tol)
    return partial(_info_kernel_gen, tol=tol)


# =====================================================================
# SparseScorer
# =====================================================================


class SparseScorer:
    r"""Score candidate supports by solving the restricted normal equations.

    Parameters
    ----------
    M : (p,) Array
        Pre-computed moment vector (cross-moments between data and
        basis functions).
    G : (p, p) Array
        Normal-equations matrix.  May be non-symmetric (e.g. when using
        Itô-shift moment estimators).  Symmetry is detected automatically
        and a Cholesky fast-path is used when possible.
    norm_X2 : float, default 0.0
        Sum of squared observations.  Only used when
        ``use_residuals=True``.
    n : int, default 1
        Sample count prefactor used in the residual-based information gain
        :math:`\tfrac{1}{2} n\,\log(\lVert X\rVert^2 / \mathrm{RSS})`.
        Only used when ``use_residuals=True``.
    pinv_tol : float, default 1e-8
        Tolerance for the diagonal preconditioning floor.
    use_residuals : bool, default False
        If *True*, the information gain is computed via the residual
        sum-of-squares expression instead of the explicit quadratic form.
    """

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
        self.M = jnp.asarray(M)
        self.G = jnp.asarray(G)
        self.p: int = int(self.M.shape[0])
        self.pinv_tol = pinv_tol
        self.norm_X2 = norm_X2
        self.n = n
        self.use_residuals = use_residuals

        # --- detect symmetry of G (once, on host) --------------------
        G_np = np.asarray(self.G)
        self.G_is_symmetric: bool = bool(np.allclose(G_np, G_np.T, atol=1e-12 * (np.abs(G_np).max() + 1e-30)))
        if self.G_is_symmetric:
            logger.debug("SparseScorer: G is symmetric → Cholesky fast path.")
        else:
            logger.debug("SparseScorer: G is non-symmetric → general LU path.")

        # --- select the right kernel ---------------------------------
        # Use shared module-level JIT cache so all instances with the
        # same config share compiled XLA code.
        self._jit_kernel = _get_single_jit(
            self.G_is_symmetric,
            self.use_residuals,
            self.pinv_tol,
            self.norm_X2,
            self.n,
        )

        # Pre-compute the full (dense) solution.
        self.total_info, self.total_C = self.info_and_coeffs(jnp.arange(self.p))

    # -----------------------------------------------------------------
    # Score a single support B
    # -----------------------------------------------------------------
    def info_and_coeffs(self, B: Array) -> Tuple[Array, Array]:
        r"""Solve :math:`G_{BB}\,C_B = M_B` and return the information gain.

        Parameters
        ----------
        B : (k,) int Array
            Indices of the active basis functions (the *support*).

        Returns
        -------
        info : scalar Array
            :math:`\tfrac{1}{2}\,C_B^\top M_B` (or the RSS variant when
            ``use_residuals=True``).
        C_B : (k,) Array
            Maximum-likelihood coefficients for the restricted support.
        """
        if B.size == 0:
            return jnp.array(0.0), jnp.zeros(0, dtype=self.M.dtype)
        return self._jit_kernel(self.M, self.G, B)

    # -----------------------------------------------------------------
    # Batched evaluation
    # -----------------------------------------------------------------

    @staticmethod
    def _pad_to_pow2(n: int) -> int:
        """Round *n* up to next power of 2 (min 1)."""
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def vmap_info(self, batch: Array) -> Tuple[Array, Array]:
        """Score a batch of supports of the *same* cardinality.

        The batch is padded to the next power-of-2 length so that
        JAX's compilation cache stays bounded (≤ ~12 unique shapes per
        support size *k* instead of one per distinct batch count).

        Parameters
        ----------
        batch : (n_supports, k) int Array
            Each row is a sorted support of length *k*.

        Returns
        -------
        infos : (n_supports,) Array
        coeffs : (n_supports, k) Array
        """
        n_real = batch.shape[0]
        k = int(batch.shape[1])

        n_padded = self._pad_to_pow2(n_real)

        if n_padded > n_real:
            # Pad with copies of the first row (valid support — avoids
            # pathological zeros).  Results from padded rows are discarded.
            pad_rows = jnp.broadcast_to(batch[0:1], (n_padded - n_real, k))
            batch = jnp.concatenate([batch, pad_rows], axis=0)

        fn = _get_vmap_jit(
            k,
            self.G_is_symmetric,
            self.use_residuals,
            self.pinv_tol,
            self.norm_X2,
            self.n,
        )
        infos, coeffs = fn(self.M, self.G, batch)

        # Slice off padding
        return infos[:n_real], coeffs[:n_real]
