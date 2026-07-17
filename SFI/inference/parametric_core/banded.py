# SFI/inference/parametric_core/banded.py
"""
Exact block-banded innovations whitening (bandwidth 1 and 2).

The flow residuals ``r_k`` have a block-banded covariance Σ (OD:
tridiagonal, ``A_k`` and lag-1 ``C_k``; UD: pentadiagonal, adding lag-2
``E_k`` — see :mod:`covariance`).  The block LDLᵀ factorisation
``Σ = L S Lᵀ`` of a banded SPD matrix has a *banded* unit-lower factor
``L`` (same bandwidth), so the innovations

    e_m = r_m − L1_m e_{m−1} [− L2_m e_{m−2}],      ẽ_m = R_m⁻¹ e_m

(``R_m = chol S_m``) whiten the stream **exactly** in one sequential
pass — no overlapping windows, no truncated precision.  Whitening the
regressor columns ``u_m = ψ_m − L1_m u_{m−1} [− L2_m u_{m−2}]``,
``ψ̃ = R⁻¹u`` alongside gives the exact GLS objects by plain sums:

    rᵀΣ⁻¹r  = Σ ẽᵀẽ,     ψᵀΣ⁻¹ψ = Σ ψ̃ᵀψ̃,
    ψᵀΣ⁻¹r  = Σ ψ̃ᵀẽ,    log|Σ| = 2 Σ log diag R.

Recursion (processing order m; ``Σ'_{m,m−1}``/``Σ'_{m,m−2}`` are the
coupling blocks in that order):

    B = 1:  L1_m = Σ'_{m,m−1} S_{m−1}⁻¹ ;   S_m = A_m − L1_m Σ'_{m,m−1}ᵀ
    B = 2:  G2_m = Σ'_{m,m−2} ;             L2_m = G2_m S_{m−2}⁻¹
            G1_m = Σ'_{m,m−1} − L2_m (L1_{m−1} S_{m−2})ᵀ
            L1_m = G1_m S_{m−1}⁻¹
            S_m  = A_m − L1_m G1_mᵀ − L2_m G2_mᵀ

(the lag-2 row needs no correction: the correction sum over shared
columns is empty for bandwidth 2).

**Direction.**  ``reverse=True`` (the default) processes each segment in
reversed physical time — the anti-causal factorisation.  This is
load-bearing for the errors-in-variables instrument: pairing the
instrument at ``k`` only against the *innovation of r_k given the
future* keeps every instrument–residual pairing η-clean, whereas the
two-sided ``ψ_instᵀΣ⁻¹r`` is *not* η-clean and is biased under
measurement noise.
For the symmetric paths (NLL, MLE Gram) the direction is irrelevant.

**Masks.**  A ``valid`` bit per step multiplies the coupling blocks
(auto-restarting the recursion across gaps, matching how
``covariance.build_*_blocks(valid_mask=...)`` decouples segments) and
zeroes the step's contributions; invalid steps carry a neutral
``S = I``.

**Chunk carry.**  The scan state is O(d² + d·n); a
:class:`BandedCarry` can be threaded across consecutive chunks of one
segment so results are chunking-invariant.

``jitter`` is relative to the block magnitude.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["BandedCarry", "init_carry", "whiten_segment", "banded_nll_gram"]


class BandedCarry(NamedTuple):
    """Scan state between consecutive steps (and across chunks).

    ``R1``/``R2`` are the Cholesky factors of the innovation covariances
    of the previous / before-previous processed step; ``L1_prev`` the
    previous lag-1 factor row; ``e*``/``u*`` the corresponding
    innovations and whitened-column numerators; ``v*`` their validity.
    """

    R1: jnp.ndarray        # (d, d)
    R2: jnp.ndarray        # (d, d)
    L1_prev: jnp.ndarray   # (d, d)
    e1: jnp.ndarray        # (d,)
    e2: jnp.ndarray        # (d,)
    u1: jnp.ndarray        # (d, n)
    u2: jnp.ndarray        # (d, n)
    v1: jnp.ndarray        # () bool
    v2: jnp.ndarray        # () bool


def init_carry(d, n_cols, dtype):
    I_d = jnp.eye(d, dtype=dtype)
    z = jnp.zeros((d,), dtype)
    zu = jnp.zeros((d, n_cols), dtype)
    f = jnp.asarray(False)
    return BandedCarry(I_d, I_d, jnp.zeros((d, d), dtype), z, z, zu, zu, f, f)


def _tri_solve(R, B):
    """Solve ``(R Rᵀ) X = B`` given the lower Cholesky factor ``R``."""
    return jax.scipy.linalg.cho_solve((R, True), B)


def _floored_chol(S, jitter):
    d = S.shape[-1]
    S = 0.5 * (S + S.T)
    scale = jnp.maximum(jnp.trace(S) / d, 1e-300)
    return jnp.linalg.cholesky(S + jitter * scale * jnp.eye(d, dtype=S.dtype))


def whiten_segment(A, c1, c2, r, cols, valid=None, *, raw_cols=None,
                   carry=None, jitter=1e-10, reverse=True):
    r"""Whiten one chunk (single particle) of a banded residual stream.

    Parameters
    ----------
    A : ``(K, d, d)``
        Diagonal covariance blocks, physical time order.
    c1 : ``(K, d, d)``
        Per-step lag-1 couplings **to the next physical row**:
        ``c1[k] = Cov(r_k, r_{k+1})``.  The entry at the chunk edge that
        has no partner is never consumed — the carry's validity bit
        zeroes it (fresh start ⇒ ``carry.v1 = False``) — so callers may
        leave it as zeros.
    c2 : ``(K, d, d)`` or None
        Lag-2 couplings ``c2[k] = Cov(r_k, r_{k+2})``; ``None`` for
        bandwidth 1.
    r : ``(K, d)`` residuals.
    cols : ``(K, d, n)`` regressor columns, co-whitened through the full
        recursion (``ψ̃ = R⁻¹(L⁻¹ψ)``) — the Σ⁻¹-pairing side.  Pass
        ``n = 0`` for an NLL-only pass.
    valid : ``(K,)`` bool or None.
    raw_cols : ``(K, d, m)`` or None
        Test-function columns whitened **scale-only** (``R_k⁻¹`` at their
        own row, no propagation): the errors-in-variables left factor,
        which pairs the raw instrument at ``k`` against the innovation
        ``ẽ_k`` — see the module docstring on the backward pairing.
    carry : :class:`BandedCarry` or None
        State from the previously processed chunk of the same segment —
        for ``reverse=True`` (the default) feed chunks last-to-first;
        ``None`` starts fresh.
    jitter : float
        Relative floor on the innovation covariance diagonal.
    reverse : bool
        Process in reversed physical time (default; required for the IV
        pairing — see module docstring).

    Returns
    -------
    e_t : ``(K, d)`` whitened innovations (flipped back to physical
        order when ``reverse``).
    cols_t : ``(K, d, n)`` whitened columns.
    raw_t : ``(K, d, m)`` scale-only-whitened raw columns (``m = 0``
        array when ``raw_cols`` is None).
    logdet : ``(K,)`` per-step ``Σ log diag R`` (0 on invalid steps).
    valid_out : ``(K,)`` the validity actually applied.
    carry_out : :class:`BandedCarry`
    """
    K, d = r.shape
    n = cols.shape[-1]
    dtype = r.dtype
    I_d = jnp.eye(d, dtype=dtype)
    if valid is None:
        valid = jnp.ones((K,), bool)
    if raw_cols is None:
        raw_cols = jnp.zeros((K, d, 0), dtype)
    band2 = c2 is not None
    if not band2:
        c2 = jnp.zeros((K, d, d), dtype)

    if reverse:
        # Step m handles physical k = K−1−m; the previously processed
        # steps are the physical FUTURE k+1, k+2, so the row blocks are
        # exactly the per-step couplings:
        #   Σ'_{m,m−1} = Cov(r_k, r_{k+1}) = c1[k],
        #   Σ'_{m,m−2} = Cov(r_k, r_{k+2}) = c2[k].
        A_p = jnp.flip(A, 0)
        r_p = jnp.flip(r, 0)
        cols_p = jnp.flip(cols, 0)
        raw_p = jnp.flip(raw_cols, 0)
        valid_p = jnp.flip(valid, 0)
        c1_p = jnp.flip(c1, 0)
        c2_p = jnp.flip(c2, 0)
    else:
        # Forward: Σ'_{m,m−1} = Cov(r_k, r_{k−1}) = c1[k−1]ᵀ, and the
        # lag-2 row block is c2[k−2]ᵀ (front entries killed by the fresh
        # carry's validity bits).
        A_p, r_p, cols_p, raw_p, valid_p = A, r, cols, raw_cols, valid
        roll1 = jnp.concatenate([c1[-1:], c1[:-1]])
        roll2 = jnp.concatenate([c2[-2:], c2[:-2]])
        c1_p = jnp.swapaxes(roll1, -1, -2)
        c2_p = jnp.swapaxes(roll2, -1, -2)

    if carry is None:
        carry = init_carry(d, n, dtype)

    def body(c, x):
        A_k, c1_k, c2_k, r_k, cols_k, raw_k, v_k = x
        link1 = v_k & c.v1
        link2 = v_k & c.v1 & c.v2 if band2 else jnp.asarray(False)

        G1 = jnp.where(link1, c1_k, 0.0)
        if band2:
            G2 = jnp.where(link2, c2_k, 0.0)
            L2 = _tri_solve(c.R2, G2.T).T
            # lag-2 correction to the lag-1 numerator:
            # Σ'_{m,m−1} − L2 (L1_{m−1} S_{m−2})ᵀ, with S_{m−2} = R2 R2ᵀ
            S2 = c.R2 @ c.R2.T
            G1 = G1 - L2 @ (c.L1_prev @ S2).T
        else:
            G2 = jnp.zeros_like(A_k)
            L2 = jnp.zeros_like(A_k)

        L1 = _tri_solve(c.R1, G1.T).T
        S = A_k - L1 @ G1.T - (L2 @ G2.T if band2 else 0.0)
        S = jnp.where(v_k, S, I_d)
        R = _floored_chol(S, jitter)

        e = jnp.where(v_k, r_k - L1 @ c.e1 - L2 @ c.e2, 0.0)
        u = jnp.where(v_k, cols_k - L1 @ c.u1 - L2 @ c.u2, 0.0)

        e_t = jax.scipy.linalg.solve_triangular(R, e, lower=True)
        u_t = jax.scipy.linalg.solve_triangular(R, u, lower=True)
        # raw (test-function) columns: scale-only, no propagation
        raw_t = jax.scipy.linalg.solve_triangular(
            R, jnp.where(v_k, raw_k, 0.0), lower=True)
        logdet = jnp.where(v_k, jnp.sum(jnp.log(jnp.diagonal(R))), 0.0)

        new = BandedCarry(R1=R, R2=c.R1, L1_prev=L1,
                          e1=e, e2=c.e1, u1=u, u2=c.u1,
                          v1=v_k, v2=c.v1)
        return new, (e_t, u_t, raw_t, logdet)

    carry_out, (e_t, cols_t, raw_t, logdet) = lax.scan(
        body, carry, (A_p, c1_p, c2_p, r_p, cols_p, raw_p, valid_p))

    if reverse:
        e_t = jnp.flip(e_t, 0)
        cols_t = jnp.flip(cols_t, 0)
        raw_t = jnp.flip(raw_t, 0)
        logdet = jnp.flip(logdet, 0)
    return e_t, cols_t, raw_t, logdet, valid, carry_out


def banded_nll_gram(A, off1, off2, r, psi=None, psi_left=None, valid=None, *,
                    jitter=1e-10, reverse=True, carry=None):
    r"""Exact NLL and (optionally) Gram/score/H for one whole segment.

    ``off1``/``off2`` follow the :mod:`covariance` convention
    (``off1[i] = Cov(r_i, r_{i+1})``, length ``K−1``; ``off2`` length
    ``K−2`` or ``None``) and are padded to the per-step layout of
    :func:`whiten_segment` internally.  ``psi`` is the regressor
    ``∂r/∂θ`` (``(K, d, n)``); ``psi_left`` an optional distinct left
    factor (the EIV-blended instrument) — ``None`` means the symmetric
    MLE (``ψ_L = ψ``).

    Returns a dict with ``nll`` (``½Σẽᵀẽ + Σ log diag R``), ``n_terms``,
    and when ``psi`` is given ``G = Σ ψ̃_Lᵀψ̃``, ``f = Σ ψ̃_Lᵀẽ``,
    ``H = Σ ψ̃_Lᵀψ̃_L``, plus the chunk ``carry``.
    """
    K, d = r.shape
    cols = psi if psi is not None else jnp.zeros((K, d, 0), r.dtype)

    c1 = jnp.concatenate([off1, jnp.zeros((K - off1.shape[0], d, d), r.dtype)])
    c2 = (None if off2 is None else
          jnp.concatenate([off2, jnp.zeros((K - off2.shape[0], d, d), r.dtype)]))

    e_t, cols_t, raw_t, logdet, valid_out, carry_out = whiten_segment(
        A, c1, c2, r, cols, valid, raw_cols=psi_left, jitter=jitter,
        reverse=reverse, carry=carry)

    out = {
        "nll": 0.5 * jnp.sum(e_t * e_t) + jnp.sum(logdet),
        "n_terms": jnp.sum(valid_out.astype(r.dtype)),
        "carry": carry_out,
    }
    if psi is not None:
        # symmetric path: both slots the fully-whitened ψ̃ (H ≡ G);
        # IV path: the left factor is the scale-only raw test function.
        lt = cols_t if psi_left is None else raw_t
        out["G"] = jnp.einsum("kdn,kdm->nm", lt, cols_t)
        out["f"] = jnp.einsum("kdn,kd->n", lt, e_t)
        out["H"] = jnp.einsum("kdn,kdm->nm", lt, lt)
    return out
