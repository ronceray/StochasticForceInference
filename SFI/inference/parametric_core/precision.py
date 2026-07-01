# SFI/inference/parametric_core/precision.py
"""
Windowed local-precision kernels for the parametric core.

The estimating equations need the precision ``P = Σ⁻¹`` of the banded
residual covariance.  ``P`` itself is *not* banded (the residuals are a
moving-average process, whose inverse covariance is AR(∞)), but its
entries decay geometrically away from the diagonal — at rate
``λ = |ρ|/(1+√(1−4ρ²))`` for bandwidth 1 with lag-1 correlation ``ρ``
(``|ρ| ≤ ½``, saturating only as the noise-to-signal ratio diverges).
So the *center row* of ``P`` is read off a small local window of ``W``
residuals with ``O(λ^{W−1})`` truncation error: assemble the
``(W·d)×(W·d)`` covariance, Cholesky-factor it once, and solve against
the right-hand sides.  At extreme measurement noise the window can be
widened (``extra_radius`` / ``n_cond``).

These are **single-window** kernels: the sliding window itself (the
``X_window:{W}`` stream), chunking, and the sum over centers are provided
by the ``SFI.integrate`` engine.  Two kernels are exposed:

* :func:`center_gram_contribution` — Gauss–Newton Gram ``G``, RHS ``f``,
  and pseudo-NLL from the center row (used for the parameter covariance,
  the optional exact GN step, and the sparsity hand-off).
* :func:`center_loss_contribution` — the per-center quadratic
  ``½ r_cᵀ P r`` with ``P`` *frozen*, whose AD-gradient w.r.t. live
  residuals is exactly the GN score.  This is the differentiable objective.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .covariance import CovarianceBlocks, assemble_dense

__all__ = [
    "center_gram_contribution",
    "center_loss_contribution",
    "center_nll_contribution",
    "center_conditional_nll_contribution",
]


def _cholesky(A_w, offdiag_w, jitter_chol):
    Sigma = assemble_dense(CovarianceBlocks(A_w, tuple(offdiag_w), len(offdiag_w)))
    Wd = Sigma.shape[0]
    Sigma = Sigma + jitter_chol * jnp.eye(Wd, dtype=Sigma.dtype)
    return jnp.linalg.cholesky(Sigma)


def center_gram_contribution(A_w, offdiag_w, r_w, psi_w, c, jitter_chol=1e-10, psi_right_w=None):
    r"""Gram / RHS / NLL contribution of the center residual ``c``.

    The estimating equation is ``⟨ψ_left P r⟩ = 0`` with ``P = Σ⁻¹`` the local
    precision.  ``ψ_left`` is the test function (an instrument — a free choice)
    and ``ψ_right = ∂r/∂θ`` is the regressor.  From the center row-block,

        G_w = ψ_left,cᵀ (P ψ_right)_c,   f_w = ψ_left,cᵀ (P r)_c,
        nll_w = ½ r_cᵀ (P r)_c − ½ log det P_{cc}.

    When ``psi_right_w is None`` the left and right factors coincide
    (``ψ_left = ψ_right = ψ_w``) — the symmetric plug-in MLE, numerically
    identical to the previous single-factor form.  A separate ``psi_right_w``
    yields the asymmetric (instrumental-variable) Gram used by the EIV path.
    ``nll_w`` is a function of ``r`` only and is unchanged by the split.

    The fourth output ``H_w = ψ_left,cᵀ (P ψ_left)_c`` is the model-based
    variance of the estimating function (``Var(ψ_leftᵀPr) = ψ_leftᵀPψ_left``
    since ``Cov(r) = P⁻¹``): the meat of the sandwich covariance
    ``Cov(θ̂) = G⁻¹ H G⁻ᵀ`` for the asymmetric (IV) path.  In the symmetric
    case ``H_w = G_w`` (the information identity) and the sandwich collapses
    to ``G⁻¹``.  It comes nearly free: ``ψ_left`` differs from ``ψ_right``
    only in row ``c``, so ``(Pψ_left)_c = (Pψ_right)_c + P_{cc}(ψ_left,c −
    ψ_right,c)`` reuses the column block already computed for the log-det.

    Parameters
    ----------
    A_w : ``(W, d, d)`` diagonal blocks.
    offdiag_w : tuple of ``(W-lag, d, d)`` off-diagonal blocks.
    r_w : ``(W, d)`` residuals.
    psi_w : ``(W, d, n_params)`` left test functions (instrument).
    c : int  center index within the window.
    jitter_chol : float
    psi_right_w : ``(W, d, n_params)`` or None
        Right factor (regressor ``∂r/∂θ``).  ``None`` → ``psi_w`` (symmetric).

    Returns
    -------
    G_w : ``(n_params, n_params)``
    f_w : ``(n_params,)``
    H_w : ``(n_params, n_params)``
    nll_w : scalar
    """
    W, d = A_w.shape[0], A_w.shape[-1]
    n_params = psi_w.shape[-1]
    Wd = W * d
    L = _cholesky(A_w, offdiag_w, jitter_chol)

    r_flat = r_w.reshape(Wd)
    Sinv_r = jax.scipy.linalg.cho_solve((L, True), r_flat)
    Pr_c = Sinv_r[c * d:(c + 1) * d]

    # P_{cc} for the log-det normaliser
    E_c = jnp.zeros((Wd, d), dtype=A_w.dtype).at[c * d:(c + 1) * d, :].set(jnp.eye(d, dtype=A_w.dtype))
    Sinv_Ec = jax.scipy.linalg.cho_solve((L, True), E_c)
    P_cc = Sinv_Ec[c * d:(c + 1) * d, :]
    _, logdet = jnp.linalg.slogdet(P_cc)
    nll_w = 0.5 * jnp.dot(r_w[c], Pr_c) - 0.5 * logdet

    sym = psi_right_w is None
    psi_right_w = psi_w if sym else psi_right_w
    psi_left_c = psi_w[c]
    psi_right_flat = psi_right_w.reshape(Wd, n_params)
    Ppsi_right_c = jax.scipy.linalg.cho_solve((L, True), psi_right_flat)[c * d:(c + 1) * d, :]
    G_w = psi_left_c.T @ Ppsi_right_c
    f_w = psi_left_c.T @ Pr_c
    if sym:
        H_w = G_w
    else:
        Ppsi_left_c = Ppsi_right_c + P_cc @ (psi_left_c - psi_right_w[c])
        H_w = psi_left_c.T @ Ppsi_left_c
    return G_w, f_w, H_w, nll_w


def center_conditional_nll_contribution(A_w, offdiag_w, r_w, c, n_cond, jitter_chol=1e-10):
    r"""NLL of residual ``c`` conditioned on its ``n_cond`` predecessors.

    Builds the joint covariance of ``[r_{c-n_cond}, …, r_c]`` from the
    window blocks, forms the Schur complement of the past block, and
    returns the conditional Gaussian NLL of the last block:

        S = A_cc − Σ_{c,p} Σ_{p,p}⁻¹ Σ_{p,c},  e = r_c − Σ_{c,p} Σ_{p,p}⁻¹ r_p,
        nll = ½ eᵀ S⁻¹ e + ½ log det S.

    Summed over centers with the full past this is the exact chain-rule
    NLL (telescoping log-det); with a fixed ``n_cond`` (a few past points)
    it is an excellent, **non-degenerate** approximation — the correct
    objective when ``D, Λ`` are free.  ``c`` and ``n_cond`` are static.
    """
    d = A_w.shape[-1]
    start = c - n_cond
    r_c = r_w[c]

    if n_cond == 0:
        L = jnp.linalg.cholesky(A_w[c] + jitter_chol * jnp.eye(d, dtype=A_w.dtype))
        Sinv_e = jax.scipy.linalg.cho_solve((L, True), r_c)
        return 0.5 * jnp.dot(r_c, Sinv_e) + jnp.sum(jnp.log(jnp.diagonal(L)))

    A_sub = A_w[start:c + 1]
    off_sub = tuple(offdiag_w[lag][start:c - lag] for lag in range(len(offdiag_w)))
    M = assemble_dense(CovarianceBlocks(A_sub, off_sub, len(off_sub)))
    nd = n_cond * d
    r_past = r_w[start:c].reshape(nd)

    M_pp = M[:nd, :nd] + jitter_chol * jnp.eye(nd, dtype=M.dtype)
    M_pc = M[:nd, nd:]
    M_cc = M[nd:, nd:]
    L_pp = jnp.linalg.cholesky(M_pp)
    rhs = jnp.concatenate([M_pc, r_past[:, None]], axis=1)
    sol = jax.scipy.linalg.cho_solve((L_pp, True), rhs)  # (nd, d+1)
    S = M_cc - M_pc.T @ sol[:, :d]
    e = r_c - M_pc.T @ sol[:, d]

    L_s = jnp.linalg.cholesky(S + jitter_chol * jnp.eye(d, dtype=S.dtype))
    Sinv_e = jax.scipy.linalg.cho_solve((L_s, True), e)
    return 0.5 * jnp.dot(e, Sinv_e) + jnp.sum(jnp.log(jnp.diagonal(L_s)))


def center_nll_contribution(A_w, offdiag_w, r_w, c, jitter_chol=1e-10):
    r"""Per-center windowed NLL ``½ r_cᵀ (P r)_c − ½ log det P_{cc}``.

    Unlike :func:`center_gram_contribution` this needs no test functions:
    it is the *full* (log-det-bearing) windowed NLL contribution, and is
    differentiable in ``θ, D, Λ`` jointly (the joint AD mode) when the
    blocks are built from live parameters.

    Returns
    -------
    nll_w : scalar
    """
    W, d = A_w.shape[0], A_w.shape[-1]
    Wd = W * d
    L = _cholesky(A_w, offdiag_w, jitter_chol)

    r_flat = r_w.reshape(Wd)
    Pr_c = jax.scipy.linalg.cho_solve((L, True), r_flat)[c * d:(c + 1) * d]
    quad = 0.5 * jnp.dot(r_w[c], Pr_c)

    E_c = jnp.zeros((Wd, d), dtype=A_w.dtype).at[c * d:(c + 1) * d, :].set(jnp.eye(d, dtype=A_w.dtype))
    P_cc = jax.scipy.linalg.cho_solve((L, True), E_c)[c * d:(c + 1) * d, :]
    _, logdet = jnp.linalg.slogdet(P_cc)
    return quad - 0.5 * logdet


def center_loss_contribution(A_w, offdiag_w, r_w, c, jitter_chol=1e-10, bandwidth=1):
    r"""Per-center quadratic ``½ r_cᵀ P_{cc} r_c + Σ_{l≥1} r_cᵀ P_{c,c+l} r_{c+l}``.

    Summed over all window centers this reproduces the global quadratic
    ``½ rᵀ P r`` exactly (for block-banded ``P``).  ``P`` is built from the
    blocks passed in; callers freeze it by passing ``stop_gradient``-ed
    blocks while leaving ``r_w`` live, so the AD-gradient w.r.t. θ is the
    Gauss–Newton score.

    Parameters
    ----------
    A_w, offdiag_w, r_w : window blocks and residuals (see above).
    c : int  center index within the window.
    jitter_chol : float
    bandwidth : int  number of lag terms to include.

    Returns
    -------
    loss : scalar
    """
    W, d = A_w.shape[0], A_w.shape[-1]
    Wd = W * d
    L = _cholesky(A_w, offdiag_w, jitter_chol)

    # Column c of P: solve Σ X = E_c  →  X[j] = P_{j,c} = P_{c,j}ᵀ
    E_c = jnp.zeros((Wd, d), dtype=A_w.dtype).at[c * d:(c + 1) * d, :].set(jnp.eye(d, dtype=A_w.dtype))
    Pcol_c = jax.scipy.linalg.cho_solve((L, True), E_c).reshape(W, d, d)

    r_c = r_w[c]
    P_cc = Pcol_c[c]
    loss = 0.5 * jnp.dot(r_c, P_cc @ r_c)
    for lag in range(1, bandwidth + 1):
        P_c_clag = Pcol_c[c + lag].T  # P_{c, c+lag}
        loss = loss + jnp.dot(r_c, P_c_clag @ r_w[c + lag])
    return loss
