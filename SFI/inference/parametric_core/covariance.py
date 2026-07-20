# SFI/inference/parametric_core/covariance.py
"""
Banded residual covariance for the parametric core.

The flow residuals are correlated only over a finite lag set by the SDE
order, so their covariance is **block-banded** with bandwidth ``B``:

    overdamped (order 1) → B = 1  (tridiagonal)
    underdamped (order 2) → B = 2  (pentadiagonal)
    order-3 SDE          → B = 3  (heptadiagonal)   [future]

A single container :class:`CovarianceBlocks` (diagonal blocks + a tuple
of off-diagonal lag blocks) represents any bandwidth, so adding a
higher-order builder is a *localized* addition — the precision engine and
assembler below are bandwidth-generic and need no change.

``D`` and ``Λ`` are full ``(d, d)`` matrices throughout (scalars / per-
axis vectors are accepted and promoted).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__all__ = ["CovarianceBlocks", "build_od_blocks", "build_ud_blocks", "assemble_dense"]


class CovarianceBlocks(NamedTuple):
    """Block-banded covariance.

    Parameters
    ----------
    A : array ``(n_res, d, d)``
        Diagonal blocks ``Cov(r_n, r_n)``.
    offdiag : tuple of arrays
        ``offdiag[lag-1]`` has shape ``(n_res-lag, d, d)`` and holds
        ``Cov(r_n, r_{n+lag})``.
    bandwidth : int
        Number of off-diagonal lags (``len(offdiag)``).
    """

    A: jnp.ndarray
    offdiag: tuple
    bandwidth: int


def _rel_jitter(A_raw, jitter, I_d):
    r"""Diagonal regulariser **scaled to the block magnitude**.

    ``jitter`` is a *relative* coefficient: the added term is
    ``jitter · ⟨diag(A_raw)⟩ · I`` where the scale is the mean diagonal
    entry of the (pre-jitter) blocks.  An absolute floor would mask the
    ``Δt^k``-small process variance at small ``Δt`` (e.g. the underdamped
    ``(4/3)Δt³D ~ 1e-9``), making ``D`` unidentifiable; a relative floor
    is always a tiny fraction of the residual variance, at any ``Δt``.
    """
    d = A_raw.shape[-1]
    scale = jnp.mean(jnp.einsum("nii->n", A_raw)) / d
    scale = jnp.maximum(scale, 1e-30)
    return jitter * scale * I_d[None]


def _as_dd(x, d, dtype):
    """Promote scalar / ``(d,)`` / ``(d, d)`` to a ``(d, d)`` matrix."""
    x = jnp.asarray(x, dtype=dtype)
    if x.ndim == 0:
        return x * jnp.eye(d, dtype=dtype)
    if x.ndim == 1:
        return jnp.diag(x)
    return x


def build_od_blocks(J, D, Lambda, dt, *, jitter=0.0, valid_mask=None, Q=None):
    r"""Overdamped (bandwidth-1) residual covariance blocks.

    For the 2-point residual ``r_k = Y_{k+1} − Y_k − Φ_k(Y_k)`` the
    leading-order covariance combines process noise (diffusion propagated
    through the linearised flow) and measurement noise (shared between
    neighbouring residuals):

        A_k = Δt(J_k D J_kᵀ + D) + J_k Λ J_kᵀ + Λ + jitter·⟨diag A⟩·I
        C_k = Cov(r_k, r_{k+1}) = −Λ J_{k+1}ᵀ

    Parameters
    ----------
    J : array ``(n_res, d, d)``
        Flow Jacobians at the residual base points.
    D, Lambda : ``(d, d)`` (or scalar / ``(d,)``)
        Diffusion and measurement-noise covariance matrices.
    dt : float or ``(n_res,)``
        Sampling interval — per-residual when a vector (each diagonal
        block takes its own ``dt``; the lag-1 coupling ``−ΛJᵀ`` carries
        no ``dt`` and is unchanged).
    jitter : float
        *Relative* diagonal regulariser (a fraction of the mean diagonal
        block magnitude — see :func:`_rel_jitter`), so it never masks the
        ``Δt``-small process variance.
    valid_mask : ``(n_res,)`` bool, optional
        Where ``False``, the diagonal block is replaced by a neutral
        ``(1+jitter)·I`` and adjacent off-diagonal couplings are zeroed,
        decoupling masked residuals.
    Q : ``(n_res, d, d)``, optional
        Precomputed (Lyapunov-exact) process-noise covariance per
        residual; replaces the endpoint-trapezoid ``Δt(J D Jᵀ + D)``.

    Returns
    -------
    CovarianceBlocks with ``bandwidth=1``.
    """
    J = jnp.asarray(J)
    d = J.shape[-1]
    dtype = J.dtype
    I_d = jnp.eye(d, dtype=dtype)
    D = jnp.asarray(D, dtype=dtype)
    Lambda = _as_dd(Lambda, d, dtype)

    if Q is not None:
        proc = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))
    else:
        dt_f = dt[:, None, None] if jnp.ndim(dt) == 1 else dt
        if D.ndim == 3:  # per-step (state-dependent) D, shape (n_res, d, d)
            proc = dt_f * (jnp.einsum("kij,kjl,kml->kim", J, D, J) + D)
        else:
            D = _as_dd(D, d, dtype)
            proc = dt_f * (jnp.einsum("kij,jl,kml->kim", J, D, J) + D[None])
    JSJ = jnp.einsum("kij,jl,kml->kim", J, Lambda, J)
    A_raw = proc + JSJ + Lambda[None]
    A = A_raw + _rel_jitter(A_raw, jitter, I_d)

    # lag-1: C_k = −Λ J_{k+1}ᵀ
    C = -jnp.einsum("ij,klj->kil", Lambda, J)[1:]  # (n_res-1, d, d)

    if valid_mask is not None:
        m = valid_mask[:, None, None]
        A = jnp.where(m, A, (1.0 + jitter) * I_d[None])
        pair = (valid_mask[:-1] & valid_mask[1:])[:, None, None]
        C = jnp.where(pair, C, jnp.zeros_like(C))

    return CovarianceBlocks(A=A, offdiag=(C,), bandwidth=1)


def build_ud_blocks(alpha_plus, alpha_zero, alpha_minus, D, Lambda, dt,
                    *, jitter=0.0, valid_mask=None):
    r"""Underdamped (bandwidth-2) residual covariance blocks.

    The 3-point shooting residual ``r_n = α₋ ε_{n-1} + α₀ ε_n + α₊ ε_{n+1}
    + ξ`` (measurement + process) has a pentadiagonal covariance:

        A_n = (4/3)Δt³ D_sym + Σ_k α_k Λ α_kᵀ + jitter·⟨diag A⟩·I
        C_n = (1/3)Δt³ D_sym + α₊ Λ α₀ᵀ|_{n,n+1} + α₀ Λ α₋ᵀ|_{n,n+1}
        E_n = α₊ Λ α₋ᵀ|_{n,n+2}

    The ``jitter`` regulariser is *relative* to the block magnitude
    (:func:`_rel_jitter`): an absolute floor would dominate the ``Δt³``-small
    process variance at small ``Δt`` and make ``D`` unidentifiable.

    where ``D_sym = (D + Dᵀ)/2`` and the leading-order process noise scales
    as ``Δt³`` (the velocity is integrated once more than the overdamped
    increment).

    Parameters
    ----------
    alpha_plus, alpha_zero, alpha_minus : ``(n_res, d, d)``
        Measurement-noise propagators (``α₊ = I`` by construction).
    D, Lambda : ``(d, d)`` (or scalar / ``(d,)``).
    dt : float or ``(n_res,)``
        A vector is accepted for shape compatibility only and is **valid
        for uniform values only**: the leading-order ``(4/3, 1/3)``
        split of the ``Δt³`` process terms assumes equal adjacent
        intervals.  Genuinely non-uniform sampling must use
        :func:`build_ud_blocks_exact`, whose per-interval Lyapunov
        tensors carry the correct interval mix (guarded upstream).
    jitter : float
    valid_mask : ``(n_res,)`` bool, optional.

    Returns
    -------
    CovarianceBlocks with ``bandwidth=2``.
    """
    ap = jnp.asarray(alpha_plus)
    a0 = jnp.asarray(alpha_zero)
    am = jnp.asarray(alpha_minus)
    d = ap.shape[-1]
    dtype = ap.dtype
    I_d = jnp.eye(d, dtype=dtype)
    D = jnp.asarray(D, dtype=dtype)
    Lambda = _as_dd(Lambda, d, dtype)

    dt3 = dt**3 if jnp.ndim(dt) == 0 else (dt**3)[:, None, None]
    if D.ndim == 3:  # per-step (state/velocity-dependent) D, shape (n_res, d, d)
        D_sym = 0.5 * (D + jnp.swapaxes(D, -1, -2))
        V_xi = (4.0 / 3.0) * dt3 * D_sym            # (n_res, d, d)
        C_proc = (1.0 / 3.0) * dt3 * D_sym
        V_xi_diag = V_xi
        C_proc_lag1 = 0.5 * (C_proc[:-1] + C_proc[1:])
    else:
        D = _as_dd(D, d, dtype)
        D_sym = 0.5 * (D + D.T)
        if jnp.ndim(dt) == 0:
            V_xi_diag = ((4.0 / 3.0) * dt3 * D_sym)[None]
            C_proc_lag1 = ((1.0 / 3.0) * dt3 * D_sym)[None]
        else:
            C_proc = (1.0 / 3.0) * dt3 * D_sym[None]
            V_xi_diag = (4.0 / 3.0) * dt3 * D_sym[None]
            C_proc_lag1 = 0.5 * (C_proc[:-1] + C_proc[1:])

    se_pp = jnp.einsum("nij,jk,nlk->nil", ap, Lambda, ap)
    se_00 = jnp.einsum("nij,jk,nlk->nil", a0, Lambda, a0)
    se_mm = jnp.einsum("nij,jk,nlk->nil", am, Lambda, am)
    A_raw = V_xi_diag + se_pp + se_00 + se_mm
    A = A_raw + _rel_jitter(A_raw, jitter, I_d)

    cross1a = jnp.einsum("nij,jk,nlk->nil", ap[:-1], Lambda, a0[1:])
    cross1b = jnp.einsum("nij,jk,nlk->nil", a0[:-1], Lambda, am[1:])
    C = C_proc_lag1 + cross1a + cross1b
    E = jnp.einsum("nij,jk,nlk->nil", ap[:-2], Lambda, am[2:])

    if valid_mask is not None:
        m = valid_mask[:, None, None]
        A = jnp.where(m, A, (1.0 + jitter) * I_d[None])
        pair = (valid_mask[:-1] & valid_mask[1:])[:, None, None]
        C = jnp.where(pair, C, jnp.zeros_like(C))
        triple = (valid_mask[:-2] & valid_mask[1:-1] & valid_mask[2:])[:, None, None]
        E = jnp.where(triple, E, jnp.zeros_like(E))

    return CovarianceBlocks(A=A, offdiag=(C, E), bandwidth=2)


def build_ud_blocks_exact(alpha_plus, alpha_zero, alpha_minus, qing, Lambda,
                          *, jitter=0.0, valid_mask=None):
    r"""Underdamped blocks with **Lyapunov-exact** linearized process noise.

    Replaces the leading-order ``(4/3, 1/3)·Δt³D`` process terms of
    :func:`build_ud_blocks` by the exact linearized covariances derived
    from the per-interval lifted noise ``ζ = (ζˣ, ζᵛ)``
    (``Q`` blocks integrated along the flow) and the shooting chain
    ``r_n = ζˣ_n + J^{xv}_{\rm out}(n)\,[ζᵛ_{n-1} − N(n)\,ζˣ_{n-1}]``:

    .. math::

        A_n^{\rm proc} &= Q^{xx}_{\rm out}(n)
          + J^{xv}_{\rm out}(n)\,S_v(n)\,J^{xv}_{\rm out}(n)^{\!\top},\\
        S_v(n) &= Q^{vv}_{\rm in} - N Q^{xv\top}_{\rm in} - Q^{xv}_{\rm in} N^{\!\top}
                  + N Q^{xx}_{\rm in} N^{\!\top}\ \big|_n ,\\
        C_n^{\rm proc} &= \bigl[Q^{xv}_{\rm out}(n) - Q^{xx}_{\rm out}(n)\,N(n{+}1)^{\!\top}\bigr]
                          J^{xv}_{\rm out}(n{+}1)^{\!\top},

    with the measurement-noise ``αΛαᵀ`` terms exactly as in
    :func:`build_ud_blocks`. ``qing`` is the dict returned by
    ``ud_multi_step_residuals_with_psi(..., D_lyap=D)``. Note
    ``Q^{xv} \equiv \Cov(ζˣ, ζᵛ)`` and the in-blocks are integrated along
    the shooting trajectory of residual ``n`` (interval ``n−1``).
    """
    ap = jnp.asarray(alpha_plus)
    a0 = jnp.asarray(alpha_zero)
    am = jnp.asarray(alpha_minus)
    d = ap.shape[-1]
    dtype = ap.dtype
    I_d = jnp.eye(d, dtype=dtype)
    Lambda = _as_dd(Lambda, d, dtype)

    N = qing["N"]
    Jxv = qing["Jxv_out"]
    Qxv_in_T = jnp.swapaxes(qing["Qxv_in"], -1, -2)
    S_v = (qing["Qvv_in"]
           - jnp.einsum("nij,njk->nik", N, qing["Qxv_in"])
           - jnp.einsum("nij,nkj->nik", Qxv_in_T, N)
           + jnp.einsum("nij,njk,nlk->nil", N, qing["Qxx_in"], N))
    V_proc = qing["Qxx_out"] + jnp.einsum("nij,njk,nlk->nil", Jxv, S_v, Jxv)
    V_proc = 0.5 * (V_proc + jnp.swapaxes(V_proc, -1, -2))
    C_proc = jnp.einsum(
        "nij,nkj->nik",
        qing["Qxv_out"][:-1]
        - jnp.einsum("nij,nkj->nik", qing["Qxx_out"][:-1], N[1:]),
        Jxv[1:])

    se_pp = jnp.einsum("nij,jk,nlk->nil", ap, Lambda, ap)
    se_00 = jnp.einsum("nij,jk,nlk->nil", a0, Lambda, a0)
    se_mm = jnp.einsum("nij,jk,nlk->nil", am, Lambda, am)
    A_raw = V_proc + se_pp + se_00 + se_mm
    A = A_raw + _rel_jitter(A_raw, jitter, I_d)

    cross1a = jnp.einsum("nij,jk,nlk->nil", ap[:-1], Lambda, a0[1:])
    cross1b = jnp.einsum("nij,jk,nlk->nil", a0[:-1], Lambda, am[1:])
    C = C_proc + cross1a + cross1b
    E = jnp.einsum("nij,jk,nlk->nil", ap[:-2], Lambda, am[2:])

    if valid_mask is not None:
        m = valid_mask[:, None, None]
        A = jnp.where(m, A, (1.0 + jitter) * I_d[None])
        pair = (valid_mask[:-1] & valid_mask[1:])[:, None, None]
        C = jnp.where(pair, C, jnp.zeros_like(C))
        triple = (valid_mask[:-2] & valid_mask[1:-1] & valid_mask[2:])[:, None, None]
        E = jnp.where(triple, E, jnp.zeros_like(E))

    return CovarianceBlocks(A=A, offdiag=(C, E), bandwidth=2)


def assemble_dense(blocks: CovarianceBlocks):
    """Assemble a :class:`CovarianceBlocks` into a dense symmetric matrix.

    Returns
    -------
    Sigma : array ``(n_res·d, n_res·d)``
    """
    A = blocks.A
    n_res, d, _ = A.shape
    Nd = n_res * d
    S = jnp.zeros((Nd, Nd), dtype=A.dtype)

    for i in range(n_res):
        S = S.at[i * d:(i + 1) * d, i * d:(i + 1) * d].set(A[i])

    for lag_idx, C_lag in enumerate(blocks.offdiag):
        lag = lag_idx + 1
        for i in range(n_res - lag):
            r0, c0 = i * d, (i + lag) * d
            S = S.at[r0:r0 + d, c0:c0 + d].set(C_lag[i])
            S = S.at[c0:c0 + d, r0:r0 + d].set(C_lag[i].T)

    return S
