# SFI/inference/parametric_core/jacobians.py
"""
RK4-composed per-particle Jacobians for interacting (multi-particle) systems.

Moved verbatim from the legacy parametric package (``_od_precompute`` /
``_underdamped``) — the frozen-background approximation: the state update
uses the full multi-particle force, while the tangent is propagated per
particle from the same-particle derivatives, keeping the flow Jacobian
block-diagonal in the particle axis and the cost O(N) per window.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

__all__ = ["rk4_composed_jacobian", "rk4_composed_jacobian_theta",
           "rk4_composed_jacobian_phase", "rk4_composed_jacobian_phase_theta"]


def rk4_composed_jacobian_theta(F_psf, dFdx_expr, dFdth_fn, n_params, X_start, theta_struct,
                                mask, extras, dt, n_substeps, integrator="rk4"):
    r"""Per-particle flow Jacobian, displacement, **and θ-sensitivity** ``B``.

    Same frozen-background RK4 composition as :func:`rk4_composed_jacobian`,
    additionally carrying the per-particle parameter sensitivity

    .. math:: B_p = \partial \Phi_p / \partial \theta \in (N, d, n_\theta)

    through the classic variation-of-constants recursion (the legacy
    ``_compute_jb_single_particle`` trick, RK4-composed): per stage
    ``dk_s/dθ = Θ_s + c_s h A_s dk_{s-1}/dθ`` with ``Θ = ∂F/∂θ`` at the
    *fixed* stage frame (exact — the force's direct θ-dependence is
    per-particle), and per substep ``B ← J_step B + B_step``.  The dropped
    cross-particle paths (θ wiggling a *neighbour's* substep motion, which
    feeds back within one Δt) are the same O(h²·coupling) terms the
    frozen-background ``J`` already drops.  Peak memory is
    O(N·d·n_θ) instead of forward-mode AD's n_θ tangents through the full
    N-body window flow.

    Parameters
    ----------
    dFdth_fn : callable ``(N, d) frame → (N, d, n_params)``
        Direct parameter derivative of the force at a fixed frame
        (e.g. ``jax.jacfwd`` over θ of the frame-level force).
    n_params : int
        Number of flat parameters (sets the ``B`` carry width).
    Other parameters as in :func:`rk4_composed_jacobian`.

    Returns
    -------
    J : ``(N, d, d)``,  Phi : ``(N, d)``,  B : ``(N, d, n_params)``
    """
    if integrator != "rk4":
        raise ValueError(
            f"rk4_composed_jacobian_theta only supports integrator='rk4', got {integrator!r}"
        )
    h = dt / n_substeps
    d = X_start.shape[-1]
    I_d = jnp.eye(d)

    def _force(X):
        return F_psf(X, params=theta_struct, mask=mask, extras=extras)

    def _dFdx(X):
        raw = dFdx_expr(X, params=theta_struct, mask=mask, extras=extras)
        return jnp.swapaxes(raw, -1, -2)

    def _mm(A, B):
        return jnp.einsum("...ij,...jk->...ik", A, B)

    def _rk4_step(carry, _):
        X, J_acc, B_acc = carry

        k1 = _force(X)
        A1 = _dFdx(X)
        T1 = dFdth_fn(X)
        X2 = X + (h / 2) * k1
        k2 = _force(X2)
        A2 = _dFdx(X2)
        T2 = dFdth_fn(X2)
        X3 = X + (h / 2) * k2
        k3 = _force(X3)
        A3 = _dFdx(X3)
        T3 = dFdth_fn(X3)
        X4 = X + h * k3
        k4 = _force(X4)
        A4 = _dFdx(X4)
        T4 = dFdth_fn(X4)

        X_new = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # x-tangent chain (as in rk4_composed_jacobian)
        S1 = I_d + (h / 2) * A1
        A2S1 = _mm(A2, S1)
        S2 = I_d + (h / 2) * A2S1
        A3S2 = _mm(A3, S2)
        S3 = I_d + h * A3S2
        J_step = I_d + (h / 6) * (A1 + 2 * A2S1 + 2 * A3S2 + _mm(A4, S3))

        # θ-tangent chain: dk_s = Θ_s + c_s·h·A_s·dk_{s-1}
        dk1 = T1
        dk2 = T2 + (h / 2) * _mm(A2, dk1)
        dk3 = T3 + (h / 2) * _mm(A3, dk2)
        dk4 = T4 + h * _mm(A4, dk3)
        B_step = (h / 6) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)

        J_acc_new = _mm(J_step, J_acc)
        B_acc_new = _mm(J_step, B_acc) + B_step
        return (X_new, J_acc_new, B_acc_new), None

    J0 = jnp.broadcast_to(I_d, X_start.shape[:-1] + (d, d))
    B0 = jnp.zeros(X_start.shape[:-1] + (d, n_params), dtype=X_start.dtype)
    (X_final, J_composed, B_composed), _ = lax.scan(
        _rk4_step, (X_start, J0, B0), None, length=n_substeps)
    return J_composed, X_final - X_start, B_composed


def rk4_composed_jacobian(F_psf, dFdx_expr, X_start, theta_struct, mask, extras, dt, n_substeps, integrator="rk4"):
    r"""Exact RK4-composed per-particle diagonal Jacobian and displacement.

    Computes the per-particle flow Jacobian :math:`\partial x_{end}^p /
    \partial x_{start}^p` and displacement :math:`\Phi^p = x_{end}^p -
    x_{start}^p` by composing exact RK4 Jacobians across substeps using
    ``dFdx_expr = F_psf.d_x(same_particle=True)``.

    The frozen-background approximation is used: cross-particle
    derivatives are ignored (other particles held fixed at their
    observed positions within each substep).

    Parameters
    ----------
    F_psf : PSF object (``particles_input=True``)
    dFdx_expr : DerivativeNode
        From ``F_psf.d_x(same_particle=True)``.  Evaluated as
        ``dFdx_expr(X, params=..., mask=..., extras=...)``
        returning ``(N, d, d)`` in the SFI convention (derivative-dim
        first: ``[p, j, i] = dF_p^i / dx_p^j``).
    X_start : ``(N, d)``
    theta_struct : structured parameters (from ``F_psf.unflatten_params``)
    mask : ``(N,)`` boolean or None
    extras : dict or None
    dt : float
    n_substeps : int
    integrator : {"rk4"}, optional
        Accepted for API symmetry with the single-particle
        :func:`._underdamped._phase_space_jacobian`. Only ``"rk4"`` is
        supported here — the substep Jacobian is the hand-rolled RK4
        chain rule below. Passing anything else raises ``ValueError`` so
        callers don't silently get RK4 when they asked for Euler.

    Returns
    -------
    J : ``(N, d, d)``
        Flow Jacobian in standard convention: ``J[p, i, j] =
        dx_{end,p}^i / dx_{start,p}^j``.
    Phi : ``(N, d)``
        Displacement ``x_{end} - x_{start}``.
    """
    if integrator != "rk4":
        raise ValueError(
            f"rk4_composed_jacobian only supports integrator='rk4', got {integrator!r}"
        )
    h = dt / n_substeps
    d = X_start.shape[-1]
    I_d = jnp.eye(d)

    def _force(X):
        return F_psf(X, params=theta_struct, mask=mask, extras=extras)

    def _dFdx(X):
        # SFI convention → standard: swap last two axes
        raw = dFdx_expr(X, params=theta_struct, mask=mask, extras=extras)
        return jnp.swapaxes(raw, -1, -2)

    def _mm(A, B):
        """Batched (N, d, d) matmul."""
        return jnp.einsum("...ij,...jk->...ik", A, B)

    def _rk4_step(carry, _):
        X, J_acc = carry

        k1 = _force(X)
        A1 = _dFdx(X)

        k2 = _force(X + (h / 2) * k1)
        A2 = _dFdx(X + (h / 2) * k1)

        k3 = _force(X + (h / 2) * k2)
        A3 = _dFdx(X + (h / 2) * k2)

        k4 = _force(X + h * k3)
        A4 = _dFdx(X + h * k3)

        # State update
        X_new = X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # RK4 Jacobian chain rule (all per-particle, (N, d, d))
        # S_i = sensitivity of stage i input w.r.t. X_start
        S1 = I_d + (h / 2) * A1
        A2S1 = _mm(A2, S1)
        S2 = I_d + (h / 2) * A2S1
        A3S2 = _mm(A3, S2)
        S3 = I_d + h * A3S2

        J_step = I_d + (h / 6) * (A1 + 2 * A2S1 + 2 * A3S2 + _mm(A4, S3))
        J_acc_new = _mm(J_step, J_acc)

        return (X_new, J_acc_new), None

    J0 = jnp.broadcast_to(I_d, X_start.shape[:-1] + (d, d))
    (X_final, J_composed), _ = lax.scan(
        _rk4_step,
        (X_start, J0),
        None,
        length=n_substeps,
    )

    Phi = X_final - X_start
    return J_composed, Phi


def rk4_composed_jacobian_phase(
    F_psf,
    dFdx_expr,
    dFdv_expr,
    X_start,
    V_start,
    theta_struct,
    mask,
    extras,
    dt,
    n_substeps,
    integrator="rk4",
):
    r"""RK4-composed per-particle phase-space Jacobian and flow.

    Multi-particle analogue of :func:`_phase_space_jacobian` for the
    flocking class: the state update uses the full multi-particle
    force :math:`F(X, V)` (so inter-particle couplings drive the
    dynamics), while the tangent is propagated **per particle** using
    the per-edge derivatives
    :math:`\partial F_p/\partial x_p` and :math:`\partial F_p/\partial v_p`
    obtained from ``F_psf.d_x(same_particle=True)`` and
    ``F_psf.d_v(same_particle=True)``. This is the frozen-background
    approximation lifted to phase space: cross-particle Jacobian blocks
    are dropped, keeping the per-particle flow Jacobian block-diagonal
    in the particle axis and the cost O(N) per window.

    Parameters
    ----------
    F_psf : PSF (``particles_input=True``, ``needs_v=True``)
    dFdx_expr : ``F_psf.d_x(same_particle=True)``
    dFdv_expr : ``F_psf.d_v(same_particle=True)``
    X_start, V_start : ``(N, d)``
    theta_struct : structured parameters (from ``F_psf.unflatten_params``)
    mask : ``(N,)`` boolean or None
    extras : dict or None
    dt : float
    n_substeps : int

    integrator : {"rk4"}, optional
        Accepted for API symmetry with the single-particle
        :func:`_phase_space_jacobian`. Only ``"rk4"`` is supported here —
        the substep Jacobian is the hand-rolled RK4 chain rule below.
        Passing anything else raises ``ValueError`` so callers don't
        silently get RK4 when they asked for Euler.

    Returns
    -------
    Phi_x, Phi_v : ``(N, d)``
        End-state position and velocity after time ``dt`` (matches
        single-particle :func:`_phase_space_jacobian` convention).
    Jxx, Jxv, Jvx, Jvv : ``(N, d, d)``
        Per-particle phase-space Jacobian blocks in standard
        convention: ``Jxx[p, i, j] = ∂x_{end,p}^i/∂x_{start,p}^j``, etc.
    """
    if integrator != "rk4":
        raise ValueError(
            f"rk4_composed_jacobian_phase only supports integrator='rk4', got {integrator!r}"
        )
    import jax.lax as lax

    h = dt / n_substeps
    d = X_start.shape[-1]
    N = X_start.shape[-2]
    I_d = jnp.eye(d)
    I_2d = jnp.eye(2 * d)

    def _force(X, V):
        return F_psf(
            X,
            v=V,
            params=theta_struct,
            mask=mask,
            extras=extras,
        )

    def _dFdx(X, V):
        # SFI convention ``[p, j, i] = dF_p^i / dx_p^j`` →
        # standard ``[p, i, j] = dF_p^i / dx_p^j`` via swap.
        raw = dFdx_expr(
            X,
            v=V,
            params=theta_struct,
            mask=mask,
            extras=extras,
        )
        return jnp.swapaxes(raw, -1, -2)

    def _dFdv(X, V):
        raw = dFdv_expr(
            X,
            v=V,
            params=theta_struct,
            mask=mask,
            extras=extras,
        )
        return jnp.swapaxes(raw, -1, -2)

    def _mm(A, B):
        """Batched ``(N, a, b) @ (N, b, c) → (N, a, c)`` matmul."""
        return jnp.einsum("...ij,...jk->...ik", A, B)

    def _M_stack(Ax, Av):
        """Build per-particle stage Jacobian

        M = [[0, I_d], [A_x, A_v]]   shape (N, 2d, 2d)
        """
        zero = jnp.zeros_like(Ax)
        top = jnp.concatenate([zero, jnp.broadcast_to(I_d, Ax.shape)], axis=-1)
        bot = jnp.concatenate([Ax, Av], axis=-1)
        return jnp.concatenate([top, bot], axis=-2)

    def _rk4_step(carry, _):
        X, V, J_acc = carry

        # Stage 1
        A1_x = _dFdx(X, V)
        A1_v = _dFdv(X, V)
        k1_x = V
        k1_v = _force(X, V)
        M1 = _M_stack(A1_x, A1_v)  # (N, 2d, 2d)
        T1 = M1  # = M1 @ I = M1

        # Stage 2
        X2 = X + (h / 2) * k1_x
        V2 = V + (h / 2) * k1_v
        A2_x = _dFdx(X2, V2)
        A2_v = _dFdv(X2, V2)
        k2_x = V2
        k2_v = _force(X2, V2)
        M2 = _M_stack(A2_x, A2_v)
        S2 = I_2d + (h / 2) * T1
        T2 = _mm(M2, S2)

        # Stage 3
        X3 = X + (h / 2) * k2_x
        V3 = V + (h / 2) * k2_v
        A3_x = _dFdx(X3, V3)
        A3_v = _dFdv(X3, V3)
        k3_x = V3
        k3_v = _force(X3, V3)
        M3 = _M_stack(A3_x, A3_v)
        S3 = I_2d + (h / 2) * T2
        T3 = _mm(M3, S3)

        # Stage 4
        X4 = X + h * k3_x
        V4 = V + h * k3_v
        A4_x = _dFdx(X4, V4)
        A4_v = _dFdv(X4, V4)
        k4_x = V4
        k4_v = _force(X4, V4)
        M4 = _M_stack(A4_x, A4_v)
        S4 = I_2d + h * T3
        T4 = _mm(M4, S4)

        # State update (full multi-particle force)
        X_new = X + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        V_new = V + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        # Per-particle 2d×2d substep Jacobian + chain compose
        J_step = I_2d + (h / 6) * (T1 + 2 * T2 + 2 * T3 + T4)
        J_acc_new = _mm(J_step, J_acc)

        return (X_new, V_new, J_acc_new), None

    J0 = jnp.broadcast_to(I_2d, (N, 2 * d, 2 * d))
    (X_final, V_final, J_composed), _ = lax.scan(
        _rk4_step,
        (X_start, V_start, J0),
        None,
        length=n_substeps,
    )

    Phi_x = X_final
    Phi_v = V_final
    Jxx = J_composed[:, :d, :d]
    Jxv = J_composed[:, :d, d:]
    Jvx = J_composed[:, d:, :d]
    Jvv = J_composed[:, d:, d:]
    return Phi_x, Phi_v, Jxx, Jxv, Jvx, Jvv


def rk4_composed_jacobian_phase_theta(
    F_psf,
    dFdx_expr,
    dFdv_expr,
    dFdth_fn,
    n_params,
    X_start,
    V_start,
    theta_struct,
    mask,
    extras,
    dt,
    n_substeps,
    integrator="rk4",
):
    r"""Phase-space per-particle Jacobian blocks, flow, **and θ-sensitivity**.

    Phase-space analogue of :func:`rk4_composed_jacobian_theta`: alongside
    the frozen-background blocks of :func:`rk4_composed_jacobian_phase`,
    carries the per-particle lifted parameter sensitivity

    .. math:: B_p = \partial (x', v')_p / \partial\theta \in (N, 2d, n_\theta)

    through the RK4 chain with lifted stage tangents
    ``Θ̃ = [0; ∂F/∂θ]`` (the position field has no direct θ-dependence) and
    ``dk_s/dθ = Θ̃_s + c_s h M_s dk_{s-1}/dθ``; per substep
    ``B ← J_step B + B_step``.  Memory O(N·2d·n_θ).

    Parameters
    ----------
    dFdth_fn : callable ``((N, d), (N, d)) → (N, d, n_params)``
        Direct parameter derivative of the force at a fixed phase frame.
    n_params : int
    Other parameters as in :func:`rk4_composed_jacobian_phase`.

    Returns
    -------
    Phi_x, Phi_v : ``(N, d)``  end-state position / velocity.
    Jxx, Jxv, Jvx, Jvv : ``(N, d, d)``  frozen-background blocks.
    Bx, Bv : ``(N, d, n_params)``  θ-sensitivity of the end state.
    """
    if integrator != "rk4":
        raise ValueError(
            f"rk4_composed_jacobian_phase_theta only supports integrator='rk4', got {integrator!r}"
        )
    import jax.lax as lax

    h = dt / n_substeps
    d = X_start.shape[-1]
    N = X_start.shape[-2]
    I_d = jnp.eye(d)
    I_2d = jnp.eye(2 * d)

    def _force(X, V):
        return F_psf(X, v=V, params=theta_struct, mask=mask, extras=extras)

    def _dFdx(X, V):
        raw = dFdx_expr(X, v=V, params=theta_struct, mask=mask, extras=extras)
        return jnp.swapaxes(raw, -1, -2)

    def _dFdv(X, V):
        raw = dFdv_expr(X, v=V, params=theta_struct, mask=mask, extras=extras)
        return jnp.swapaxes(raw, -1, -2)

    def _mm(A, B):
        return jnp.einsum("...ij,...jk->...ik", A, B)

    def _M_stack(Ax, Av):
        zero = jnp.zeros_like(Ax)
        top = jnp.concatenate([zero, jnp.broadcast_to(I_d, Ax.shape)], axis=-1)
        bot = jnp.concatenate([Ax, Av], axis=-1)
        return jnp.concatenate([top, bot], axis=-2)

    def _lift(T):
        # (N, d, n) → (N, 2d, n): zero position block on top
        return jnp.concatenate([jnp.zeros_like(T), T], axis=-2)

    def _rk4_step(carry, _):
        X, V, J_acc, B_acc = carry

        A1x, A1v = _dFdx(X, V), _dFdv(X, V)
        k1_x, k1_v = V, _force(X, V)
        M1 = _M_stack(A1x, A1v)
        T1 = _lift(dFdth_fn(X, V))

        X2, V2 = X + (h / 2) * k1_x, V + (h / 2) * k1_v
        A2x, A2v = _dFdx(X2, V2), _dFdv(X2, V2)
        k2_x, k2_v = V2, _force(X2, V2)
        M2 = _M_stack(A2x, A2v)
        T2 = _lift(dFdth_fn(X2, V2))

        X3, V3 = X + (h / 2) * k2_x, V + (h / 2) * k2_v
        A3x, A3v = _dFdx(X3, V3), _dFdv(X3, V3)
        k3_x, k3_v = V3, _force(X3, V3)
        M3 = _M_stack(A3x, A3v)
        T3 = _lift(dFdth_fn(X3, V3))

        X4, V4 = X + h * k3_x, V + h * k3_v
        A4x, A4v = _dFdx(X4, V4), _dFdv(X4, V4)
        k4_x, k4_v = V4, _force(X4, V4)
        M4 = _M_stack(A4x, A4v)
        T4 = _lift(dFdth_fn(X4, V4))

        # x-tangent chain (identical to rk4_composed_jacobian_phase)
        S2 = I_2d + (h / 2) * M1
        T2_ = _mm(M2, S2)
        S3 = I_2d + (h / 2) * T2_
        T3_ = _mm(M3, S3)
        S4 = I_2d + h * T3_
        T4_ = _mm(M4, S4)
        J_step = I_2d + (h / 6) * (M1 + 2 * T2_ + 2 * T3_ + T4_)

        # θ-tangent chain on the lifted system
        dk1 = T1
        dk2 = T2 + (h / 2) * _mm(M2, dk1)
        dk3 = T3 + (h / 2) * _mm(M3, dk2)
        dk4 = T4 + h * _mm(M4, dk3)
        B_step = (h / 6) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)

        X_new = X + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        V_new = V + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        J_acc_new = _mm(J_step, J_acc)
        B_acc_new = _mm(J_step, B_acc) + B_step
        return (X_new, V_new, J_acc_new, B_acc_new), None

    J0 = jnp.broadcast_to(I_2d, (N, 2 * d, 2 * d))
    B0 = jnp.zeros((N, 2 * d, n_params), dtype=X_start.dtype)
    (X_final, V_final, J_composed, B_composed), _ = lax.scan(
        _rk4_step, (X_start, V_start, J0, B0), None, length=n_substeps)
    return (
        X_final, V_final,
        J_composed[:, :d, :d], J_composed[:, :d, d:],
        J_composed[:, d:, :d], J_composed[:, d:, d:],
        B_composed[:, :d, :], B_composed[:, d:, :],
    )
