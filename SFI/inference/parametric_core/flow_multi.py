# SFI/inference/parametric_core/flow_multi.py
"""
Multiparticle residuals — one function for both the non-interacting and
interacting cases, dispatched on the force model's particle contract.

The covariance/precision machinery is **per particle**, so all this layer
does is produce per-particle residuals ``r`` and flow Jacobians ``J`` from
a window of multiparticle positions:

* **non-interacting** (``particles_input=False``): the same force is
  applied to each particle independently — the frame is a pure batch axis.
* **interacting** (``particles_input=True``): the displacement uses the
  *full* multiparticle force (couplings matter), while the per-particle
  flow Jacobian uses the frozen-background same-particle derivative
  ``F_psf.d_x(same_particle=True)`` — via
  :func:`jacobians.rk4_composed_jacobian` (O(N) per window).

The θ-sensitivity ``ψ = ∂r/∂θ`` and the flow Jacobian come from the SAME
per-stage recursion (:func:`_jpb`) for both contracts: for interacting
models this is the frozen-background approximation; for non-interacting
models there are no cross-particle terms and the recursion is the exact
chain rule — replacing forward-mode AD with n_params tangents through
the flow (the historical non-interacting path) at a cost of one force,
one ∂F/∂x, and one ∂F/∂θ evaluation per stage.

Single particle is just ``N=1``; the integrate engine reduces + masks
over the particle axis (``reduce_over_particles``), so there is no
separate multiparticle code path in the solver — exactly like
``infer_force_linear``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .flow import flow_displacement, od_window_residuals
from .flow_ud import _phase_field, ud_window_residuals

__all__ = ["multi_step_residuals", "multi_step_residuals_with_psi",
           "ud_multi_step_residuals", "ud_multi_step_residuals_with_psi",
           "multi_od_instrument", "multi_ud_instrument"]


def _d_expr(F_psf, var="x"):
    """Per-particle derivative expression, dispatched on the particle contract.

    ``same_particle=True`` (the frozen-background diagonal) for interacting
    models; the plain derivative for non-interacting ones, where the particle
    axis is a batch axis and the same-particle block IS the full derivative.
    """
    same = bool(getattr(F_psf, "particles_input", False))
    kw = {"same_particle": True} if same else {}
    return F_psf.d_x(**kw) if var == "x" else F_psf.d_v(**kw)


def _dFdth_builder(F_psf, theta_flat, extras, *, needs_v=False):
    r"""``∂F/∂θ`` at a fixed frame → ``(N, d, n_params)``.

    Fast path for a linear model (``Basis.to_psf()``: a parameter-free
    feature stack under a single ``CoeffNode``): ``∂F/∂θ_a = b_a`` is the
    feature stack itself — one primal basis evaluation, no AD.  General
    PSFs fall back to ``jacfwd`` over θ (the primal features of θ-independent
    subtrees are still evaluated once; only genuine θ-dependence carries
    tangents).
    """
    from SFI.statefunc.basis import Basis
    from SFI.statefunc.nodes.ops.linear import CoeffNode

    root = getattr(F_psf, "root", None)
    if (isinstance(root, CoeffNode) and root.children
            and not root.children[0].param_suite):
        feats = Basis(root.children[0])
        if needs_v:
            return lambda X, V: feats(X, v=V, extras=extras)
        return lambda X: feats(X, extras=extras)

    if needs_v:
        def dFdth_v(X, V):
            return jax.jacfwd(
                lambda th: F_psf(X, v=V, params=F_psf.unflatten_params(th), extras=extras)
            )(theta_flat)                               # (N, d, n_params)
        return dFdth_v

    def dFdth(X):
        return jax.jacfwd(
            lambda th: F_psf(X, params=F_psf.unflatten_params(th), extras=extras)
        )(theta_flat)                                   # (N, d, n_params)
    return dFdth


def _jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator, D_lyap=None,
         return_stages=False):
    r"""Frame → ``(J, Φ, B[, Q])`` closure via the per-stage θ-recursion.

    ``B = ∂Φ/∂θ`` per particle via :func:`jacobians.rk4_composed_jacobian_theta`
    — O(N·d·n_params) carry instead of forward-mode tangents through the
    window flow.  For interacting models the dropped cross-particle paths
    (θ wiggling a *neighbour's* substep motion, which feeds back within one
    Δt) are the same O(h²·coupling) terms the frozen-background ``J``
    already drops; for non-interacting models the recursion is exact.
    ``D_lyap`` additionally requests the Lyapunov-exact process covariance
    ``Q`` accrued over the interval (see :func:`jacobians`).

    The returned closure accepts an optional per-call ``dt_step`` (a traced
    scalar) overriding the constructor ``dt`` — the hook for per-interval
    (non-uniform) time steps.
    """
    from .jacobians import rk4_composed_jacobian_theta

    struct = F_psf.unflatten_params(theta_flat)
    dFdx = _d_expr(F_psf, "x")
    n_params = int(theta_flat.shape[-1])
    dFdth = _dFdth_builder(F_psf, theta_flat, extras)

    def jpb(X_start, dt_step=None):
        return rk4_composed_jacobian_theta(
            F_psf, dFdx, dFdth, n_params, X_start, struct, None, extras,
            dt if dt_step is None else dt_step, n_substeps, integrator,
            D_lyap=D_lyap, return_stages=return_stages)

    return jpb


def multi_step_residuals_with_psi(F_psf, theta_flat, X_w, extras, dt, n_substeps,
                                  integrator="rk4", D_lyap=None):
    r"""Residuals + flow Jacobians + regressor ``ψ_right`` (both contracts).

    Same residuals/Jacobians as :func:`multi_step_residuals`, additionally
    returning the per-particle regressor ``ψ_right = ∂r/∂θ = −B`` from the
    per-stage θ-recursion — the memory-scalable replacement for
    ``jax.jacfwd`` through the flow (which holds n_params tangents alive
    across the whole window graph).  Exact for non-interacting models;
    frozen-background for interacting ones (cross-particle O(h²) feedback
    dropped, consistently with ``J``).

    ``dt`` may be a scalar (uniform sampling) or a ``(W-1,)`` vector of
    per-interval steps, ``dt[k]`` spanning ``(k → k+1)``.

    Returns
    -------
    r : ``(W-1, N, d)``,  J : ``(W-1, N, d, d)``,  psi : ``(W-1, N, d, n_params)``
    [, Q : ``(W-1, N, d, d)`` when ``D_lyap`` is given]
    """
    jpb = _jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator,
               D_lyap=D_lyap)

    if D_lyap is not None:
        def _step_q(X_start, X_end, dt_k):
            J, Phi, B, Q = jpb(X_start, dt_k)
            return X_end - X_start - Phi, J, -B, Q

        step = _step_q
    else:
        def _step(X_start, X_end, dt_k):
            J, Phi, B = jpb(X_start, dt_k)
            return X_end - X_start - Phi, J, -B

        step = _step

    if jnp.ndim(dt) > 0:
        return jax.vmap(step)(X_w[:-1], X_w[1:], dt)
    return jax.vmap(step, in_axes=(0, 0, None))(X_w[:-1], X_w[1:], dt)


def multi_step_residuals(F_psf, theta_struct, X_w, extras, dt, n_substeps, integrator="rk4"):
    r"""Per-particle residuals and flow Jacobians over a multiparticle window.

    Parameters
    ----------
    F_psf : PSF
    theta_struct : structured parameters (from ``F_psf.unflatten_params``).
    X_w : ``(W, N, d)`` window of multiparticle positions.
    extras : dict or None
    dt, n_substeps, integrator : flow settings.

    Returns
    -------
    r : ``(W-1, N, d)``
    J : ``(W-1, N, d, d)``
    """
    if getattr(F_psf, "particles_input", False):
        from .jacobians import rk4_composed_jacobian

        dFdx = F_psf.d_x(same_particle=True)

        def _step(X_start, X_end):  # (N, d), (N, d)
            J, Phi = rk4_composed_jacobian(
                F_psf, dFdx, X_start, theta_struct, None, extras, dt, n_substeps, integrator)
            return X_end - X_start - Phi, J

        return jax.vmap(_step)(X_w[:-1], X_w[1:])  # (W-1, N, d), (W-1, N, d, d)

    def _drift(x):
        return F_psf(x[None], params=theta_struct, extras=extras)[0]

    def _per_particle(Xp):  # (W, d)
        return od_window_residuals(_drift, Xp, dt, n_substeps, integrator)

    r, J = jax.vmap(_per_particle, in_axes=1)(X_w)  # (N, W-1, d), (N, W-1, d, d)
    return jnp.swapaxes(r, 0, 1), jnp.swapaxes(J, 0, 1)


def ud_multi_step_residuals(F_psf, theta_struct, Y_w, extras, dt, n_substeps, integrator="rk4"):
    r"""Per-particle underdamped 3-point residuals + α-propagators + ``v̂``.

    * *Non-interacting* (``particles_input=False``): ``vmap`` the
      single-particle :func:`flow_ud.ud_window_residuals` over particles.
    * *Interacting* (``particles_input=True``): multi-particle phase-space
      shooting (full force drives the flow) + frozen-background per-particle
      phase Jacobian via ``rk4_composed_jacobian_phase`` — the underdamped
      analogue of the overdamped interacting path.

    Returns ``r, α₊, α₀, α₋, v̂`` each shaped ``(W-2, N, ...)``.
    """
    if getattr(F_psf, "particles_input", False):
        return _ud_multi_interacting(F_psf, theta_struct, Y_w, extras, dt, n_substeps, integrator)

    def _force(x, v):
        return F_psf(x[None], v=v[None], params=theta_struct, extras=extras)[0]

    def _per_particle(Yp):  # (W, d)
        r, ap, a0, am, _, vhat = ud_window_residuals(_force, Yp, dt, n_substeps, integrator)
        return r, ap, a0, am, vhat

    r, ap, a0, am, vhat = jax.vmap(_per_particle, in_axes=1)(Y_w)  # each (N, W-2, ...)

    def sw(x):
        return jnp.swapaxes(x, 0, 1)

    return sw(r), sw(ap), sw(a0), sw(am), sw(vhat)


def multi_od_instrument(F_psf, theta, X_base, extras, dt, n_substeps,
                        integrator="rk4", dt_prop=None):
    r"""η-clean overdamped instrument per particle — ``(N, d, n_params)``.

    Trapezoidal pair of θ-sensitivities at the η-clean base frame and its
    deterministic forward image (cf. :func:`flow.od_instrument`),

        ψ_inst = −½(B(X_base) + B(X_base + Φ)),

    computed with the per-stage θ-recursion for BOTH particle contracts —
    O(N·d·n_params) memory, vs jacfwd through the (N-body) flow.  For
    interacting models the instrument flow must be the *same N-body flow*
    the residual uses: evaluating the force on isolated single-particle
    frames zeroes (or crashes) every pair-feature column, which makes the
    IV Gram structurally singular on the interaction parameters — the
    ABP-port plateau.  For non-interacting models the recursion is the
    exact ``∂Φ/∂θ`` and reproduces the legacy vmapped single-particle
    instrument.

    Parameters
    ----------
    F_psf : PSF
    theta : ``(n_params,)`` flat parameters.
    X_base : ``(N, d)`` η-clean base frame (the reserved front position).
    extras : dict or None
    dt, n_substeps, integrator : flow settings.  ``dt`` is the *residual's*
        interval (both θ-sensitivities are evaluated over it).
    dt_prop : optional traced scalar
        Per-interval sampling: the base frame's own interval
        ``dt_{k−1}``, over which the base is propagated to its clean
        forward image ``≈ x_k``.  ``None`` (uniform sampling) keeps the
        historical single-``dt`` construction, where the propagation
        displacement comes for free with ``B(X_base)``.
    """
    jpb = _jpb(F_psf, theta, extras, dt, n_substeps, integrator)
    if dt_prop is None:
        _, Phi0, B0 = jpb(X_base)
        _, _, B1 = jpb(jax.lax.stop_gradient(X_base + Phi0))
        return -0.5 * (B0 + B1)
    _, _, B0 = jpb(X_base)
    struct = F_psf.unflatten_params(theta)

    def _drift(X):
        return F_psf(X, params=struct, extras=extras)

    Phi0 = flow_displacement(_drift, X_base, dt_prop, n_substeps, integrator)
    _, _, B1 = jpb(jax.lax.stop_gradient(X_base + Phi0))
    return -0.5 * (B0 + B1)


def multi_ud_instrument(F_psf, theta, Y_a, Y_b, extras, dt, n_substeps,
                        integrator="rk4", n_predict=2):
    r"""η-clean underdamped instrument per particle — ``(N, d, n_params)``.

    Frame-level construction for BOTH particle contracts, built from the
    same protocols as the residual path — shooting at ``(Y_a → Y_b)`` with
    the per-particle ``J^{xv}`` (:func:`jacobians.rk4_composed_jacobian_
    phase`, as in the residual's ``v̂``), phase-flowing the full frame
    forward ``n_predict`` intervals, then differentiating the one-step
    position flow in θ at the clean (stop-gradient) phase point via the
    phase θ-recursion.  Everything is a function of the two front frames
    only, so the instrument stays η-clean of the whole residual block.

    For interacting models the flow is the full N-body flow (isolated
    single-particle frames would zero every pair-feature column); for
    non-interacting models every ingredient is the exact per-particle
    chain rule and this reproduces the legacy vmapped
    :func:`flow_ud.ud_instrument`.

    ``dt`` may be a scalar (uniform sampling) or an ``(n_predict + 2,)``
    vector ``(dt_shoot, dt_pred_1, …, dt_pred_n, dt_eval)`` — for residual
    ``n`` with base pair ``(Y_{n−2}, Y_{n−1})`` this is
    ``(dt[n−2], dt[n−1], dt[n], dt[n+1])``: shoot over the pair's own
    interval, flow forward over the two physical intervals to the clean
    centre ``≈ z_{n+1}``, and differentiate the one-step flow over the
    residual's *prediction* interval (matching ``ψ``'s ``B^x_out``).
    """
    from .jacobians import rk4_composed_jacobian_phase

    per_step = jnp.ndim(dt) > 0
    dt_shoot = dt[0] if per_step else dt
    struct0 = F_psf.unflatten_params(theta)
    dFdx = _d_expr(F_psf, "x")
    dFdv = _d_expr(F_psf, "v")
    N = Y_a.shape[0]

    def frame_phase_field(th_struct):
        def force(X, V):
            return F_psf(X, v=V, params=th_struct, extras=extras)
        return _phase_field(force, N)

    # 1. shooting at (Y_a → Y_b): arrival velocity at Y_b (residual protocol).
    V0 = (Y_b - Y_a) / dt_shoot
    Phi_x0, Phi_v0, _Jxx, Jxv, _Jvx, Jvv = rk4_composed_jacobian_phase(
        F_psf, dFdx, dFdv, Y_a, V0, struct0, None, extras, dt_shoot,
        n_substeps, integrator)
    dV = jnp.linalg.solve(Jxv, (Y_b - Phi_x0)[..., None]).squeeze(-1)
    v_b = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv, dV)

    # 2. full-force phase flow to the center base (θ fixed, then "data").
    z = jnp.concatenate([Y_b, v_b], axis=0)            # (2N, d) lifted frame
    g0 = frame_phase_field(struct0)
    for i in range(n_predict):
        z = z + flow_displacement(g0, z, dt[1 + i] if per_step else dt,
                                  n_substeps, integrator)
    z_c = jax.lax.stop_gradient(z)

    # 3. ψ_inst = −∂Φˣ/∂θ at the clean phase point — per-particle phase
    # θ-sensitivity (Bx), O(N·d·n_params) memory.
    jpb = _phase_jpb(F_psf, theta, extras,
                     dt[n_predict + 1] if per_step else dt,
                     n_substeps, integrator)
    x_c, v_c = z_c[:N], z_c[N:]
    Bx = jpb(x_c, v_c)[6]
    return -Bx


def _phase_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator,
               D_lyap=None, return_stages=False):
    """Phase frame → (Φx, Φv, Jxx, Jxv, Jvx, Jvv, Bx, Bv[, Q]) closure
    (both contracts; ``Q`` is the lifted Lyapunov covariance when
    ``D_lyap`` is given).  The closure accepts an optional per-call
    ``dt_step`` overriding the constructor ``dt`` (per-interval steps)."""
    from .jacobians import rk4_composed_jacobian_phase_theta

    struct = F_psf.unflatten_params(theta_flat)
    dFdx = _d_expr(F_psf, "x")
    dFdv = _d_expr(F_psf, "v")
    n_params = int(theta_flat.shape[-1])
    dFdth = _dFdth_builder(F_psf, theta_flat, extras, needs_v=True)

    def jpb(X, V, dt_step=None):
        return rk4_composed_jacobian_phase_theta(
            F_psf, dFdx, dFdv, dFdth, n_params, X, V, struct, None, extras,
            dt if dt_step is None else dt_step, n_substeps, integrator,
            D_lyap=D_lyap, return_stages=return_stages)

    return jpb


def ud_multi_step_residuals_with_psi(F_psf, theta_flat, Y_w, extras, dt, n_substeps,
                                     integrator="rk4", D_lyap=None,
                                     with_stages=False):
    r"""UD residuals + α-propagators + ``ψ_right`` (both particle contracts).

    Same shooting protocol as the plain residual path, additionally
    returning ``ψ_right = ∂r/∂θ`` from the per-particle phase θ-recursion:

    .. math::

        \partial\hat v/\partial\theta = B^v_{\rm in}
            - J^{vv} (J^{xv})^{-1} B^x_{\rm in},\qquad
        ψ = -\bigl(B^x_{\rm out} + J^{xv}_{\rm out}\,
            \partial\hat v/\partial\theta\bigr),

    treating the (O(h)) Jacobian blocks as θ-independent.  Differentiating
    them exactly would need ``∂²F/∂x∂θ`` — the second-order AD this path
    avoids.  The dropped terms are O(Δt³) absolute on a ψ of magnitude
    O(Δt²), i.e. **O(Δt) relative**; as an estimating-equation test
    function this preserves consistency, costs O(rel²) efficiency, and is
    what the interacting path always shipped.  Pinned by the dt-scaling
    test in ``test_parametric_core_psi_recursion``.

    Returns ``r, α₊, α₀, α₋, v̂, ψ`` with ``ψ : (W-2, N, d, n_params)``.
    With ``D_lyap`` given, additionally returns a dict of the
    Lyapunov-exact process ingredients per residual: the lifted noise
    blocks of the shooting interval (``Qxx_in, Qxv_in, Qvv_in``), the
    position blocks of the forward interval (``Qxx_out, Qxv_out``), and
    the propagators ``N = J^{vv}(J^{xv})^{-1}`` (shooting) and
    ``Jxv_out`` needed to assemble the exact-linearized pentadiagonal
    blocks (see :func:`covariance.build_ud_blocks_exact`).

    ``dt`` may be a scalar or a ``(W-1,)`` vector of per-interval steps:
    residual ``n`` then shoots over ``dt[n]`` (interval ``n → n+1``) and
    predicts over ``dt[n+1]`` (interval ``n+1 → n+2``).  The α-propagator
    algebra below is already written in per-interval Jacobian blocks, so
    it holds verbatim for unequal intervals.
    """
    with_q = D_lyap is not None
    jpb = _phase_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator,
                     D_lyap=D_lyap, return_stages=with_stages)
    d = Y_w.shape[-1]
    I_d = jnp.eye(d, dtype=Y_w.dtype)

    def _step(X_prev, X_cur, X_next, dt_in, dt_out):   # frames (N, d)
        V0 = (X_cur - X_prev) / dt_in
        out_in = jpb(X_prev, V0, dt_in)
        Phi_x0, Phi_v0, Jxx_in, Jxv_in, Jvx_in, Jvv_in, Bx_in, Bv_in = out_in[:8]
        Jxv_in_inv = jnp.linalg.inv(Jxv_in)
        dV = jnp.einsum("nij,nj->ni", Jxv_in_inv, X_cur - Phi_x0)
        v_hat = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv_in, dV)
        dvhat_dth = Bv_in - jnp.einsum("nij,njk->nik", Jvv_in,
                                       jnp.einsum("nij,njk->nik", Jxv_in_inv, Bx_in))

        out_out = jpb(X_cur, v_hat, dt_out)
        Phi_x_out, _, Jxx_out, Jxv_out = out_out[0], out_out[1], out_out[2], out_out[3]
        Bx_out = out_out[6]
        r = X_next - Phi_x_out
        psi = -(Bx_out + jnp.einsum("nij,njk->nik", Jxv_out, dvhat_dth))

        M = Jvx_in - Jvv_in @ Jxv_in_inv @ Jxx_in
        Nmat = Jvv_in @ Jxv_in_inv
        alpha_p1 = jnp.broadcast_to(I_d, Jxx_out.shape)
        alpha_0 = -Jxx_out - Jxv_out @ Nmat
        alpha_m1 = -Jxv_out @ M
        base = (r, alpha_p1, alpha_0, alpha_m1, v_hat, psi)
        if with_q:
            Q_in, Q_out = out_in[8], out_out[8]
            base = base + ({
                "Qxx_in": Q_in[:, :d, :d], "Qxv_in": Q_in[:, :d, d:],
                "Qvv_in": Q_in[:, d:, d:],
                "Qxx_out": Q_out[:, :d, :d], "Qxv_out": Q_out[:, :d, d:],
                "N": Nmat, "Jxv_out": Jxv_out,
            },)
        if with_stages:
            # the lifted stage matrices of the two interval flows — the
            # fixed-θ cache from which Q_in/Q_out are recomputable for any
            # D via lyapunov_from_stages (the profile / diffusion fast path)
            base = base + ({
                "stages_in": out_in[-1], "stages_out": out_out[-1],
                "N": Nmat, "Jxv_out": Jxv_out,
            },)
        return base

    if jnp.ndim(dt) > 0:
        return jax.vmap(_step)(Y_w[:-2], Y_w[1:-1], Y_w[2:], dt[:-1], dt[1:])
    return jax.vmap(_step, in_axes=(0, 0, 0, None, None))(
        Y_w[:-2], Y_w[1:-1], Y_w[2:], dt, dt)


def _ud_multi_interacting(F_psf, theta_struct, Y_w, extras, dt, n_substeps, integrator):
    """Interacting underdamped: per-particle shooting + frozen-background phase Jacobian."""
    from .jacobians import rk4_composed_jacobian_phase

    dFdx = F_psf.d_x(same_particle=True)
    dFdv = F_psf.d_v(same_particle=True)
    d = Y_w.shape[-1]
    I_d = jnp.eye(d, dtype=Y_w.dtype)

    def _phase(X, V):
        return rk4_composed_jacobian_phase(
            F_psf, dFdx, dFdv, X, V, theta_struct, None, extras, dt, n_substeps, integrator)

    def _step(X_prev, X_cur, X_next):              # each (N, d)
        V0 = (X_cur - X_prev) / dt
        Phi_x0, Phi_v0, Jxx_in, Jxv_in, Jvx_in, Jvv_in = _phase(X_prev, V0)
        dV = jnp.linalg.solve(Jxv_in, (X_cur - Phi_x0)[..., None]).squeeze(-1)  # (N, d)
        v_hat = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv_in, dV)

        Phi_x_out, _, Jxx_out, Jxv_out, _, _ = _phase(X_cur, v_hat)
        r = X_next - Phi_x_out

        Jxv_in_inv = jnp.linalg.inv(Jxv_in)
        M = Jvx_in - Jvv_in @ Jxv_in_inv @ Jxx_in
        Nmat = Jvv_in @ Jxv_in_inv
        alpha_p1 = jnp.broadcast_to(I_d, Jxx_out.shape)
        alpha_0 = -Jxx_out - Jxv_out @ Nmat
        alpha_m1 = -Jxv_out @ M
        return r, alpha_p1, alpha_0, alpha_m1, v_hat

    return jax.vmap(_step)(Y_w[:-2], Y_w[1:-1], Y_w[2:])  # each (W-2, N, ...)
