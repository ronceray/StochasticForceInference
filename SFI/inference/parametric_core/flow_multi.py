# SFI/inference/parametric_core/flow_multi.py
"""
Multiparticle residuals — one function for both the non-interacting and
interacting cases, dispatched on the force model's particle contract.

The covariance/precision machinery is **per particle**, so all this layer
does is produce per-particle residuals ``r`` and flow Jacobians ``J`` from
a window of multiparticle positions:

* **non-interacting** (``particles_input=False``): the same force is
  applied to each particle independently — ``vmap`` the single-particle
  :func:`flow.od_window_residuals` over the particle axis.
* **interacting** (``particles_input=True``): the displacement uses the
  *full* multiparticle force (couplings matter), while the per-particle
  flow Jacobian uses the frozen-background same-particle derivative
  ``F_psf.d_x(same_particle=True)`` — via
  :func:`jacobians.rk4_composed_jacobian` (O(N) per window).

Single particle is just ``N=1``; the integrate engine reduces + masks
over the particle axis (``reduce_over_particles``), so there is no
separate multiparticle code path in the solver — exactly like
``infer_force_linear``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .flow import flow_displacement, od_instrument, od_window_residuals
from .flow_ud import _phase_field, ud_instrument, ud_window_residuals

__all__ = ["multi_step_residuals", "multi_step_residuals_with_psi",
           "ud_multi_step_residuals", "ud_multi_step_residuals_with_psi",
           "multi_od_instrument", "multi_ud_instrument"]


def _interacting_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator):
    r"""Frame → ``(J, Φ, B)`` closure for an interacting force (frozen background).

    ``B = ∂Φ/∂θ`` per particle via :func:`jacobians.rk4_composed_jacobian_theta`
    — the legacy same-particle trick: O(N·d·n_params) carry instead of
    forward-mode tangents through the full N-body window flow.  ``∂F/∂θ`` at a
    fixed frame is exact and per-particle (jacfwd of one force evaluation);
    only the O(h²) cross-particle substep feedback is dropped, consistently
    with the frozen-background ``J``.
    """
    from .jacobians import rk4_composed_jacobian_theta

    struct = F_psf.unflatten_params(theta_flat)
    dFdx = F_psf.d_x(same_particle=True)
    n_params = int(theta_flat.shape[-1])

    def dFdth(X):
        return jax.jacfwd(
            lambda th: F_psf(X, params=F_psf.unflatten_params(th), extras=extras)
        )(theta_flat)                                   # (N, d, n_params)

    def jpb(X_start):
        return rk4_composed_jacobian_theta(
            F_psf, dFdx, dFdth, n_params, X_start, struct, None, extras,
            dt, n_substeps, integrator)

    return jpb


def multi_step_residuals_with_psi(F_psf, theta_flat, X_w, extras, dt, n_substeps,
                                  integrator="rk4"):
    r"""Interacting residuals + flow Jacobians + frozen-background ``ψ_right``.

    Same residuals/Jacobians as the interacting branch of
    :func:`multi_step_residuals`, additionally returning the per-particle
    regressor ``ψ_right = ∂r/∂θ = −B`` from the same-particle θ-recursion —
    the memory-scalable replacement for ``jax.jacfwd`` through the full
    N-body flow (which holds n_params tangents alive across the whole
    window graph and scales O(K·N²·n_params) per chunk).

    Returns
    -------
    r : ``(W-1, N, d)``,  J : ``(W-1, N, d, d)``,  psi : ``(W-1, N, d, n_params)``
    """
    jpb = _interacting_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator)

    def _step(X_start, X_end):
        J, Phi, B = jpb(X_start)
        return X_end - X_start - Phi, J, -B

    return jax.vmap(_step)(X_w[:-1], X_w[1:])


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


def multi_od_instrument(F_psf, theta, X_base, extras, dt, n_substeps, integrator="rk4"):
    r"""η-clean overdamped instrument per particle — ``(N, d, n_params)``.

    Multiparticle dispatcher for :func:`flow.od_instrument`, mirroring
    :func:`multi_step_residuals`:

    * **non-interacting**: the single-particle instrument, ``vmap``-ed over
      the particle axis of the base frame.
    * **interacting** (``particles_input=True``): the instrument flow must be
      the *same N-body flow* the residual uses — the sensitivity is taken of
      the full-frame displacement ``Φ(X;θ)``, seeded at the η-clean base frame.
      Evaluating the force on isolated single-particle frames instead zeroes
      (or crashes) every pair-feature column, which makes the IV Gram
      structurally singular on the interaction parameters — the ABP-port
      plateau.  ``od_instrument`` is shape-agnostic, so the frame-level call
      only needs a frame-level drift closure.

    Parameters
    ----------
    F_psf : PSF
    theta : ``(n_params,)`` flat parameters.
    X_base : ``(N, d)`` η-clean base frame (the reserved front position).
    extras : dict or None
    dt, n_substeps, integrator : flow settings.
    """
    if getattr(F_psf, "particles_input", False):
        # Frozen-background trapeze: ψ_inst = −½(B(X_base) + B(X_base + Φ)),
        # the same-particle θ-sensitivity at the η-clean frame and its forward
        # image — O(N·d·n_params) memory, vs jacfwd through the N-body flow.
        jpb = _interacting_jpb(F_psf, theta, extras, dt, n_substeps, integrator)
        _, Phi0, B0 = jpb(X_base)
        _, _, B1 = jpb(jax.lax.stop_gradient(X_base + Phi0))
        return -0.5 * (B0 + B1)

    def drift_of_theta(th):
        struct = F_psf.unflatten_params(th)
        return lambda y: F_psf(y[None], params=struct, extras=extras)[0]

    return jax.vmap(
        lambda xb: od_instrument(drift_of_theta, theta, xb, dt, n_substeps, integrator)
    )(X_base)


def multi_ud_instrument(F_psf, theta, Y_a, Y_b, extras, dt, n_substeps,
                        integrator="rk4", n_predict=2):
    r"""η-clean underdamped instrument per particle — ``(N, d, n_params)``.

    Multiparticle dispatcher for :func:`flow_ud.ud_instrument`:

    * **non-interacting**: the single-particle instrument, ``vmap``-ed over
      the particle axis of the lagged pair ``(Y_a, Y_b)``.
    * **interacting** (``particles_input=True``): frame-level analogue built
      from the same protocols as the interacting residual path — shooting at
      ``(Y_a → Y_b)`` with the frozen-background per-particle ``J^{xv}``
      (:func:`jacobians.rk4_composed_jacobian_phase`, as in the residual's
      ``v̂``), phase-flowing the full frame forward ``n_predict`` intervals,
      then differentiating the full-force one-step position flow in θ at the
      clean (stop-gradient) phase point.  Everything is a function of the two
      front frames only, so the instrument stays η-clean of the whole
      residual block.
    """
    if getattr(F_psf, "particles_input", False):
        from .jacobians import rk4_composed_jacobian_phase

        struct0 = F_psf.unflatten_params(theta)
        dFdx = F_psf.d_x(same_particle=True)
        dFdv = F_psf.d_v(same_particle=True)
        N = Y_a.shape[0]

        def frame_phase_field(th_struct):
            def force(X, V):
                return F_psf(X, v=V, params=th_struct, extras=extras)
            return _phase_field(force, N)

        # 1. shooting at (Y_a → Y_b): arrival velocity at Y_b (residual protocol).
        V0 = (Y_b - Y_a) / dt
        Phi_x0, Phi_v0, _Jxx, Jxv, _Jvx, Jvv = rk4_composed_jacobian_phase(
            F_psf, dFdx, dFdv, Y_a, V0, struct0, None, extras, dt, n_substeps, integrator)
        dV = jnp.linalg.solve(Jxv, (Y_b - Phi_x0)[..., None]).squeeze(-1)
        v_b = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv, dV)

        # 2. full-force phase flow to the center base (θ fixed, then "data").
        z = jnp.concatenate([Y_b, v_b], axis=0)            # (2N, d) lifted frame
        g0 = frame_phase_field(struct0)
        for _ in range(n_predict):
            z = z + flow_displacement(g0, z, dt, n_substeps, integrator)
        z_c = jax.lax.stop_gradient(z)

        # 3. ψ_inst = −∂Φˣ/∂θ at the clean phase point — frozen-background
        # per-particle phase θ-sensitivity (Bx), O(N·d·n_params) memory.
        jpb = _interacting_phase_jpb(F_psf, theta, extras, dt, n_substeps, integrator)
        x_c, v_c = z_c[:N], z_c[N:]
        Bx = jpb(x_c, v_c)[6]
        return -Bx

    def force_of_theta(th):
        struct = F_psf.unflatten_params(th)
        return lambda x, v: F_psf(x[None], v=v[None], params=struct, extras=extras)[0]

    return jax.vmap(
        lambda a, b: ud_instrument(force_of_theta, theta, a, b, dt, n_substeps,
                                   integrator, n_predict=n_predict)
    )(Y_a, Y_b)


def _interacting_phase_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator):
    """Phase frame → (Φx, Φv, Jxx, Jxv, Jvx, Jvv, Bx, Bv) closure (frozen background)."""
    from .jacobians import rk4_composed_jacobian_phase_theta

    struct = F_psf.unflatten_params(theta_flat)
    dFdx = F_psf.d_x(same_particle=True)
    dFdv = F_psf.d_v(same_particle=True)
    n_params = int(theta_flat.shape[-1])

    def dFdth(X, V):
        return jax.jacfwd(
            lambda th: F_psf(X, v=V, params=F_psf.unflatten_params(th), extras=extras)
        )(theta_flat)                                   # (N, d, n_params)

    def jpb(X, V):
        return rk4_composed_jacobian_phase_theta(
            F_psf, dFdx, dFdv, dFdth, n_params, X, V, struct, None, extras,
            dt, n_substeps, integrator)

    return jpb


def ud_multi_step_residuals_with_psi(F_psf, theta_flat, Y_w, extras, dt, n_substeps,
                                     integrator="rk4"):
    r"""Interacting UD residuals + α-propagators + frozen-background ``ψ_right``.

    Same shooting protocol as :func:`_ud_multi_interacting`, additionally
    returning ``ψ_right = ∂r/∂θ`` from the per-particle phase θ-recursion:

    .. math::

        \partial\hat v/\partial\theta = B^v_{\rm in}
            - J^{vv} (J^{xv})^{-1} B^x_{\rm in},\qquad
        ψ = -\bigl(B^x_{\rm out} + J^{xv}_{\rm out}\,
            \partial\hat v/\partial\theta\bigr),

    treating the (O(h)) Jacobian blocks as θ-independent — the same
    second-order cross terms the frozen-background blocks already drop.

    Returns ``r, α₊, α₀, α₋, v̂, ψ`` with ``ψ : (W-2, N, d, n_params)``.
    """
    jpb = _interacting_phase_jpb(F_psf, theta_flat, extras, dt, n_substeps, integrator)
    d = Y_w.shape[-1]
    I_d = jnp.eye(d, dtype=Y_w.dtype)

    def _step(X_prev, X_cur, X_next):              # each (N, d)
        V0 = (X_cur - X_prev) / dt
        Phi_x0, Phi_v0, Jxx_in, Jxv_in, Jvx_in, Jvv_in, Bx_in, Bv_in = jpb(X_prev, V0)
        Jxv_in_inv = jnp.linalg.inv(Jxv_in)
        dV = jnp.einsum("nij,nj->ni", Jxv_in_inv, X_cur - Phi_x0)
        v_hat = Phi_v0 + jnp.einsum("nij,nj->ni", Jvv_in, dV)
        dvhat_dth = Bv_in - jnp.einsum("nij,njk->nik", Jvv_in,
                                       jnp.einsum("nij,njk->nik", Jxv_in_inv, Bx_in))

        Phi_x_out, _, Jxx_out, Jxv_out, _, _, Bx_out, _ = jpb(X_cur, v_hat)
        r = X_next - Phi_x_out
        psi = -(Bx_out + jnp.einsum("nij,njk->nik", Jxv_out, dvhat_dth))

        M = Jvx_in - Jvv_in @ Jxv_in_inv @ Jxx_in
        Nmat = Jvv_in @ Jxv_in_inv
        alpha_p1 = jnp.broadcast_to(I_d, Jxx_out.shape)
        alpha_0 = -Jxx_out - Jxv_out @ Nmat
        alpha_m1 = -Jxv_out @ M
        return r, alpha_p1, alpha_0, alpha_m1, v_hat, psi

    return jax.vmap(_step)(Y_w[:-2], Y_w[1:-1], Y_w[2:])


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
