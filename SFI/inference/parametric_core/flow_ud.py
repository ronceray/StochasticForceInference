# SFI/inference/parametric_core/flow_ud.py
"""
Underdamped (2nd-order) flow: phase-space propagation, shooting velocity,
and the 3-point position residual with its measurement-noise propagators.

The velocity is unobserved, so the residual is built from three
consecutive positions:

1. **Shooting** — resolve the velocity ``v̂_n`` that carries ``Y_{n-1}``
   to ``Y_n`` by one Newton step from the secant velocity (the position
   block of the phase-space flow is inverted via ``J^{xv}``).
2. **Residual** — propagate ``(Y_n, v̂_n)`` one step and compare to the
   next observed position: ``r_n = Y_{n+1} − Φˣ(Y_n, v̂_n; θ)``.
3. **α-propagators** — how the measurement noise at the three samples
   enters ``r_n`` (``α₊ = I``, ``α₀``, ``α₋``); these set the bandwidth-2
   (pentadiagonal) covariance structure.

The phase-space flow is just :func:`flow.flow_step` applied to the lifted
field ``ż = (v, F(x, v))`` on ``ℝ^{2d}`` — RK4/Euler and the Jacobian come
for free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .flow import flow_step

__all__ = ["phase_space_flow_jac", "shooting_velocity", "ud_step_quantities",
           "ud_window_residuals", "ud_instrument"]


def _phase_field(force_fn, d):
    def g(z):
        return jnp.concatenate([z[d:], force_fn(z[:d], z[d:])])
    return g


def phase_space_flow_jac(force_fn, x, v, dt, n_substeps, integrator="rk4"):
    r"""Phase-space arrival state and ``2d×2d`` flow Jacobian blocks.

    Returns ``Φ_x, Φ_v`` (arrival position, velocity) and the four
    ``d×d`` blocks ``J^{xx}, J^{xv}, J^{vx}, J^{vv}`` of ``∂(x',v')/∂(x,v)``.
    """
    d = x.shape[0]
    z0 = jnp.concatenate([x, v])
    Phi_z, J = flow_step(_phase_field(force_fn, d), z0, dt, n_substeps, integrator)
    z_final = z0 + Phi_z
    return (
        z_final[:d], z_final[d:],
        J[:d, :d], J[:d, d:], J[d:, :d], J[d:, d:],
    )


def shooting_velocity(force_fn, Y_prev, Y_cur, dt, n_substeps, integrator="rk4"):
    r"""Velocity at ``Y_prev`` that flows to ``Y_cur`` (one Newton step).

    Returns ``v̂`` (the *arrival* velocity at ``Y_cur``) and the incoming
    phase-space Jacobian blocks evaluated at ``(Y_prev, v₀)``.
    """
    v0 = (Y_cur - Y_prev) / dt
    Phi_x0, Phi_v0, Jxx, Jxv, Jvx, Jvv = phase_space_flow_jac(force_fn, Y_prev, v0, dt, n_substeps, integrator)
    dv = jnp.linalg.solve(Jxv, Y_cur - Phi_x0)
    v_hat = Phi_v0 + Jvv @ dv
    return v_hat, Jxx, Jxv, Jvx, Jvv


def ud_step_quantities(force_fn, Y_prev, Y_cur, Y_next, dt, n_substeps, integrator="rk4"):
    r"""Residual and α-propagators for one interior position ``Y_cur``.

    Returns ``r, α₊, α₀, α₋, J^{vv}_out, v̂`` where ``r = Y_next − Φˣ(Y_cur,
    v̂)``, the α's propagate the measurement noise at ``Y_{n-1}, Y_n,
    Y_{n+1}`` into ``r``, and ``v̂`` is the shooting velocity at ``Y_cur``
    (used to evaluate a velocity-dependent diffusion ``D(x, v)``).
    """
    d = Y_cur.shape[0]
    I_d = jnp.eye(d, dtype=Y_cur.dtype)

    v_hat, Jxx_in, Jxv_in, Jvx_in, Jvv_in = shooting_velocity(force_fn, Y_prev, Y_cur, dt, n_substeps, integrator)
    Phi_x_out, _, Jxx_out, Jxv_out, _, Jvv_out = phase_space_flow_jac(
        force_fn, Y_cur, v_hat, dt, n_substeps, integrator
    )

    r = Y_next - Phi_x_out

    Jxv_in_inv = jnp.linalg.inv(Jxv_in)
    M = Jvx_in - Jvv_in @ Jxv_in_inv @ Jxx_in
    N = Jvv_in @ Jxv_in_inv

    alpha_p1 = I_d
    alpha_0 = -Jxx_out - Jxv_out @ N
    alpha_m1 = -Jxv_out @ M
    return r, alpha_p1, alpha_0, alpha_m1, Jvv_out, v_hat


def ud_window_residuals(force_fn, Y_w, dt, n_substeps, integrator="rk4"):
    r"""3-point residuals and α-propagators over a position window.

    For ``W`` consecutive positions returns the ``W-2`` interior residuals
    and their propagators.

    Returns
    -------
    r : ``(W-2, d)``
    alpha_plus, alpha_zero, alpha_minus : ``(W-2, d, d)``
    Jvv_out : ``(W-2, d, d)``
    v_hat : ``(W-2, d)``  shooting velocities (for velocity-dependent D)
    """
    def _step(Yp, Yc, Yn):
        return ud_step_quantities(force_fn, Yp, Yc, Yn, dt, n_substeps, integrator)

    return jax.vmap(_step)(Y_w[:-2], Y_w[1:-1], Y_w[2:])


def ud_instrument(force_of_theta, theta, Y_a, Y_b, dt, n_substeps, integrator="rk4", n_predict=2):
    r"""η-clean instrument (left test function) for one underdamped residual.

    The center 3-point residual ``r_c = Y_{c+2} − Φˣ(Y_{c+1}, v̂)`` is
    contaminated by measurement noise at ``Y_c, Y_{c+1}, Y_{c+2}`` through the
    shooting velocity ``v̂`` (reconstructed from ``Y_c, Y_{c+1}``) — the
    errors-in-variables source that is *amplified* in the underdamped case
    because ``v̂ ∼ ΔY/Δt`` divides the position noise by ``Δt``.

    The instrument is the regressor ``∂r/∂θ = −∂Φˣ(Y_{c+1}, v̂)/∂θ`` evaluated
    at a **clean phase point** instead of the noisy ``(Y_{c+1}, v̂)``.  The clean
    phase point is reconstructed from the lagged pair ``(Y_a, Y_b) =
    (Y_{c-2}, Y_{c-1})`` — whose measurement noise is independent of the
    residual's — by shooting the velocity at ``Y_b`` and phase-flowing forward
    ``n_predict`` intervals to the center base; that point is held fixed
    (stop-gradient, "data"), and only the final one-step position displacement
    is differentiated in θ:

    .. math:: ψ_{\mathrm{inst}} = -\left.\frac{∂Φˣ(x,v;θ)}{∂θ}\right|_{(\tilde x_{c+1},\,\tilde v_{c+1})}.

    Because it is the *same* one-step displacement sensitivity as the regressor
    (only the evaluation phase point is clean), every parameter column — in
    particular the velocity-damping (``γ``) column, which depends on the
    reconstructed velocity — keeps the regressor's scale, so the asymmetric
    Gram stays well-conditioned.  It is the underdamped analogue of the
    overdamped lag-1 shift instrument :func:`flow.od_instrument`.

    Parameters
    ----------
    force_of_theta : callable  ``θ_flat → (force_fn : (x, v) → accel)``
    theta : array ``(n_params,)``
    Y_a, Y_b : arrays ``(d,)``  clean lagged pair ``(Y_{c-2}, Y_{c-1})``.
    n_predict : int  phase-flow steps from ``Y_b`` to the center base
        (``= 2`` for the centered 7-residual window: ``Y_{c-1} → Y_{c+1}``).

    Returns
    -------
    psi_inst : array ``(d, n_params)``
    """
    force0 = force_of_theta(theta)
    v_b, *_ = shooting_velocity(force0, Y_a, Y_b, dt, n_substeps, integrator)
    x, v = Y_b, v_b
    for _ in range(n_predict):
        x, v, *_ = phase_space_flow_jac(force0, x, v, dt, n_substeps, integrator)
    x_c = jax.lax.stop_gradient(x)
    v_c = jax.lax.stop_gradient(v)

    def disp_x(th):
        x_next, _, *_ = phase_space_flow_jac(
            force_of_theta(th), x_c, v_c, dt, n_substeps, integrator)
        return x_next

    return -jax.jacfwd(disp_x)(theta)
