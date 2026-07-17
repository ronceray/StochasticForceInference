# SFI/inference/parametric_core/flow.py
"""
Deterministic flow, Jacobian, and residuals for the parametric core.

The estimator linearises the SDE around the *deterministic* flow of the
drift field.  For a single observation interval Δt the displacement and
its sensitivity to the start point are

    Φ(y;θ) = ψ_Δt(y;θ) − y,        J(y;θ) = I + ∂Φ/∂y = ∂ψ_Δt/∂y,

where ``ψ_Δt`` is the RK4 (or single-step Euler) flow of ``ẋ = F(x;θ)``.
Choosing ``integrator="euler", n_substeps=1`` recovers the bare Euler
predictor ``Φ = F(y;θ)·Δt`` — useful for *independently* assessing the
effect of flow propagation versus the covariance/precision model.

These are pure, JAX-traceable functions (parameters are closed over in
``drift_fn``); batching over windows/particles is done by the caller via
``vmap`` or the ``SFI.integrate`` engine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from SFI.integrate.rk4 import select_flow

__all__ = ["flow_step", "flow_displacement", "od_window_residuals", "od_instrument"]


def flow_displacement(drift_fn, y, dt, n_substeps, integrator="rk4"):
    """Deterministic displacement ``Φ(y) = flow(y) − y`` over one interval Δt.

    Parameters
    ----------
    drift_fn : callable ``(d,) → (d,)``
        Autonomous vector field with parameters already closed over.
    y : array ``(d,)``
    dt : float
    n_substeps : int  (static)
    integrator : {"rk4", "euler"}

    Returns
    -------
    Phi : array ``(d,)``
    """
    flow = select_flow(integrator)
    return flow(drift_fn, y, dt, n_substeps) - y


def flow_step(drift_fn, y, dt, n_substeps, integrator="rk4"):
    """Displacement ``Φ`` and flow Jacobian ``J = I + ∂Φ/∂y`` at a point.

    Parameters
    ----------
    drift_fn : callable ``(d,) → (d,)``
    y : array ``(d,)``
    dt : float
    n_substeps : int  (static)
    integrator : {"rk4", "euler"}

    Returns
    -------
    Phi : array ``(d,)``
    J : array ``(d, d)``
        ``J = ∂(flow)/∂y`` — the linear tangent map of the flow.
    """
    d = y.shape[-1]

    def disp(z):
        return flow_displacement(drift_fn, z, dt, n_substeps, integrator)

    Phi = disp(y)
    J = jax.jacfwd(disp)(y) + jnp.eye(d, dtype=y.dtype)
    return Phi, J


def od_window_residuals(drift_fn, X_w, dt, n_substeps, integrator="rk4"):
    """Overdamped 2-point residuals and flow Jacobians over a position window.

    For a window ``X_w`` of ``W`` consecutive positions, returns the
    ``W-1`` residuals ``r_k = X_{k+1} − X_k − Φ_k(X_k)`` together with the
    flow Jacobians ``J_k`` at each window base point.

    Parameters
    ----------
    drift_fn : callable ``(d,) → (d,)``
    X_w : array ``(W, d)``
    dt : float
    n_substeps : int  (static)
    integrator : {"rk4", "euler"}

    Returns
    -------
    r : array ``(W-1, d)``
    J : array ``(W-1, d, d)``
    """
    def _step(y):
        return flow_step(drift_fn, y, dt, n_substeps, integrator)

    Phi, J = jax.vmap(_step)(X_w[:-1])
    r = X_w[1:] - X_w[:-1] - Phi
    return r, J


def od_instrument(drift_of_theta, theta, x_base, dt, n_substeps, integrator="rk4"):
    r"""η-clean instrument (left test function) for one overdamped residual.

    The residual ``r_c = X_{c+1} − X_c − Φ(X_c;θ)`` has regressor
    ``ψ_right = ∂r_c/∂θ = −∂Φ/∂θ|_{X_c}``, which is correlated with the
    measurement noise ``η_c`` sitting in ``X_c`` (and in ``r_c``) — the
    errors-in-variables bias.  The instrument replaces that left factor with
    the same sensitivity built from a base point ``x_base`` whose noise is
    *independent* of the residual's (the core uses the lagged ``X_{c-1}``),
    flow-propagated forward one interval so it stays geometrically aligned
    with the center:

    .. math::

        ψ_{\mathrm{inst}} = -\tfrac12\!\left(
            \left.\frac{∂Φ}{∂θ}\right|_{x_{\mathrm{base}}}
          + \left.\frac{∂Φ}{∂θ}\right|_{x_{\mathrm{base}}+Φ(x_{\mathrm{base}})}
        \right).

    The trapezoidal pair (base point and its deterministic forward image) is
    the parametric analogue of the legacy lag-1 *shift* instrument that SFI
    auto-selects under measurement noise.  The forward point is evaluated at
    the current ``θ`` (stop-gradient), so the θ-derivative is taken of the
    sensitivity *at fixed points* — a valid test function for ``∂r/∂θ``.

    Parameters
    ----------
    drift_of_theta : callable  ``θ_flat → (drift_fn : (d,) → (d,))``
    theta : array ``(n_params,)``
    x_base : array ``(d,)``
        η-clean base point (independent of the center residual's noise).
    dt, n_substeps, integrator : flow controls (as elsewhere).

    Returns
    -------
    psi_inst : array ``(d, n_params)``
    """
    def disp_theta_jac(y):
        return jax.jacfwd(
            lambda th: flow_displacement(drift_of_theta(th), y, dt, n_substeps, integrator))(theta)

    Phi0 = flow_displacement(drift_of_theta(theta), x_base, dt, n_substeps, integrator)
    x_fwd = jax.lax.stop_gradient(x_base + Phi0)
    return -0.5 * (disp_theta_jac(x_base) + disp_theta_jac(x_fwd))
