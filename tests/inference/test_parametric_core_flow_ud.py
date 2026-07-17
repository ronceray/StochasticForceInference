"""Tests for SFI.inference.parametric_core.flow_ud — phase-space flow + shooting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _harmonic_force(k, gamma):
    def F(x, v):
        return -k * x - gamma * v
    return F


def test_phase_space_jacobian_matches_expm_for_linear_ud():
    """Linear UD force F=-kx-γv: phase-space Jacobian blocks == expm(A_phase·dt)."""
    from SFI.inference.parametric_core.flow_ud import phase_space_flow_jac

    k, gamma, dt = 1.3, 0.4, 0.1
    x = jnp.array([0.5]); v = jnp.array([-0.2])
    F = _harmonic_force(k, gamma)
    _, _, Jxx, Jxv, Jvx, Jvv = phase_space_flow_jac(F, x, v, dt, n_substeps=8, integrator="rk4")

    A_phase = jnp.array([[0.0, 1.0], [-k, -gamma]])
    E = np.asarray(expm(A_phase * dt))
    np.testing.assert_allclose(np.asarray(Jxx).ravel()[0], E[0, 0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(Jxv).ravel()[0], E[0, 1], atol=1e-7)
    np.testing.assert_allclose(np.asarray(Jvx).ravel()[0], E[1, 0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(Jvv).ravel()[0], E[1, 1], atol=1e-7)


def _deterministic_ud_trajectory(F, x0, v0, dt, n_sub, T):
    """Integrate the deterministic UD flow to make a noiseless position series."""
    from SFI.inference.parametric_core.flow_ud import phase_space_flow_jac
    xs = [x0]
    x, v = x0, v0
    for _ in range(T - 1):
        x_next, v_next, *_ = phase_space_flow_jac(F, x, v, dt, n_sub, "rk4")
        xs.append(x_next)
        x, v = x_next, v_next
    return jnp.stack(xs, axis=0)  # (T, d)


def test_shooting_recovers_velocity_on_deterministic_trajectory():
    from SFI.inference.parametric_core.flow_ud import phase_space_flow_jac, shooting_velocity

    k, gamma, dt, n_sub = 1.0, 0.3, 0.05, 4
    F = _harmonic_force(k, gamma)
    x0, v0 = jnp.array([0.4]), jnp.array([0.1])
    Y = _deterministic_ud_trajectory(F, x0, v0, dt, n_sub, T=20)

    # the true velocity at step 1 is v after one flow from (x0, v0)
    _, v_true_1, *_ = phase_space_flow_jac(F, x0, v0, dt, n_sub, "rk4")
    v_hat_1, *_ = shooting_velocity(F, Y[0], Y[1], dt, n_sub, "rk4")
    np.testing.assert_allclose(np.asarray(v_hat_1), np.asarray(v_true_1), atol=1e-6)


def test_ud_residuals_vanish_on_deterministic_trajectory():
    """At the true force, the 3-point shooting residuals are ~0 (noiseless)."""
    from SFI.inference.parametric_core.flow_ud import ud_window_residuals

    k, gamma, dt, n_sub = 1.0, 0.3, 0.05, 4
    F = _harmonic_force(k, gamma)
    Y = _deterministic_ud_trajectory(F, jnp.array([0.4]), jnp.array([0.1]), dt, n_sub, T=12)

    r, ap, a0, am, Jvv, vhat = ud_window_residuals(F, Y, dt, n_sub, "rk4")
    assert r.shape == (Y.shape[0] - 2, 1)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-7)
    # alpha_plus == I by construction
    np.testing.assert_allclose(np.asarray(ap), np.broadcast_to(np.eye(1), ap.shape), atol=1e-10)
