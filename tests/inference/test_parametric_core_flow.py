"""Tests for SFI.inference.parametric_core.flow — RK4/Euler flow, Jacobian, residuals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _linear_drift(A):
    def drift(x):
        return A @ x
    return drift


def test_euler_single_step_linear_drift_is_exact():
    """Euler, 1 substep, linear drift F(x)=Ax: Phi = A y dt, J = I + A dt (exact)."""
    from SFI.inference.parametric_core.flow import flow_step

    A = jnp.array([[0.0, 1.0], [-1.0, -0.3]])
    y = jnp.array([0.7, -0.2])
    dt = 0.05

    Phi, J = flow_step(_linear_drift(A), y, dt, n_substeps=1, integrator="euler")

    np.testing.assert_allclose(np.asarray(Phi), np.asarray(A @ y * dt), atol=1e-12)
    np.testing.assert_allclose(np.asarray(J), np.asarray(jnp.eye(2) + A * dt), atol=1e-12)


def test_rk4_flow_matches_matrix_exponential_for_linear_drift():
    """RK4 with enough substeps reproduces the exact linear flow expm(A dt)."""
    from SFI.inference.parametric_core.flow import flow_step

    A = jnp.array([[0.0, 1.0], [-1.0, -0.3]])
    y = jnp.array([0.7, -0.2])
    dt = 0.1

    Phi, J = flow_step(_linear_drift(A), y, dt, n_substeps=8, integrator="rk4")
    eAdt = expm(A * dt)

    np.testing.assert_allclose(np.asarray(J), np.asarray(eAdt), atol=1e-7)
    np.testing.assert_allclose(np.asarray(y + Phi), np.asarray(eAdt @ y), atol=1e-7)


def test_flow_jacobian_matches_finite_difference_nonlinear():
    """Flow Jacobian J = I + dPhi/dy matches finite differences for a nonlinear drift."""
    from SFI.inference.parametric_core.flow import flow_step

    def drift(x):
        return jnp.array([x[1], -jnp.sin(x[0]) - 0.2 * x[1]])  # pendulum

    y = jnp.array([0.4, -0.1])
    dt = 0.05
    _, J = flow_step(drift, y, dt, n_substeps=4, integrator="rk4")

    def disp(z):
        from SFI.integrate.rk4 import ode_flow
        return ode_flow(drift, z, dt, 4) - z

    eps = 1e-6
    J_fd = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        J_fd[:, j] = np.asarray((disp(y + jnp.array(e)) - disp(y - jnp.array(e))) / (2 * eps))
    J_fd += np.eye(2)

    np.testing.assert_allclose(np.asarray(J), J_fd, atol=1e-6)


def test_od_window_residuals_shapes_and_values():
    """od_window_residuals returns r, J over a window with correct shapes/values."""
    from SFI.inference.parametric_core.flow import od_window_residuals

    A = jnp.array([[0.0, 1.0], [-1.0, -0.3]])
    # 4-position window
    X_w = jnp.array([[0.1, 0.2], [0.15, 0.18], [0.19, 0.14], [0.21, 0.09]])
    dt = 0.05

    r, J = od_window_residuals(_linear_drift(A), X_w, dt, n_substeps=1, integrator="euler")

    # 3 residuals for a 4-position window
    assert r.shape == (3, 2)
    assert J.shape == (3, 2, 2)
    # residual r_k = X_{k+1} - X_k - Phi_k, with Phi_k = A X_k dt (euler)
    Phi = jnp.einsum("ij,kj->ki", A, X_w[:-1]) * dt
    r_expect = X_w[1:] - X_w[:-1] - Phi
    np.testing.assert_allclose(np.asarray(r), np.asarray(r_expect), atol=1e-12)
