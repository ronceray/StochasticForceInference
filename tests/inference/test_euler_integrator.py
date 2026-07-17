# TODO: review this file
"""Tests for the ``integrator`` parameter in parametric inference backends.

Covers:
- OD ``infer_force`` with ``integrator="rk4"`` and ``integrator="euler"``
- UD ``infer_force`` with both integrators
- Invalid integrator raises ``ValueError``
- RK4 and Euler produce different (but both finite) results on a simple system
- Unit test: Euler n_substeps=1 constant force gives exact residual y_{k+1}-y_k-a*dt
"""
import jax.numpy as jnp
import pytest

from SFI import OverdampedLangevinInference, UnderdampedLangevinInference
from SFI import TrajectoryCollection
from SFI.statefunc import set_jit
from SFI.integrate.rk4 import euler_flow, ode_flow


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def disable_jit():
    """Speed up tests by disabling JIT compilation."""
    set_jit(False)
    yield
    set_jit(True)


def _make_od_collection(seed=0, T=200, dt=0.05):
    """Simulate a simple 1D overdamped Ornstein–Uhlenbeck process."""
    import jax
    key = jax.random.PRNGKey(seed)
    gamma, D = 1.0, 0.5
    X = jnp.zeros((T + 1, 1, 1))
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, 1, 1))
    xs = [jnp.zeros((1, 1))]
    for t in range(T):
        x = xs[-1]
        dx = -gamma * x * dt + jnp.sqrt(2 * D * dt) * noise[t]
        xs.append(x + dx)
    X = jnp.stack(xs)          # (T+1, 1, 1)
    return TrajectoryCollection.from_arrays(X=X, dt=dt)


def _make_ud_collection(seed=1, T=200, dt=0.05):
    """Simulate a simple 1D underdamped harmonic oscillator."""
    import jax
    key = jax.random.PRNGKey(seed)
    gamma, k, D = 1.0, 1.0, 0.5
    xs = [jnp.zeros((1, 1))]
    vs = [jnp.zeros((1, 1))]
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, 1, 1))
    for t in range(T):
        x, v = xs[-1], vs[-1]
        f = -k * x - gamma * v
        xn = x + v * dt
        vn = v + f * dt + jnp.sqrt(2 * D * dt) * noise[t]
        xs.append(xn)
        vs.append(vn)
    X = jnp.stack(xs)          # (T+1, 1, 1)
    return TrajectoryCollection.from_arrays(X=X, dt=dt)


def _make_od_psf():
    from SFI.bases import monomials_up_to
    return monomials_up_to(1, dim=1, rank="vector").to_psf()


def _od_psf():
    """Return a simple OD PSF (linear in x, vectorial)."""
    from SFI.bases import monomials_up_to
    return monomials_up_to(1, dim=1, rank="vector").to_psf()


def _ud_psf():
    """Return a simple UD PSF (linear in x,v)."""
    from SFI.bases import monomials_up_to
    return monomials_up_to(1, dim=1, rank="vector", include_v=True).to_psf()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestODIntegratorParam:
    def test_rk4_runs(self):
        col = _make_od_collection()
        inf = OverdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="WeakNoise")
        F_psf = _od_psf()
        inf.infer_force(F_psf, n_substeps=2, integrator="rk4")
        assert inf.force_inferred is not None

    def test_euler_runs(self):
        col = _make_od_collection()
        inf = OverdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="WeakNoise")
        F_psf = _od_psf()
        inf.infer_force(F_psf, n_substeps=2, integrator="euler")
        assert inf.force_inferred is not None

    def test_invalid_integrator_raises(self):
        col = _make_od_collection()
        inf = OverdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="WeakNoise")
        F_psf = _od_psf()
        with pytest.raises(ValueError, match="integrator"):
            inf.infer_force(F_psf, n_substeps=2, integrator="heun")

    def test_rk4_euler_differ(self):
        """RK4 and Euler should give different (non-identical) results."""
        col = _make_od_collection()
        F_psf = _od_psf()

        inf_rk4 = OverdampedLangevinInference(col)
        inf_rk4.compute_diffusion_constant(method="WeakNoise")
        inf_rk4.infer_force(F_psf, n_substeps=2, integrator="rk4")

        inf_euler = OverdampedLangevinInference(col)
        inf_euler.compute_diffusion_constant(method="WeakNoise")
        inf_euler.infer_force(F_psf, n_substeps=2, integrator="euler")

        theta_rk4 = inf_rk4.force_inferred.flatten_params()
        theta_euler = inf_euler.force_inferred.flatten_params()

        # Both should be finite
        assert jnp.all(jnp.isfinite(theta_rk4))
        assert jnp.all(jnp.isfinite(theta_euler))
        # But not identical
        assert not jnp.allclose(theta_rk4, theta_euler)

    def test_default_is_rk4(self):
        """Default integrator should match explicit rk4."""
        col = _make_od_collection()
        F_psf = _od_psf()

        inf_default = OverdampedLangevinInference(col)
        inf_default.compute_diffusion_constant(method="WeakNoise")
        inf_default.infer_force(F_psf, n_substeps=2)

        inf_rk4 = OverdampedLangevinInference(col)
        inf_rk4.compute_diffusion_constant(method="WeakNoise")
        inf_rk4.infer_force(F_psf, n_substeps=2, integrator="rk4")

        assert jnp.allclose(
            inf_default.force_inferred.flatten_params(),
            inf_rk4.force_inferred.flatten_params(),
        )


class TestUDIntegratorParam:
    def test_rk4_runs(self):
        col = _make_ud_collection()
        inf = UnderdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="auto")
        F_psf = _ud_psf()
        inf.infer_force(F_psf, n_substeps=2, integrator="rk4")
        assert inf.force_inferred is not None

    def test_euler_runs(self):
        col = _make_ud_collection()
        inf = UnderdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="auto")
        F_psf = _ud_psf()
        inf.infer_force(F_psf, n_substeps=2, integrator="euler")
        assert inf.force_inferred is not None

    def test_invalid_integrator_raises(self):
        col = _make_ud_collection()
        inf = UnderdampedLangevinInference(col)
        inf.compute_diffusion_constant(method="auto")
        F_psf = _ud_psf()
        with pytest.raises(ValueError, match="integrator"):
            inf.infer_force(F_psf, n_substeps=2, integrator="bogus")


# ── Unit-level numerical checks ───────────────────────────────────────────────

class TestEulerFlowNumerics:
    """Low-level checks that euler_flow gives exact results for simple cases."""

    def test_constant_force_residual(self):
        """Euler with n_substeps=1 and constant force gives y_{k+1} - y_k - a*dt exactly."""
        a = jnp.array([1.5, -0.3])
        x0 = jnp.array([2.0, -1.0])
        x1 = jnp.array([2.2, -0.8])  # arbitrary next position
        dt = 0.1

        x_flow = euler_flow(lambda x: a, x0, dt, n_substeps=1)
        residual = x1 - x_flow

        expected_residual = x1 - x0 - a * dt
        assert jnp.allclose(residual, expected_residual, atol=1e-7)

    def test_euler_matches_rk4_for_linear_flow(self):
        """For linear ODE dx/dt = A x with one substep, Euler and RK4 differ by O(h²)."""
        x0 = jnp.array([1.0, 0.5])
        dt = 1e-4   # tiny dt so Euler ≈ RK4

        def f(x):
            return jnp.array([-x[0], x[1]])  # dx=-x, dy=y

        x_euler = euler_flow(f, x0, dt, n_substeps=1)
        x_rk4 = ode_flow(f, x0, dt, n_substeps=1)
        # At tiny dt, difference should be O(dt²) ≈ 1e-8
        assert jnp.allclose(x_euler, x_rk4, atol=1e-6)

    def test_euler_n_substeps_refines(self):
        """More substeps improves Euler accuracy for nonlinear force."""
        x0 = jnp.array([1.0])
        dt = 0.5
        f = lambda x: -x   # true solution: exp(-t)
        x_true = x0 * jnp.exp(-dt)

        x_1 = euler_flow(f, x0, dt, n_substeps=1)
        x_10 = euler_flow(f, x0, dt, n_substeps=10)

        err_1 = jnp.abs(x_1 - x_true)
        err_10 = jnp.abs(x_10 - x_true)
        assert err_10 < err_1, "More substeps should reduce Euler error"
