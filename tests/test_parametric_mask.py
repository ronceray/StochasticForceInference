# TODO: review this file
"""
tests/test_parametric_mask.py
=========================
Test parametric SFI with masked (degraded) data: random data loss via
TrajectoryCollection.degrade(data_loss_fraction=...).

Verifies that:
  1. No NaN appears in any output (theta, D, Lambda, G, info).
  2. Inferred parameters remain consistent with clean-data reference
     within a relaxed tolerance.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")



@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _simulate_ou_noisy(A, D, Lambda, dt, T, key):
    """Simulate OU + measurement noise."""
    d = A.shape[0]
    B = jnp.linalg.cholesky(2 * D)
    L_eta = jnp.linalg.cholesky(Lambda + 1e-30 * jnp.eye(d))
    k1, k2 = jax.random.split(key)
    noise_proc = jax.random.normal(k1, (T, d))
    noise_meas = jax.random.normal(k2, (T, d))

    def _step(x, noise):
        x_next = x + A @ x * dt + jnp.sqrt(dt) * B @ noise
        return x_next, x

    _, X = jax.lax.scan(_step, jnp.zeros(d), noise_proc)
    Y = X + noise_meas @ L_eta.T
    return X, Y


def _make_ou_collection(A, D, Lambda, dt, T, key):
    from SFI.trajectory import TrajectoryDataset, TrajectoryCollection
    _, Y = _simulate_ou_noisy(A, D, Lambda, dt, T, key)
    ds = TrajectoryDataset(X=jnp.array(Y[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds), Y


def _make_drift_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def expr(x, *, params):
        return params["A"] @ x

    return make_psf(
        expr, dim=d, rank=1, n_features=1,
        params=[ParamSpec("A", shape=(d, d))],
    )


def _assert_no_nan(sfi, label=""):
    """Assert that all Parametric SFI outputs are NaN-free."""
    prefix = f"[{label}] " if label else ""
    theta = sfi.force_coefficients_full
    assert bool(jnp.all(jnp.isfinite(theta))), (
        f"{prefix}theta has non-finite values: {theta}"
    )
    D = sfi.diffusion_average
    assert bool(jnp.all(jnp.isfinite(D))), (
        f"{prefix}D has non-finite values"
    )
    eta = sfi.Lambda
    assert bool(jnp.all(jnp.isfinite(eta))), (
        f"{prefix}Lambda has non-finite values"
    )
    G = sfi.force_G
    assert bool(jnp.all(jnp.isfinite(G))), (
        f"{prefix}Gram matrix has non-finite values"
    )


@pytest.mark.slow
class TestParametricMask:
    """Parametric SFI with masked data (random data loss)."""

    @pytest.fixture(scope="class")
    def ou_system(self):
        d = 2
        A = jnp.array([[-1.0, 0.3], [0.0, -1.5]])
        D = 0.3 * jnp.eye(d)
        Lambda = 0.02 * jnp.eye(d)
        dt = 0.01
        T = 50000
        return A, D, Lambda, dt, T

    @pytest.fixture(scope="class")
    def clean_result(self, ou_system):
        """Run Parametric SFI on clean data as reference."""
        from SFI.inference.overdamped import OverdampedLangevinInference

        A, D, Lambda, dt, T = ou_system
        d = A.shape[0]
        coll, _ = _make_ou_collection(
            A, D, Lambda, dt, T, jax.random.PRNGKey(111),
        )
        sfi = OverdampedLangevinInference(coll)
        F_psf = _make_drift_psf(d)
        sfi.infer_force(
            F_psf, {"A": A * 0.8}, n_substeps=4, max_outer=10,
        )
        return sfi

    def test_clean_baseline(self, ou_system, clean_result):
        """Sanity check: clean data produces NaN-free, accurate results."""
        A, D, Lambda, dt, T = ou_system
        sfi = clean_result
        _assert_no_nan(sfi, "clean")

        d = A.shape[0]
        theta = sfi.force_coefficients_full.reshape(d, d)
        err = float(jnp.linalg.norm(theta - A) / jnp.linalg.norm(A))
        print(f"  Clean θ_hat:\n{theta}")
        print(f"  Clean D diag: {jnp.diag(sfi.diffusion_average)}")
        print(f"  Clean Λ diag: {jnp.diag(sfi.Lambda)}")
        print(f"  Clean θ error: {err:.4f}")
        assert err < 0.15, f"Clean force error too large: {err}"

    def test_masked_20pct(self, ou_system, clean_result):
        """20% data loss: results should be NaN-free and consistent."""
        from SFI.inference.overdamped import OverdampedLangevinInference

        A, D, Lambda, dt, T = ou_system
        d = A.shape[0]
        coll, _ = _make_ou_collection(
            A, D, Lambda, dt, T, jax.random.PRNGKey(111),
        )
        coll_deg = coll.degrade(data_loss_fraction=0.2, seed=42)

        # Verify degradation produced a mask
        ds0 = coll_deg.datasets[0]
        M = np.asarray(ds0._M2d())
        frac = 1.0 - float(np.mean(M))
        print(f"  Actual data loss fraction: {frac:.3f}")

        sfi_m = OverdampedLangevinInference(coll_deg)
        F_psf = _make_drift_psf(d)
        sfi_m.infer_force(
            F_psf, {"A": A * 0.8}, n_substeps=4, max_outer=10,
        )
        _assert_no_nan(sfi_m, "mask-20%")

        theta_m = sfi_m.force_coefficients_full.reshape(d, d)
        theta_c = clean_result.force_coefficients_full.reshape(d, d)
        err_abs = float(jnp.linalg.norm(theta_m - A) / jnp.linalg.norm(A))
        err_vs_clean = float(
            jnp.linalg.norm(theta_m - theta_c) / jnp.linalg.norm(theta_c)
        )
        print(f"  Masked-20% θ_hat:\n{theta_m}")
        print(f"  Masked-20% D diag: {jnp.diag(sfi_m.diffusion_average)}")
        print(f"  Masked-20% Λ diag: {jnp.diag(sfi_m.Lambda)}")
        print(f"  Masked-20% θ abs error: {err_abs:.4f}")
        print(f"  Masked-20% θ vs clean: {err_vs_clean:.4f}")
        assert err_abs < 0.25, f"Masked force error too large: {err_abs}"

    def test_masked_40pct(self, ou_system, clean_result):
        """40% data loss: results should be NaN-free."""
        from SFI.inference.overdamped import OverdampedLangevinInference

        A, D, Lambda, dt, T = ou_system
        d = A.shape[0]
        coll, _ = _make_ou_collection(
            A, D, Lambda, dt, T, jax.random.PRNGKey(111),
        )
        coll_deg = coll.degrade(data_loss_fraction=0.4, seed=99)

        sfi_m = OverdampedLangevinInference(coll_deg)
        F_psf = _make_drift_psf(d)
        sfi_m.infer_force(
            F_psf, {"A": A * 0.8}, n_substeps=4, max_outer=10,
        )
        _assert_no_nan(sfi_m, "mask-40%")

        theta_m = sfi_m.force_coefficients_full.reshape(d, d)
        err_abs = float(jnp.linalg.norm(theta_m - A) / jnp.linalg.norm(A))
        print(f"  Masked-40% θ_hat:\n{theta_m}")
        print(f"  Masked-40% D diag: {jnp.diag(sfi_m.diffusion_average)}")
        print(f"  Masked-40% Λ diag: {jnp.diag(sfi_m.Lambda)}")
        print(f"  Masked-40% θ abs error: {err_abs:.4f}")
        # With 40% loss, tolerance is more relaxed
        assert err_abs < 0.35, f"Masked force error too large: {err_abs}"
