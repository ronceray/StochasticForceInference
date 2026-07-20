# TODO: review this file
"""
tests/test_dual_mask.py
========================
Test dual-mask support in Parametric SFI:
  - static mask: particle position is known (for flow evaluation)
  - dynamic mask: increment is reliable (for fitting weights)

Key scenarios:
  1. TrajectoryDataset stores and validates dynamic_mask.
  2. Backward compat: dynamic_mask=None behaves as before.
  3. Single-particle OU with explicit dynamic_mask.
  4. Multi-particle ABP with per-particle dynamic masking
     (all positions known, some increments excluded).
  5. degrade() propagates dynamic_mask.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np



# ── Helpers ──


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _simulate_ou_noisy(A, D, Lambda, dt, T, key):
    """Simulate OU + measurement noise (single particle)."""
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


def _make_ou_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def expr(x, *, params):
        return params["A"] @ x

    return make_psf(
        expr, dim=d, rank=1, n_features=1,
        params=[ParamSpec("A", shape=(d, d))],
    )


def _assert_no_nan(sfi, label=""):
    prefix = f"[{label}] " if label else ""
    for name, val in [
        ("theta", sfi.force_coefficients_full),
        ("D", sfi.diffusion_average),
        ("Lambda", sfi.Lambda),
        ("G", sfi.force_G),
    ]:
        assert bool(jnp.all(jnp.isfinite(val))), (
            f"{prefix}{name} has non-finite values"
        )


# ── Tests ──

class TestDualMaskDataset:
    """TrajectoryDataset with dynamic_mask field."""

    def test_dynamic_mask_none(self):
        """dynamic_mask=None is the default; _dynamic_M2d falls back to _M2d."""
        from SFI.trajectory import TrajectoryDataset
        ds = TrajectoryDataset.from_arrays(
            X=np.ones((10, 2)), dt=0.01,
        )
        assert ds.dynamic_mask is None
        M = ds._M2d()
        Md = ds._dynamic_M2d()
        np.testing.assert_array_equal(M, Md)

    def test_dynamic_mask_subset(self):
        """dynamic_mask is stored and must be a subset of mask."""
        from SFI.trajectory import TrajectoryDataset
        mask = np.ones((10, 2), dtype=bool)
        dyn = mask.copy()
        dyn[3, 0] = False
        dyn[7, 1] = False
        ds = TrajectoryDataset.from_arrays(
            X=np.ones((10, 2, 2)), dt=0.01,
            mask=mask, dynamic_mask=dyn,
        )
        assert ds.dynamic_mask is not None
        np.testing.assert_array_equal(ds._dynamic_M2d(), dyn)

    def test_dynamic_mask_not_subset_raises(self):
        """dynamic_mask entries outside mask should raise."""
        from SFI.trajectory import TrajectoryDataset
        mask = np.ones((10, 2), dtype=bool)
        mask[5, 0] = False
        dyn = np.ones((10, 2), dtype=bool)  # True where mask is False → invalid
        with pytest.raises(ValueError, match="subset"):
            TrajectoryDataset.from_arrays(
                X=np.ones((10, 2, 2)), dt=0.01,
                mask=mask, dynamic_mask=dyn,
            )


class TestDualMaskDegrade:
    """degrade() propagates dynamic_mask."""

    def test_degrade_preserves_dynamic_mask(self):
        from SFI.trajectory import TrajectoryDataset

        T, N, d = 100, 3, 2
        X = np.random.randn(T, N, d)
        mask = np.ones((T, N), dtype=bool)
        dyn = mask.copy()
        # mark 10 random entries as dynamically invalid
        rng = np.random.default_rng(0)
        for _ in range(10):
            t, n = rng.integers(T), rng.integers(N)
            dyn[t, n] = False

        ds = TrajectoryDataset.from_arrays(
            X=X, dt=0.01, mask=mask, dynamic_mask=dyn,
        )
        from SFI.trajectory.degrade import degrade_dataset
        ds_deg = degrade_dataset(ds, downsample=2, seed=7)

        assert ds_deg.dynamic_mask is not None
        M_out = np.asarray(ds_deg._M2d())
        Md_out = np.asarray(ds_deg._dynamic_M2d())
        # dynamic ⊆ static
        assert np.all(Md_out <= M_out)

    def test_degrade_no_dynamic_mask_stays_none(self):
        from SFI.trajectory import TrajectoryDataset

        ds = TrajectoryDataset.from_arrays(
            X=np.random.randn(100, 2), dt=0.01,
        )
        from SFI.trajectory.degrade import degrade_dataset
        ds_deg = degrade_dataset(ds, downsample=2, seed=7)
        assert ds_deg.dynamic_mask is None

    def test_degrade_data_loss_drops_both_masks(self):
        from SFI.trajectory import TrajectoryDataset

        T, N, d = 200, 2, 2
        mask = np.ones((T, N), dtype=bool)
        dyn = mask.copy()
        ds = TrajectoryDataset.from_arrays(
            X=np.random.randn(T, N, d), dt=0.01,
            mask=mask, dynamic_mask=dyn,
        )
        from SFI.trajectory.degrade import degrade_dataset
        ds_deg = degrade_dataset(ds, data_loss_fraction=0.3, seed=42)

        M_out = np.asarray(ds_deg._M2d())
        Md_out = np.asarray(ds_deg._dynamic_M2d())
        # Both masks lost the same entries
        np.testing.assert_array_equal(M_out, Md_out)
        # Some entries are False
        assert float(np.mean(M_out)) < 1.0
