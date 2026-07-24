"""Tests for SFI.diagnostics.parametric_four_point_diagnostic.

Under correct model specification and i.i.d. measurement noise, midpoint
flow residuals separated by two steps share no noise source, so their
cross-covariance should vanish.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_ou_data(T=500, dt=0.02, d=2, D_diag=None, sigma_noise=0.0, seed=42):
    """Simulate an OU process: dX = -k X dt + sqrt(2D) dW, with optional
    measurement noise η ~ N(0, σ² I)."""
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    if D_diag is None:
        D_diag = np.array([0.5, 0.3], dtype=np.float64)[:d]
    D = np.diag(D_diag)
    sqrt2D = np.sqrt(2.0 * D)

    X = np.zeros((T, d), dtype=np.float64)
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        dW = rng.normal(size=d) * np.sqrt(dt)
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ dW

    Y = X.copy()
    if sigma_noise > 0:
        Y = Y + rng.normal(scale=sigma_noise, size=Y.shape)

    return Y, k, D, sigma_noise


def _make_collection(Y, dt):
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset
    ds = TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


class TestFourPointDiagnostic:

    def test_diagnostic_correct_model(self):
        """With correct model and no noise, C_02 should be near zero."""
        from SFI.diagnostics import parametric_four_point_diagnostic

        d = 2
        dt = 0.02
        Y, k, D_true, _ = _make_ou_data(T=2000, dt=dt, d=d, sigma_noise=0.0, seed=77)
        coll = _make_collection(Y, dt)

        drift_fn = lambda x: -k @ x
        result = parametric_four_point_diagnostic(coll, drift_fn, dt, n_substeps=4)

        assert "C_02" in result
        assert "frobenius_norm" in result
        # With long clean trajectory and correct model, C_02 ≈ 0
        assert result["frobenius_norm"] < 0.05

    def test_diagnostic_with_noise(self):
        """With i.i.d. noise and correct model, C_02 still ≈ 0
        (only consecutive residuals are correlated)."""
        from SFI.diagnostics import parametric_four_point_diagnostic

        d = 2
        dt = 0.02
        Y, k, _, _ = _make_ou_data(T=3000, dt=dt, d=d, sigma_noise=0.1, seed=88)
        coll = _make_collection(Y, dt)

        drift_fn = lambda x: -k @ x
        result = parametric_four_point_diagnostic(coll, drift_fn, dt, n_substeps=4)

        # With correct model and i.i.d. noise, C_02 should be small
        assert result["frobenius_norm"] < 0.1
