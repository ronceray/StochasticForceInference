"""
Regression tests for issue #18:
simulate_bootstrapped_trajectory fails when original data has NaN fill values at masked positions.

Root cause: peek_row returns first in-bounds index, which may be a masked (NaN) time step.
The NaN propagates through simulation, then _traj_to_collection raises "X contains Nans
in unmasked positions" because the simulated dataset has no mask.

Fix: find first fully-finite X row for the initial condition.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


def _make_ou_trajectory(T=300, d=2, dt=0.01, k=1.0, D=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, d), dtype=np.float32)
    for t in range(T - 1):
        X[t + 1] = X[t] - k * X[t] * dt + np.sqrt(2 * D * dt) * rng.normal(size=d).astype(np.float32)
    return X


def _make_masked_collection(T=300, d=2, dt=0.01, n_masked_start=10, seed=0):
    """
    Trajectory where the first ``n_masked_start`` time steps are masked.
    Masked positions carry NaN fill values, as a user would store them.
    """
    X = _make_ou_trajectory(T=T, d=d, dt=dt, seed=seed)
    mask = np.ones(T, dtype=bool)
    mask[:n_masked_start] = False

    X_with_nan = X.copy()
    X_with_nan[~mask] = np.nan

    ds = TrajectoryDataset.from_arrays(X=X_with_nan, dt=dt, mask=mask)
    return TrajectoryCollection.from_dataset(ds)


def _linear_basis(d):
    from SFI.statefunc.factory import make_basis
    def f(x, **kw):
        return x[..., :, None] * jnp.eye(d, dtype=x.dtype)
    return make_basis(f, dim=d, rank=1, n_features=d)


# ---------------------------------------------------------------------------
# Overdamped
# ---------------------------------------------------------------------------

class TestSimulateBootstrappedOverdamped:
    def test_simulate_does_not_raise_with_nan_masked_start(self):
        """simulate_bootstrapped_trajectory must not fail when t=0 is masked (NaN)."""
        d = 2
        coll = _make_masked_collection(T=300, d=d, n_masked_start=10)

        inf = OverdampedLangevinInference(coll)
        basis = _linear_basis(d)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(basis)

        key = jax.random.PRNGKey(0)
        result, proc = inf.simulate_bootstrapped_trajectory(key)

        X_sim = np.asarray(result.datasets[0].X)
        assert np.all(np.isfinite(X_sim)), "Simulated trajectory should contain no NaN"

    def test_simulate_unmasked_data_unchanged(self):
        """simulate_bootstrapped_trajectory should work identically when data has no NaN."""
        d = 2
        X = _make_ou_trajectory(T=300, d=d, seed=1)
        ds = TrajectoryDataset.from_arrays(X=X, dt=0.01)
        coll = TrajectoryCollection.from_dataset(ds)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(_linear_basis(d))

        key = jax.random.PRNGKey(1)
        result, _ = inf.simulate_bootstrapped_trajectory(key)

        X_sim = np.asarray(result.datasets[0].X)
        assert np.all(np.isfinite(X_sim))

    def test_simulate_with_scattered_nan_mask(self):
        """Masked positions scattered throughout the trajectory (not just the start)."""
        d = 2
        T = 400
        dt = 0.01
        rng = np.random.default_rng(7)
        X = _make_ou_trajectory(T=T, d=d, dt=dt, seed=7)

        # Mask ~30% of time steps, including t=0
        mask = rng.random(T) > 0.3
        mask[0] = False  # ensure t=0 is masked

        X_with_nan = X.copy()
        X_with_nan[~mask] = np.nan

        ds = TrajectoryDataset.from_arrays(X=X_with_nan, dt=dt, mask=mask)
        coll = TrajectoryCollection.from_dataset(ds)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(_linear_basis(d))

        key = jax.random.PRNGKey(7)
        result, _ = inf.simulate_bootstrapped_trajectory(key)

        X_sim = np.asarray(result.datasets[0].X)
        assert np.all(np.isfinite(X_sim))
