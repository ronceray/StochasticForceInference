"""Engine-integration tests: OverdampedLangevinInference.infer_force (minimal parametric)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_ou_data(T=8000, dt=0.01, d=2, seed=42):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    D = np.diag(np.array([0.5, 0.3], dtype=np.float64)[:d])
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ (rng.normal(size=d) * np.sqrt(dt))
    return X, k, D


def _make_collection(Y, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _make_linear_drift_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x

    return make_psf(
        f, dim=d, rank=1, n_features=1,
        params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)],
    )


def test_infer_force_wires_result_surface():
    from SFI.inference.overdamped import OverdampedLangevinInference

    X, k, _ = _make_ou_data()
    coll = _make_collection(X, 0.01)
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_make_linear_drift_psf(2), n_substeps=4, max_outer=5)

    # standard result surface present
    assert inf.metadata["force_method"] == "parametric"
    assert inf.force_G.shape == (4, 4)
    assert inf.force_coefficients_full.shape == (4,)

    # force_inferred is a usable callable field that recovers the drift
    Fx = inf.force_inferred(jnp.array([[1.0, 0.0]]))
    np.testing.assert_allclose(np.asarray(Fx).ravel(), [-1.0, -0.1], atol=0.2)

    # parameter covariance available
    assert inf.force_inferred.param_cov is not None

    # the profiled constant-D field is exposed as a callable, so the full
    # result surface (incl. simulate_bootstrapped_trajectory) works without
    # a separate compute_diffusion_constant() call
    assert hasattr(inf, "diffusion_inferred")
    Dx = inf.diffusion_inferred(jnp.array([[0.3, -0.2]]))
    assert np.all(np.isfinite(np.asarray(Dx)))

    # downstream error analysis runs without error
    inf.compute_force_error()
    assert hasattr(inf, "force_inferred")
