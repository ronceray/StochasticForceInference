"""End-to-end solves on the exact banded core.

The fitted parameters must recover the ground truth within the
statistical error; the underlying kernels are pinned against dense
GLS references in ``test_parametric_core_exact_vs_dense``.
"""

import jax
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _ou(T=6000, dt=0.02, d=2, seed=7, noise=0.03):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]])[:d, :d]
    D = np.diag([0.5, 0.3][:d])
    s2D = np.sqrt(2 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + s2D @ (rng.normal(size=d) * np.sqrt(dt))
    return X + noise * rng.standard_normal(X.shape), k


def _coll(X, dt=0.02):
    import jax.numpy as jnp

    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    ds = TrajectoryDataset.from_arrays(X=jnp.asarray(X[:, None, :]), dt=dt)
    return TrajectoryCollection.from_dataset(ds)


@pytest.mark.parametrize("eiv", [False, True])
def test_exact_core_od_recovers_drift(eiv):
    import jax.numpy as jnp
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    X, k = _ou()
    coll = _coll(X)
    basis = monomials_up_to(1, dim=2, rank="vector")

    res_e = solve_force_od(coll, basis, max_outer=6, eiv=eiv)
    th_e = np.asarray(res_e.theta)
    assert np.all(np.isfinite(th_e))
    assert np.all(np.isfinite(np.asarray(res_e.theta_cov)))
    # evaluate the fitted force field F̂(x) and compare to the true −k x at
    # a few probe points (layout-agnostic)
    psf = basis.to_psf()
    struct = psf.unflatten_params(jnp.asarray(th_e))
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=2)
        Fhat = np.asarray(psf(jnp.asarray(x)[None], params=struct)[0])
        np.testing.assert_allclose(Fhat, -(k @ x), atol=0.15, rtol=0.1)


def test_exact_core_ud_recovers_drift():
    import jax.numpy as jnp

    from SFI.inference.parametric_core.solve import solve_force_ud
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    rng = np.random.default_rng(7)
    T, dt, k_t, g_t, D_t = 8000, 0.05, 1.0, 0.5, 0.1
    X = np.zeros((T, 1))
    V = np.zeros((T, 1))
    X[0], V[0] = rng.normal(0, 0.3, 1), rng.normal(0, 0.3, 1)
    for t in range(T - 1):
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k_t * X[t] - g_t * V[t]) * dt \
            + np.sqrt(2 * D_t * dt) * rng.normal(size=1)
    coll = _coll(X, dt=dt)

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    psf = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                   params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                           ParamSpec("gamma", shape=(), dtype=jnp.float64)])
    res_e = solve_force_ud(coll, psf, max_outer=6, inner="gn", eiv=False)
    th_e = np.asarray(res_e.theta)
    assert np.all(np.isfinite(th_e))
    assert abs(th_e[0] - k_t) < 0.2 and abs(th_e[1] - g_t) < 0.2