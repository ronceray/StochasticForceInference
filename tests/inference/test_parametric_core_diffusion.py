"""Tests for SFI.inference.parametric_core.solve_diffusion_od (state-dependent D)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest




@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` leaks float64 into every test
    collected later in the session (order-dependent numerics)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_ou_1d(T=12000, dt=0.01, k=1.0, D0=0.5, seed=0):
    rng = np.random.default_rng(seed)
    s = np.sqrt(2 * D0)
    X = np.zeros((T, 1))
    X[0] = rng.normal(0, np.sqrt(D0 / k))
    for t in range(T - 1):
        X[t + 1] = X[t] - k * X[t] * dt + s * rng.normal() * np.sqrt(dt)
    return X


def _make_collection(Y, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _linear_force_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x
    return make_psf(f, dim=1, rank=1, n_features=1, params=[ParamSpec("A", shape=(1, 1), dtype=jnp.float64)])


def _quad_D_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return jnp.array([[params["c0"] + params["c1"] * x[0] ** 2]])
    return make_psf(
        f, dim=1, rank=2, n_features=1,
        params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                ParamSpec("c1", shape=(), dtype=jnp.float64)],
    )


def test_solve_diffusion_recovers_constant_D_with_quadratic_model():
    """On constant-D data, the quadratic D(x)=c0+c1 x² recovers c0≈D, c1≈0."""
    from SFI.inference.parametric_core.solve import solve_diffusion_od

    k, D0 = 1.0, 0.5
    X = _make_ou_1d(T=20000, dt=0.01, k=k, D0=D0, seed=1)
    coll = _make_collection(X, 0.01)
    F_psf = _linear_force_psf()
    theta_F = jnp.asarray(F_psf.flatten_params({"A": jnp.array([[-k]])}), dtype=jnp.float64)
    D_psf = _quad_D_psf()

    res = solve_diffusion_od(
        coll, F_psf, theta_F, D_psf, Lambda=jnp.array([[1e-8]]),
        theta_D0={"c0": 0.4, "c1": 0.0}, n_substeps=4, maxiter=100)
    p = D_psf.unflatten_params(res.theta_D)
    c0, c1 = float(np.asarray(p["c0"])), float(np.asarray(p["c1"]))

    assert abs(c0 - D0) / D0 < 0.10, f"c0={c0} vs D0={D0}"
    assert abs(c1) < 0.10, f"c1={c1} should be ~0 (constant D)"


def test_lbfgs_backtracks_through_nan_region():
    """_lbfgs must not abort at x0 when a line-search trial hits NaN.

    Regression for the silent stuck-at-init failure: an infeasible trial
    point (e.g. non-PSD diffusion proposal → Cholesky NaN) used to abort
    scipy's line search and return the initial point unchanged.
    """
    from SFI.inference.parametric_core.solve import _lbfgs

    def obj_and_grad(x):
        # Quadratic with minimum at 0.9, NaN for x > 1 (infeasible region).
        v = jnp.where(x[0] > 1.0, jnp.nan, (x[0] - 0.9) ** 2)
        g = jnp.where(x[0] > 1.0, jnp.nan, 2.0 * (x[0] - 0.9))
        return v, g.reshape(1)

    # Steep start: the first line-search trial overshoots into the NaN region.
    x0 = jnp.array([-50.0])
    x, _ = _lbfgs(obj_and_grad, x0, jnp.float64, maxiter=200)
    assert abs(float(x[0]) - 0.9) < 1e-4, f"stuck at {float(x[0])} (started at -50)"


def test_infer_diffusion_engine_wiring():
    """infer_force then infer_diffusion wires a usable diffusion field."""
    from SFI.inference.overdamped import OverdampedLangevinInference

    k, D0 = 1.0, 0.5
    X = _make_ou_1d(T=12000, dt=0.01, k=k, D0=D0, seed=2)
    coll = _make_collection(X, 0.01)
    inf = OverdampedLangevinInference(coll)
    inf.infer_force(_linear_force_psf(), n_substeps=4, max_outer=5)
    inf.infer_diffusion()  # default symmetric_matrix_basis → constant D

    assert inf.metadata["diffusion_method"] == "parametric"
    D_eval = np.asarray(inf.diffusion_inferred(jnp.array([[0.3]])))
    assert D_eval.shape[-2:] == (1, 1)
    assert abs(float(D_eval.ravel()[0]) - D0) / D0 < 0.15, f"D={D_eval} vs {D0}"
