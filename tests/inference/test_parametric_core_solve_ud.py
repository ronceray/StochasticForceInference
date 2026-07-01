"""End-to-end test for SFI.inference.parametric_core.solve_force_ud (underdamped)."""

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


def _simulate_ud_harmonic(N=8000, dt=0.05, d=1, seed=42, k=1.0, gamma=0.5, D=0.1):
    rng = np.random.default_rng(seed)
    sqrt2D = np.sqrt(2 * D)
    X = np.zeros((N, d)); V = np.zeros((N, d))
    X[0] = rng.normal(0, 0.3, size=d); V[0] = rng.normal(0, 0.3, size=d)
    for t in range(N - 1):
        dW = rng.normal(size=d) * np.sqrt(dt)
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k * X[t] - gamma * V[t]) * dt + sqrt2D * dW
    return X


def _make_ud_collection(X_pos, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X_pos[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _make_ud_linear_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    return make_psf(
        f, dim=d, rank=1, n_features=1, needs_v=True,
        params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                ParamSpec("gamma", shape=(), dtype=jnp.float64)],
    )


def test_solve_force_ud_recovers_harmonic():
    from SFI.inference.parametric_core.solve import solve_force_ud

    k_true, gamma_true, D_true = 1.0, 0.5, 0.1
    X = _simulate_ud_harmonic(N=12000, dt=0.05, d=1, seed=7,
                              k=k_true, gamma=gamma_true, D=D_true)
    coll = _make_ud_collection(X, dt=0.05)
    F_psf = _make_ud_linear_psf(1)

    res = solve_force_ud(coll, F_psf, n_substeps=4, max_outer=6)
    p = F_psf.unflatten_params(res.theta)
    k_hat = float(np.asarray(p["k"])); g_hat = float(np.asarray(p["gamma"]))
    D_hat = float(np.asarray(res.D).ravel()[0])

    assert abs(k_hat - k_true) / k_true < 0.15, f"k: {k_hat} vs {k_true}"
    assert abs(g_hat - gamma_true) / gamma_true < 0.30, f"gamma: {g_hat} vs {gamma_true}"
    assert abs(D_hat - D_true) / D_true < 0.30, f"D: {D_hat} vs {D_true}"


def test_solve_force_ud_gn_matches_lbfgs_clean():
    """On CLEAN underdamped data the direct-GN inner solver reaches the same
    fixed point as L-BFGS — proving the UD GN path is correct, so any
    high-noise divergence is the (deferred) velocity-EIV bias, not a GN bug."""
    from SFI.inference.parametric_core.solve import solve_force_ud

    X = _simulate_ud_harmonic(N=12000, dt=0.05, d=1, seed=11, k=1.0, gamma=0.5, D=0.1)
    coll = _make_ud_collection(X, dt=0.05)

    th_lbfgs = np.asarray(solve_force_ud(coll, _make_ud_linear_psf(1), n_substeps=4, max_outer=6, inner="lbfgs", eiv=False).theta)
    res_gn = solve_force_ud(coll, _make_ud_linear_psf(1), n_substeps=4, max_outer=6, inner="gn", eiv=False)
    th_gn = np.asarray(res_gn.theta)

    rel = np.linalg.norm(th_gn - th_lbfgs) / (np.linalg.norm(th_lbfgs) + 1e-30)
    assert rel < 5e-2, f"UD GN vs L-BFGS θ disagree on clean data: rel={rel:.3e}\n gn={th_gn}\n lbfgs={th_lbfgs}"
    assert res_gn.info.get("inner") == "gn"


def test_infer_force_ud_engine_wiring():
    """UnderdampedLangevinInference.infer_force wires a usable result."""
    from SFI.inference.underdamped import UnderdampedLangevinInference

    X = _simulate_ud_harmonic(N=8000, dt=0.05, d=1, seed=3)
    coll = _make_ud_collection(X, dt=0.05)
    inf = UnderdampedLangevinInference(coll)
    inf.infer_force(_make_ud_linear_psf(1), n_substeps=4, max_outer=5)

    assert inf.metadata["force_method"] == "parametric"
    assert inf.force_G.shape == (2, 2)
    # callable inferred force field
    Fxv = inf.force_inferred(jnp.array([[1.0]]), v=jnp.array([[0.0]]))
    assert np.all(np.isfinite(np.asarray(Fxv)))
    assert inf.force_inferred.param_cov is not None


def _quad_vD_psf():
    """Velocity-dependent diffusion D(x,v) = c0 + c1 v² (1x1, needs_v)."""
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return jnp.array([[params["c0"] + params["c1"] * v[0] ** 2]])
    return make_psf(
        f, dim=1, rank=2, n_features=1, needs_v=True,
        params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                ParamSpec("c1", shape=(), dtype=jnp.float64)],
    )


def test_infer_diffusion_ud_velocity_model():
    """UD engine: infer_force then infer_diffusion with D(x,v)=c0+c1 v²
    recovers c0≈D, c1≈0 on constant-D data (uses the consistent force-solved θ)."""
    from SFI.inference.underdamped import UnderdampedLangevinInference

    D_true = 0.1
    X = _simulate_ud_harmonic(N=16000, dt=0.05, d=1, seed=4, k=1.0, gamma=0.5, D=D_true)
    coll = _make_ud_collection(X, dt=0.05)
    inf = UnderdampedLangevinInference(coll)
    inf.infer_force(_make_ud_linear_psf(1), n_substeps=4, max_outer=6)
    inf.infer_diffusion(_quad_vD_psf(), theta_D0={"c0": 0.08, "c1": 0.0})

    assert inf.metadata["diffusion_method"] == "parametric"
    p = _quad_vD_psf().unflatten_params(inf.diffusion_coefficients)
    c0, c1 = float(np.asarray(p["c0"])), float(np.asarray(p["c1"]))
    assert abs(c0 - D_true) / D_true < 0.20, f"c0={c0} vs D={D_true}"
    assert abs(c1) < 0.05, f"c1={c1} should be ~0 (constant D)"
    # diffusion field is callable with velocity
    Dxv = np.asarray(inf.diffusion_inferred(jnp.array([[0.2]]), v=jnp.array([[0.3]])))
    assert Dxv.shape[-2:] == (1, 1) and np.all(np.isfinite(Dxv))
