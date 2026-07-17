"""Stage-5: nonlinear-in-θ (PSF) force solves on the exact core.

* ``inner="auto"`` resolves PSFs to damped Gauss–Newton (the exact Gram
  is PSF-capable via the θ-recursion's AD fallback), matching the
  ground truth;
* explicit ``inner="lbfgs"`` runs direct exact-NLL L-BFGS (the stub that
  used to raise) and agrees with the GN fit;
* the underdamped PSF path solves through the same machinery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _coll(X, dt):
    ds = TrajectoryDataset.from_arrays(X=jnp.asarray(X[:, None, :]), dt=dt)
    return TrajectoryCollection.from_dataset(ds)


def _tanh_psf():
    """Genuinely nonlinear in θ: F(x) = −a·x + b·tanh(c·x)."""
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params, mask=None, extras=None):
        return -params["a"] * x + params["b"] * jnp.tanh(params["c"] * x)

    return make_psf(f, dim=1, rank=1, n_features=1,
                    params=[ParamSpec("a", shape=(), dtype=jnp.float64),
                            ParamSpec("b", shape=(), dtype=jnp.float64),
                            ParamSpec("c", shape=(), dtype=jnp.float64)])


def _tanh_data(T=20000, dt=0.02, seed=0, a=2.0, b=1.5, c=2.0, D=0.4):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 1))
    for t in range(T - 1):
        F = -a * X[t] + b * np.tanh(c * X[t])
        X[t + 1] = X[t] + F * dt + np.sqrt(2 * D * dt) * rng.normal()
    return X


THETA0 = {"a": 1.5, "b": 1.0, "c": 1.5}
TRUTH = np.array([2.0, 1.5, 2.0])


def test_psf_exact_gn_recovers_truth():
    from SFI.inference.parametric_core.solve import solve_force_od

    psf = _tanh_psf()
    X = _tanh_data(seed=1)
    coll = _coll(X, 0.02)

    res_e = solve_force_od(coll, psf, theta0=THETA0, max_outer=8, eiv=False)
    assert res_e.info["inner"] == "gn"          # PSF-auto → damped GN
    th_e = np.asarray(res_e.theta)
    sig = np.sqrt(np.abs(np.diag(np.asarray(res_e.theta_cov))))
    assert np.all(np.isfinite(th_e))
    # the nonlinear tanh PSF recovers (a, b, c) within the error bars
    assert np.all(np.abs(th_e - TRUTH) < 4 * sig + 0.15), (th_e, sig)


def test_psf_exact_explicit_lbfgs_agrees_with_gn():
    from SFI.inference.parametric_core.solve import solve_force_od

    psf = _tanh_psf()
    X = _tanh_data(T=8000, seed=2)
    coll = _coll(X, 0.02)

    res_gn = solve_force_od(coll, psf, theta0=THETA0, max_outer=8,
                            eiv=False, _core="exact")
    res_lb = solve_force_od(coll, psf, theta0=THETA0, max_outer=8,
                            inner="lbfgs", eiv=False, _core="exact")
    assert res_lb.info["inner"] == "lbfgs"
    th_gn, th_lb = np.asarray(res_gn.theta), np.asarray(res_lb.theta)
    sig = np.sqrt(np.abs(np.diag(np.asarray(res_gn.theta_cov))))
    assert np.all(np.isfinite(th_lb))
    # quasi-score GN root vs full-likelihood optimum: same estimate to
    # well within the statistical error on this problem
    assert np.all(np.abs(th_gn - th_lb) < 1.0 * sig + 0.05), (th_gn, th_lb,
                                                              sig)


def test_psf_exact_eiv_stays_active():
    """PSF models keep the errors-in-variables instrument on the
    Gauss–Newton path (explicit ``inner="lbfgs"`` downgrades w→0)."""
    from SFI.inference.parametric_core.solve import solve_force_od

    psf = _tanh_psf()
    rng = np.random.default_rng(3)
    X = _tanh_data(T=12000, seed=3) + 0.02 * rng.standard_normal((12000, 1))
    coll = _coll(X, 0.02)
    res = solve_force_od(coll, psf, theta0=THETA0, max_outer=8, eiv=True,
                         _core="exact")
    assert float(res.info["eiv_w"]) == 1.0
    th = np.asarray(res.theta)
    sig = np.sqrt(np.abs(np.diag(np.asarray(res.theta_cov))))
    assert np.all(np.isfinite(th))
    assert np.all(np.abs(th - TRUTH) < 4 * sig + 0.3), (th, sig)


def test_psf_exact_ud_lbfgs_redirects_to_gn():
    """The UD frozen-precision quadratic is unbounded along the damping
    direction on the exact core (the shooting velocity re-solved at θ_live
    absorbs the misfit of over-damped models), so an explicit
    ``inner="lbfgs"`` warns and solves via damped Gauss–Newton — which
    recovers (k, γ) where the naive objective drifted to γ̂ ≈ 2×."""
    from SFI.inference.parametric_core.solve import solve_force_ud
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    rng = np.random.default_rng(4)
    T, dt, k_t, g_t, D_t = 8000, 0.05, 1.0, 0.5, 0.1
    X = np.zeros((T, 1))
    xi, vi = 0.3, 0.0
    X[0] = xi
    for t in range(T - 1):
        X[t + 1] = xi = xi + vi * dt
        vi = vi + (-k_t * xi - g_t * vi) * dt + np.sqrt(2 * D_t * dt) * rng.normal()
    coll = _coll(X, dt)

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    psf = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                   params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                           ParamSpec("gamma", shape=(), dtype=jnp.float64)])
    with pytest.warns(RuntimeWarning, match="damping direction"):
        res = solve_force_ud(coll, psf, max_outer=6, inner="lbfgs",
                             eiv=False, _core="exact")
    assert res.info["inner"] == "gn"
    th = np.asarray(res.theta)
    assert np.all(np.isfinite(th))
    assert abs(th[0] - k_t) < 0.2 and abs(th[1] - g_t) < 0.2, th
