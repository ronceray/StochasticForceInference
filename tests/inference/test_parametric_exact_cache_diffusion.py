"""Stage-4: the fixed-θ tensor cache and the exact-core diffusion solve.

* ``lyapunov_from_stages`` on cached stage Jacobians reproduces the fused
  in-scan Q (shared per-substep helper — identical arithmetic);
* ``prepare`` + ``nll_cached`` equal the plain ``nll`` at any (D, Λ) —
  value and (D,Λ)-gradient — across chunk sizes, with zero basis
  evaluations per call, and fall back transparently when over budget;
* ``solve_diffusion_od/ud`` recover state-dependent D on uniform and
  non-uniform grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases import monomials_up_to
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


def _ou(T=3000, dt=0.02, seed=0, k=1.0, D=0.5, noise=0.02):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 1))
    for t in range(T - 1):
        X[t + 1] = X[t] - k * X[t] * dt + np.sqrt(2 * D * dt) * rng.normal()
    return X + noise * rng.standard_normal(X.shape)


def _ud_lin_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    return make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def _ud_harmonic(T=10000, dt=0.05, seed=0, k=1.0, g=0.5, D=0.1):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 1))
    xi, vi = 0.3, 0.0
    X[0] = xi
    for t in range(T - 1):
        X[t + 1] = xi = xi + vi * dt
        vi = vi + (-k * xi - g * vi) * dt + np.sqrt(2 * D * dt) * rng.normal()
    return X


# ── lyapunov_from_stages ≡ fused in-scan Q ───────────────────────────────


def test_od_lyapunov_from_stages_matches_fused():
    from SFI.inference.parametric_core.jacobians import lyapunov_from_stages
    from SFI.inference.parametric_core.precompute import od_point_tensors

    k, D, dt, n_sub = 1.25, 0.5, 0.05, 3
    F = monomials_up_to(1, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.0, -k])
    X = jnp.asarray(np.linspace(-1, 1, 7))[:, None, None]
    fused = od_point_tensors(F, theta, X, None, dt, n_sub, "rk4",
                             with_psi=False, D_lyap=jnp.asarray([[D]]))
    cached = od_point_tensors(F, theta, X, None, dt, n_sub, "rk4",
                              with_psi=False, with_stages=True)
    twoD = 2.0 * jnp.asarray([[D]])
    Q = jax.vmap(lambda st: lyapunov_from_stages(st, dt / n_sub, twoD))(
        cached["stages"])
    np.testing.assert_allclose(np.asarray(Q), np.asarray(fused["Q"]),
                               rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(np.asarray(cached["J"]),
                               np.asarray(fused["J"]), rtol=1e-15)


def test_ud_lyapunov_from_stages_matches_fused():
    from SFI.inference.parametric_core.jacobians import lyapunov_from_stages
    from SFI.inference.parametric_core.precompute import ud_point_tensors

    psf = _ud_lin_psf()
    theta = jnp.asarray([1.0, 0.8])
    D, dt, n_sub = 0.7, 0.05, 2
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.normal(0, 0.5, size=(8, 1, 1)))
    fused = ud_point_tensors(psf, theta, Y, None, dt, n_sub, "rk4",
                             with_psi=False, D_lyap=jnp.asarray([[D]]))
    cached = ud_point_tensors(psf, theta, Y, None, dt, n_sub, "rk4",
                              with_psi=False, with_stages=True)
    c = cached["cache"]
    d = 1
    B = jnp.zeros((2 * d, 2 * d)).at[d:, d:].set(2.0 * jnp.asarray([[D]]))
    h = dt / n_sub
    Q_in = jax.vmap(lambda st: lyapunov_from_stages(st, h, B))(c["stages_in"])
    Q_out = jax.vmap(lambda st: lyapunov_from_stages(st, h, B))(c["stages_out"])
    q = fused["qing"]
    np.testing.assert_allclose(np.asarray(Q_in[..., :d, :d]),
                               np.asarray(q["Qxx_in"]), rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(np.asarray(Q_in[..., d:, d:]),
                               np.asarray(q["Qvv_in"]), rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(np.asarray(Q_out[..., :d, d:]),
                               np.asarray(q["Qxv_out"]), rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(np.asarray(c["N"]), np.asarray(q["N"]),
                               rtol=1e-15)


# ── nll_cached ≡ nll ─────────────────────────────────────────────────────


def _z_mats(z, d):
    from SFI.inference.parametric_core.solve import _chol_vec_to_mat

    nsym = d * (d + 1) // 2
    return _chol_vec_to_mat(z[:nsym], d), _chol_vec_to_mat(z[nsym:], d)


@pytest.mark.parametrize("chunk_pts", [149, 100_000])
def test_nll_cached_identity_od(chunk_pts):
    from SFI.inference.parametric_core.runner import make_exact_runs_od
    from SFI.inference.parametric_core.solve import _mat_to_chol_vec

    X = _ou(T=2500, seed=1)
    basis = monomials_up_to(1, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.05, -0.9])
    runs = make_exact_runs_od(_coll(X, 0.02), basis, dt=None, n_substeps=1,
                              integrator="rk4", w=1.0, chunk_pts=chunk_pts)
    runs.prepare(theta)
    D = jnp.asarray([[0.43]])
    Lam = jnp.asarray([[3.7e-4]])
    a = float(runs.nll(theta, D, Lam))
    b = float(runs.nll_cached(D, Lam))
    np.testing.assert_allclose(b, a, rtol=1e-12)

    # gradient parity w.r.t. the (D, Λ) parametrization
    z = jnp.concatenate([_mat_to_chol_vec(D, 1), _mat_to_chol_vec(Lam, 1)])
    g_plain = jax.grad(lambda zz: runs.nll(theta, *_z_mats(zz, 1)))(z)
    g_cache = jax.grad(lambda zz: runs.nll_cached(*_z_mats(zz, 1)))(z)
    np.testing.assert_allclose(np.asarray(g_cache), np.asarray(g_plain),
                               rtol=1e-10, atol=1e-12)


def test_nll_cached_identity_ud():
    from SFI.inference.parametric_core.runner import make_exact_runs_ud

    X = _ud_harmonic(T=2500, seed=2)
    psf = _ud_lin_psf()
    theta = jnp.asarray([0.9, 0.45])
    runs = make_exact_runs_ud(_coll(X, 0.05), psf, dt=None, n_substeps=1,
                              integrator="rk4", w=0.0, chunk_pts=173)
    runs.prepare(theta)
    D = jnp.asarray([[0.12]])
    Lam = jnp.asarray([[1e-6]])
    a = float(runs.nll(theta, D, Lam))
    b = float(runs.nll_cached(D, Lam))
    np.testing.assert_allclose(b, a, rtol=1e-12)


def test_nll_cached_no_phase_a_after_prepare(monkeypatch):
    """After prepare, cached evaluations must not touch the flow/basis."""
    import SFI.inference.parametric_core.runner as runner_mod
    from SFI.inference.parametric_core.runner import make_exact_runs_od

    X = _ou(T=800, seed=3)
    basis = monomials_up_to(1, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.0, -1.0])
    runs = make_exact_runs_od(_coll(X, 0.02), basis, dt=None, n_substeps=1,
                              integrator="rk4", w=0.0)
    runs.prepare(theta)

    def _boom(*a, **k):
        raise AssertionError("Phase A re-evaluated on the cached path")

    monkeypatch.setattr(runner_mod, "od_point_tensors", _boom)
    v = float(runs.nll_cached(jnp.asarray([[0.5]]), jnp.asarray([[1e-4]])))
    assert np.isfinite(v)


def test_nll_cached_fallback_over_budget(monkeypatch):
    """Over-budget prepare stores nothing; nll_cached transparently
    recomputes and equals the plain nll."""
    from SFI.inference.parametric_core.runner import ExactRuns, make_exact_runs_od

    X = _ou(T=800, seed=4)
    basis = monomials_up_to(1, dim=1, rank="vector").to_psf()
    theta = jnp.asarray([0.0, -1.0])
    monkeypatch.setattr(ExactRuns, "_cache_bytes_per_point",
                        lambda self, N, d: 1e15)
    runs = make_exact_runs_od(_coll(X, 0.02), basis, dt=None, n_substeps=1,
                              integrator="rk4", w=0.0)
    runs.prepare(theta)
    assert runs._cache_data is None
    D, Lam = jnp.asarray([[0.5]]), jnp.asarray([[1e-4]])
    np.testing.assert_allclose(float(runs.nll_cached(D, Lam)),
                               float(runs.nll(theta, D, Lam)), rtol=1e-12)


# ── exact-core diffusion ─────────────────────────────────────────────────


def test_diffusion_exact_od_parity_constant_D():
    from SFI.bases import symmetric_matrix_basis
    from SFI.inference.parametric_core.solve import (
        solve_diffusion_od,
        solve_force_od,
    )

    rng = np.random.default_rng(5)
    T, dt, k = 12000, 0.02, np.array([[1.0, 0.1], [0.1, 1.5]])
    D_true = np.diag([0.5, 0.3])
    s2D = np.sqrt(2 * D_true)
    X = np.zeros((T, 2))
    for t in range(T - 1):
        X[t + 1] = X[t] - (k @ X[t]) * dt + s2D @ rng.normal(size=2) * np.sqrt(dt)
    coll = _coll(X, dt)
    basis = monomials_up_to(1, dim=2, rank="vector")
    fres = solve_force_od(coll, basis, max_outer=6)
    D_basis = symmetric_matrix_basis(2)

    d_e = solve_diffusion_od(coll, basis, fres.theta, D_basis,
                             Lambda=fres.Lambda)
    D_psf = D_basis.to_psf()

    def _D_at0(th):
        return np.asarray(D_psf(jnp.zeros(2)[None],
                                params=D_psf.unflatten_params(th))[0])

    De = _D_at0(d_e.theta_D)
    # the (constant) diffusion is recovered within a few per cent
    assert np.linalg.norm(De - D_true) / np.linalg.norm(D_true) < 0.15, De


def test_diffusion_exact_od_state_dependent_uneven_dt():
    """D(x) = c0 + c1 x² recovered from a jittered-dt trajectory."""
    from SFI.inference.parametric_core.solve import (
        solve_diffusion_od,
        solve_force_od,
    )
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    rng = np.random.default_rng(6)
    T, k = 20000, 1.0
    c0_t, c1_t = 0.3, 0.2
    dts = 0.02 * (1.0 + 0.3 * (rng.uniform(size=T - 1) - 0.5) * 2)
    n_micro = 20
    X = np.zeros((T, 1))
    xi = 0.0
    for t in range(T - 1):
        h = dts[t] / n_micro
        for _ in range(n_micro):
            xi = xi - k * xi * h + np.sqrt(2 * (c0_t + c1_t * xi**2) * h) \
                * rng.normal()
        X[t + 1] = xi
    coll = _coll(X, np.concatenate([dts, dts[-1:]]))
    basis = monomials_up_to(1, dim=1, rank="vector")
    fres = solve_force_od(coll, basis, max_outer=6, eiv=False, _core="exact")

    def fD(x, *, params, mask=None, extras=None):
        return jnp.array([[params["c0"] + params["c1"] * x[0] ** 2]])

    D_psf = make_psf(fD, dim=1, rank=2, n_features=1,
                     params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                             ParamSpec("c1", shape=(), dtype=jnp.float64)])
    dres = solve_diffusion_od(coll, basis, fres.theta, D_psf,
                              Lambda=jnp.zeros((1, 1)),
                              theta_D0={"c0": 0.25, "c1": 0.0},
                              _core="exact")
    p = D_psf.unflatten_params(dres.theta_D)
    c0, c1 = float(np.asarray(p["c0"])), float(np.asarray(p["c1"]))
    assert abs(c0 - c0_t) / c0_t < 0.15, c0
    assert abs(c1 - c1_t) / c1_t < 0.35, c1


def test_diffusion_exact_ud_velocity_model():
    from SFI.inference.parametric_core.solve import (
        solve_diffusion_ud,
        solve_force_ud,
    )
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    D_true = 0.1
    X = _ud_harmonic(T=14000, dt=0.05, seed=7, D=D_true)
    coll = _coll(X, 0.05)
    fres = solve_force_ud(coll, _ud_lin_psf(), max_outer=6, inner="gn",
                          eiv=False, _core="exact")

    def fD(x, *, v, params, mask=None, extras=None):
        return jnp.array([[params["c0"] + params["c1"] * v[0] ** 2]])

    D_psf = make_psf(fD, dim=1, rank=2, n_features=1, needs_v=True,
                     params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                             ParamSpec("c1", shape=(), dtype=jnp.float64)])
    dres = solve_diffusion_ud(coll, _ud_lin_psf(), fres.theta, D_psf,
                              Lambda=fres.Lambda,
                              theta_D0={"c0": 0.08, "c1": 0.0},
                              _core="exact")
    p = D_psf.unflatten_params(dres.theta_D)
    c0, c1 = float(np.asarray(p["c0"])), float(np.asarray(p["c1"]))
    assert abs(c0 - D_true) / D_true < 0.30, c0
    assert abs(c1) < 0.06, c1
