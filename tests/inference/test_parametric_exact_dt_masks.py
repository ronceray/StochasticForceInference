"""Exact-runner plumbing: per-interval dt streams, centre/neighbour
masks, instrument gap validity, byte-budgeted chunking.

These exercise the ``ExactRuns`` engine layer (not just the kernels):

* dt is derived per dataset from scalar ``dt`` / per-step ``dt`` array /
  absolute ``t`` and threaded as a traced operand, so uniform sampling
  supplied any of the three ways gives the identical Gram at 1e-12;
* results are invariant to the chunk size (the reverse-time carry) under
  non-uniform dt;
* residual/instrument validity follows the dynamic-at-centre /
  static-at-neighbours blend, and garbage in a masked
  gap does not perturb the fit;
* ``chunk_target_bytes`` bounds the chunk and does not change the result.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases import monomials_up_to
from SFI.inference.parametric_core.gram import unpack_gram
from SFI.inference.parametric_core.runner import make_exact_runs_od
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _ou(T=4000, dt=0.02, d=1, seed=7, noise=0.02, k=1.0, D=0.5):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.4, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] - k * X[t] * dt + np.sqrt(2 * D * dt) * rng.normal(size=d)
    return X + noise * rng.standard_normal(X.shape)


def _ds(X, *, dt=None, t=None, mask=None, dynamic_mask=None):
    return TrajectoryDataset.from_arrays(
        X=jnp.asarray(X[:, None, :]), dt=dt, t=t, mask=mask,
        dynamic_mask=dynamic_mask)


def _model(d=1, k=1.0, D=0.5, Lam=4e-4):
    basis = monomials_up_to(1, dim=d, rank="vector").to_psf()
    theta = jnp.asarray(np.concatenate([[0.0], [-k]]))
    return basis, theta, jnp.asarray([[D]]), jnp.asarray([[Lam]])


def _gram(coll, *, w=0.0, dt=None, chunk_pts=200_000, chunk_target_bytes=None):
    basis, theta, D, Lam = _model()
    runs = make_exact_runs_od(coll, basis, dt=dt, n_substeps=1,
                              integrator="rk4", w=w, chunk_pts=chunk_pts,
                              chunk_target_bytes=chunk_target_bytes)
    return unpack_gram(runs.gram(theta, D, Lam), int(basis.template.size))


# ── dt supplied three ways ───────────────────────────────────────────────


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_uniform_dt_scalar_array_and_times_agree(w):
    X = _ou()
    T, dt = X.shape[0], 0.02
    c_scalar = TrajectoryCollection.from_dataset(_ds(X, dt=dt))
    c_array = TrajectoryCollection.from_dataset(_ds(X, dt=np.full(T, dt)))
    c_times = TrajectoryCollection.from_dataset(
        _ds(X, t=np.arange(T) * dt))
    g0, f0, H0, _ = _gram(c_scalar, w=w)
    for coll in (c_array, c_times):
        g, f, H, _ = _gram(coll, w=w)
        np.testing.assert_allclose(g, g0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f, f0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(H, H0, rtol=1e-12, atol=1e-12)


# ── chunk invariance under non-uniform dt ────────────────────────────────


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_uneven_dt_chunk_invariance(w):
    X = _ou(T=1500)
    T = X.shape[0]
    rng = np.random.default_rng(1)
    dt = 0.02 * (1.0 + 0.3 * (rng.uniform(size=T) - 0.5))     # ±15 %
    coll = TrajectoryCollection.from_dataset(_ds(X, dt=dt))
    ref = _gram(coll, w=w, chunk_pts=200_000)
    small = _gram(coll, w=w, chunk_pts=137)
    # the reverse-time carry accumulates in a different order per chunking;
    # agreement is float64-reassociation tight, not bitwise.
    for a, b in zip(ref, small):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-8, atol=1e-9)


def test_chunk_target_bytes_shrinks_and_preserves():
    X = _ou(T=1500)
    coll = TrajectoryCollection.from_dataset(_ds(X, dt=0.02))
    basis, theta, D, Lam = _model()
    small = make_exact_runs_od(coll, basis, dt=None, n_substeps=1,
                               integrator="rk4", w=1.0,
                               chunk_target_bytes=200_000)
    # the byte budget must clamp well below the full length
    assert small._resolve_chunk_pts(coll.datasets[0]) < X.shape[0]
    g_s = unpack_gram(small.gram(theta, D, Lam), int(basis.template.size))
    g_f = _gram(coll, w=1.0)
    for a, b in zip(g_f, g_s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-8, atol=1e-9)


# ── mask validity: the centre/neighbour blend ────────────────────────────


def test_residual_and_instrument_validity_rules():
    X = _ou(T=40)
    T = X.shape[0]
    static = np.ones((T, 1), bool)
    dynamic = np.ones((T, 1), bool)
    static[20, 0] = False                      # a masked position at t=20
    dynamic[20, 0] = False                     # (dynamic ⊆ static, enforced)
    dynamic[10, 0] = False                     # an unreliable increment at t=10
    coll = TrajectoryCollection.from_dataset(
        _ds(X, dt=0.02, mask=static, dynamic_mask=dynamic))
    basis, *_ = _model()
    runs = make_exact_runs_od(coll, basis, dt=None, n_substeps=1,
                              integrator="rk4", w=1.0)
    ds = coll.datasets[0]
    a, b, lo, hi, Xb, dtb, ms, md = next(iter(runs._iter_blocks(ds)))
    lead = a - lo
    n_res = b - a
    vr = np.asarray(runs._residual_validity(ms, md, n_res, lead))[:, 0]
    iv = np.asarray(runs._instrument_validity(ms, n_res, lead))[:, 0]
    # OD residual k uses increment base k (dynamic) and position k+1 (static):
    # the static hole at 20 kills residual 19; the dynamic hole at 10 kills
    # residual 10.
    assert not vr[19] and not vr[10]
    assert vr[9] and vr[11] and vr[18] and vr[21]
    # instrument base of residual k is position k−1: the static hole at 20
    # invalidates the instrument of residual 21.
    assert not iv[21] and iv[20] and iv[22]
    assert not iv[0]                           # dataset front, no clean base


def test_garbage_in_masked_gap_does_not_perturb_fit():
    from SFI.inference.parametric_core.solve import solve_force_od

    X = _ou(T=6000, seed=3)
    T = X.shape[0]
    basis = monomials_up_to(1, dim=1, rank="vector")

    clean = TrajectoryCollection.from_dataset(_ds(X, dt=0.02))
    res_clean = solve_force_od(clean, basis, max_outer=6, eiv=True,
                               _core="exact")

    # blank a 200-frame gap: mask it AND fill the positions with garbage
    mask = np.ones((T, 1), bool)
    mask[3000:3200, 0] = False
    Xg = X.copy()
    Xg[3000:3200, 0] = 1e6
    gapped = TrajectoryCollection.from_dataset(
        _ds(Xg, dt=0.02, mask=mask, dynamic_mask=mask))
    res_gap = solve_force_od(gapped, basis, max_outer=6, eiv=True,
                             _core="exact")

    th_c = np.asarray(res_clean.theta)
    th_g = np.asarray(res_gap.theta)
    assert np.all(np.isfinite(th_g))
    # the load-bearing assertion: masking a gap and filling it with 1e6
    # garbage produces essentially the SAME fit as the clean data (only the
    # 200 dropped rows' worth of statistical shift) — the garbage does not
    # leak into any residual, coupling, or instrument.
    sig = np.sqrt(np.abs(np.diag(np.asarray(res_clean.theta_cov))))
    assert np.all(np.abs(th_c - th_g) < 0.3 * sig + 1e-3), (th_c, th_g, sig)
    # both remain consistent with the ground-truth slope −k = −1 (this short
    # noisy-OU EIV fit is high-variance, so allow ~2σ)
    assert abs(th_g[1] + 1.0) < 2.0 * sig[1] + 0.05


# ── UD uneven-dt guard ───────────────────────────────────────────────────


def test_ud_uneven_dt_requires_lyapunov(monkeypatch):
    from SFI.inference.parametric_core.solve import solve_force_ud
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    rng = np.random.default_rng(5)
    T = 400
    X = np.cumsum(rng.normal(0, 0.01, size=(T, 1)), axis=0)
    dt = 0.05 * (1.0 + 0.3 * (rng.uniform(size=T) - 0.5))
    coll = TrajectoryCollection.from_dataset(_ds(X, dt=dt))

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    psf = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                   params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                           ParamSpec("gamma", shape=(), dtype=jnp.float64)])

    monkeypatch.setenv("SFI_EXACT_UPGRADES", "0")
    with pytest.raises(ValueError, match="Lyapunov"):
        solve_force_ud(coll, psf, max_outer=2, inner="gn", eiv=False,
                       _core="exact")
