"""Stage-3: end-to-end non-uniform-dt force solves + moment-init.

* Moment initialisation: the generalized Vestergaard denominator
  ``(dt + dt⁻)/2`` makes ``D̂`` exactly unbiased at any spacing (the naive
  centre-``dt`` division is ~33 % biased on an alternating 1:3 grid); the
  ULI local-mean ``dt̄³`` is validated as init-grade under jitter.
* The sharp end-to-end discriminator: an OU process sampled on an
  alternating coarse grid ``dt ∈ {0.05, 0.55}`` (k·dt up to 0.55, flow
  strongly nonlinear).  A fit that pretends the mean ``dt̄ = 0.3`` is
  provably biased: ``e^{−k̂ dt̄} = E[e^{−k dt}]`` gives ``k̂ ≈ 0.90`` and
  ``D̂ ≈ 0.41`` for ``(k, D) = (1, 0.5)``, while the per-step exact-core
  fit must recover the truth.
* UD: a jittered-dt harmonic oscillator (±30 %) fit with the
  Lyapunov-exact blocks recovers (k, γ, D).
* Masks and uneven dt combine.
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


def _coll(X, *, dt=None, t=None, mask=None, dynamic_mask=None):
    ds = TrajectoryDataset.from_arrays(
        X=jnp.asarray(X[:, None, :]), dt=dt, t=t, mask=mask,
        dynamic_mask=dynamic_mask)
    return TrajectoryCollection.from_dataset(ds)


def _ou_exact(dts, *, k=1.0, D=0.5, noise=0.0, seed=0):
    """OU sampled EXACTLY on the grid of per-step intervals ``dts``."""
    rng = np.random.default_rng(seed)
    T = len(dts) + 1
    x = np.zeros(T)
    x[0] = rng.normal(0, np.sqrt(D / k))
    for i, dt in enumerate(dts):
        decay = np.exp(-k * dt)
        sig = np.sqrt(D * (1.0 - decay**2) / k)
        x[i + 1] = x[i] * decay + sig * rng.normal()
    y = x + noise * rng.standard_normal(T)
    return y[:, None]


def _ud_fine(dts, *, k=1.0, g=0.5, D=0.1, noise=0.0, seed=0, n_micro=40):
    """Underdamped harmonic oscillator on per-step intervals (fine-grid)."""
    rng = np.random.default_rng(seed)
    T = len(dts) + 1
    x = np.zeros(T)
    xi, vi = rng.normal(0, 0.3), rng.normal(0, 0.3)
    x[0] = xi
    for i, dt in enumerate(dts):
        h = dt / n_micro
        sq = np.sqrt(2 * D * h)
        for _ in range(n_micro):
            a = -k * xi - g * vi
            xi = xi + vi * h
            vi = vi + a * h + sq * rng.normal()
        x[i + 1] = xi
    y = x + noise * rng.standard_normal(T)
    return y[:, None]


# ── moment initialisation ────────────────────────────────────────────────


def test_moment_init_od_alternating_dt_unbiased():
    """Alternating 1:3 grid + localization noise: the generalized
    Vestergaard denominator is exactly unbiased, where the naive
    centre-``dt`` division has bias E[(dt+dt⁻)/(2dt)] − 1 = 1/3."""
    from SFI.inference.parametric_core.solve import _moment_init

    k, D, Lam = 1.0, 0.5, 4e-4
    T = 40000
    dts = np.where(np.arange(T - 1) % 2 == 0, 0.01, 0.03)
    X = _ou_exact(dts, k=k, D=D, noise=np.sqrt(Lam), seed=1)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))
    D0, L0 = _moment_init(coll, dynamics="od", d=1, dtype=jnp.float64,
                          drop_se=False)
    assert abs(float(D0[0, 0]) / D - 1.0) < 0.08, float(D0[0, 0])
    # Λ̂ carries the moment estimator's pre-existing O(D·k·dt·dt⁻) drift
    # bias (≈ +1.5e-4 here, uniform-dt included) — init-grade decade check;
    # the exact-NLL profile refines it.
    assert 0.3 * Lam < float(L0[0, 0]) < 3.0 * Lam, float(L0[0, 0])


def test_moment_init_od_uniform_bitwise_unchanged():
    """(dt + dt)/2 is exactly dt in floating point: the generalized
    estimator is bitwise-identical on uniform data."""
    from SFI.inference.overdamped import _D_noisy

    rng = np.random.default_rng(2)
    n, d = 256, 2
    dX = jnp.asarray(rng.normal(size=(n, d)))
    dXm = jnp.asarray(rng.normal(size=(n, d)))
    dt = jnp.full((n,), 0.0173)
    new = _D_noisy(dX=dX, dX_minus=dXm, dt=dt, dt_minus=dt)
    # the historical estimator: every term divided by the centre dt
    invdt = (1.0 / dt)[:, None]
    ref = 0.25 * (jnp.einsum("nm,nk->nmk", dX, dX * invdt)
                  + 2 * jnp.einsum("nm,nk->nmk", dX, dXm * invdt)
                  + 2 * jnp.einsum("nm,nk->nmk", dXm, dX * invdt)
                  + jnp.einsum("nm,nk->nmk", dXm, dXm * invdt))
    assert jnp.array_equal(new, ref), "not bitwise-identical at uniform dt"


def test_weaknoise_od_alternating_dt_unbiased():
    """Clean data on an alternating 1:3 grid: ``method='auto'`` selects the
    weak-noise estimator, whose centre-dt division was ~+33 % biased; the
    two-interval-mean denominator makes it exactly unbiased."""
    from SFI.inference.overdamped import OverdampedLangevinInference

    k, D = 1.0, 0.5
    T = 40000
    dts = np.where(np.arange(T - 1) % 2 == 0, 0.01, 0.03)
    X = _ou_exact(dts, k=k, D=D, seed=10)              # no localization noise
    inf = OverdampedLangevinInference(
        _coll(X, dt=np.concatenate([dts, dts[-1:]])), verbosity=0)
    inf.compute_diffusion_constant(method="WeakNoise")
    D_hat = float(np.asarray(inf.diffusion_average).ravel()[0])
    assert abs(D_hat / D - 1.0) < 0.08, D_hat


def test_moment_init_ud_uniform_bitwise_unchanged():
    """Q ≡ 0 and dt̄ ≡ dt exactly at uniform sampling: the corrected ULI
    estimator is bitwise-identical to the historical one there."""
    from SFI.inference.underdamped import _D_noisy_uli, _symmetrize_imn

    rng = np.random.default_rng(8)
    n, d = 256, 2
    dX = jnp.asarray(rng.normal(size=(n, d)))
    dXm = jnp.asarray(rng.normal(size=(n, d)))
    dXp = jnp.asarray(rng.normal(size=(n, d)))
    dt = jnp.full((n,), 0.0417)
    new = _D_noisy_uli(dX=dX, dX_minus=dXm, dX_plus=dXp,
                       dt=dt, dt_minus=dt, dt_plus=dt)
    a = jnp.einsum("im,in->imn", dX, dX)
    b = jnp.einsum("im,in->imn", dXm, dXm)
    c = jnp.einsum("im,in->imn", dXp, dXp)
    dd = jnp.einsum("im,in->imn", dXp, dXm)
    e = jnp.einsum("im,in->imn", dX, dXp)
    f = jnp.einsum("im,in->imn", dX, dXm)
    M = _symmetrize_imn(-a + b + c - 3.0 * dd + e + f)
    ref = (3.0 / 11.0) * M / (dt[:, None, None] ** 3)
    assert jnp.array_equal(new, ref), "not bitwise-identical at uniform dt"


def test_moment_init_ud_alternating_dt_ballistic_correction():
    """Alternating grid: the ULI stencil's ⟨v²⟩ cancellation
    −u²+v²+w²−3wv+uw+uv is interval-pattern-dependent; uncorrected it
    leaks −(a−b)²⟨v²⟩/dt̄³ (a *negative* D̂ here).  The η-clean
    Q-correction must restore a positive, decade-correct estimate."""
    from SFI.inference.parametric_core.solve import _moment_init

    D_t = 0.1
    T = 20000
    dts = np.where(np.arange(T - 1) % 2 == 0, 0.035, 0.065)
    X = _ud_fine(dts, k=1.0, g=0.5, D=D_t, seed=9)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))
    D0, _ = _moment_init(coll, dynamics="ud", d=1, dtype=jnp.float64,
                         drop_se=True)
    v = float(D0[0, 0])
    assert v > 0.0, f"ballistic leakage resurfaced: D0 = {v}"
    assert abs(v / D_t - 1.0) < 0.6, v


def test_moment_init_ud_jittered_dt_init_grade():
    from SFI.inference.parametric_core.solve import _moment_init

    rng = np.random.default_rng(3)
    T = 20000
    dts = 0.05 * (1.0 + 0.3 * (rng.uniform(size=T - 1) - 0.5) * 2)
    X = _ud_fine(dts, k=1.0, g=0.5, D=0.1, seed=3)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))
    D0, _ = _moment_init(coll, dynamics="ud", d=1, dtype=jnp.float64,
                         drop_se=True)
    # init-grade: the local-mean dt̄³ keeps D0 within ~30 % under ±30 %
    # jitter (Jensen residue of the cubic normalisation, measured +0.29
    # here); the exact-NLL profile refines D̂ — the init only needs the
    # right decade.
    assert abs(float(D0[0, 0]) / 0.1 - 1.0) < 0.40, float(D0[0, 0])


# ── end-to-end: the sharp mean-dt discriminator (OD) ─────────────────────


def test_od_uneven_solve_recovers_where_mean_dt_fit_is_biased():
    from SFI.inference.parametric_core.solve import solve_force_od

    k, D = 1.0, 0.5
    T = 20000
    dts = np.where(np.arange(T - 1) % 2 == 0, 0.05, 0.55)
    X = _ou_exact(dts, k=k, D=D, seed=4)
    basis = monomials_up_to(1, dim=1, rank="vector")

    # per-step fit (exact core, n_substeps=2 keeps RK4 error ≪ the bias)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))
    res = solve_force_od(coll, basis, max_outer=6, eiv=False, n_substeps=2,
                         _core="exact")
    k_hat = -float(res.theta[1])
    sig_k = float(np.sqrt(np.abs(np.asarray(res.theta_cov)[1, 1])))
    D_hat = float(np.asarray(res.D)[0, 0])

    # deliberately-wrong fit: same data, pretend the mean dt̄ = 0.3
    coll_wrong = _coll(X, dt=float(np.mean(dts)))
    res_w = solve_force_od(coll_wrong, basis, max_outer=6, eiv=False,
                           n_substeps=2, _core="exact")
    k_wrong = -float(res_w.theta[1])
    D_wrong = float(np.asarray(res_w.D)[0, 0])

    # per-step fit: unbiased (well within 3σ and 5 % absolute)
    assert abs(k_hat - k) < max(3 * sig_k, 0.05), (k_hat, sig_k)
    assert abs(D_hat - D) / D < 0.06, D_hat
    # mean-dt fit: the analytic bias e^{−k̂ dt̄} = E[e^{−k dt}] → k̂ ≈ 0.90,
    # and D̂ absorbs the flow mismatch (≈ −18 %)
    assert k_wrong < k - 0.06, k_wrong
    assert abs(k_wrong - k) > 2.5 * abs(k_hat - k), (k_hat, k_wrong)
    assert D_wrong < D * 0.92, D_wrong


def test_od_uneven_solve_with_noise_and_eiv():
    """Measurement noise + the η-clean instrument on a jittered grid."""
    from SFI.inference.parametric_core.solve import solve_force_od

    rng = np.random.default_rng(5)
    k, D, noise = 1.0, 0.5, 0.03
    T = 12000
    dts = 0.02 * (1.0 + 0.3 * (rng.uniform(size=T - 1) - 0.5) * 2)
    X = _ou_exact(dts, k=k, D=D, noise=noise, seed=5)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))
    basis = monomials_up_to(1, dim=1, rank="vector")
    res = solve_force_od(coll, basis, max_outer=6, eiv=True, _core="exact")
    k_hat = -float(res.theta[1])
    sig_k = float(np.sqrt(np.abs(np.asarray(res.theta_cov)[1, 1])))
    assert np.all(np.isfinite(np.asarray(res.theta)))
    assert abs(k_hat - k) < 3 * sig_k + 0.05, (k_hat, sig_k)
    # the profiled localization noise is in the right decade
    assert 0.2 * noise**2 < float(np.asarray(res.Lambda)[0, 0]) < 5 * noise**2


# ── end-to-end: UD jittered grid ─────────────────────────────────────────


def test_ud_uneven_solve_recovers():
    from SFI.inference.parametric_core.solve import solve_force_ud
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    rng = np.random.default_rng(6)
    k_t, g_t, D_t = 1.0, 0.5, 0.1
    T = 8000
    dts = 0.05 * (1.0 + 0.3 * (rng.uniform(size=T - 1) - 0.5) * 2)
    X = _ud_fine(dts, k=k_t, g=g_t, D=D_t, seed=6)
    coll = _coll(X, dt=np.concatenate([dts, dts[-1:]]))

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    psf = make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                   params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                           ParamSpec("gamma", shape=(), dtype=jnp.float64)])
    res = solve_force_ud(coll, psf, max_outer=6, inner="gn", eiv=False,
                         _core="exact")
    k_hat, g_hat = float(res.theta[0]), float(res.theta[1])
    sig = np.sqrt(np.abs(np.diag(np.asarray(res.theta_cov))))
    assert abs(k_hat - k_t) < 3 * sig[0] + 0.15, (k_hat, sig[0])
    assert abs(g_hat - g_t) < 3 * sig[1] + 0.15, (g_hat, sig[1])
    assert abs(float(np.asarray(res.D)[0, 0]) - D_t) / D_t < 0.25


# ── masks + uneven dt combined ───────────────────────────────────────────


def test_od_uneven_dt_with_masked_gap():
    from SFI.inference.parametric_core.solve import solve_force_od

    k, D = 1.0, 0.5
    T = 8000
    dts = np.where(np.arange(T - 1) % 2 == 0, 0.015, 0.03)
    X = _ou_exact(dts, k=k, D=D, seed=7)
    mask = np.ones((T, 1), bool)
    mask[4000:4200, 0] = False
    Xg = X.copy()
    Xg[4000:4200, 0] = 1e6                       # garbage in the gap
    coll = _coll(Xg, dt=np.concatenate([dts, dts[-1:]]), mask=mask,
                 dynamic_mask=mask)
    basis = monomials_up_to(1, dim=1, rank="vector")
    res = solve_force_od(coll, basis, max_outer=6, eiv=False, _core="exact")
    k_hat = -float(res.theta[1])
    sig_k = float(np.sqrt(np.abs(np.asarray(res.theta_cov)[1, 1])))
    assert np.all(np.isfinite(np.asarray(res.theta)))
    assert abs(k_hat - k) < 3 * sig_k + 0.08, (k_hat, sig_k)
