"""EIV instrument / estimating-equation tests for parametric_core.

Covers the asymmetric-Gram (left/right test-function) generalisation, the
eta-clean OD/UD skip instruments, and the end-to-end measurement-noise
recovery that motivates the whole port.
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


# ── OD η-clean instrument, end-to-end EIV recovery ───────────────────────────

def _make_ou_noisy(T=20000, dt=0.01, d=2, seed=7, noise=0.0):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]])[:d, :d]
    Dm = np.diag([0.5, 0.3][:d]); s = np.sqrt(2 * Dm)
    X = np.zeros((T, d)); X[0] = rng.normal(0, 0.5, d)
    for t in range(T - 1):
        X[t + 1] = X[t] - (k @ X[t]) * dt + s @ (rng.normal(size=d) * np.sqrt(dt))
    Y = X + noise * rng.standard_normal(X.shape) if noise else X
    return Y, k


def _coll(Y, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    return TrajectoryCollection.from_dataset(
        TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, :]), dt=float(dt)))


def _lin_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec
    def f(x, *, params):
        return params["A"] @ x
    return make_psf(f, dim=d, rank=1, n_features=1,
                    params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)])


def test_od_iv_recovers_drift_under_measurement_noise():
    """Pure-IV (eiv=True) recovers A where symmetric GN (eiv=False) is EIV-biased."""
    from SFI.inference.parametric_core.solve import solve_force_od
    sigma = 0.04
    Y, k = _make_ou_noisy(T=40000, dt=0.01, d=2, seed=3, noise=sigma)
    coll = _coll(Y, 0.01)
    A_sym = np.asarray(_lin_psf(2).unflatten_params(
        solve_force_od(coll, _lin_psf(2), n_substeps=4, max_outer=6, inner="gn", eiv=False).theta)["A"])
    A_iv = np.asarray(_lin_psf(2).unflatten_params(
        solve_force_od(coll, _lin_psf(2), n_substeps=4, max_outer=6, inner="gn", eiv=True).theta)["A"])
    err_sym = np.linalg.norm(A_sym - (-k)) / np.linalg.norm(k)
    err_iv = np.linalg.norm(A_iv - (-k)) / np.linalg.norm(k)
    assert err_iv < 0.6 * err_sym, f"IV did not reduce EIV bias: iv={err_iv:.3f} sym={err_sym:.3f}"
    assert err_iv < 0.15, f"IV error too large: {err_iv:.3f}"


def test_od_skip_instrument_recovers_drift():
    """The skip ("maintain a lag") instrument also recovers the OD drift under
    measurement noise — its base is the reserved front position, outside the
    tridiagonal-coupled block."""
    from SFI.inference.parametric_core.solve import solve_force_od
    Y, k = _make_ou_noisy(T=40000, dt=0.01, d=2, seed=3, noise=0.04)
    coll = _coll(Y, 0.01)
    A_skip = np.asarray(_lin_psf(2).unflatten_params(
        solve_force_od(coll, _lin_psf(2), n_substeps=4, max_outer=6,
                       inner="gn", eiv=True).theta)["A"])
    err = np.linalg.norm(A_skip - (-k)) / np.linalg.norm(k)
    assert err < 0.15, f"skip IV error too large: {err:.3f}"


# ── Task 5: underdamped instrument rescues the measurement-noise collapse ─────

def _simulate_ud_harmonic_noisy(N, dt, seed, k, gamma, D, noise):
    rng = np.random.default_rng(seed)
    s = np.sqrt(2 * D)
    X = np.zeros((N, 1)); V = np.zeros((N, 1))
    X[0] = rng.normal(0, 0.3, 1); V[0] = rng.normal(0, 0.3, 1)
    for t in range(N - 1):
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k * X[t] - gamma * V[t]) * dt + s * rng.normal(size=1) * np.sqrt(dt)
    return X + noise * rng.standard_normal(X.shape)


def _ud_linear_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec
    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v
    return make_psf(f, dim=1, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def test_ud_instrument_rescues_restoring_coefficient_under_noise():
    """Under measurement noise the symmetric UD-GN is badly velocity-EIV
    biased — at its own optimum the damping blows up (γ ≈ 16 ≫ 0.5: the noise
    reads as huge velocity decorrelation) and k is biased; the η-clean
    instrument (eiv=True) rescues k.  (Before the GN line-search merit fix the
    symmetric solve froze near a k→0 collapse instead of reaching its biased
    optimum; the γ blow-up is the true symmetric fixed point.  γ keeps a
    residual σ²-bias on the IV path too — see notes — so this locks in only
    the validated k-rescue.)"""
    from SFI.inference.parametric_core.solve import solve_force_ud
    k_true, gamma_true, D = 1.0, 0.5, 0.1
    X = _simulate_ud_harmonic_noisy(20000, 0.01, 1, k_true, gamma_true, D, noise=0.01)
    coll = _coll(X, 0.01)
    p_sym = _ud_linear_psf().unflatten_params(
        solve_force_ud(coll, _ud_linear_psf(), n_substeps=4, max_outer=6, inner="gn", eiv=False).theta)
    k_sym, g_sym = float(np.asarray(p_sym["k"])), float(np.asarray(p_sym["gamma"]))
    k_iv = float(np.asarray(_ud_linear_psf().unflatten_params(
        solve_force_ud(coll, _ud_linear_psf(), n_substeps=4, max_outer=6, inner="gn", eiv=True).theta)["k"]))
    sym_biased = abs(g_sym - gamma_true) / gamma_true > 2.0 or abs(k_sym - k_true) / k_true > 0.5
    assert sym_biased, f"expected a strong symmetric EIV bias, got k={k_sym:.3f} γ={g_sym:.3f}"
    assert abs(k_iv - k_true) / k_true < 0.5, f"instrument did not rescue k: {k_iv:.3f}"


def test_ud_skip_instrument_removes_gamma_bias():
    """The skip ("maintain a lag") instrument keeps its base outside the whole
    residual block the pentadiagonal precision couples in, so the velocity-
    damping γ stays consistent under measurement noise (a center-propagated
    instrument would leave a large σ² overlap bias, γ ≈ +16 vs true 0.5)."""
    from SFI.inference.parametric_core.solve import solve_force_ud
    X = _simulate_ud_harmonic_noisy(40000, 0.01, 2, k=1.0, gamma=0.5, D=0.1, noise=0.01)
    coll = _coll(X, 0.01)
    g_skip = float(np.asarray(_ud_linear_psf().unflatten_params(
        solve_force_ud(coll, _ud_linear_psf(), n_substeps=4, max_outer=6,
                       inner="gn", eiv=True).theta)["gamma"]))
    assert abs(g_skip - 0.5) < 1.0, f"skip did not keep γ consistent: {g_skip:.3f}"


# ── Multi-particle (interacting) instrument: OD and UD ───────────────────────


def _trap_spring_psf(dim=2):
    from SFI.bases.pairs import parametric_radial_kernel, pair_direction
    from SFI.statefunc import make_psf

    F_trap = make_psf(lambda x, *, params: -params["k_trap"] * x,
                      params={"k_trap": ()}, dim=dim, rank=1)
    e_ij = pair_direction(dim=dim)
    k_spring = parametric_radial_kernel(lambda r, p: p["k_spring"] * r,
                                        params={"k_spring": ()}, dim=dim)
    return F_trap + (k_spring * e_ij).dispatch_pairs(return_as="psf")


def test_od_iv_recovers_interacting_under_measurement_noise():
    """Trap + pair springs with measurement noise: the frame-level N-body
    instrument removes the EIV bias on BOTH the self and the pair coefficient.
    The old isolated-frame instrument zeroed the pair column (singular IV Gram
    → ridge-pinned plateau, FINDINGS #9); the symmetric path is ~2× biased."""
    from jax import random
    from SFI.inference.parametric_core.solve import solve_force_od
    from SFI.langevin import OverdampedProcess

    dim, N, dt, sigma = 2, 5, 0.02, 0.1
    proc = OverdampedProcess(_trap_spring_psf(dim), D=jnp.eye(dim) * 0.5,
                             theta_F={"k_trap": 1.0, "k_spring": 0.5})
    proc.initialize(random.normal(random.PRNGKey(0), (N, dim)) * 0.5)
    coll = proc.simulate(dt=dt, Nsteps=4000, key=random.PRNGKey(1),
                         prerun=300, oversampling=10).degrade(noise=sigma, seed=42)

    truth = np.array([1.0, 0.5])

    def _fit(eiv):
        F = _trap_spring_psf(dim)
        res = solve_force_od(coll, F, inner="gn", eiv=eiv, max_outer=5)
        p = F.unflatten_params(jnp.asarray(res.theta))
        return np.array([float(p["k_trap"]), float(p["k_spring"])])

    fit_sym, fit_iv = _fit(False), _fit(True)
    err_sym = np.linalg.norm(fit_sym - truth) / np.linalg.norm(truth)
    err_iv = np.linalg.norm(fit_iv - truth) / np.linalg.norm(truth)
    assert err_sym > 0.4, f"expected a strong EIV bias on the symmetric path: {err_sym:.3f}"
    # ~±20% finite-sample IV scatter on k_trap at this data size; the pair
    # coefficient (the column the old instrument zeroed) recovers tightly.
    assert err_iv < 0.30, f"interacting IV did not recover: {err_iv:.3f}"
    assert err_iv < 0.5 * err_sym
    assert abs(fit_iv[1] - 0.5) / 0.5 < 0.20, f"pair coefficient off: {fit_iv[1]:.3f}"


def _flock_pair_psf(dim=2):
    from SFI.statefunc import make_interactor
    from SFI.statefunc.params import ParamSpec

    def _pair(x, *, v, params):
        dr = x[1] - x[0]
        dv = v[1] - v[0]
        return (params["k_coh"] * dr + params["k_alg"] * dv)[..., None]

    return make_interactor(
        _pair, dim=dim, rank=1, K=2, n_features=1, needs_v=True,
        params=[ParamSpec("k_coh", shape=(), default=0.5),
                ParamSpec("k_alg", shape=(), default=0.8)],
    ).dispatch_pairs(drop_features=True)


def test_ud_iv_rescues_interacting_under_measurement_noise():
    """Underdamped cohesion+alignment (pure pair force) with measurement
    noise, on the exact core.  The old isolated-frame instrument had ALL
    columns zero here (pure pair force) and returned θ = 0; the frame-level
    N-body instrument, with the converged (D, Λ) profile, recovers the
    cohesion coefficient with AND without the instrument — the alignment
    (velocity-coupling) coefficient keeps the known residual O(σ²) bias, so
    only the cohesion recovery is asserted."""
    from jax import random
    from SFI.inference.parametric_core.solve import solve_force_ud
    from SFI.langevin import UnderdampedProcess

    dim, N, dt, sigma = 2, 5, 0.01, 0.005
    proc = UnderdampedProcess(_flock_pair_psf(dim), D=jnp.eye(dim) * 0.05)
    proc.initialize(random.normal(random.PRNGKey(3), (N, dim)) * 0.5,
                    v0=random.normal(random.PRNGKey(4), (N, dim)) * 0.2)
    coll = proc.simulate(dt=dt, Nsteps=6000, key=random.PRNGKey(5),
                         prerun=200, oversampling=10).degrade(noise=sigma, seed=43)

    def _fit(eiv):
        F = _flock_pair_psf(dim)
        res = solve_force_ud(coll, F, inner="gn", eiv=eiv, max_outer=5)
        p = F.unflatten_params(jnp.asarray(res.theta))
        return float(p["k_coh"]), float(p["k_alg"])

    kc_iv, _ = _fit(True)
    kc_sym, _ = _fit(False)
    assert np.isfinite(kc_iv) and np.isfinite(kc_sym)
    assert abs(kc_iv - 0.5) / 0.5 < 0.30, f"exact IV k_coh: {kc_iv:.3f}"
    assert abs(kc_sym - 0.5) / 0.5 < 0.30, f"exact sym k_coh: {kc_sym:.3f}"
