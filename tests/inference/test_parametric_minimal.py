"""Robustness gate for the v2.0 *minimal* parametric estimator.

The minimal path is ``parametric_core`` restricted to a single RK4 step
(``integrator="rk4"``, ``n_substeps=1``), residual-covariance NLL, skip-trick
Gauss–Newton for a linear ``Basis`` / loss-minimisation L-BFGS for a ``PSF``,
with constant ``(D, Λ)`` profiling.  (A single Euler step cannot carry the
force into the underdamped position update, so the skip-trick instrument is
degenerate there — hence RK4 is the single-step default; see the guard test.)
These tests pin the *defaults* (so the public ``infer_force`` wired to them in
the next phase inherits a robust contract) and assert finiteness + sane
recovery — robust, not necessarily competitive with the parked research paths.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest



# ── data generators (reuse the existing parametric_core test patterns) ──


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    """Run this module in float64, restoring the global flag afterwards.

    An import-time `jax.config.update` runs at pytest collection and
    leaks float64 into the whole session (order-independent poison)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _make_ou_data(T=8000, dt=0.02, d=2, seed=42, noise=0.0):
    rng = np.random.default_rng(seed)
    k = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)[:d, :d]
    D = np.diag(np.array([0.5, 0.3], dtype=np.float64)[:d])
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, d))
    X[0] = rng.normal(0, 0.5, size=d)
    for t in range(T - 1):
        X[t + 1] = X[t] + (-k @ X[t]) * dt + sqrt2D @ (rng.normal(size=d) * np.sqrt(dt))
    Y = X + noise * rng.standard_normal(X.shape) if noise else X
    return Y, k, D


def _make_dw_data(T=12000, dt=0.01, seed=1, D=0.5):
    """1D double well: F(x) = x - x^3 (wells at ±1)."""
    rng = np.random.default_rng(seed)
    sqrt2D = np.sqrt(2.0 * D)
    X = np.zeros((T, 1))
    X[0] = 1.0
    for t in range(T - 1):
        x = X[t, 0]
        X[t + 1, 0] = x + (x - x**3) * dt + sqrt2D * rng.normal() * np.sqrt(dt)
    return X


def _simulate_ud_harmonic(N=10000, dt=0.05, d=1, seed=42, k=1.0, gamma=0.5, D=0.1):
    rng = np.random.default_rng(seed)
    sqrt2D = np.sqrt(2 * D)
    X = np.zeros((N, d)); V = np.zeros((N, d))
    X[0] = rng.normal(0, 0.3, size=d); V[0] = rng.normal(0, 0.3, size=d)
    for t in range(N - 1):
        dW = rng.normal(size=d) * np.sqrt(dt)
        X[t + 1] = X[t] + V[t] * dt
        V[t + 1] = V[t] + (-k * X[t] - gamma * V[t]) * dt + sqrt2D * dW
    return X


def _coll(X_pos, dt):
    from SFI.trajectory.dataset import TrajectoryDataset
    from SFI.trajectory.collection import TrajectoryCollection
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X_pos[:, None, :]), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds)


def _linear_drift_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["A"] @ x

    return make_psf(f, dim=d, rank=1, n_features=1,
                    params=[ParamSpec("A", shape=(d, d), dtype=jnp.float64)])


def _ud_linear_psf(d):
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, v, params, mask=None, extras=None):
        return -params["k"] * x - params["gamma"] * v

    return make_psf(f, dim=d, rank=1, n_features=1, needs_v=True,
                    params=[ParamSpec("k", shape=(), dtype=jnp.float64),
                            ParamSpec("gamma", shape=(), dtype=jnp.float64)])


def _finite(x):
    return bool(np.all(np.isfinite(np.asarray(x))))


def _force_nmse_linear(basis, theta, X_eval, k):
    """NMSE of the inferred force field vs the true linear drift F(x) = -k x,
    evaluated on sample points (order-independent over basis layout)."""
    from SFI.statefunc import SF
    Fp = basis.to_psf()
    sf = SF(Fp, Fp.unflatten_params(jnp.asarray(theta)))
    F_hat = np.asarray(sf(jnp.asarray(X_eval)))
    F_true = -(np.asarray(X_eval) @ k.T)
    return np.mean((F_hat - F_true) ** 2) / np.mean(F_true ** 2)


# ── contract: the minimal defaults are a single RK4 step ──

def test_minimal_defaults_are_rk4_single_step():
    from SFI.inference.parametric_core import solve as S
    for name in ("solve_force_od", "solve_force_ud",
                 "solve_diffusion_od", "solve_diffusion_ud"):
        sig = inspect.signature(getattr(S, name))
        assert sig.parameters["integrator"].default == "rk4", name
        assert sig.parameters["n_substeps"].default == 1, name


def test_ud_euler_single_step_eiv_degeneracy_guard():
    """A single Euler step can't support the UD skip-trick (∂Φˣ/∂θ≡0).  The
    solver must guard it: warn, fall back to eiv=False, and return a finite,
    sane fit — never a silent singular Gram / blow-up."""
    import warnings
    from SFI.inference.parametric_core.solve import solve_force_ud

    X = _simulate_ud_harmonic(N=12000, dt=0.05, seed=7)
    coll = _coll(X, dt=0.05)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = solve_force_ud(coll, _ud_linear_psf(1), integrator="euler",
                             n_substeps=1, eiv=True, max_outer=6)
    assert any("skip-trick" in str(w.message) for w in rec), "expected degeneracy warning"
    assert res.info.get("eiv_w") == 0.0          # instrument disabled
    assert _finite(res.theta) and _finite(res.G)  # no blow-up
    p = _ud_linear_psf(1).unflatten_params(res.theta)
    assert abs(float(p["k"]) - 1.0) < 0.30, f"k={float(p['k'])}"


# ── OD force: GN / skip-trick path (linear Basis) ──

def test_od_force_minimal_basis_gn_recovers_ou():
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, D_true = _make_ou_data(T=10000, dt=0.02, seed=7)
    coll = _coll(Y, dt=0.02)
    basis = monomials_up_to(1, dim=2, rank="vector")

    res = solve_force_od(coll, basis, max_outer=6)  # all-default minimal config
    assert res.info.get("inner") == "gn"
    assert res.info.get("n_substeps") == 1
    assert _finite(res.theta) and _finite(res.G) and _finite(res.D)
    assert np.linalg.norm(np.asarray(res.D) - D_true) / np.linalg.norm(D_true) < 0.10
    # Gram non-singular
    assert np.linalg.cond(np.asarray(res.G)) < 1e8


def test_od_force_minimal_skip_trick_noise_robust():
    """eiv=True (skip-trick) default stays finite and roughly recovers drift
    under measurement noise — the consistent estimating equation."""
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, _ = _make_ou_data(T=12000, dt=0.02, seed=5, noise=0.03)
    coll = _coll(Y, dt=0.02)
    basis = monomials_up_to(1, dim=2, rank="vector")

    res = solve_force_od(coll, basis, max_outer=6)  # eiv=True by default
    assert _finite(res.theta) and _finite(res.G)
    nmse = _force_nmse_linear(basis, res.theta, Y[::10], k)
    assert nmse < 0.10, f"noisy drift field not recovered: NMSE={nmse:.3e}"


def test_od_force_minimal_double_well_basis():
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    X = _make_dw_data(T=15000, dt=0.01, seed=1)
    coll = _coll(X, dt=0.01)
    basis = monomials_up_to(3, dim=1, rank="vector")

    res = solve_force_od(coll, basis, max_outer=6)
    assert _finite(res.theta) and _finite(res.G)
    # F(x)=x - x^3 → coeffs [c0,c1,c2,c3] ≈ [0, 1, 0, -1]
    c = np.asarray(res.theta).ravel()
    assert abs(c[1] - 1.0) < 0.35 and abs(c[3] + 1.0) < 0.35, c


# ── OD force: loss path (PSF → L-BFGS IRLS) ──

def test_od_force_minimal_psf_loss_path():
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, _ = _make_ou_data(T=10000, dt=0.02, seed=9)
    coll = _coll(Y, dt=0.02)

    res = solve_force_od(coll, _linear_drift_psf(2), max_outer=6)  # PSF → lbfgs
    assert res.info.get("inner") == "lbfgs"
    A = np.asarray(_linear_drift_psf(2).unflatten_params(res.theta)["A"])
    assert _finite(A)
    assert np.linalg.norm(A - (-k)) / np.linalg.norm(k) < 0.20


# ── UD force ──

def test_ud_force_minimal_recovers_harmonic():
    from SFI.inference.parametric_core.solve import solve_force_ud

    k_t, g_t, D_t = 1.0, 0.5, 0.1
    X = _simulate_ud_harmonic(N=12000, dt=0.05, seed=7, k=k_t, gamma=g_t, D=D_t)
    coll = _coll(X, dt=0.05)

    res = solve_force_ud(coll, _ud_linear_psf(1), max_outer=6)
    p = _ud_linear_psf(1).unflatten_params(res.theta)
    k_h, g_h = float(np.asarray(p["k"])), float(np.asarray(p["gamma"]))
    D_h = float(np.asarray(res.D).ravel()[0])
    assert _finite(res.theta) and _finite(res.G)
    assert abs(k_h - k_t) / k_t < 0.20, f"k={k_h}"
    assert abs(g_h - g_t) / g_t < 0.40, f"gamma={g_h}"
    assert abs(D_h - D_t) / D_t < 0.35, f"D={D_h}"


# ── state-dependent diffusion (minimal config) ──

def test_od_diffusion_minimal_constant_D():
    from SFI.bases import monomials_up_to, symmetric_matrix_basis
    from SFI.inference.parametric_core.solve import solve_diffusion_od, solve_force_od

    Y, k, D_true = _make_ou_data(T=12000, dt=0.02, seed=3)
    coll = _coll(Y, dt=0.02)
    basis = monomials_up_to(1, dim=2, rank="vector")
    fres = solve_force_od(coll, basis, max_outer=6)

    D_basis = symmetric_matrix_basis(2)
    dres = solve_diffusion_od(coll, basis, fres.theta, D_basis,
                              Lambda=fres.Lambda)
    assert _finite(dres.theta_D)
    # The default θ_D init (moment-D̂ projection) must actually solve, not
    # return the singular θ_D=0 start: recovered D(x) ≈ D_true.
    D_psf = D_basis.to_psf()
    D_hat = np.asarray(D_psf(jnp.zeros(2)[None],
                             params=D_psf.unflatten_params(dres.theta_D))[0])
    rel = np.linalg.norm(D_hat - D_true) / np.linalg.norm(D_true)
    assert rel < 0.25, f"constant D not recovered (rel={rel:.3f}):\n{D_hat}\nvs\n{D_true}"


def test_ud_diffusion_minimal_velocity_model():
    from SFI.inference.parametric_core.solve import solve_diffusion_ud, solve_force_ud
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    D_true = 0.1
    X = _simulate_ud_harmonic(N=14000, dt=0.05, seed=4, D=D_true)
    coll = _coll(X, dt=0.05)
    fres = solve_force_ud(coll, _ud_linear_psf(1), max_outer=6)

    def fD(x, *, v, params, mask=None, extras=None):
        return jnp.array([[params["c0"] + params["c1"] * v[0] ** 2]])

    D_psf = make_psf(fD, dim=1, rank=2, n_features=1, needs_v=True,
                     params=[ParamSpec("c0", shape=(), dtype=jnp.float64),
                             ParamSpec("c1", shape=(), dtype=jnp.float64)])
    dres = solve_diffusion_ud(coll, _ud_linear_psf(1), fres.theta, D_psf,
                              Lambda=fres.Lambda, theta_D0={"c0": 0.08, "c1": 0.0})
    p = D_psf.unflatten_params(dres.theta_D)
    c0, c1 = float(np.asarray(p["c0"])), float(np.asarray(p["c1"]))
    assert _finite(dres.theta_D)
    assert abs(c0 - D_true) / D_true < 0.30, f"c0={c0}"
    assert abs(c1) < 0.06, f"c1={c1}"


# ── IV sandwich covariance ──

def test_iv_sandwich_error_bars_calibrated():
    """On noisy data the eiv path's parameter covariance is the sandwich
    G⁻¹HG⁻ᵀ.  Check (a) it differs from the naive G⁻¹, (b) predicted
    stderr matches the empirical seed-to-seed scatter of θ̂ within a
    factor ~2, (c) the symmetric path's H collapses to G."""
    import numpy as np
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    sigma = 0.05
    B = monomials_up_to(order=1, dim=1, rank="vector")
    thetas, preds = [], []
    for seed in range(8):
        Y, k, _ = _make_ou_data(T=4000, dt=0.05, d=1, seed=100 + seed,
                                noise=sigma)
        coll = _coll(Y, dt=0.05)
        res = solve_force_od(coll, B, max_outer=6)
        thetas.append(np.asarray(res.theta))
        preds.append(np.sqrt(np.diag(np.asarray(res.theta_cov))))
        if seed == 0:
            # sandwich ≠ naive G⁻¹ on the IV path under noise
            naive = np.linalg.inv(np.asarray(res.G))
            rel = np.linalg.norm(np.asarray(res.theta_cov) - naive) / np.linalg.norm(naive)
            assert rel > 1e-3, f"sandwich identical to naive G⁻¹ (rel={rel:.2e})"
    thetas = np.array(thetas)
    emp = thetas.std(axis=0, ddof=1)
    pred = np.mean(preds, axis=0)
    # the linear coefficient (largest-signal component)
    j = int(np.argmax(np.abs(thetas.mean(axis=0))))
    ratio = pred[j] / (emp[j] + 1e-30)
    assert 0.35 < ratio < 3.0, f"stderr calibration off: pred={pred[j]:.3g} emp={emp[j]:.3g}"


def test_symmetric_path_sandwich_collapses_to_Ginv():
    """eiv=False ⇒ H = G ⇒ theta_cov = G⁻¹ (information identity)."""
    import numpy as np
    from SFI.bases import monomials_up_to
    from SFI.inference.parametric_core.solve import solve_force_od

    Y, k, _ = _make_ou_data(T=3000, dt=0.05, d=1, seed=3)
    coll = _coll(Y, dt=0.05)
    res = solve_force_od(coll, monomials_up_to(order=1, dim=1, rank="vector"),
                         eiv=False, max_outer=4)
    naive = np.linalg.inv(np.asarray(res.G))
    rel = np.linalg.norm(np.asarray(res.theta_cov) - naive) / np.linalg.norm(naive)
    assert rel < 1e-6, f"symmetric sandwich ≠ G⁻¹ (rel={rel:.2e})"


def test_eiv_auto_resolution():
    """eiv="auto" → instrument ON for every model, interacting included (the
    instrument now uses the same N-body flow as the residual; the earlier
    auto-False for particles worked around an isolated-frame instrument whose
    pair-feature columns were identically zero).  Explicit values pass through."""
    from SFI.inference.parametric_core.solve import _resolve_eiv_auto

    class _Plain:
        particles_input = False

    class _Interacting:
        particles_input = True

    assert _resolve_eiv_auto("auto", _Plain()) is True
    assert _resolve_eiv_auto("auto", _Interacting()) is True
    assert _resolve_eiv_auto(True, _Interacting()) is True
    assert _resolve_eiv_auto(False, _Plain()) is False
    assert _resolve_eiv_auto(0.5, _Interacting()) == 0.5
