"""Tests for the OD-vs-UD dynamics classifier (SFI.diagnostics.dynamics_order).

The classifier rests on the lag-resolved displacement covariance
``C_k = <Delta x_t . Delta x_{t+k}>``.  Two analytic signatures pin the
covariance backbone:

* **Pure localization noise** (a static point seen through i.i.d. Gaussian
  measurement error sigma): ``C0 = 2 sigma^2 I``, ``C1 = -sigma^2 I``,
  ``C2 = 0`` — the lag>=2 measurement-noise immunity that makes the test robust.
* **Constant velocity** (a smooth, ballistic path): ``C0 = C1 = C2 = (v0 v0^T) dt^2``
  — full momentum persistence, the underdamped limit.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import TrajectoryCollection
from SFI.bases import unit_axes, x_components
from SFI.bases.constants import identity_matrix_basis
from SFI.diagnostics.dynamics_order import (
    DynamicsOrderReport,
    _covariances_vs_dt,
    _decide_verdict,
    _dynamics_model,
    _fit_dynamics,
    _increment_covariances,
    _scaling_statistics,
    classify_dynamics,
)
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc.factory import make_basis


def _isotropic_scan(h, c0, c1, c2, d=2, n_eff=1e6):
    """Build a (S, d, d) isotropic covariance scan from scalar per-axis lags."""
    eye = np.eye(d)
    return {
        "strides": np.arange(len(h)),
        "dt": np.asarray(h, dtype=float),
        "C0": c0[:, None, None] * eye,
        "C1": c1[:, None, None] * eye,
        "C2": c2[:, None, None] * eye,
        "n_eff": np.full(len(h), n_eff),
    }


def test_localization_noise_covariance_signature():
    """White localization noise lives only at lag 0 and lag 1."""
    rng = np.random.default_rng(0)
    T, d = 20000, 2
    sigma = 0.3
    X = sigma * rng.standard_normal((T, d))  # static point + i.i.d. noise
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1)

    cov = _increment_covariances(coll)
    C0 = np.asarray(cov["C0"])
    C1 = np.asarray(cov["C1"])
    C2 = np.asarray(cov["C2"])

    s2 = sigma**2
    assert np.allclose(C0, 2.0 * s2 * np.eye(d), atol=0.02)
    assert np.allclose(C1, -s2 * np.eye(d), atol=0.02)
    assert np.allclose(C2, np.zeros((d, d)), atol=0.02)  # noise-immune at lag 2


def test_constant_velocity_full_persistence():
    """A smooth ballistic path has identical covariance at every lag."""
    T = 5000
    dt = 0.1
    v0 = np.array([1.0, -0.5])
    X = (np.arange(T)[:, None] * dt) * v0[None, :]  # x_t = v0 * t * dt
    coll = TrajectoryCollection.from_arrays(X=X, dt=dt)

    cov = _increment_covariances(coll)
    expected = np.outer(v0, v0) * dt**2
    assert np.allclose(np.asarray(cov["C0"]), expected, rtol=1e-3, atol=1e-9)
    assert np.allclose(np.asarray(cov["C1"]), expected, rtol=1e-3, atol=1e-9)
    assert np.allclose(np.asarray(cov["C2"]), expected, rtol=1e-3, atol=1e-9)


def test_covariances_vs_dt_scan():
    """The dt scan coarse-grains by stride: dt and covariances scale accordingly."""
    T = 4000
    dt = 0.05
    v0 = np.array([1.0, -0.5])
    X = (np.arange(T)[:, None] * dt) * v0[None, :]  # ballistic
    coll = TrajectoryCollection.from_arrays(X=X, dt=dt)

    strides = (1, 2, 4)
    scan = _covariances_vs_dt(coll, strides=strides)

    assert np.allclose(scan["dt"], np.asarray(strides) * dt, rtol=1e-6)
    for i, s in enumerate(strides):
        expected = np.outer(v0, v0) * (s * dt) ** 2  # ballistic: C ~ dt^2
        assert np.allclose(scan["C0"][i], expected, rtol=1e-3)
        assert np.allclose(scan["C2"][i], expected, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Forward covariance model (parametric layer)
# --------------------------------------------------------------------------- #
def test_model_pure_localization_noise():
    """Localization noise: +2 sigma^2 at lag 0, -sigma^2 at lag 1, 0 at lag 2."""
    h = np.array([0.1, 0.2, 0.4])
    s2 = 0.09
    c0, c1, c2 = _dynamics_model(h, sigma2=s2, D=0.0, V=0.0, gamma=1.0)
    assert np.allclose(c0, 2.0 * s2)
    assert np.allclose(c1, -s2)
    assert np.allclose(c2, 0.0)


def test_model_pure_diffusion():
    """Overdamped free diffusion: 2 D h at lag 0, nothing at lags 1, 2."""
    h = np.array([0.1, 0.2, 0.4])
    D = 0.5
    c0, c1, c2 = _dynamics_model(h, sigma2=0.0, D=D, V=0.0, gamma=1.0)
    assert np.allclose(c0, 2.0 * D * h)
    assert np.allclose(c1, 0.0)
    assert np.allclose(c2, 0.0)


def test_model_inertia_ballistic_small_h():
    """Inertia at fine sampling is ballistic: C0=C1=C2=V h^2, so rho2 -> 1."""
    V, gamma = 2.0, 1.0
    h = np.array([1e-3])
    c0, c1, c2 = _dynamics_model(h, sigma2=0.0, D=0.0, V=V, gamma=gamma)
    assert np.allclose(c0, V * h**2, rtol=1e-2)
    assert np.allclose(c1, V * h**2, rtol=1e-2)
    assert np.allclose(c2, V * h**2, rtol=1e-2)
    assert c2[0] / c0[0] > 0.99  # full momentum persistence


def test_model_inertia_decorrelates_at_coarse_h():
    """At coarse sampling (gamma h >> 1) the inertial lag-2 term collapses."""
    V, gamma = 2.0, 1.0
    h = np.array([10.0])  # gamma h = 10
    c0, c1, c2 = _dynamics_model(h, sigma2=0.0, D=0.0, V=V, gamma=gamma)
    # lag-2 correlation is exp(-2 gamma h) of the lag-1 scale: negligible
    assert c2[0] / c0[0] < 0.01


def test_model_stable_at_extreme_gamma_h():
    """The forward model must stay finite when the fit explores huge gamma."""
    c0, c1, c2 = _dynamics_model(np.array([1.0]), sigma2=0.0, D=0.0, V=1.0, gamma=1e5)
    assert np.all(np.isfinite(c0)) and np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))
    assert 0.0 <= c2[0] < c0[0]


def test_fit_recovers_known_parameters():
    """The fit inverts the forward model (exact-data round trip)."""
    true = dict(sigma2=0.02, D=0.3, V=1.5, gamma=2.0)
    dt0 = 0.02
    strides = np.array([1, 2, 4, 8, 16, 32, 64])
    h = strides * dt0
    c0, c1, c2 = _dynamics_model(h, **true)
    scan = _isotropic_scan(h, c0, c1, c2)

    fit = _fit_dynamics(scan)
    p = fit["params"]
    assert np.isclose(p["sigma2"], true["sigma2"], rtol=0.05, atol=1e-3)
    assert np.isclose(p["D"], true["D"], rtol=0.05)
    assert np.isclose(p["V"], true["V"], rtol=0.05)
    assert np.isclose(p["gamma"], true["gamma"], rtol=0.1)
    assert np.isclose(fit["tau_v"], 1.0 / true["gamma"], rtol=0.1)


def test_fit_prefers_ud_for_inertial_data_and_od_for_diffusive():
    """AICc comparison favours the inertial model only when inertia is present."""
    dt0 = 0.02
    strides = np.array([1, 2, 4, 8, 16, 32, 64])
    h = strides * dt0

    # Inertial data -> UD preferred (delta_aicc = AICc_OD - AICc_UD > 0)
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.0, V=1.5, gamma=2.0)
    fit_ud = _fit_dynamics(_isotropic_scan(h, c0, c1, c2))
    assert fit_ud["delta_aicc"] > 0.0

    # Purely diffusive + noise -> OD preferred (no inertia to find)
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.3, V=0.0, gamma=1.0)
    fit_od = _fit_dynamics(_isotropic_scan(h, c0, c1, c2))
    assert fit_od["delta_aicc"] < 0.0


# --------------------------------------------------------------------------- #
# Model-free scaling statistics (Layer 1)
# --------------------------------------------------------------------------- #
def test_scaling_statistics_separates_od_ud():
    """Noise-immune rho2 and apparent-KE slope distinguish OD from UD."""
    dt0 = 0.02
    strides = np.array([1, 2, 4, 8, 16, 32])
    h = strides * dt0

    # Overdamped (noise + diffusion): rho2 ~ 0, K ~ 1/h (slope ~ -1)
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.3, V=0.0, gamma=1.0)
    st = _scaling_statistics(_isotropic_scan(h, c0, c1, c2))
    assert np.all(np.abs(st["rho2"]) < 0.05)
    assert st["beta"] < -0.7

    # Underdamped (inertia, finely sampled): positive lag-2 persistence, flatter K
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.0, V=1.5, gamma=1.0)
    st = _scaling_statistics(_isotropic_scan(h, c0, c1, c2))
    assert st["rho2"][0] > 0.2
    assert st["beta"] > -0.6


# --------------------------------------------------------------------------- #
# Verdict logic
# --------------------------------------------------------------------------- #
def _verdict_from_lags(h, c0, c1, c2):
    scan = _isotropic_scan(h, c0, c1, c2)
    return _decide_verdict(_fit_dynamics(scan), _scaling_statistics(scan), scan)


def test_verdict_overdamped():
    dt0 = 0.02
    h = np.array([1, 2, 4, 8, 16, 32]) * dt0
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.3, V=0.0, gamma=1.0)
    assert _verdict_from_lags(h, c0, c1, c2) == "OD"


def test_verdict_underdamped_resolved():
    dt0 = 0.02
    h = np.array([1, 2, 4, 8, 16, 32]) * dt0
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.0, V=1.5, gamma=1.0)  # gamma*h_min = 0.02
    assert _verdict_from_lags(h, c0, c1, c2) == "UD"


def test_verdict_inconclusive_when_underresolved():
    dt0 = 0.2
    h = np.array([1, 2, 4, 8]) * dt0  # gamma*h_min = 0.8: marginal momentum resolution
    c0, c1, c2 = _dynamics_model(h, sigma2=0.02, D=0.0, V=2.0, gamma=4.0)
    assert _verdict_from_lags(h, c0, c1, c2) == "inconclusive"


# --------------------------------------------------------------------------- #
# End-to-end classification on simulated trajectories (physics validation)
# --------------------------------------------------------------------------- #
def _simulate_ou(k=1.0, D0=0.2, dt=0.01, Nsteps=20000, seed=0):
    """1-D Ornstein--Uhlenbeck (overdamped)."""
    def ou_force(x, *, mask=None):
        return jnp.array([-k * x[0]])

    F = make_basis(ou_force, dim=1, rank=1, n_features=1)
    proc = OverdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([D0])})
    proc.initialize(jnp.zeros(1, dtype=jnp.float32))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=100, oversampling=5)


def _simulate_friction_ud(gamma=1.0, D0=1.0, dt=0.02, Nsteps=20000, seed=1):
    """1-D free particle with friction (underdamped; OU velocity, <v^2>=D0/gamma)."""
    def friction_force(x, *, v, mask=None):
        return jnp.array([-gamma * v[0]])

    F = make_basis(friction_force, dim=1, rank=1, n_features=1, needs_v=True)
    proc = UnderdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([D0])})
    proc.initialize(jnp.zeros(1, dtype=jnp.float32), v0=jnp.zeros(1, dtype=jnp.float32))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=200, oversampling=5)


@pytest.fixture(scope="module")
def ou_coll():
    return _simulate_ou()


@pytest.fixture(scope="module")
def ud_coll():
    return _simulate_friction_ud()


def test_classify_overdamped(ou_coll):
    rep = classify_dynamics(ou_coll, cross_check=False)
    assert isinstance(rep, DynamicsOrderReport)
    assert rep.verdict == "OD"


def test_classify_overdamped_robust_to_localization_noise(ou_coll):
    """Localization noise inflates C0/C1 but lag>=2 is immune: verdict stays OD."""
    noisy = ou_coll.degrade(noise=0.05, seed=7)
    rep = classify_dynamics(noisy, cross_check=False)
    assert rep.verdict == "OD"


def test_classify_underdamped(ud_coll):
    rep = classify_dynamics(ud_coll, cross_check=False)
    assert rep.verdict == "UD"
    assert rep.fit["tau_v"] > 0.0


def test_cross_check_flags_od_misfit_on_underdamped(ud_coll):
    """An overdamped fit to underdamped data leaves autocorrelated residuals."""
    rep = classify_dynamics(ud_coll, cross_check=True)
    p = rep.cross_check["ljung_box_pvalue"]
    assert p is not None and p < 0.05


def test_report_serialization_and_summary(ou_coll, capsys):
    rep = classify_dynamics(ou_coll, cross_check=False)
    d = rep.to_dict()
    assert d["verdict"] == "OD"
    rep.print_summary()
    out = capsys.readouterr().out
    assert "verdict" in out.lower()


# --------------------------------------------------------------------------- #
# Worked gallery scenarios (mirror examples/gallery/dynamics_order_demo.py)
# --------------------------------------------------------------------------- #
def _simulate_2d_rotational_od(k=1.0, omega=1.0, D0=0.15, dt=0.02, Nsteps=12000, seed=0):
    """2D overdamped particle in a harmonic trap with a non-conservative
    rotational (curl) force ``F = -k x + omega (z x x)``.

    Driven out of equilibrium (steady probability current) but still
    first-order, so the verdict must stay ``OD``.  Exercises the d>1 isotropic
    component-pooling path.
    """
    x0, x1 = x_components(2)
    e0, e1 = unit_axes(2)
    F_rot = (-k * x0 - omega * x1) * e0 + (-k * x1 + omega * x0) * e1
    proc = OverdampedProcess(F_rot, D=D0)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=200, oversampling=10)


def _simulate_free_inertial_coarse(gamma=1.0, D0=1.0, dt=0.7, Nsteps=500_000, seed=1):
    """Freely diffusing inertial particle (``F = -gamma v``) sampled coarsely.

    At ``gamma*dt ~ 0.7`` momentum is only half-resolved.  Mirrors the gallery
    demo's ``inconclusive`` scenario; a large sample is needed for the weak
    velocity signal to register (``V_z > 3``).
    """
    def free_inertial(x, *, v, mask=None):
        return jnp.array([-gamma * v[0]])

    F = make_basis(free_inertial, dim=1, rank=1, n_features=1, needs_v=True)
    proc = UnderdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([D0])})
    proc.initialize(jnp.zeros(1, dtype=jnp.float32), v0=jnp.zeros(1, dtype=jnp.float32))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=500, oversampling=10)


def test_classify_2d_rotational_overdamped_driven():
    """A driven 2D overdamped system (non-conservative rotational force) is out
    of equilibrium but first-order: verdict OD, and d=2 pooling must work."""
    coll = _simulate_2d_rotational_od().degrade(noise=0.06, seed=12)
    rep = classify_dynamics(coll, cross_check=False)
    assert rep.meta["d"] == 2
    assert rep.verdict == "OD"


@pytest.mark.slow
def test_classify_inconclusive_coarse_sampling():
    """A coarsely-sampled inertial particle sits on the resolution boundary:
    the fit still detects inertia but the noise-immune lag-2 persistence has
    decayed into the gray zone, so the classifier abstains."""
    coll = _simulate_free_inertial_coarse()
    rep = classify_dynamics(coll, cross_check=False)
    assert rep.verdict == "inconclusive"
    assert rep.meta["gamma_dt_min"] > 0.3  # coarsely sampled: momentum marginal
