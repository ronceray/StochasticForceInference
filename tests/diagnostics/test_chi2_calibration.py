"""Finite-Δt calibration of the χ² (MSE-consistency) test.

The expansions behind ``residuals._chi2_calibration`` are re-derived from the
exact Gaussian moments of linear SDEs (no simulation), then the calibrated
test is checked on simulated data: centred where the plain ``χ²_d`` reference
reads several sigma low, with the diffusion estimator's own variance.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov as lyap

from SFI.bases.monomials import monomials_up_to
from SFI.diagnostics.residual_tests import mse_consistency
from SFI.diagnostics.residuals import (
    _D_STENCIL_BIAS,
    _LAMBDA_STENCIL_BIAS,
    build_residuals,
)
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import make_sf

# --------------------------------------------------------------------- #
# Exact Gaussian moments
# --------------------------------------------------------------------- #
_J_OD = np.array([[-1.0, 0.7], [-0.4, -2.0]])
_D2 = np.array([[0.6, 0.15], [0.15, 0.4]])


def _od_moments(dt):
    """E[dX_i dX_jᵀ] (lags 0, 1) and the one-step residual covariance, exact."""
    Sig = lyap(_J_OD, -2 * _D2)

    def E(a, b):
        return Sig @ expm(_J_OD.T * (b - a) * dt) if b >= a else expm(_J_OD * (a - b) * dt) @ Sig

    def inc(i, j):
        return E(i + 1, j + 1) - E(i + 1, j) - E(i, j + 1) + E(i, j)

    M = expm(_J_OD * dt) - np.eye(2) - _J_OD * dt
    resid = M @ Sig @ M.T + Sig - expm(_J_OD * dt) @ Sig @ expm(_J_OD.T * dt)
    B = _J_OD @ Sig @ _J_OD.T + _J_OD @ _D2 + _D2 @ _J_OD.T
    return inc(0, 0), inc(0, 1), resid, B


def test_overdamped_expansions_exact():
    """OD: residual covariance 2DΔt + Δt²(JD + DJᵀ); stencil biases β Δt B; Λ̂ = −Δt² B."""
    for dt in (4e-3, 2e-3):
        a, b, resid, B = _od_moments(dt)
        K = _J_OD @ _D2
        assert np.abs(resid - 2 * _D2 * dt - dt**2 * (K + K.T)).max() < 5 * dt**3
        stencils = {
            "MSD": 0.5 * a / dt,
            "WeakNoise": 0.25 * (2 * a - b - b.T) / dt,
            "noisy": 0.25 * (2 * a + 2 * b + 2 * b.T) / dt,
        }
        for name, Dh in stencils.items():
            beta = _D_STENCIL_BIAS["OD"][name][0]
            assert np.abs(Dh - _D2 - beta * dt * B).max() < 5 * dt**2, name
        lam = -0.5 * (b + b.T)
        assert np.abs(lam + _LAMBDA_STENCIL_BIAS["OD"] * dt**2 * B).max() < 5 * dt**3


def _ud_moments(Kx, Kv, D, dt):
    d = D.shape[0]
    Ay = np.block([[np.zeros((d, d)), np.eye(d)], [Kx, Kv]])
    Qn = np.zeros((2 * d, 2 * d))
    Qn[d:, d:] = 2 * D
    Sy = lyap(Ay, -Qn)

    def EX(a, b):
        C = Sy @ expm(Ay.T * (b - a) * dt) if b >= a else expm(Ay * (a - b) * dt) @ Sy
        return C[:d, :d]

    idx = [-1, 0, 1, 2]
    G = np.block([[EX(a, b) for b in idx] for a in idx])

    def cov(ca, cb):
        return np.kron(np.array(ca)[None], np.eye(d)) @ G @ np.kron(np.array(cb)[None], np.eye(d)).T

    dXm, dX, dXp = [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]
    a_, b_, c_ = cov(dX, dX), cov(dXm, dXm), cov(dXp, dXp)
    d_, e_, f_ = cov(dXp, dXm), cov(dX, dXp), cov(dX, dXm)

    def sym(M):
        return 0.5 * (M + M.T)

    stencils = {
        "MSD": 0.75 * cov([1, -2, 1, 0], [1, -2, 1, 0]) / dt**3,
        "WeakNoise": 0.5 * cov([-1, 3, -3, 1], [-1, 3, -3, 1]) / dt**3,
        "noisy": (3 / 11) * sym(-a_ + b_ + c_ - 3 * d_ + e_ + f_) / dt**3,
    }
    lam = sym(10 * a_ + b_ + c_ + 8 * d_ - 10 * e_ - 10 * f_) / 44
    R = (np.kron(np.array([1, -2, 1, 0])[None] / dt**2, np.eye(d))
         - Kx @ np.kron(np.array([1, 1, 1, 0])[None] / 3, np.eye(d))
         - Kv @ np.kron(np.array([-1, 0, 1, 0])[None] / (2 * dt), np.eye(d)))
    resid = R @ G @ R.T
    S = np.hstack([Kx, Kv]) @ Sy @ np.hstack([Kx, Kv]).T
    B = S + Kv @ D + D @ Kv.T
    return stencils, lam, resid, B


def test_underdamped_expansions_exact():
    """UD: residual covariance (4/3)D/Δt with no O(1) term; biases β Δt B_v; Λ̂ = −(3/22)Δt⁴ B_v."""
    D = np.array([[0.6, 0.1], [0.1, 0.3]])
    Kx = -np.array([[2.0, 0.6], [-0.4, 4.0]])
    Kv = -np.array([[0.7, 0.3], [-0.2, 1.1]])
    lims = {}
    for dt in (4e-3, 2e-3):
        stencils, lam, resid, B = _ud_moments(Kx, Kv, D, dt)
        # O(Δt) absolute remainder = O(Δt²) relative to (4/3)D/Δt.
        assert np.abs(resid - (4 / 3) * D / dt).max() < 2 * dt
        for name, Dh in stencils.items():
            lims.setdefault(name, []).append((Dh - D) / dt)
        lims.setdefault("Lambda", []).append(lam / dt**4)
    for name in ("MSD", "WeakNoise", "noisy"):
        rich = 2 * lims[name][1] - lims[name][0]  # Richardson: removes the O(Δt) remainder
        assert np.allclose(rich, _D_STENCIL_BIAS["UD"][name][0] * B, atol=2e-3 * np.abs(B).max()), name
    rich = 2 * lims["Lambda"][1] - lims["Lambda"][0]
    assert np.allclose(rich, -_LAMBDA_STENCIL_BIAS["UD"] * B, atol=2e-3 * np.abs(B).max())


# --------------------------------------------------------------------- #
# Simulated data
# --------------------------------------------------------------------- #
_J_SIM = np.array([[-1.0, 0.5], [-0.5, -2.0]])


def _ou_collection(seed=0, Nsteps=20000, dt=0.02):
    F = make_sf(lambda x: jnp.asarray(_J_SIM) @ x, dim=2, rank=1)
    proc = OverdampedProcess(F, D=0.5)
    proc.initialize(jnp.zeros(2))
    return proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=200, oversampling=10)


def _ou_fit(coll, method):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method=method)
    inf.infer_force_linear(monomials_up_to(order=1, dim=2, rank="vector"), preset="clean")
    return inf


@pytest.fixture(scope="module")
def ou_coll():
    return _ou_collection()


def test_overdamped_expected_norm_and_stderr(ou_coll):
    """E‖z‖² = d + Δt⟨∇·F̂⟩ (WeakNoise); stderr 1/√2 (WeakNoise), √2 (noisy) of √(2d/n)."""
    dt = 0.02
    inf = _ou_fit(ou_coll, "WeakNoise")
    b = build_residuals(inf)
    assert b.chi2_calibration["finite_dt"] == "applied"
    assert abs((b.chi2_expected - 2) - dt * np.trace(_J_SIM)) < 0.15 * dt * abs(np.trace(_J_SIM))
    naive = np.sqrt(2 * b.d / b.n_obs)
    assert 0.6 < b.chi2_stderr / naive < 0.85
    out = mse_consistency(inf, b)
    z_plain = (out["mean_mahalanobis_norm"] - b.d) / naive
    assert z_plain < -2.5          # the uncalibrated reference reads several sigma low
    assert abs(out["excess_z"]) < 2.5

    b_noisy = build_residuals(_ou_fit(ou_coll, "noisy"))
    assert 1.25 < b_noisy.chi2_stderr / naive < 1.65


def test_msd_in_sample_is_uninformative(ou_coll):
    """With the MSD diffusion on the fitting data the excess is fixed by the fit."""
    inf = _ou_fit(ou_coll, "MSD")
    out = mse_consistency(inf, build_residuals(inf))
    assert np.isnan(out["excess_z"]) and np.isnan(out["realised_NMSE"])
    assert "MSD" in out["note"]
    # Held out, the training D̂ is independent of the test residuals: informative again.
    train, test = ou_coll.split_time(0.7)
    inf_t = _ou_fit(train, "MSD")
    out_t = mse_consistency(inf_t, build_residuals(inf_t, data=test))
    assert np.isfinite(out_t["excess_z"]) and abs(out_t["excess_z"]) < 4.0


def test_underdamped_weaknoise_has_no_finite_dt_term():
    F = make_sf(lambda x, v: -x - 0.5 * v, dim=1, rank=1, needs_v=True)
    proc = UnderdampedProcess(F, D=0.3)
    proc.initialize(jnp.zeros(1), v0=jnp.zeros(1))
    coll = proc.simulate(dt=0.02, Nsteps=20000, key=random.PRNGKey(1), prerun=200, oversampling=40)
    B = monomials_up_to(order=1, dim=1, rank="vector", include_v=True)
    zs = {}
    for method in ("WeakNoise", "noisy"):
        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method=method)
        inf.infer_force_linear(B, preset="clean")
        b = build_residuals(inf)
        out = mse_consistency(inf, b)
        zs[method] = out["excess_z"]
        if method == "WeakNoise":
            assert b.chi2_expected == pytest.approx(1.0, abs=1e-6)  # float32 whitening
        else:
            # D̂_noisy = D + (15/11) Δt B_v with B_v > 0 here: the norms read low.
            assert b.chi2_expected < 1.0
    assert all(abs(z) < 3.0 for z in zs.values())
