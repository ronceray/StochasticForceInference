"""Tests for UnderdampedLangevinInference.compute_entropy_production().

Benchmark: the inertial driven trap

    dx/dt = v,   dv/dt = -k x + eps (z_hat x x) - gamma v + sqrt(2 D) xi,

a linear system, hence a phase-space OU process with exact entropy
production rate from the PARITY-AWARE Lyapunov formula (x even, v odd):

    sigma = Tr[M^T D_z^+ M S],   M = (A + E A E)/2 + D_z S^-1,

with S the stationary covariance (A S + S A^T = -2 D_z), E = diag(I, -I)
and D_z^+ the velocity-block pseudoinverse.  Note the all-even-variable
formula A - D S^-1 over-counts by including the reversible Hamiltonian
part.  Cross-checked in the helper against the Sekimoto heat rate
(gamma <|v|^2> - d D)/T with T = D/gamma.

The estimator's fluctuation scale is set by the odd quadratic
<F^- D^-1 F^-> = gamma^2 <|v|^2>/D (the log path-ratio's two large
contributions fluctuate even where their means cancel), which the
provisional error bar must reflect.
"""

import logging

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax import random

from SFI.bases import monomials_up_to, unit_axes, v_components, x_components
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import UnderdampedProcess

K_TRAP = 1.0
GAMMA = 1.0
D0 = 0.5
EPS = 0.5
DT = 0.01
NSTEPS = 40_000


def sigma_exact_lyapunov(k, gamma, D0, eps):
    """Exact EP rate of the linear UD trap+curl (parity-aware; Sekimoto-pinned)."""
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    A = np.block([
        [np.zeros((2, 2)), np.eye(2)],
        [-k * np.eye(2) + eps * J, -gamma * np.eye(2)],
    ])
    D_z = np.zeros((4, 4))
    D_z[2:, 2:] = D0 * np.eye(2)
    S = scipy.linalg.solve_lyapunov(A, -2.0 * D_z)
    E = np.diag([1.0, 1.0, -1.0, -1.0])
    M = 0.5 * (A + E @ A @ E) + D_z @ np.linalg.inv(S)
    sigma = float(np.trace(M.T @ np.linalg.pinv(D_z) @ M @ S))
    v2 = float(np.trace(S[2:, 2:]))
    sigma_heat = (gamma * v2 - 2 * D0) / (D0 / gamma)
    assert abs(sigma - sigma_heat) < 1e-8 * max(1.0, abs(sigma))
    return sigma, v2


def _make_force(eps, k=K_TRAP, gamma=GAMMA):
    x0c, x1c = x_components(2)
    v0c, v1c = v_components(2)
    e0, e1 = unit_axes(2)
    return (
        (-k * x0c - eps * x1c - gamma * v0c) * e0
        + (-k * x1c + eps * x0c - gamma * v1c) * e1
    )


def _simulate(eps, *, seed, dt=DT, Nsteps=NSTEPS, oversampling=8):
    proc = UnderdampedProcess(_make_force(eps), D=D0)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    return proc.simulate(
        dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed),
        prerun=500, oversampling=oversampling, compute_observables=True,
    )


def _base_basis():
    # {x0, x1, v0, v1} x {e0, e1}: 8 features, contains the true model.
    return monomials_up_to(
        1, dim=2, include_constant=False, include_v=True, rank="vector"
    )


def _fit(coll, basis=None):
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis if basis is not None else _base_basis())
    return inf


@pytest.fixture(scope="module")
def coll_neq():
    return _simulate(EPS, seed=0)


@pytest.fixture(scope="module")
def fit_neq(coll_neq):
    return _fit(coll_neq)


def test_driven_rate_matches_exact(fit_neq):
    sigma, _ = sigma_exact_lyapunov(K_TRAP, GAMMA, D0, EPS)
    out = fit_neq.compute_entropy_production()
    window = 4.0 * out["Sdot_error"] + out["Sdot_bias"] + 0.1 * sigma
    assert abs(out["Sdot"] - sigma) < window, (out, sigma)
    # Attributes for print_report / report_dict
    assert fit_neq.DeltaS == pytest.approx(out["DeltaS"])
    assert fit_neq.error_DeltaS == pytest.approx(out["error_DeltaS"])


def test_equilibrium_null():
    coll_eq = _simulate(0.0, seed=1)
    inf = _fit(coll_eq)
    out = inf.compute_entropy_production()
    assert abs(out["Sdot"]) <= out["Sdot_bias"] + 3.0 * out["Sdot_error"], out
    # The odd sector carries plenty of information (friction) but almost no
    # irreversibility: the dimensionless ratio is small at equilibrium.
    assert out["Q_odd"] > 10.0 * abs(out["DeltaS"])
    assert abs(inf.entropy_odd_ratio) < 0.1


def test_parity_rank_counting(coll_neq, fit_neq):
    # Base basis: odd features are {v0, v1} x {e0, e1} -> N- = 4.
    out = fit_neq.compute_entropy_production()
    assert out["Nminus"] == 4
    assert out["Sdot_bias"] == pytest.approx(2.0 * 4 / out["tauN"], rel=1e-6)
    # Debiased fields: odd-sector fluctuation bias subtracted.
    assert out["DeltaS_debiased"] == pytest.approx(out["DeltaS"] - 2.0 * 4)
    assert out["Sdot_debiased"] == pytest.approx(out["Sdot"] - out["Sdot_bias"])

    # Augmented basis (order 2, with constant): odd scalars are
    # {v0, v1, x0 v0, x0 v1, x1 v0, x1 v1} -> 12 odd features of 30.
    B2 = monomials_up_to(2, dim=2, include_constant=True, include_v=True, rank="vector")
    inf2 = _fit(coll_neq, basis=B2)
    out2 = inf2.compute_entropy_production()
    assert out2["Nminus"] == 12
    # Same trajectory, larger (even-augmented) basis: compatible estimates.
    tol = 3.0 * (out["Sdot_error"] + out2["Sdot_error"]) + out2["Sdot_bias"]
    assert abs(out["Sdot"] - out2["Sdot"]) < tol, (out["Sdot"], out2["Sdot"])


def test_simulator_observable_cross_check(coll_neq, fit_neq):
    out = fit_neq.compute_entropy_production()
    S_sim = coll_neq.datasets[0].meta["observables"]["entropy"]
    assert abs(out["DeltaS"] - S_sim) < 5.0 * out["error_DeltaS"] + 0.15 * abs(S_sim), (
        out["DeltaS"],
        S_sim,
    )


def test_support_after_sparsify(coll_neq):
    B2 = monomials_up_to(2, dim=2, include_constant=True, include_v=True, rank="vector")
    inf = _fit(coll_neq, basis=B2)
    out_full = inf.compute_entropy_production(support="full")
    inf.sparsify_force(criterion="PASTIS")
    out_cur = inf.compute_entropy_production(support="current")
    # The plug-in value is set by the (sparsified) fit; support only changes
    # the reported odd-sector dof count.
    assert out_cur["Nminus"] <= out_full["Nminus"]
    assert out_cur["Sdot_bias"] == pytest.approx(
        2.0 * out_cur["Nminus"] / out_cur["tauN"], rel=1e-6
    )
    sigma, _ = sigma_exact_lyapunov(K_TRAP, GAMMA, D0, EPS)
    window = 4.0 * out_cur["Sdot_error"] + out_cur["Sdot_bias"] + 0.1 * sigma
    assert abs(out_cur["Sdot"] - sigma) < window, (out_cur, sigma)


def test_resolution_guard_warns(caplog):
    # gamma * dt = 2 >> 0.5: velocities are unresolved.
    coll = _simulate(EPS, seed=2, dt=2.0, Nsteps=2_000, oversampling=200)
    inf = _fit(coll)
    with caplog.at_level(logging.WARNING, logger="SFI.inference.underdamped"):
        inf.compute_entropy_production()
    assert any("underresolved" in rec.message for rec in caplog.records), (
        [rec.message for rec in caplog.records]
    )


def test_requires_linear_fit(coll_neq):
    inf = UnderdampedLangevinInference(coll_neq)
    inf.compute_diffusion_constant(method="WeakNoise")
    with pytest.raises(RuntimeError, match="infer_force_linear"):
        inf.compute_entropy_production()
    inf.metadata["force_method"] = "parametric"
    with pytest.raises(RuntimeError, match="not supported"):
        inf.compute_entropy_production()


def test_time_reversal_split(fit_neq):
    F_even, F_odd = fit_neq.time_reversal_split()
    x = jnp.array([[0.4, -0.3], [1.0, 0.2]])
    v = jnp.array([[0.5, 1.1], [-0.7, 0.3]])
    total = np.asarray(F_even(x, v) + F_odd(x, v))
    direct = np.asarray(fit_neq.force_inferred(x, v=v))
    np.testing.assert_allclose(total, direct, rtol=1e-5, atol=1e-6)
    # Odd part vanishes at v=0; even part is v-independent for this fit.
    np.testing.assert_allclose(np.asarray(F_odd(x, 0.0 * v)), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(F_even(x, v)), np.asarray(F_even(x, -v)), rtol=1e-5, atol=1e-6
    )


def test_crossfit_coefficients_override(coll_neq):
    """Cross-fitting (fit on one half, evaluate on the other) removes the
    same-sample plug-in correlation bias; the override hook must accept a
    full-basis coefficient vector and validate its shape."""
    from SFI.trajectory import TrajectoryCollection

    _, X, _ = coll_neq.to_arrays(dataset=0)
    X = jnp.asarray(X)
    T = int(X.shape[0]) // 2
    collA = TrajectoryCollection.from_arrays(X=X[:T], dt=DT)
    collB = TrajectoryCollection.from_arrays(X=X[T:], dt=DT)
    infA, infB = _fit(collA), _fit(collB)

    outAB = infB.compute_entropy_production(coefficients=infA.force_coefficients_full)
    outBA = infA.compute_entropy_production(coefficients=infB.force_coefficients_full)
    rate = 0.5 * (outAB["Sdot"] + outBA["Sdot"])
    err = 0.5 * (outAB["Sdot_error"] ** 2 + outBA["Sdot_error"] ** 2) ** 0.5

    sigma, _ = sigma_exact_lyapunov(K_TRAP, GAMMA, D0, EPS)
    assert abs(rate - sigma) < 4.0 * err + outAB["Sdot_bias"] + 0.1 * sigma, (rate, sigma)

    with pytest.raises(ValueError, match="coefficients"):
        infB.compute_entropy_production(coefficients=jnp.zeros(3))


def test_print_report_and_report_dict_smoke(fit_neq, capsys):
    fit_neq.compute_entropy_production()
    fit_neq.print_report()
    assert "Entropy production" in capsys.readouterr().out
    d = fit_neq.report_dict()
    assert isinstance(d["DeltaS"], float)
    assert isinstance(d["error_DeltaS"], float)


def test_gamma_dt_calibration():
    """Empirical finite-Delta-t calibration on the exact benchmark.

    Documents the estimator's bias envelope as the sampling interval
    approaches the velocity correlation time; the printed table feeds the
    research note.
    """
    sigma, _v2 = sigma_exact_lyapunov(K_TRAP, GAMMA, D0, EPS)
    total_time = 300.0
    rows = []
    for i, gdt in enumerate([0.02, 0.05, 0.1, 0.2, 0.4]):
        dt = gdt / GAMMA
        Nsteps = int(total_time / dt)
        oversampling = max(8, int(round(dt / 0.0025)))
        coll = _simulate(EPS, seed=10 + i, dt=dt, Nsteps=Nsteps, oversampling=oversampling)
        out = _fit(coll).compute_entropy_production()
        rows.append((gdt, out["Sdot"], out["Sdot_error"], out["Sdot_bias"]))
        # Loose envelope: statistical window + a linear-in-gdt bias allowance.
        window = 4.0 * out["Sdot_error"] + out["Sdot_bias"] + 3.0 * sigma * gdt + 0.1 * sigma
        assert abs(out["Sdot"] - sigma) < window, (gdt, out, sigma)

    print(f"\ngamma*dt calibration (sigma_exact = {sigma:.4f}):")
    print(f"{'g*dt':>6} {'Sdot':>8} {'err':>7} {'bias':>7} {'rel.dev':>8}")
    for gdt, sdot, err, bias in rows:
        print(f"{gdt:>6.2f} {sdot:>8.4f} {err:>7.4f} {bias:>7.4f} {(sdot - sigma) / sigma:>+8.1%}")
    # Fine-sampling points must be accurate.
    for gdt, sdot, err, bias in rows[:2]:
        assert abs(sdot - sigma) < 4.0 * err + bias + 0.1 * sigma, (gdt, sdot, sigma)
