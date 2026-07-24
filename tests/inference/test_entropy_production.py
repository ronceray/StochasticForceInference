"""Tests for OverdampedLangevinInference.compute_entropy_production().

Physical pinning system: a 2D isotropic harmonic trap with a rotational
(curl) drive,

    F(x) = -k x + eps (z_hat x x),   D = D0 * Id.

The non-equilibrium steady state keeps the Boltzmann density of the trap
(the curl force is divergence-free and tangent to the density levels), the
mean local velocity is v(x) = eps (z_hat x x), and the entropy production
rate is exactly

    sigma = <v . D^-1 . v> = eps^2 <|x|^2> / D0 = 2 eps^2 / k

(independent of D0).  With k = 1, eps = 1 this pins sigma = 2, which
catches any x2 normalization slip (A = 2 D_bar) at ~13 standard errors.
"""

import jax.numpy as jnp
import pytest
from jax import random

from SFI.bases import monomials_up_to, unit_axes, x_components
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.langevin import OverdampedProcess

K_TRAP = 1.0
EPS = 1.0
D0 = 0.5
DT = 0.01
NSTEPS = 20_000
SIGMA_EXACT = 2.0 * EPS**2 / K_TRAP


def _simulate(eps, *, seed, Nsteps=NSTEPS):
    x0, x1 = x_components(2)
    e0, e1 = unit_axes(2)
    F = (-K_TRAP * x0 - eps * x1) * e0 + (-K_TRAP * x1 + eps * x0) * e1
    proc = OverdampedProcess(F, D=D0)
    proc.initialize(jnp.array([0.5, 0.0], dtype=jnp.float32))
    return proc.simulate(
        dt=DT,
        Nsteps=Nsteps,
        key=random.PRNGKey(seed),
        prerun=500,
        oversampling=4,
        compute_observables=True,
    )


def _linear_vector_basis():
    # {x0, x1} x {e0, e1}: 4 features, contains the true model exactly.
    return monomials_up_to(1, dim=2, include_constant=False, rank="vector")


@pytest.fixture(scope="module")
def coll_neq():
    return _simulate(EPS, seed=0)


@pytest.fixture(scope="module")
def fit_strato(coll_neq):
    inf = OverdampedLangevinInference(coll_neq)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_linear_vector_basis(), M_mode="Strato", G_mode="rectangle")
    return inf


def test_trap_curl_rate_matches_exact_strato_path(fit_strato):
    out = fit_strato.compute_entropy_production()
    # Rate pinned to sigma = 2 eps^2 / k; bias is reported, not subtracted.
    window = 4.0 * out["Sdot_error"] + out["Sdot_bias"] + 0.15
    assert abs(out["Sdot"] - SIGMA_EXACT) < window, out
    assert out["DeltaS"] >= 0.0
    assert out["Nb"] == 4
    # Attributes for print_report / report_dict
    assert fit_strato.DeltaS == pytest.approx(out["DeltaS"])
    assert fit_strato.error_DeltaS == pytest.approx(out["error_DeltaS"])
    # v1.0 error convention: error_DeltaS = sqrt(2 DeltaS + (2 Nb)^2)
    assert out["error_DeltaS"] == pytest.approx(
        (2.0 * out["DeltaS"] + (2.0 * out["Nb"]) ** 2) ** 0.5, rel=1e-6
    )
    # Debiased (AIC-corrected) fields: bias subtracted, exposed both ways.
    assert out["DeltaS_debiased"] == pytest.approx(out["DeltaS"] - 2.0 * out["Nb"])
    assert out["Sdot_debiased"] == pytest.approx(out["Sdot"] - out["Sdot_bias"])
    assert fit_strato.Sdot_debiased == pytest.approx(out["Sdot_debiased"])


def test_ito_preset_computes_v_moments_on_demand(coll_neq, fit_strato):
    inf = OverdampedLangevinInference(coll_neq)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_linear_vector_basis(), M_mode="Ito", G_mode="trapeze")
    assert not hasattr(inf, "force_v_moments")
    out = inf.compute_entropy_production()
    assert hasattr(inf, "force_v_moments")
    ref = fit_strato.compute_entropy_production()
    tol = 3.0 * (out["Sdot_error"] + ref["Sdot_error"])
    assert abs(out["Sdot"] - ref["Sdot"]) < tol, (out["Sdot"], ref["Sdot"])


def test_equilibrium_control_zero_within_errorbars():
    coll = _simulate(0.0, seed=1)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_linear_vector_basis(), M_mode="Strato", G_mode="rectangle")
    out = inf.compute_entropy_production()
    assert out["Sdot"] >= 0.0  # rectangle Gram: exact quadratic form
    assert out["Sdot"] <= out["Sdot_bias"] + 3.0 * out["Sdot_error"], out
    # Debiased estimate fluctuates around zero at equilibrium (sign-free).
    assert abs(out["Sdot_debiased"]) <= 3.0 * out["Sdot_error"], out


def test_bias_formula_and_support_scaling(coll_neq):
    inf = OverdampedLangevinInference(coll_neq)
    inf.compute_diffusion_constant(method="WeakNoise")
    B = monomials_up_to(2, dim=2, include_constant=True, rank="vector")  # 12 features
    inf.infer_force_linear(B, M_mode="Strato", G_mode="rectangle")

    out_full = inf.compute_entropy_production(support="full")
    assert out_full["Nb"] == 12
    assert out_full["Sdot_bias"] == pytest.approx(2.0 * 12 / out_full["tauN"], rel=1e-6)

    inf.sparsify_force(criterion="PASTIS")
    out_cur = inf.compute_entropy_production(support="current")
    assert out_cur["Nb"] == len(inf.force_support)
    assert out_cur["Sdot_bias"] == pytest.approx(2.0 * out_cur["Nb"] / out_cur["tauN"], rel=1e-6)

    # Both rates remain compatible with the exact value.
    for out in (out_full, out_cur):
        window = 4.0 * out["Sdot_error"] + out["Sdot_bias"] + 0.15
        assert abs(out["Sdot"] - SIGMA_EXACT) < window, out


def test_simulator_observable_cross_check(coll_neq, fit_strato):
    out = fit_strato.compute_entropy_production()
    S_sim = coll_neq.datasets[0].meta["observables"]["entropy"]
    assert abs(out["DeltaS"] - S_sim) < 5.0 * out["error_DeltaS"] + 0.1 * abs(S_sim), (
        out["DeltaS"],
        S_sim,
    )


def test_print_report_and_report_dict_smoke(fit_strato, capsys):
    fit_strato.compute_entropy_production()
    fit_strato.print_report()
    captured = capsys.readouterr().out
    assert "Entropy production" in captured
    d = fit_strato.report_dict()
    assert isinstance(d["DeltaS"], float)
    assert isinstance(d["error_DeltaS"], float)


def test_requires_linear_fit(coll_neq):
    inf = OverdampedLangevinInference(coll_neq)
    inf.compute_diffusion_constant(method="WeakNoise")
    with pytest.raises(RuntimeError, match="infer_force_linear"):
        inf.compute_entropy_production()
    # Parametric fits reuse force_basis/force_G_full with a different
    # (NLL-Hessian) normalization; the guard must key on force_method.
    inf.metadata["force_method"] = "parametric"
    with pytest.raises(RuntimeError, match="not supported"):
        inf.compute_entropy_production()


def test_invalid_support_raises(fit_strato):
    with pytest.raises(ValueError, match="support"):
        fit_strato.compute_entropy_production(support="bogus")
