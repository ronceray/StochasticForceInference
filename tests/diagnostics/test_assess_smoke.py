# tests/diagnostics/test_assess_smoke.py
"""Smoke tests for ``SFI.diagnostics.assess`` on small OU/VdP problems."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
from jax import random

from SFI.bases.constants import identity_matrix_basis, unit_vector_basis
from SFI.bases.monomials import monomials_up_to
from SFI.diagnostics import DiagnosticsReport, assess
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc.factory import make_basis


def _monomial_vector_basis(dim: int, degree: int):
    Bsc = monomials_up_to(
        order=degree, dim=dim, include_constant=True, include_x=True,
        include_v=False,
    )
    U = unit_vector_basis(dim)
    return Bsc * U


def _make_ou(dim=2, K=(0.8, 1.2), D0=0.2):
    K_arr = jnp.asarray(K, dtype=jnp.float32)

    def ou_force(x, *, mask=None):
        return jnp.stack([-K_arr[m] * x[m] for m in range(dim)], axis=-1)

    F_basis = make_basis(ou_force, dim=dim, rank=1, n_features=1)
    D_basis = identity_matrix_basis(dim)
    proc = OverdampedProcess(F_basis.to_psf(), D=D_basis.to_psf())
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([D0])},
    )
    return proc


@pytest.fixture(scope="module")
def ou_inf():
    """OD-linear inference on a 2-D OU sample."""
    proc = _make_ou()
    proc.initialize(jnp.zeros(2, dtype=jnp.float32))
    coll = proc.simulate(
        dt=0.01, Nsteps=4000, key=random.PRNGKey(0),
        prerun=50, oversampling=5,
    )
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    B = _monomial_vector_basis(2, degree=1)
    inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()
    return inf


def test_assess_minimal_runs(ou_inf):
    rep = assess(ou_inf, level="minimal")
    assert isinstance(rep, DiagnosticsReport)
    moments = rep.residuals["moments"]
    assert moments["n"] > 100
    # Whitened residuals should be roughly N(0, 1):
    assert abs(moments["mean"]) < 0.2
    assert 0.5 < moments["std"] < 2.0


def test_assess_standard_runs_and_passes(ou_inf):
    rep = assess(ou_inf, level="standard")
    # All residual-consistency sections populated
    assert "moments" in rep.residuals
    assert "autocorr" in rep.residuals
    assert "normality" in rep.residuals
    assert "mse_consistency" in rep.residuals

    # OU is well specified — the normality p-value should not be tiny.
    p_ks = rep.residuals["normality"]["ks"]["pvalue"]
    assert np.isfinite(p_ks) and p_ks > 1e-6

    # Predicted vs realised NMSE should both be small for well-specified OU.
    cons = rep.residuals["mse_consistency"]
    pred = cons.get("predicted_NMSE")
    real = cons.get("realised_NMSE")
    if pred is not None and np.isfinite(pred):
        assert 0.0 < pred < 1.0
    if real is not None and np.isfinite(real):
        assert 0.0 <= real < 1.0


def test_removed_full_level_raises(ou_inf):
    # 'full' was a placeholder level; it is no longer accepted.
    with pytest.raises(ValueError):
        assess(ou_inf, level="full")


def test_invalid_level_raises(ou_inf):
    with pytest.raises(ValueError):
        assess(ou_inf, level="bogus")


def test_inferer_diagnose_method(ou_inf):
    rep = ou_inf.diagnose(level="minimal")
    assert isinstance(rep, DiagnosticsReport)
    assert rep.meta["regime"] == "OD"
    assert rep.meta["d"] == 2


def test_report_serialisation(ou_inf):
    rep = assess(ou_inf, level="standard")
    d = rep.to_dict()
    assert "residuals" in d and "meta" in d
    js = rep.to_json()
    assert isinstance(js, str) and "moments" in js


# ---------------------------------------------------------------------- #
# Underdamped path
# ---------------------------------------------------------------------- #


def _make_harmonic_ud(D0=0.1, k=1.0, gamma=0.5):
    def force(x, *, v, mask=None):
        return jnp.array([-k * x[0] - gamma * v[0]])

    F_basis = make_basis(force, dim=1, rank=1, n_features=1, needs_v=True)
    D_basis = identity_matrix_basis(1)
    proc = UnderdampedProcess(F_basis.to_psf(), D=D_basis.to_psf())
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([D0])},
    )
    return proc


@pytest.fixture(scope="module")
def ud_inf():
    proc = _make_harmonic_ud()
    proc.initialize(
        jnp.array([0.5], dtype=jnp.float32),
        v0=jnp.array([0.0], dtype=jnp.float32),
    )
    coll = proc.simulate(
        dt=0.01, Nsteps=4000, key=random.PRNGKey(1), prerun=100, oversampling=5,
    )
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="auto")
    # Linear force basis: φ = [1, x, v]  ⊗  e_x
    Bsc = monomials_up_to(order=1, dim=1, include_constant=True,
                          include_x=True, include_v=True)
    U = unit_vector_basis(1)
    B = Bsc * U
    inf.infer_force_linear(B)
    inf.compute_force_error()
    return inf


def test_assess_underdamped(ud_inf):
    rep = assess(ud_inf, level="standard")
    assert rep.meta["regime"] == "UD"
    moments = rep.residuals["moments"]
    assert moments["n"] > 100
    # UD secant residuals are noisier; only loose bounds.
    assert np.isfinite(moments["mean"])
    assert np.isfinite(moments["std"]) and moments["std"] > 0


# ---------------------------------------------------------------------- #
# Multi-particle + masked + multi-dataset compatibility
# ---------------------------------------------------------------------- #
def test_assess_multiparticle_masked():
    """Independent OU particles + a NaN-mask on a fraction of rows.

    Routes through ``TrajectoryDataset.make_batch_producer`` so this
    locks in compatibility with multi-particle and dynamic-mask data.
    """
    from SFI.trajectory import TrajectoryCollection

    rng = np.random.default_rng(0)
    T, N, d = 1500, 3, 2
    dt = 0.02
    K = np.array([0.7, 1.3])
    sigma = np.sqrt(2 * 0.15 * dt)

    X = np.zeros((T, N, d), dtype=np.float32)
    for t in range(T - 1):
        drift = -K[None, :] * X[t]
        noise = sigma * rng.standard_normal((N, d)).astype(np.float32)
        X[t + 1] = X[t] + drift * dt + noise

    # Drop ~15% of rows for particles 1 and 2 at random times to exercise
    # the dynamic mask path.
    mask = np.ones((T, N), dtype=bool)
    drop = rng.random((T, N)) < 0.15
    drop[:, 0] = False  # keep particle 0 fully observed
    mask[drop] = False

    coll = TrajectoryCollection.from_arrays(X=X, dt=dt, mask=mask)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    B = _monomial_vector_basis(2, degree=1)
    inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()

    rep = assess(inf, level="standard")
    assert rep.meta["n_particles"] == N
    # ``n_obs`` (in meta) counts d-vector residual rows; ``moments['n']``
    # counts scalar z values (= n_obs * d). Use n_obs for the mask budget.
    n_rows = rep.meta["n_obs"]
    # Each accepted t requires both X[t] and X[t+1] valid; expected
    # fraction is roughly 0.85 * 0.85 ≈ 0.72 for particles 1 and 2
    # plus 1.0 for particle 0. Lower bound is conservative.
    assert n_rows > N * (T - 1) * 0.65
    assert n_rows < N * (T - 1)  # i.e. some were actually masked out
    assert rep.residuals["moments"]["n"] == n_rows * d
    # Whitened residuals should still be roughly unit-variance.
    assert 0.7 < rep.residuals["moments"]["std"] < 1.4


# ---------------------------------------------------------------------- #
# Plotting smoke test
# ---------------------------------------------------------------------- #
def test_diagnostics_plot_summary(ou_inf):
    """plot_summary() must build a 1x3 figure without raising."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from SFI.diagnostics import (
        plot_qq, plot_residual_acf, plot_residual_histogram,
        plot_summary,
    )

    rep = assess(ou_inf, level="standard")
    fig = plot_summary(rep)
    assert fig is not None
    # 1x3 grid -> 3 axes
    assert len(fig.axes) == 3
    plt.close(fig)

    # Individual panels accept an ``ax``
    fig2, axes = plt.subplots(1, 3)
    plot_qq(rep, ax=axes[0])
    plot_residual_histogram(rep, ax=axes[1])
    plot_residual_acf(rep, ax=axes[2])
    plt.close(fig2)


# ---------------------------------------------------------------------- #
# Misspecified fit must trigger flag_issues
# ---------------------------------------------------------------------- #
def test_misspecified_fit_flags_issues():
    """Fit a linear basis to a 1-D double-well; whitened squared
    residuals should show clear volatility clustering and the issue
    list non-empty.
    """
    def doublewell(x, *, mask=None):
        return x - x ** 3

    F_basis = make_basis(doublewell, dim=1, rank=1, n_features=1)
    proc = OverdampedProcess(
        F_basis.to_psf(), D=identity_matrix_basis(1).to_psf(),
    )
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([0.15])},
    )
    proc.initialize(jnp.array([0.5], dtype=jnp.float32))
    coll = proc.simulate(
        dt=0.005, Nsteps=8000, key=random.PRNGKey(2), prerun=200,
        oversampling=10,
    )

    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    # Deliberately under-parameterised: linear basis on a cubic force.
    Bsc = monomials_up_to(order=1, dim=1, include_constant=True,
                          include_x=True)
    U = unit_vector_basis(1)
    B = Bsc * U
    inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()

    rep = assess(inf, level="standard")
    issues = rep.flag_issues(alpha=0.05)
    # We expect at least one diagnostic to fire — typically
    # ``ljung_box_squared`` (volatility clustering at the wells) or
    # ``mse_consistency`` (chi^2 z-score on the residual excess).
    assert len(issues) >= 1, (
        f"Misspecified linear-on-cubic fit returned no warnings.\n"
        f"autocorr: {rep.residuals.get('autocorr')}\n"
        f"mse:      {rep.residuals.get('mse_consistency')}"
    )
