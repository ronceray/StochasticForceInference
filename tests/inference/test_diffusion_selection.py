"""Diffusion model selection (sparsify_diffusion) and parametric diffusion errors.

Test system: the Landauer blowtorch, a double well with
``D(x) = D0 (1 + g x)`` (Itô), fitted on the polynomial basis
``{1, x, x², x³}`` — true support ``{1, x}``.

* Linear route: the selection information is the efficient minimum-distance
  likelihood in the sandwich metric ``F = Gᵀ S⁻¹ G``; it is dimensionless
  (invariant under a change of length unit) and reproduces the sandwich
  covariance on the full basis.
* Parametric route: the covariance is the inverse NLL Hessian.  For
  Euler-scale Δt the Gaussian NLL's Fisher information of a linear diffusion
  model is ``½ Σ_t b(x_t) b(x_t)ᵀ / D(x_t)²``, which the Hessian must match.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import SFI
from SFI.bases import identity_matrix_basis, monomials_up_to, unit_axes, x_components
from SFI.langevin import OverdampedProcess

D0, G_GRAD, DT = 0.5, 0.3, 0.01


def _blowtorch(seed, Nsteps=50_000):
    (x,) = x_components(1)
    (ex,) = unit_axes(1)
    proc = OverdampedProcess(F=(x - x**3) * ex, D=D0 * (1.0 + G_GRAD * x) * identity_matrix_basis(1),
                             theta_F=jnp.array([1.0]), theta_D=jnp.array([1.0]))
    proc.initialize(jnp.array([0.5]))
    return proc.simulate(dt=DT, Nsteps=Nsteps, key=random.PRNGKey(seed), prerun=200, oversampling=10)


def _linear_fit(coll, order=3):
    inf = SFI.OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(3, dim=1, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(monomials_up_to(order, dim=1, rank="symmetric_matrix"), M_mode="WeakNoise")
    return inf


@pytest.fixture(scope="module")
def blowtorch_coll():
    return _blowtorch(seed=7)


# ── linear route ─────────────────────────────────────────────────────────────


def test_sparsify_diffusion_selects_blowtorch_support(blowtorch_coll):
    inf = _linear_fit(blowtorch_coll)
    res = inf.sparsify_diffusion(criterion="PASTIS", p=0.05)
    assert sorted(np.asarray(inf.diffusion_support).tolist()) == [0, 1]
    assert res.selection["criterion"] == "PASTIS"
    C = np.asarray(inf.diffusion_coefficients_full)
    np.testing.assert_allclose(C[:2], [D0, D0 * G_GRAD], atol=0.02)
    assert np.all(C[2:] == 0.0)

    # Covariance of the selected (efficient restricted) estimator = F_BB⁻¹,
    # embedded in the full parameter order on the fitted field.
    inf.compute_diffusion_error()
    F = np.asarray(inf.diffusion_fisher_full)
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_covariance),
                               np.linalg.inv(F[:2, :2]), rtol=1e-6)
    cov_full = np.asarray(inf.diffusion_inferred.param_cov)
    assert cov_full.shape == (4, 4) and np.all(cov_full[2:, :] == 0.0)
    band = np.asarray(inf.diffusion_inferred.predict_ci(jnp.array([[0.0], [1.0]]))["std"])
    assert np.all(np.isfinite(band)) and np.all(band > 0)


def test_selection_metric_is_the_sandwich_on_the_full_basis(blowtorch_coll):
    """F⁻¹ = G⁻¹ S G⁻¹ and the full-support coefficients are the OLS fit."""
    inf = _linear_fit(blowtorch_coll)
    C_ols = np.asarray(inf.diffusion_coefficients_full)
    F, m = (np.asarray(a) for a in inf._diffusion_selection_metric())
    inf.compute_diffusion_error()
    np.testing.assert_allclose(np.linalg.inv(F), np.asarray(inf.diffusion_coefficients_covariance), rtol=1e-4)  # float32 Gram
    np.testing.assert_allclose(np.linalg.solve(F, m), C_ols, rtol=1e-4, atol=1e-7)


def test_diffusion_information_is_dimensionless():
    """Changing the length unit leaves the selection information unchanged.

    (The plain least-squares quadratic form ½ MᵀG⁻¹M would scale as length⁴.)
    """
    X = np.asarray(_blowtorch(seed=3, Nsteps=20_000).to_arrays(dataset=0)[1])
    infos = []
    for lam in (1.0, 3.0):
        inf = _linear_fit(SFI.TrajectoryCollection.from_arrays(X=X * lam, dt=DT))
        F, m = (np.asarray(a) for a in inf._diffusion_selection_metric())
        infos.append([0.5 * m[B] @ np.linalg.solve(F[np.ix_(B, B)], m[B]) for B in ([0], [0, 1], [0, 1, 2, 3])])
    np.testing.assert_allclose(infos[0], infos[1], rtol=1e-4)


@pytest.mark.slow
def test_spurious_information_is_chi2_calibrated():
    """2[I(full) − I(true)] over seeds averages to the number of spurious terms (χ²₂)."""
    stats = []
    for seed in range(40):
        inf = _linear_fit(_blowtorch(seed=100 + seed, Nsteps=20_000))
        F, m = (np.asarray(a) for a in inf._diffusion_selection_metric())
        I_full = 0.5 * m @ np.linalg.solve(F, m)
        I_true = 0.5 * m[:2] @ np.linalg.solve(F[:2, :2], m[:2])
        stats.append(2.0 * (I_full - I_true))
    stats = np.asarray(stats)
    # χ²₂: mean 2, sd of the mean over 40 draws = 2/√40 ≈ 0.32.
    assert abs(stats.mean() - 2.0) < 3 * 2.0 / np.sqrt(len(stats))


def test_sparsify_diffusion_requires_a_fit(blowtorch_coll):
    inf = SFI.OverdampedLangevinInference(blowtorch_coll)
    with pytest.raises(RuntimeError, match="requires a diffusion fit"):
        inf.sparsify_diffusion()


# ── parametric route ─────────────────────────────────────────────────────────


def _exact_force_psf():
    from SFI.statefunc.factory import make_psf
    from SFI.statefunc.params import ParamSpec

    def f(x, *, params):
        return params["a"] * (x - x**3)

    return make_psf(f, dim=1, rank=1, n_features=1, params=[ParamSpec("a", shape=(), dtype=jnp.float64)])


def _solve_D(coll, Bd):
    from SFI.inference.parametric_core.solve import _as_psf, solve_diffusion_od

    F_psf = _exact_force_psf()
    return solve_diffusion_od(coll, F_psf, jnp.array([1.0]), _as_psf(Bd), Lambda=jnp.zeros((1, 1)))


def test_parametric_hessian_is_the_fisher_information(blowtorch_coll):
    """NLL Hessian ≈ ½ Σ b bᵀ / D² (Gaussian-increment Fisher information)."""
    Bd = monomials_up_to(1, dim=1, rank="symmetric_matrix")
    res = _solve_D(blowtorch_coll, Bd)
    np.testing.assert_allclose(np.asarray(res.theta_D), [D0, D0 * G_GRAD], atol=0.02)
    _, X, _ = blowtorch_coll.to_arrays(dataset=0)
    x = np.asarray(X[:-1, 0, 0], dtype=float)
    b = np.stack([np.ones_like(x), x], axis=1)
    D = D0 * (1.0 + G_GRAD * x)
    F_an = 0.5 * np.einsum("ti,tj,t->ij", b, b, 1.0 / D**2)
    np.testing.assert_allclose(np.asarray(res.hessian), F_an, rtol=0.03)


@pytest.mark.slow
def test_parametric_stderr_matches_seed_scatter():
    """z = (θ̂ − θ)/σ̂ from the inverse Hessian has unit spread across seeds."""
    Bd = monomials_up_to(1, dim=1, rank="symmetric_matrix")
    z = []
    for seed in range(30):
        res = _solve_D(_blowtorch(seed=200 + seed, Nsteps=10_000), Bd)
        se = np.sqrt(np.diag(np.linalg.inv(np.asarray(res.hessian))))
        z.append((np.asarray(res.theta_D) - [D0, D0 * G_GRAD]) / se)
    z = np.asarray(z)
    # sd of a 30-draw sample sd ≈ 1/√60 ≈ 0.13 → 3σ band.
    np.testing.assert_array_less(np.abs(z.std(axis=0) - 1.0), 0.4)
    np.testing.assert_array_less(np.abs(z.mean(axis=0)), 3.0 / np.sqrt(len(z)))


@pytest.fixture(scope="module")
def parametric_inf(blowtorch_coll):
    inf = SFI.OverdampedLangevinInference(blowtorch_coll)
    inf.infer_force(monomials_up_to(3, dim=1, rank="vector"))
    inf.infer_diffusion(monomials_up_to(2, dim=1, rank="symmetric_matrix"))
    return inf


def test_infer_diffusion_returns_inverse_fisher(parametric_inf):
    inf = parametric_inf
    cov = np.asarray(inf.diffusion_inferred.param_cov)
    H = np.asarray(inf._diffusion_parametric_full["hessian"])
    np.testing.assert_allclose(cov, np.linalg.inv(H), rtol=1e-4)  # float32 storage
    inf.compute_diffusion_error()
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_stderr), np.sqrt(np.diag(cov)), rtol=1e-4)
    assert np.isfinite(inf.diffusion_predicted_MSE)


def test_sparsify_parametric_diffusion_refits_selected_support(parametric_inf):
    inf = parametric_inf
    inf.sparsify_diffusion(criterion="PASTIS", p=0.05)
    assert sorted(np.asarray(inf.diffusion_support).tolist()) == [0, 1]
    C = np.asarray(inf.diffusion_coefficients_full)
    assert C[2] == 0.0
    np.testing.assert_allclose(C[:2], [D0, D0 * G_GRAD], atol=0.02)
    inf.compute_diffusion_error()
    H_sub = np.asarray(inf._diffusion_parametric_current["hessian"])[:2, :2]
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_covariance), np.linalg.inv(H_sub), rtol=1e-4)
    cov_full = np.asarray(inf.diffusion_inferred.param_cov)
    assert cov_full.shape == (3, 3) and np.all(cov_full[2, :] == 0.0)


# ── underdamped (linear route) ───────────────────────────────────────────────


def test_sparsify_diffusion_underdamped_velocity_noise():
    """UD: D(v) = D0 (1 + v²) on {1, x, v, x², xv, v²} → support {1, v²}."""
    from SFI.bases import v_components
    from SFI.langevin import UnderdampedProcess

    (x,) = x_components(1)
    (v,) = v_components(1)
    (ex,) = unit_axes(1)
    F_model = (-1.0 * x - 1.0 * v) * ex
    D_model = 0.3 * (1.0 + v**2) * identity_matrix_basis(1)
    proc = UnderdampedProcess(F=F_model, D=D_model, theta_F=jnp.ones(F_model.n_features),
                              theta_D=jnp.ones(D_model.n_features))
    proc.initialize(jnp.array([0.0]), v0=jnp.array([0.0]))
    coll = proc.simulate(dt=0.01, Nsteps=80_000, key=random.PRNGKey(5), prerun=500, oversampling=20)

    B_diff = monomials_up_to(order=2, dim=1, include_v=True, rank="symmetric_matrix")
    inf = SFI.UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(order=1, dim=1, include_v=True, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(B_diff)
    inf.sparsify_diffusion()
    labels = [B_diff.labels[i] for i in np.asarray(inf.diffusion_support)]
    assert sorted(labels) == sorted(["1·S00", "v0^2·S00"]), labels
    inf.compute_diffusion_error()
    assert np.all(np.isfinite(np.asarray(inf.diffusion_coefficients_stderr)))


# ── fit state across successive fits; covariance after selection ─────────────


@pytest.fixture(scope="module")
def small_blowtorch_coll():
    return _blowtorch(seed=11, Nsteps=10_000)


def test_linear_diffusion_fit_after_parametric_is_a_clean_linear_fit(small_blowtorch_coll):
    """infer_force → infer_diffusion → infer_diffusion_linear.

    The linear fit used to inherit the parametric method flag: its error
    analysis read the parametric Hessian (a "not positive definite" error, or
    a covariance of the wrong model) and sparsify_diffusion silently re-fitted
    the parametric model over the linear fit.  It must behave exactly as a
    linear fit on a fresh object.
    """
    coll = small_blowtorch_coll
    B_lin = monomials_up_to(3, dim=1, rank="symmetric_matrix")
    inf = SFI.OverdampedLangevinInference(coll)
    inf.infer_force(monomials_up_to(3, dim=1, rank="vector"))
    inf.infer_diffusion(monomials_up_to(1, dim=1, rank="symmetric_matrix"))
    inf.compute_diffusion_error()
    inf.infer_diffusion_linear(B_lin, M_mode="WeakNoise")
    assert inf.metadata["diffusion_method"] == "linear"
    assert "diffusion_parametric_info" not in inf.metadata
    for attr in ("_diffusion_parametric_full", "_diffusion_parametric_current", "_diffusion_parametric_spec",
                 "diffusion_coefficients_covariance", "diffusion_coefficients_stderr"):
        assert not hasattr(inf, attr), attr
    assert inf.diffusion_inferred.param_cov is None  # not the parametric fit's (2, 2) covariance

    ref = SFI.OverdampedLangevinInference(coll)
    ref.infer_diffusion_linear(B_lin, M_mode="WeakNoise")
    for obj in (inf, ref):
        obj.compute_diffusion_error()
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_stderr),
                               np.asarray(ref.diffusion_coefficients_stderr), rtol=1e-6)
    for obj in (inf, ref):
        obj.sparsify_diffusion(criterion="PASTIS", p=0.05)
    assert inf.metadata["diffusion_method"] == "linear"
    np.testing.assert_array_equal(np.asarray(inf.diffusion_support), np.asarray(ref.diffusion_support))
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_full),
                               np.asarray(ref.diffusion_coefficients_full), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(inf.diffusion_inferred.param_cov),
                               np.asarray(ref.diffusion_inferred.param_cov), rtol=1e-6)


def test_parametric_diffusion_fit_after_linear_selection(small_blowtorch_coll):
    """The linear selection metric and error analysis do not survive a parametric
    fit; a refused re-fit leaves the fit untouched."""
    inf = SFI.OverdampedLangevinInference(small_blowtorch_coll)
    inf.infer_force(monomials_up_to(3, dim=1, rank="vector"))
    inf.infer_diffusion_linear(monomials_up_to(3, dim=1, rank="symmetric_matrix"), M_mode="WeakNoise")
    inf.compute_diffusion_error()
    inf.sparsify_diffusion(criterion="PASTIS", p=0.05)
    assert inf.diffusion_fisher_full is not None

    B_par = monomials_up_to(1, dim=1, rank="symmetric_matrix")
    inf.infer_diffusion(B_par)
    assert "diffusion_fisher_full" not in vars(inf)
    assert inf.metadata["diffusion_method"] == "parametric"
    assert not hasattr(inf, "diffusion_coefficients_covariance")  # the linear fit's
    assert list(inf.diffusion_basis_labels) == list(B_par.labels)
    inf.compute_diffusion_error()
    H = np.asarray(inf._diffusion_parametric_full["hessian"])
    np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_covariance), np.linalg.inv(H), rtol=1e-4)

    with pytest.raises(RuntimeError, match="already been inferred"):
        inf.infer_diffusion_linear(monomials_up_to(2, dim=1, rank="symmetric_matrix"), M_mode="WeakNoise")
    assert inf.diffusion_basis is B_par
    assert inf.metadata["diffusion_method"] == "parametric"
    assert inf.metadata["diffusion_M_mode"] == "WeakNoise"


def test_diffusion_param_cov_follows_the_selection(blowtorch_coll):
    """After sparsify_diffusion, param_cov is the selected estimator's covariance in
    the full parameter order — never the pre-selection covariance (it used to keep
    it until the next compute_diffusion_error, and a support-sized one after a
    second selection broke predict_ci)."""
    inf = _linear_fit(blowtorch_coll)
    inf.compute_diffusion_error()
    for criterion in ("PASTIS", "AIC"):
        inf.sparsify_diffusion(criterion=criterion, p=0.05)
        sup = np.asarray(inf.diffusion_support)
        F = np.asarray(inf.diffusion_fisher_full)
        cov_S = np.linalg.inv(F[np.ix_(sup, sup)])
        np.testing.assert_allclose(np.asarray(inf.diffusion_coefficients_covariance), cov_S, rtol=1e-5)
        cov = np.asarray(inf.diffusion_inferred.param_cov)
        want = np.zeros((4, 4))
        want[np.ix_(sup, sup)] = cov_S
        np.testing.assert_allclose(cov, want, rtol=1e-5, atol=1e-12)
        x = 0.5
        J = x ** np.arange(4)
        std = np.asarray(inf.diffusion_inferred.predict_ci(jnp.array([[x]]))["std"]).ravel()
        np.testing.assert_allclose(std, np.sqrt(J @ want @ J), rtol=1e-4)

    # No error analysis before the selection: no covariance after it either.
    inf2 = _linear_fit(blowtorch_coll)
    inf2.sparsify_diffusion(criterion="PASTIS", p=0.05)
    assert inf2.diffusion_inferred.param_cov is None


def test_coeff_block_locates_diffusion_basis_blocks(blowtorch_coll):
    """diffusion_basis_labels is recorded, so coeff_block takes a Basis for the
    diffusion field; predict_time_profile names the diffusion error method."""
    from SFI.bases import monomials_degree

    B = monomials_up_to(3, dim=1, rank="symmetric_matrix")
    inf = SFI.OverdampedLangevinInference(blowtorch_coll)
    inf.infer_diffusion_linear(B, M_mode="WeakNoise")
    assert list(inf.diffusion_basis_labels) == list(B.labels)
    coeffs, cov = inf.coeff_block(monomials_degree(2, dim=1, rank="symmetric_matrix"), field="diffusion")
    np.testing.assert_array_equal(coeffs, np.asarray(inf.diffusion_coefficients_full)[2:3])
    assert cov is None
    with pytest.raises(RuntimeError, match=r"compute_diffusion_error\(\)"):
        inf.predict_time_profile(B, np.linspace(0.0, 1.0, 3), field="diffusion", return_std=True)
    inf.compute_diffusion_error()
    _, cov = inf.coeff_block(B, field="diffusion")
    np.testing.assert_allclose(cov, np.asarray(inf.diffusion_coefficients_covariance))
