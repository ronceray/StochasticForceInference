"""Coefficient covariance after sparsify_force on the parametric (EIV) path.

The errors-in-variables fit solves an asymmetric estimating equation; its
covariance is the sandwich G_raw⁻¹ H G_raw⁻ᵀ.  After selection, the kept
coefficients must be the efficient restricted estimator in the metric
F = G_rawᵀ H⁻¹ G_raw (the inverse sandwich), with covariance F_SS⁻¹ —
never the inverse of the symmetrised Gram restricted to the support, which
is neither (the flocking demo's stderrs doubled after selection).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


SUPPORT = np.array([1])  # true support of F(x) = -x in [1, x, x², x³]


def _noisy_ou(seed, T=20000, dt=0.01, noise=0.05):
    from SFI.trajectory.collection import TrajectoryCollection
    from SFI.trajectory.dataset import TrajectoryDataset

    rng = np.random.default_rng(seed)
    X = np.zeros(T)
    X[0] = rng.normal()
    xi = rng.normal(size=T)
    for t in range(T - 1):
        X[t + 1] = X[t] - X[t] * dt + np.sqrt(dt) * xi[t]
    Y = X + noise * rng.standard_normal(T)
    return TrajectoryCollection.from_dataset(
        TrajectoryDataset.from_arrays(X=jnp.array(Y[:, None, None]), dt=dt))


def _fit(coll, basis, eiv=True):
    import SFI

    inf = SFI.OverdampedLangevinInference(coll)
    inf.infer_force(basis, eiv=eiv)
    inf.compute_force_error()
    return inf


def _restrict(inf, support):
    """Force the selection onto ``support`` (as sparsify_force would)."""
    _, coeffs = inf.force_scorer.info_and_coeffs(jnp.asarray(support))
    inf._update_force_coefficients(coeffs, support)
    inf.compute_force_error()
    return inf


def test_eiv_restricted_covariance_is_inverse_fisher_on_support():
    from SFI.bases import monomials_up_to

    inf = _fit(_noisy_ou(0), monomials_up_to(3, dim=1, rank="vector"))
    assert inf.force_fisher_full is not None
    cov_full = np.asarray(inf.force_coefficients_covariance)
    F = np.asarray(inf.force_fisher_full)
    # The selection metric is the inverse of the solver's sandwich.
    np.testing.assert_allclose(np.linalg.inv(F), cov_full, rtol=1e-6)

    theta_full = np.asarray(inf.force_coefficients_full)
    _restrict(inf, SUPPORT)
    ix = np.ix_(SUPPORT, SUPPORT)
    cov_S = np.asarray(inf.force_coefficients_covariance)
    np.testing.assert_allclose(cov_S, np.linalg.inv(F[ix]), rtol=1e-8)
    # Efficient restricted estimator: θ_S = F_SS⁻¹ (F θ̂)_S.
    np.testing.assert_allclose(
        np.asarray(inf.force_coefficients),
        np.linalg.solve(F[ix], (F @ theta_full)[SUPPORT]), rtol=1e-8)
    # Dropping terms can only shrink the kept variance (Schur complement).
    assert np.all(np.diag(cov_S) <= np.diag(cov_full)[SUPPORT] * (1 + 1e-10))


def test_mle_path_keeps_gram_metric():
    from SFI.bases import monomials_up_to

    inf = _fit(_noisy_ou(0, noise=0.0), monomials_up_to(3, dim=1, rank="vector"), eiv=False)
    assert inf.force_fisher_full is None
    _restrict(inf, SUPPORT)
    G = np.asarray(inf.force_G_full)
    np.testing.assert_allclose(
        np.asarray(inf.force_coefficients_covariance),
        np.linalg.pinv(G[np.ix_(SUPPORT, SUPPORT)]), rtol=1e-8)


def test_param_cov_follows_sparsify_force():
    """The fit-time param_cov (the sandwich) becomes the restricted estimator's
    F_SS⁻¹ in the full parameter order after sparsify_force (the overdamped
    engine dropped it, the underdamped one kept the full-basis matrix)."""
    import SFI
    from SFI.bases import monomials_up_to

    inf = SFI.OverdampedLangevinInference(_noisy_ou(0))
    inf.infer_force(monomials_up_to(3, dim=1, rank="vector"))
    F = np.asarray(inf.force_fisher_full)
    np.testing.assert_allclose(np.asarray(inf.force_inferred.param_cov), np.linalg.inv(F), rtol=1e-6)
    inf.sparsify_force(criterion="PASTIS", p=0.05)
    sup = np.asarray(inf.force_support)
    assert 0 < sup.size < 4
    want = np.zeros((4, 4))
    want[np.ix_(sup, sup)] = np.linalg.inv(F[np.ix_(sup, sup)])
    np.testing.assert_allclose(np.asarray(inf.force_inferred.param_cov), want, rtol=1e-8, atol=1e-14)
    assert not hasattr(inf, "force_coefficients_covariance")  # no error analysis was run
    inf.compute_force_error()
    np.testing.assert_allclose(np.asarray(inf.force_inferred.param_cov), want, rtol=1e-8, atol=1e-14)


def test_linear_force_fit_after_psf_fit_starts_clean():
    """A linear fit after a parametric PSF fit inherits none of its state; a
    refused linear re-fit after a Basis fit leaves the fit untouched."""
    import SFI
    from SFI.bases import monomials_up_to

    coll = _noisy_ou(1, T=5000, noise=0.0)
    B = monomials_up_to(1, dim=1, rank="vector")
    inf = SFI.OverdampedLangevinInference(coll)
    inf.infer_force(B.to_psf(), eiv=False)
    inf.compute_force_error()
    inf.infer_force_linear(B, preset="clean")
    assert inf.metadata["force_method"] == "linear"
    assert "force_parametric_info" not in inf.metadata
    for attr in ("force_psf", "force_fisher_full", "force_coefficients_covariance", "force_coefficients_stderr"):
        assert not hasattr(inf, attr), attr
    assert inf.force_inferred.param_cov is None
    inf.compute_force_error()
    np.testing.assert_allclose(np.asarray(inf.force_inferred.param_cov),
                               np.linalg.pinv(np.asarray(inf.force_G_full)), rtol=1e-8)
    with pytest.raises(RuntimeError, match="parametric force inference"):
        inf.infer_diffusion()

    inf2 = SFI.OverdampedLangevinInference(coll)
    inf2.infer_force(B, eiv=False)
    with pytest.raises(RuntimeError, match="already been inferred"):
        inf2.infer_force_linear(monomials_up_to(3, dim=1, rank="vector"), preset="clean")
    assert inf2.force_basis is B
    assert inf2.metadata["force_method"] == "parametric" and "force_preset" not in inf2.metadata


@pytest.mark.slow
def test_eiv_restricted_stderr_matches_seed_scatter():
    """Predicted stderr after selection vs Monte-Carlo scatter over seeds.

    Noisy underdamped Van der Pol (the gallery setup), support fixed to the
    three true terms.  The symmetrised-Gram inverse over-states the scatter
    3-4x here; the restricted inverse Fisher matches it.
    """
    from jax import random

    from SFI import UnderdampedLangevinInference
    from SFI.bases import monomials_up_to, named_scalar, unit_axes, v_components, x_components
    from SFI.langevin import UnderdampedProcess

    (x,), (v,), (ex,) = x_components(1), v_components(1), unit_axes(1)
    F = (named_scalar("mu", default=3.0) * (1 - x * x) * v - x) * ex
    B = monomials_up_to(order=3, dim=1, include_v=True, rank="vector")
    labels = list(B.labels)
    support = np.array([labels.index(k) for k in ("x0·e0", "v0·e0", "(x0^2·v0)·e0")])

    n_seeds = 20
    est, se = [], []
    for s in range(n_seeds):
        proc = UnderdampedProcess(F, D=jnp.array([[0.05]]))
        proc.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))
        coll = proc.simulate(dt=0.02, Nsteps=4000, key=random.PRNGKey(10 + s),
                             prerun=200, oversampling=8).degrade(noise=0.005, seed=100 + s)
        inf = UnderdampedLangevinInference(coll)
        inf.infer_force(B)
        _restrict(inf, support)
        est.append(np.asarray(inf.force_coefficients))
        se.append(np.asarray(inf.force_coefficients_stderr))
    sd_mc = np.std(est, axis=0, ddof=1)
    # Sampling error of a standard deviation: sd / sqrt(2 (n − 1)); 3σ band.
    tol = 3.0 / np.sqrt(2 * (n_seeds - 1))
    np.testing.assert_array_less(np.abs(np.mean(se, axis=0) / sd_mc - 1.0), tol)
