# TODO: review this file
# tests/inference/test_confidence_intervals.py
"""Tests for pointwise confidence intervals on force and diffusion predictions."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.statefunc.factory import make_basis
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


# --------------- helpers -------------------------------------------------- #


def _make_ou_collection(T=2000, dt=0.01, d=2, k_diag=(0.7, 1.1), D_diag=(0.2, 0.2), seed=42):
    """Simulate diagonal OU process → TrajectoryCollection."""
    rng = np.random.default_rng(seed)
    k = np.array(k_diag, dtype=np.float64)
    D = np.array(D_diag, dtype=np.float64)
    x = np.zeros((T, 1, d), dtype=np.float64)
    for t in range(T - 1):
        eta = rng.normal(size=(1, d))
        x[t + 1, 0] = x[t, 0] + dt * (-k * x[t, 0]) + np.sqrt(2.0 * D * dt) * eta[0]
    ds = TrajectoryDataset.from_arrays(X=jnp.array(x), dt=float(dt))
    return TrajectoryCollection.from_dataset(ds), k, D


def _coord_linear_vector_basis(dim: int):
    """Vector basis: B(x)[i,m,a] = δ_{m,a} x_{i,a}.  Features = dim."""

    def f(x, **kw):
        I = jnp.eye(dim, dtype=x.dtype)
        return jnp.einsum("m,ma->ma", x, I)

    return make_basis(f, dim=dim, rank=1, n_features=dim)


def _diag_rank2_basis(dim: int):
    """Rank-2 diffusion basis: B0 = I (constant), B1 = diag(x²).

    Works with both batched (..., dim) and unbatched (dim,) inputs.
    """

    def f(x, **kw):
        I = jnp.eye(dim, dtype=x.dtype)
        x2 = x[..., :dim] ** 2                       # (..., dim)
        B0 = jnp.broadcast_to(I, x2.shape + (dim,))  # (..., dim, dim)
        B1 = x2[..., :, None] * I                     # (..., dim, dim)
        return jnp.stack([B0, B1], axis=-1)            # (..., dim, dim, 2)

    return make_basis(f, dim=dim, rank=2, n_features=2)


def _run_force_inference(T=4000, d=2, seed=42):
    """Return the inferer after force inference + error estimation."""
    coll, k, D = _make_ou_collection(T=T, dt=0.01, d=d, seed=seed)
    basis = _coord_linear_vector_basis(d)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis)
    inf.compute_force_error()
    return inf, k, D


# --------------- tests ---------------------------------------------------- #


class TestPredictVar:
    """Tests for InferenceResultSF.predict_var()."""

    def test_no_param_cov_raises(self):
        """predict_var raises if param_cov is None."""
        coll, _, _ = _make_ou_collection(T=200, d=2)
        basis = _coord_linear_vector_basis(2)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(basis)
        # Before compute_force_error → param_cov is None
        with pytest.raises(RuntimeError, match="covariance.*not available"):
            inf.force_inferred.predict_var(jnp.zeros((5, 2)))

    def test_shape_matches_force(self):
        """predict_var output shape matches force output shape."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.linspace(-2, 2, 20).reshape(10, 2)
        var = inf.force_inferred.predict_var(x)
        assert var.shape == (10, 2), f"Expected (10,2), got {var.shape}"

    def test_variance_positive(self):
        """Variance should be non-negative everywhere."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.linspace(-2, 2, 20).reshape(10, 2)
        var = inf.force_inferred.predict_var(x)
        assert jnp.all(var >= 0), "Variance should be non-negative"

    def test_linear_model_exact_formula(self):
        """For linear model: Var[F_m(x)] = Σ_ab J_{m,a}(x) Σ_{ab} J_{m,b}(x).

        With CoeffNode, J_{m,a} = basis(x)_{m,a}, so we can verify against
        the explicit formula.
        """
        d = 2
        inf, _, _ = _run_force_inference(T=4000, d=d)
        x = jnp.array([[1.0, 0.5], [-0.3, 0.7], [0.0, 0.0]])

        # Predicted variance via new API
        var_pred = inf.force_inferred.predict_var(x)

        # Manual computation: basis(x) gives (N, d, p); covariance is (p, p)
        basis = inf.force_basis
        B = basis(x)  # (3, d, p)
        Cov = jnp.asarray(inf.force_coefficients_covariance)  # (p, p)

        # Var_m(x) = B_{m,:} @ Cov @ B_{m,:}^T — take diagonal
        var_manual = jnp.einsum("...mp,pq,...mq->...m", B, Cov, B)

        npt.assert_allclose(np.array(var_pred), np.array(var_manual), rtol=1e-5)


class TestPredictCov:
    """Tests for InferenceResultSF.predict_cov()."""

    def test_shape_for_force(self):
        """predict_cov returns (N, d, d) for a rank-1 force model."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.linspace(-2, 2, 20).reshape(10, 2)
        cov = inf.force_inferred.predict_cov(x)
        assert cov.shape == (10, 2, 2), f"Expected (10,2,2), got {cov.shape}"

    def test_diagonal_matches_predict_var(self):
        """Diagonal of predict_cov should equal predict_var."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.array([[1.0, 0.5], [-0.3, 0.7]])
        var = inf.force_inferred.predict_var(x)
        cov = inf.force_inferred.predict_cov(x)
        diag = jnp.diagonal(cov, axis1=-2, axis2=-1)  # (N, d)
        npt.assert_allclose(np.array(var), np.array(diag), rtol=1e-6)

    def test_symmetric(self):
        """Covariance matrix should be symmetric at each point."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.array([[1.0, 0.5]])
        cov = inf.force_inferred.predict_cov(x)
        npt.assert_allclose(
            np.array(cov[0]), np.array(cov[0].T), rtol=1e-4, atol=1e-5
        )


class TestPredictCI:
    """Tests for InferenceResultSF.predict_ci()."""

    def test_returns_correct_keys(self):
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.array([[1.0, 0.5]])
        ci = inf.force_inferred.predict_ci(x)
        assert set(ci.keys()) == {"mean", "std", "lower", "upper"}

    def test_mean_matches_call(self):
        """CI mean should be the same as direct evaluation."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.array([[1.0, 0.5], [0.0, 0.0]])
        ci = inf.force_inferred.predict_ci(x)
        direct = inf.force_inferred(x)
        npt.assert_allclose(np.array(ci["mean"]), np.array(direct), rtol=1e-6)

    def test_lower_upper_bracket_mean(self):
        """lower <= mean <= upper for all components."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.linspace(-2, 2, 20).reshape(10, 2)
        ci = inf.force_inferred.predict_ci(x, alpha=0.95)
        assert jnp.all(ci["lower"] <= ci["mean"] + 1e-10)
        assert jnp.all(ci["upper"] >= ci["mean"] - 1e-10)

    def test_narrower_ci_at_lower_alpha(self):
        """90% CI should be narrower than 99% CI."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        x = jnp.array([[1.0, 0.5]])
        ci_90 = inf.force_inferred.predict_ci(x, alpha=0.90)
        ci_99 = inf.force_inferred.predict_ci(x, alpha=0.99)
        width_90 = ci_90["upper"] - ci_90["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert jnp.all(width_90 < width_99)

    def test_coverage_on_ou(self):
        """95% CI should contain the exact force for ~90%+ of grid points.

        We use a generous threshold (80%) to account for finite-sample effects,
        discretization bias, and the fact that coverage is only asymptotically exact.
        """
        d = 2
        inf, k_true, _ = _run_force_inference(T=6000, d=d, seed=123)
        # Grid points
        x1 = jnp.linspace(-1.5, 1.5, 15)
        x2 = jnp.linspace(-1.5, 1.5, 15)
        xx1, xx2 = jnp.meshgrid(x1, x2)
        x_grid = jnp.stack([xx1.ravel(), xx2.ravel()], axis=-1)

        ci = inf.force_inferred.predict_ci(x_grid, alpha=0.95)
        F_exact = -jnp.array(k_true) * x_grid  # (N, d)

        contained = (F_exact >= ci["lower"]) & (F_exact <= ci["upper"])
        coverage = float(contained.mean())
        assert coverage > 0.80, f"Coverage {coverage:.2%} too low (expected >80%)"


class TestParamCovPropagation:
    """Verify that param_cov flows correctly through the inference pipeline."""

    def test_param_cov_set_after_compute_error(self):
        """force_inferred.param_cov should be set after compute_force_error()."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        assert inf.force_inferred.param_cov is not None

    def test_param_cov_shape(self):
        """param_cov should be (p, p) where p is the support size."""
        d = 2
        inf, _, _ = _run_force_inference(T=2000, d=d)
        p = len(inf.force_support)
        assert inf.force_inferred.param_cov.shape == (p, p)

    def test_param_cov_matches_covariance_attribute(self):
        """param_cov should match force_coefficients_covariance."""
        inf, _, _ = _run_force_inference(T=2000, d=2)
        npt.assert_allclose(
            np.array(inf.force_inferred.param_cov),
            np.array(inf.force_coefficients_covariance),
            rtol=1e-6,
        )


class TestDiffusionError:
    """Tests for compute_diffusion_error()."""

    def test_diffusion_error_smoke(self):
        """compute_diffusion_error should set covariance and predicted MSE."""
        d = 2
        coll, _, _ = _make_ou_collection(T=3000, d=d, seed=77)
        basis_D = _diag_rank2_basis(d)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(_coord_linear_vector_basis(d))
        inf.infer_diffusion_linear(basis_D)
        inf.compute_diffusion_error()

        assert hasattr(inf, "diffusion_coefficients_covariance")
        assert hasattr(inf, "diffusion_coefficients_stderr")
        p_D = len(inf.diffusion_support)
        assert inf.diffusion_coefficients_covariance.shape == (p_D, p_D)
        assert inf.diffusion_coefficients_stderr.shape == (p_D,)

    def test_diffusion_param_cov_propagated(self):
        """diffusion_inferred.param_cov should be set after compute_diffusion_error."""
        d = 2
        coll, _, _ = _make_ou_collection(T=3000, d=d, seed=77)
        basis_D = _diag_rank2_basis(d)
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(_coord_linear_vector_basis(d))
        inf.infer_diffusion_linear(basis_D)
        inf.compute_diffusion_error()
        assert inf.diffusion_inferred.param_cov is not None
