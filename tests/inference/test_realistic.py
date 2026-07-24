# TODO: review this file
# tests/inference/test_realistic.py
"""
Realistic end-to-end inference tests modelled after the examples/ workflows.

Each test follows the simulate → infer → validate pattern from the
Lorenz / Van der Pol examples, but uses tiny parameters (short trajectories,
low dimension, small bases) to keep wall-clock time under ~5 s per test.

Coverage gaps addressed:
  - Full OverdampedProcess → OverdampedLangevinInference pipeline
  - UnderdampedProcess → UnderdampedLangevinInference pipeline
  - Nonlinear (PSF-based) force inference
  - Sparsification via beam-search (overcomplete basis → prune)
  - Data degradation (downsample, noise, data loss) + inference
  - Bootstrapped trajectory synthesis from inferred models
  - Serialization round-trip (save_results / load_results, save_model / load_model)
  - compare_to_exact with model objects (SF, not lambdas)
  - compute_force_error
  - Monomial vector basis in inference
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases.constants import identity_matrix_basis, unit_vector_basis
from SFI.bases.monomials import monomials_up_to
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import StateExpr
from SFI.statefunc.factory import make_basis, make_psf


# ======================================================================
# Helpers
# ======================================================================


def _monomial_vector_basis(dim: int, degree: int, *, include_v: bool = False):
    """Vectorised scalar monomials  e_mu * phi_a(x)  up to ``degree``."""
    Bsc = monomials_up_to(
        order=degree,
        dim=dim,
        include_constant=True,
        include_x=True,
        include_v=include_v,
    )
    U = unit_vector_basis(dim)
    return Bsc * U


# ======================================================================
# Fixtures – shared simulation data (cached per session)
# ======================================================================

# ---------- overdamped 2-D OU ----------

_OU_DIM = 2
_OU_K = jnp.array([0.8, 1.2], dtype=jnp.float32)
_OU_D0 = 0.2


def _make_ou_process():
    """2-D OU process  F(x) = -K x,  D = D0 I."""

    def ou_force(x, *, mask=None):
        return jnp.stack([-_OU_K[0] * x[0], -_OU_K[1] * x[1]], axis=-1)  # (dim,)

    F_basis = make_basis(ou_force, dim=_OU_DIM, rank=1, n_features=1)
    D_basis = identity_matrix_basis(_OU_DIM)

    F_psf = F_basis.to_psf()
    D_psf = D_basis.to_psf()

    proc = OverdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([_OU_D0])},
    )
    return proc


@pytest.fixture(scope="module")
def ou_sim():
    """Simulate a 2-D OU trajectory (3 000 steps, dt = 0.01)."""
    proc = _make_ou_process()
    proc.initialize(jnp.zeros(_OU_DIM, dtype=jnp.float32))

    key = random.PRNGKey(42)
    coll = proc.simulate(
        dt=0.01, Nsteps=3000, key=key, prerun=50, oversampling=5
    )
    return proc, coll


# ---------- underdamped 1-D Van der Pol ----------

_VDP_MU = 2.0
_VDP_D0 = 0.1


def _make_vdp_process():
    """1-D Van der Pol:  dv/dt = mu(1 - x^2)v - x  +  noise."""

    def vdp_force(x, *, v, mask=None):
        x0, v0 = x[0], v[0]
        return jnp.array([_VDP_MU * (1.0 - x0**2) * v0 - x0])

    F_basis = make_basis(vdp_force, dim=1, rank=1, n_features=1, needs_v=True)
    D_basis = identity_matrix_basis(1)
    F_psf, D_psf = F_basis.to_psf(), D_basis.to_psf()

    proc = UnderdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([_VDP_D0])},
    )
    return proc


@pytest.fixture(scope="module")
def vdp_sim():
    """Simulate a short 1-D Van der Pol trajectory (3 000 steps, dt = 0.005)."""
    proc = _make_vdp_process()
    proc.initialize(
        jnp.array([1.0], dtype=jnp.float32),
        v0=jnp.array([0.0], dtype=jnp.float32),
    )

    key = random.PRNGKey(7)
    coll = proc.simulate(
        dt=0.005, Nsteps=3000, key=key, prerun=100, oversampling=5
    )
    return proc, coll


# ======================================================================
# 1) Overdamped end-to-end  (simulate → infer → validate)
# ======================================================================


class TestOverdampedEndToEnd:
    """Mirrors the Lorenz / OU examples but in 2-D with a linear basis."""

    def test_linear_force_inference_recovers_ou(self, ou_sim):
        """Monomial basis → infer_force_linear → NMSE < 0.5."""
        proc, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        # coefficients should be finite
        assert np.isfinite(np.asarray(inf.force_coefficients_full)).all()

        # compare to the exact model object rather than private bound fields
        inf.compare_to_exact(model_exact=proc, maxpoints=500)
        assert hasattr(inf, "NMSE_force")
        assert float(inf.NMSE_force) < 0.5

    def test_compute_force_error(self, ou_sim):
        """compute_force_error produces finite predicted MSE."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
        inf.compute_force_error()

        assert hasattr(inf, "force_predicted_MSE")
        assert np.isfinite(inf.force_predicted_MSE)
        assert hasattr(inf, "force_coefficients_covariance")
        assert np.isfinite(np.asarray(inf.force_coefficients_covariance)).all()
        assert hasattr(inf, "force_coefficients_stderr")
        assert np.isfinite(np.asarray(inf.force_coefficients_stderr)).all()

    def test_compute_force_error_matches_gaussian_coefficient_theory(self, ou_sim):
        """Predicted NMSE matches the coefficient-space Gaussian error model."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
        inf.compute_force_error()

        coeffs = np.asarray(inf.force_coefficients_full)
        moments = np.asarray(inf.force_moments)
        G = np.asarray(inf.force_G)
        cov = np.asarray(inf.force_coefficients_covariance)
        force_energy = float(coeffs @ moments)

        predicted_nmse = float(inf.force_predicted_MSE)
        theory_nmse = float(np.trace(G @ cov) / force_energy)
        np.testing.assert_allclose(predicted_nmse, theory_nmse, rtol=1e-6, atol=1e-10)

        rng = np.random.default_rng(0)
        evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
        sqrt_cov = (evecs * np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T
        deltas = rng.standard_normal((20000, cov.shape[0])) @ sqrt_cov.T
        mc_nmse = np.einsum("si,ij,sj->s", deltas, G, deltas) / force_energy
        np.testing.assert_allclose(mc_nmse.mean(), predicted_nmse, rtol=6e-2, atol=0.0)

    def test_diffusion_estimate_reasonable(self, ou_sim):
        """Inferred diffusion should be within ~50 % of D0·I."""
        _, coll = ou_sim
        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")

        D_est = np.asarray(inf.diffusion_average)
        D_true = _OU_D0 * np.eye(_OU_DIM)
        np.testing.assert_allclose(D_est, D_true, rtol=0.5, atol=0.05)

    @pytest.mark.parametrize("M_mode", ["Ito", "Strato"])
    @pytest.mark.parametrize("G_mode", ["rectangle", "trapeze"])
    def test_modes_produce_finite_results(self, ou_sim, M_mode, G_mode):
        """All M/G mode combinations yield finite G, moments, and coefficients."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode=M_mode, G_mode=G_mode)

        assert np.isfinite(np.asarray(inf.force_G_full)).all()
        assert np.isfinite(np.asarray(inf.force_moments)).all()
        assert np.isfinite(np.asarray(inf.force_coefficients_full)).all()


# ======================================================================
# 2) Underdamped end-to-end  (Van der Pol)
# ======================================================================


class TestUnderdampedEndToEnd:
    """Mirrors the vanderpol.py example."""

    def test_linear_force_inference_finite(self, vdp_sim):
        """Monomial basis with velocity → finite coefficients."""
        _, coll = vdp_sim
        B = _monomial_vector_basis(1, degree=2, include_v=True)

        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="auto")
        inf.infer_force_linear(B, M_mode="symmetric", G_mode="trapeze")

        assert np.isfinite(np.asarray(inf.force_coefficients_full)).all()
        assert np.isfinite(np.asarray(inf.force_G_full)).all()

    def test_diffusion_estimate_reasonable(self, vdp_sim):
        """Diffusion within ~2× of D0."""
        _, coll = vdp_sim
        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="auto")

        D_est = float(jnp.trace(inf.diffusion_average))
        assert 0.01 < D_est < 1.0  # D0 = 0.1, so 0.01 – 1.0 is generous

    def test_compute_force_error(self, vdp_sim):
        """compute_force_error works in underdamped setting."""
        _, coll = vdp_sim
        B = _monomial_vector_basis(1, degree=2, include_v=True)

        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="auto")
        inf.infer_force_linear(B, M_mode="symmetric", G_mode="trapeze")
        inf.compute_force_error()

        assert np.isfinite(inf.force_predicted_MSE)

    def test_compare_to_exact_with_model(self, vdp_sim):
        """compare_to_exact accepts an initialized exact model object."""
        proc, coll = vdp_sim
        B = _monomial_vector_basis(1, degree=2, include_v=True)

        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="auto")
        inf.infer_force_linear(B, M_mode="symmetric", G_mode="trapeze")

        inf.compare_to_exact(model_exact=proc, maxpoints=500)
        assert hasattr(inf, "NMSE_force")
        assert np.isfinite(float(inf.NMSE_force))

    @pytest.mark.parametrize("method", ["auto", "WeakNoise", "noisy"])
    def test_diffusion_methods(self, vdp_sim, method):
        """All diffusion methods yield finite PSD diffusion."""
        _, coll = vdp_sim
        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method=method)

        D = np.asarray(inf.diffusion_average)
        assert np.isfinite(D).all()
        eig = np.linalg.eigvalsh(D)
        assert np.all(eig > -1e-6)  # PSD


# ======================================================================
# 3) Nonlinear (PSF-based) force inference
# ======================================================================


class TestParametricPSFInference:
    """PSF-parameterised force families through the minimal parametric path."""

    def test_psf_ou_recovers_finite_params(self, ou_sim):
        """PSF-based parametric inference on OU → finite parameters."""
        _, coll = ou_sim

        # Build a PSF parameterisation for F(x; k) = -diag(k) x
        def ou_force_single(x, *, params, mask=None, extras=None):
            k = params["k"]
            return -k * x

        F_psf = make_psf(
            ou_force_single,
            dim=_OU_DIM,
            rank=1,
            n_features=1,
            params={"k": (_OU_DIM,)},
            labels=["F = -k*x"],
        )
        theta0 = {"k": jnp.ones(_OU_DIM) * 0.5}

        inf = OverdampedLangevinInference(coll)
        inf.infer_force(F_psf, theta0)

        # Parameters should be finite and in the right ballpark
        k_fit = np.asarray(F_psf.unflatten_params(inf.force_coefficients_full)["k"])
        assert np.isfinite(k_fit).all()
        # OU has k = [0.8, 1.2]; check within 50%
        np.testing.assert_allclose(k_fit, np.asarray(_OU_K), rtol=0.5, atol=0.1)

    def test_psf_sets_force_inferred(self, ou_sim):
        """After parametric inference, force_inferred should be callable."""
        _, coll = ou_sim

        def ou_force_single(x, *, params, mask=None, extras=None):
            k = params["k"]
            return -k * x

        F_psf = make_psf(
            ou_force_single,
            dim=_OU_DIM,
            rank=1,
            n_features=1,
            params={"k": (_OU_DIM,)},
        )
        theta0 = {"k": jnp.ones(_OU_DIM) * 0.5}

        inf = OverdampedLangevinInference(coll)
        inf.infer_force(F_psf, theta0)

        # force_inferred should be callable
        x_test = jnp.ones((1, _OU_DIM))
        F_val = inf.force_inferred(x_test)
        assert F_val.shape == (1, _OU_DIM)
        assert np.isfinite(np.asarray(F_val)).all()


# ======================================================================
# 4) Sparsification
# ======================================================================


class TestSparsification:
    """Overcomplete monomial basis → sparsify_force should prune features."""

    def test_sparsify_reduces_features(self, ou_sim):
        """
        Degree-3 monomial basis in 2-D has 20 features for a 2-param truth.
        Sparsification should select a small subset.
        """
        _, coll = ou_sim

        # overcomplete basis: degree 3 → many features
        B = _monomial_vector_basis(_OU_DIM, degree=3)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        n_full = inf.force_coefficients_full.shape[0]
        assert n_full > 4, "Basis should be overcomplete"

        inf.sparsify_force(criterion="PASTIS", beam_width=2, max_k=6)

        # After sparsification, some coefficients should be zeroed out
        n_nonzero = int(np.sum(np.asarray(inf.force_coefficients_full) != 0))
        assert n_nonzero < n_full, "Sparsification should have pruned some features"

    def test_sparsify_force_inferred_still_callable(self, ou_sim):
        """After sparsification, force_inferred should still work."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=2)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")
        inf.sparsify_force(criterion="PASTIS", beam_width=2)

        x_test = jnp.ones((1, _OU_DIM))
        F_val = inf.force_inferred(x_test)
        assert F_val.shape == (1, _OU_DIM)
        assert np.isfinite(np.asarray(F_val)).all()


# ======================================================================
# 5) Degradation + inference
# ======================================================================


class TestDegradedData:
    """Test that inference still works after data degradation."""

    def test_downsample_and_noise(self, ou_sim):
        """
        Downsample + measurement noise → force is still recoverable (noisier).
        """
        proc, coll = ou_sim
        degraded = coll.degrade(downsample=3, noise=0.02, seed=0)

        B = _monomial_vector_basis(_OU_DIM, degree=1)
        inf = OverdampedLangevinInference(degraded)
        inf.compute_diffusion_constant(method="auto")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        assert np.isfinite(np.asarray(inf.force_coefficients_full)).all()

        # NMSE should be finite even on degraded data
        inf.compare_to_exact(model_exact=proc, maxpoints=300)
        assert np.isfinite(float(inf.NMSE_force))

    def test_data_loss(self, ou_sim):
        """
        Losing 10% of frames → inference still finite.
        """
        _, coll = ou_sim
        degraded = coll.degrade(data_loss_fraction=0.10, seed=1)

        B = _monomial_vector_basis(_OU_DIM, degree=1)
        inf = OverdampedLangevinInference(degraded)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        assert np.isfinite(np.asarray(inf.force_coefficients_full)).all()
        assert np.isfinite(np.asarray(inf.force_G_full)).all()


# ======================================================================
# 6) Bootstrapped trajectory
# ======================================================================


class TestBootstrap:
    """Re-simulate from inferred model (overdamped and underdamped)."""

    def test_overdamped_bootstrap(self, ou_sim):
        """Bootstrap trajectory should be a valid TrajectoryCollection."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        key = random.PRNGKey(99)
        boot_coll, boot_proc = inf.simulate_bootstrapped_trajectory(
            key, oversampling=2
        )

        from SFI.trajectory.collection import TrajectoryCollection

        assert isinstance(boot_coll, TrajectoryCollection)
        X = np.asarray(boot_coll.datasets[0].X)
        assert np.isfinite(X).all()
        # trajectory should have moved
        assert np.std(X) > 0.01

    def test_underdamped_bootstrap(self, vdp_sim):
        """Bootstrap trajectory from underdamped inference."""
        _, coll = vdp_sim
        B = _monomial_vector_basis(1, degree=2, include_v=True)

        inf = UnderdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="auto")
        inf.infer_force_linear(B, M_mode="symmetric", G_mode="trapeze")

        key = random.PRNGKey(77)
        boot_coll, boot_proc = inf.simulate_bootstrapped_trajectory(
            key, oversampling=2
        )

        from SFI.trajectory.collection import TrajectoryCollection

        assert isinstance(boot_coll, TrajectoryCollection)
        X = np.asarray(boot_coll.datasets[0].X)
        assert np.isfinite(X).all()


# ======================================================================
# 7) Serialization round-trip
# ======================================================================


class TestSerialization:
    """save_results / load_results  and  save_model / load_model."""

    def test_save_load_results(self, ou_sim, tmp_path):
        """Lightweight .npz round-trip preserves report_dict data."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        path = tmp_path / "test_results"
        inf.save_results(str(path))

        from SFI.inference.serialization import load_results

        loaded = load_results(str(path))
        assert "diffusion_average" in loaded
        np.testing.assert_allclose(
            loaded["diffusion_average"],
            np.asarray(inf.diffusion_average),
            atol=1e-7,
        )

    def test_save_load_model(self, ou_sim, tmp_path):
        """Equinox model round-trip preserves callable predictions."""
        _, coll = ou_sim
        B = _monomial_vector_basis(_OU_DIM, degree=1)

        inf = OverdampedLangevinInference(coll)
        inf.compute_diffusion_constant(method="WeakNoise")
        inf.infer_force_linear(B, M_mode="Ito", G_mode="trapeze")

        result_sf = inf.force_inferred
        path = tmp_path / "test_model"
        result_sf.save(str(path))

        from SFI.inference.result import InferenceResultSF
        from SFI.statefunc import SF

        # Build a template from the same basis (zero params)
        P = B.to_psf()
        zero_params = {
            s.name: jnp.zeros(s.shape) for s in P.template.specs
        }
        template_sf = SF(P, zero_params)
        template = InferenceResultSF(template_sf)
        loaded = InferenceResultSF.load(str(path), template=template)

        x_test = jnp.ones((1, _OU_DIM))
        np.testing.assert_allclose(
            np.asarray(loaded(x_test)),
            np.asarray(result_sf(x_test)),
            atol=1e-6,
        )
