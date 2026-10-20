"""coeff_block after sparsification, and predict_time_profile on a time-Fourier block."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import SFI
from SFI.bases import X, extra_scalar, time_fourier
from SFI.langevin import OverdampedProcess
from SFI.trajectory import time_series_extra


@pytest.fixture(scope="module")
def fitted():
    dt, T = 0.02, 1500
    t = np.arange(T) * dt
    w = 2 * np.pi / t[-1]
    k_t = 2.0 + 0.8 * np.cos(w * t)
    proc = OverdampedProcess(F=extra_scalar("neg_k") * X(dim=1), D=0.1, theta_F=jnp.array([1.0]))
    proc.set_extras(extras_global={"neg_k": time_series_extra(-k_t)})
    runs = []
    for i in range(16):
        proc.initialize(jnp.array([0.3 * (i % 4 - 1.5)]))
        runs.append(proc.simulate(dt=dt, Nsteps=T, key=random.PRNGKey(i), oversampling=2))
    coll = runs[0].merge(runs[1:])

    n_modes = 2
    B = time_fourier(n_modes) * X(dim=1)
    inf = SFI.OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(B, preset="clean")
    inf.compute_force_error()
    inf.sparsify_force(criterion="PASTIS")
    inf.compute_force_error()
    return inf, B, t, w, k_t, n_modes


def test_coeff_block_embeds_support_covariance(fitted):
    inf, B, *_ = fitted
    coeffs, cov = inf.coeff_block(B)
    n = B.n_features
    assert coeffs.shape == (n,) and cov.shape == (n, n)
    support = np.asarray(inf.force_support)
    pruned = np.setdiff1d(np.arange(n), support)
    assert len(pruned) > 0
    np.testing.assert_array_equal(cov[pruned], 0.0)
    np.testing.assert_allclose(cov[np.ix_(support, support)], np.asarray(inf.force_coefficients_covariance))


def test_predict_time_profile_matches_design(fitted):
    inf, B, t, w, k_t, n_modes = fitted
    prof, std = inf.predict_time_profile(B, t, x=np.ones(1), return_std=True)
    coeffs, cov = inf.coeff_block(B)
    rows = [np.ones_like(t)]
    for j in range(1, n_modes + 1):
        rows += [np.cos(j * w * t), np.sin(j * w * t)]
    Fm = np.stack(rows, axis=1)
    np.testing.assert_allclose(prof, Fm @ coeffs, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(std, np.sqrt(np.einsum("ti,ij,tj->t", Fm, cov, Fm)), rtol=1e-4, atol=1e-6)
    # the reconstruction tracks -k(t) within a few standard errors
    assert np.mean(np.abs(prof + k_t) / std) < 4.0
