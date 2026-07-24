"""Held-out validation: split_time + holdout_score."""

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
from SFI.statefunc.factory import make_basis


def _vector_monomials(dim, degree, include_v=False):
    Bsc = monomials_up_to(order=degree, dim=dim, include_constant=True, include_x=True, include_v=include_v)
    return Bsc * unit_vector_basis(dim)


def _simulate_ou(Nsteps=6000, key=0):
    F = make_basis(lambda x, *, mask=None: -x, dim=1, rank=1, n_features=1)
    proc = OverdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([0.2])})
    proc.initialize(jnp.zeros(1, dtype=jnp.float32))
    return proc.simulate(dt=0.01, Nsteps=Nsteps, key=random.PRNGKey(key), prerun=100, oversampling=5)


def test_holdout_clean_ou_matches_predicted():
    coll = _simulate_ou()
    train, test = coll.split_time(0.7)

    inf = OverdampedLangevinInference(train)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_vector_monomials(1, 1), M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()

    out = inf.holdout_score(test)
    assert np.isfinite(out["holdout_NMSE"])
    assert out["holdout_NMSE"] >= 0.0
    # Well-specified fit: held-out error small, comparable to predicted.
    assert out["holdout_NMSE"] < max(3.0 * out["predicted_NMSE"], 0.05)
    assert inf.force_holdout_NMSE == out["holdout_NMSE"]


def test_holdout_detects_misspecification():
    """Linear fit on a double-well: held-out error far above predicted."""

    F = make_basis(lambda x, *, mask=None: x - x**3, dim=1, rank=1, n_features=1)
    proc = OverdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([0.15])})
    proc.initialize(jnp.array([0.5], dtype=jnp.float32))
    coll = proc.simulate(dt=0.005, Nsteps=8000, key=random.PRNGKey(2), prerun=200, oversampling=10)
    train, test = coll.split_time(0.7)

    inf = OverdampedLangevinInference(train)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_vector_monomials(1, 1), M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()

    out = inf.holdout_score(test)
    assert np.isfinite(out["holdout_NMSE"])
    assert out["ratio"] > 10.0


def test_holdout_underdamped_smoke():
    F = make_basis(
        lambda x, v, *, mask=None: -x - 0.5 * v, dim=1, rank=1, n_features=1, needs_v=True
    )
    proc = UnderdampedProcess(F.to_psf(), D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([0.3])})
    proc.initialize(jnp.zeros(1, dtype=jnp.float32), v0=jnp.zeros(1, dtype=jnp.float32))
    coll = proc.simulate(dt=0.02, Nsteps=6000, key=random.PRNGKey(3), prerun=100, oversampling=5)
    train, test = coll.split_time(0.7)

    inf = UnderdampedLangevinInference(train)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(_vector_monomials(1, 1, include_v=True))
    inf.compute_force_error()

    out = inf.holdout_score(test)
    assert np.isfinite(out["holdout_NMSE"])
    assert out["n_obs"] > 100
    # The NMSE *units* are coarse for this weakly-forced UD system (chi^2
    # resolution); the calibrated statement is the z-score: a
    # well-specified fit must not flag bias.
    assert abs(out["excess_z"]) < 5.0


def test_assess_accepts_holdout_data():
    coll = _simulate_ou(Nsteps=4000, key=5)
    train, test = coll.split_time(0.7)
    inf = OverdampedLangevinInference(train)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(_vector_monomials(1, 1), M_mode="Ito", G_mode="trapeze")
    inf.compute_force_error()
    rep = inf.diagnose(data=test)
    assert rep.residuals["moments"]["n"] > 100
