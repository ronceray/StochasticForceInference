"""The linear force estimators warn when the force Gram is ill-conditioned.

A collinear dictionary (two features proportional to each other) makes the
Gram singular: the pseudo-inverse solve still returns coefficients, but they
and their error estimates are meaningless.  The engine records the condition
number in ``metadata["force_gram_cond"]`` and warns above a dtype-aware
threshold; a well-conditioned basis is silent.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases import monomials_up_to
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.inference.underdamped import UnderdampedLangevinInference
from SFI.statefunc import make_basis
from SFI.trajectory import TrajectoryCollection


def _ou_collection(T=3000, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 1), dtype=np.float32)
    for t in range(T - 1):
        x[t + 1] = x[t] - dt * x[t] + np.sqrt(2.0 * 0.5 * dt) * rng.normal(size=1)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(x), dt=dt)


def _collinear_basis():
    return make_basis(lambda x, mask=None: jnp.stack([x, 2.0 * x], axis=-1), dim=1, rank=1, n_features=2)


def _no_gram_warning(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    assert not [w for w in rec if "condition number" in str(w.message)]


def test_overdamped_collinear_basis_warns():
    inf = OverdampedLangevinInference(_ou_collection())
    inf.compute_diffusion_constant()
    with pytest.warns(UserWarning, match="condition number"):
        inf.infer_force_linear(_collinear_basis(), preset="clean")
    assert inf.metadata["force_gram_cond"] > 1e6


def test_overdamped_well_conditioned_basis_is_silent():
    inf = OverdampedLangevinInference(_ou_collection())
    inf.compute_diffusion_constant()
    _no_gram_warning(lambda: inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="clean"))
    assert 1.0 <= inf.metadata["force_gram_cond"] < 1e3


def test_underdamped_collinear_basis_warns():
    rng = np.random.default_rng(1)
    T, dt = 3000, 0.01
    x, v = np.zeros((T, 1), dtype=np.float32), np.zeros(1, dtype=np.float32)
    for t in range(T - 1):
        v = v + dt * (-x[t] - v) + np.sqrt(2.0 * 0.5 * dt) * rng.normal(size=1)
        x[t + 1] = x[t] + dt * v
    coll = TrajectoryCollection.from_arrays(X=jnp.asarray(x), dt=dt)
    inf = UnderdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    B = make_basis(lambda x, v, mask=None: jnp.stack([v, 3.0 * v], axis=-1), dim=1, rank=1, n_features=2, needs_v=True)
    with pytest.warns(UserWarning, match="condition number"):
        inf.infer_force_linear(B, preset="clean")
    assert inf.metadata["force_gram_cond"] > 1e6
