"""A parametric force fit that returns the all-zero solution must say so.

Regression: in the noise-dominated fine-Δt regime (Van der Pol μ = 3,
D = 0.05, Δt = 0.02, localisation noise σ = 0.05 — the gallery's stride-1
fit) the underdamped parametric ``infer_force`` returned θ ≡ 0 silently:
the Gauss–Newton solver never left its zero initial guess, so
``force_information = 0`` and ``force_predicted_MSE`` is NaN.  The result is
unchanged; a ``RuntimeWarning`` now flags it.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI import UnderdampedLangevinInference
from SFI.bases import identity_matrix_basis, monomials_up_to
from SFI.inference.base import BaseLangevinInference
from SFI.langevin import UnderdampedProcess
from SFI.statefunc.factory import make_basis

MSG = "all-zero force"


def test_helper_fires_only_on_the_all_zero_solution():
    warn = BaseLangevinInference._warn_if_null_parametric_force
    with pytest.warns(RuntimeWarning, match=MSG):
        assert warn(jnp.zeros(4), {"outer_iterations": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not warn(jnp.array([0.0, 1e-12, 0.0]))  # any non-zero coefficient
        assert not warn(jnp.zeros(0))  # parameter-free model
        assert not warn(jnp.array([0.0, np.nan]))  # failed fit: reported elsewhere


def _noisy_vdp(Nsteps=2000):
    mu = 3.0

    def vdp_force(x, *, v, mask=None):
        return jnp.array([mu * (1.0 - x[0] ** 2) * v[0] - x[0]])

    F = make_basis(vdp_force, dim=1, rank=1, n_features=1, needs_v=True).to_psf()
    proc = UnderdampedProcess(F, D=identity_matrix_basis(1).to_psf())
    proc.set_params(theta_F={"coeff": jnp.array([1.0])}, theta_D={"coeff": jnp.array([0.05])})
    proc.initialize(jnp.array([1.0]), v0=jnp.array([0.0]))
    coll = proc.simulate(dt=0.02, Nsteps=Nsteps, key=random.PRNGKey(0), prerun=100, oversampling=20)
    return coll.degrade(noise=0.05, seed=3)


def test_underdamped_null_fit_warns_without_changing_the_result():
    inf = UnderdampedLangevinInference(_noisy_vdp())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        inf.infer_force(monomials_up_to(order=3, dim=1, include_v=True, rank="vector"))
    fired = [w for w in rec if issubclass(w.category, RuntimeWarning) and MSG in str(w.message)]
    theta = np.asarray(inf.force_coefficients_full)
    null_fit = not np.any(theta != 0)
    # The warning fires exactly on the null solution (reproduced on this data
    # when the test was written: θ ≡ 0 after one outer iteration).
    assert bool(fired) == null_fit
    if null_fit:
        assert fired[0].filename == __file__  # points at the user's call
        inf.compute_force_error()
        assert inf.force_information == 0.0
        assert np.isnan(inf.force_predicted_MSE)
