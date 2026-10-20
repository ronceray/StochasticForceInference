"""``psd_fraction`` and the PSD check that precedes a bootstrap simulation.

A diffusion dictionary without a constant term fitted on additive-noise
data gives ``D(x) = c x`` with ``c ≈ 0``: the field changes sign across the
data, is clipped silently by the simulator, and must be reported.  A
constant or a well-posed fit is PSD everywhere and silent.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases import monomials_up_to
from SFI.bases.constants import constant_array
from SFI.diagnostics import psd_fraction
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.statefunc import make_basis, make_sf
from SFI.trajectory import TrajectoryCollection


def _ou(T=4000, dt=0.01, D0=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 1), dtype=np.float32)
    for t in range(T - 1):
        x[t + 1] = x[t] - dt * x[t] + np.sqrt(2.0 * D0 * dt) * rng.normal(size=1)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(x), dt=dt)


def test_psd_fraction_on_hand_made_fields():
    coll = _ou()
    const = constant_array(jnp.array([[0.5]]))
    r = psd_fraction(const, coll)
    assert r["available"] and r["fraction"] == 0.0 and r["min_eig"] > 0
    sign = make_sf(lambda x: x[0] * jnp.eye(1), dim=1, rank=2)
    r = psd_fraction(sign, coll)
    assert r["available"] and 0.3 < r["fraction"] < 0.7, r      # ≈ P(x < 0)
    assert r["min_eig"] < 0 and r["n_negative"] > 0


def test_bootstrap_simulation_warns_on_non_psd_fit_and_is_silent_otherwise():
    coll = _ou()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="clean")
    inf.infer_diffusion_linear(
        make_basis(lambda x, mask=None: (x[0] * jnp.eye(1))[..., None], dim=1, rank=2, n_features=1),
        M_mode="WeakNoise",
    )
    with pytest.warns(UserWarning, match="not positive semi-definite"):
        inf.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False)
    assert 0.3 < inf.metadata["diffusion_psd_fraction"] < 0.7

    good = OverdampedLangevinInference(coll)
    good.compute_diffusion_constant(method="WeakNoise")
    good.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="clean")
    good.infer_diffusion_linear(M_mode="WeakNoise")   # constant symmetric matrix
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        good.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False)
    assert not [w for w in rec if "semi-definite" in str(w.message)]
    assert good.metadata["diffusion_psd_fraction"] == 0.0
