"""Drift and diffusion must treat measurement noise consistently.

Under localization noise the pre-point Itô drift carries an offset
-Λ ∇ln P / Δt and the raw diffusion stencils an offset Λ/Δt; ``preset="auto"``
corrects both (Stratonovich-shift drift + Vestergaard diffusion) or neither.
Correcting only one side turns the leftover offset into a spurious force
along ∇ln P.  The engines warn about such a mix before simulating the
fitted model and in ``print_report``; consistent pairs and clean data are
silent.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases import monomials_up_to
from SFI.inference.overdamped import OverdampedLangevinInference
from SFI.trajectory import TrajectoryCollection
from SFI.trajectory.degrade import degrade_collection


def _ou(T=6000, dt=0.01, D0=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 1), dtype=np.float32)
    for t in range(T - 1):
        x[t + 1] = x[t] - dt * x[t] + np.sqrt(2.0 * D0 * dt) * rng.normal(size=1)
    return TrajectoryCollection.from_arrays(X=jnp.asarray(x), dt=dt)


def _fit(coll, *, method, preset):
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method=method)
    inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset=preset)
    return inf


def _silent(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    assert not [w for w in rec if "measurement-noise" in str(w.message)]


def test_mixed_treatments_warn_under_noise():
    noisy = degrade_collection(_ou(), noise=0.05, seed=1)
    inf = _fit(noisy, method="noisy", preset="clean")
    assert inf.Lambda_trace > 0
    with pytest.warns(UserWarning, match="measurement-noise"):
        inf.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False)
    inf2 = _fit(noisy, method="WeakNoise", preset="robust")
    with pytest.warns(UserWarning, match="measurement-noise"):
        inf2.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False)
    assert inf._noise_consistency_issue() and inf2._noise_consistency_issue()


def test_consistent_pairs_and_clean_data_are_silent():
    noisy = degrade_collection(_ou(), noise=0.05, seed=1)
    both = _fit(noisy, method="auto", preset="auto")
    assert both.metadata["force_preset"] == "robust" and both.metadata["diffusion_constant_method"] == "noisy"
    _silent(lambda: both.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False))
    raw = _fit(noisy, method="WeakNoise", preset="clean")
    _silent(lambda: raw.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False))


def test_mixed_treatments_are_silent_without_detected_noise():
    # The raw Lambda estimate carries an O(dt^2) confinement bias and fluctuates
    # around zero on clean data, so the no-noise branch is exercised by
    # overriding the detected level rather than by hoping for a negative draw.
    mixed = _fit(degrade_collection(_ou(), noise=0.05, seed=1), method="noisy", preset="clean")
    assert mixed._noise_consistency_issue() is not None
    mixed.Lambda_trace = -1e-4
    _silent(lambda: mixed.simulate_bootstrapped_trajectory(random.PRNGKey(0), simulate=False))
    assert mixed._noise_consistency_issue() is None
