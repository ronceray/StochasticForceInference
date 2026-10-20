"""Replica batches: ``x0`` of shape ``(B, d)`` with a ``pdepth=0`` force.

The ``B`` rows are integrated as independent copies of the same process and
land in the ``(T, B, d)`` dataset layout, i.e. exactly what a ``pdepth=1``
particle system produces.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import make_sf


def _harmonic(k=1.0, dim=2):
    return make_sf(lambda x, **kw: -k * x, dim=dim, rank=1)


def _damped_harmonic(dim=2):
    return make_sf(lambda x, v, **kw: -x - v, dim=dim, rank=1, needs_v=True)


# ----------------------------- Overdamped -------------------------------


def test_overdamped_replicas_shape_and_metadata():
    B, d, T = 5, 2, 30
    proc = OverdampedProcess(_harmonic(dim=d), D=0.3)
    x0 = jnp.asarray(np.random.default_rng(0).normal(size=(B, d)))
    proc.initialize(x0)
    coll = proc.simulate(dt=0.01, Nsteps=T, key=random.PRNGKey(0), oversampling=2)

    ds = coll.datasets[0]
    assert ds.X.shape == (T, B, d)
    assert ds.N == B
    assert ds.meta["pdepth"] == 0
    assert ds.meta["num_particles"] == B
    obs = ds.meta["observables"]
    assert np.isfinite(obs["information"]) and np.isfinite(obs["entropy"])


def test_overdamped_single_replica_keeps_squeezed_output():
    proc = OverdampedProcess(_harmonic(), D=0.3)
    proc.initialize(jnp.array([[1.0, 0.0]]))  # (1, d) -> (d,)
    coll = proc.simulate(dt=0.01, Nsteps=10, key=random.PRNGKey(0))
    assert coll.datasets[0].X.shape == (10, 2)
    assert coll.datasets[0].meta["num_particles"] == 1


def test_overdamped_replicas_match_independent_runs_in_the_small_noise_limit():
    """Each row of a replica batch follows the same flow as a separate run."""
    B, d, T, dt = 4, 2, 40, 0.01
    F = _harmonic(k=1.5, dim=d)
    x0 = jnp.asarray(np.random.default_rng(1).normal(size=(B, d)))

    proc = OverdampedProcess(F, D=1e-12)
    proc.initialize(x0)
    Xb = proc.simulate(dt=dt, Nsteps=T, key=random.PRNGKey(3), oversampling=1, compute_observables=False)
    Xb = np.asarray(Xb.datasets[0].X)

    for i in range(B):
        p = OverdampedProcess(F, D=1e-12)
        p.initialize(x0[i])
        Xi = p.simulate(dt=dt, Nsteps=T, key=random.PRNGKey(3), oversampling=1, compute_observables=False)
        np.testing.assert_allclose(Xb[:, i], np.asarray(Xi.datasets[0].X), atol=1e-5, rtol=0)


def test_overdamped_replicas_are_independent_and_ergodic():
    """Stationary variance D/k in every replica, no cross-replica correlation."""
    B, k, D_val, dt = 100, 2.0, 0.5, 0.05
    proc = OverdampedProcess(_harmonic(k=k, dim=1), D=D_val)
    proc.initialize(jnp.zeros((B, 1)))
    coll = proc.simulate(
        dt=dt, Nsteps=2000, key=random.PRNGKey(7), oversampling=4, prerun=200, compute_observables=False
    )
    X = np.asarray(coll.datasets[0].X)[:, :, 0]  # (T, B)

    exact_var = D_val / k
    assert abs(X.var() - exact_var) / exact_var < 0.1

    C = np.corrcoef(X.T)
    off = C[~np.eye(B, dtype=bool)]
    assert np.abs(off).mean() < 0.15  # shared noise across rows would give ~1


# ----------------------------- Underdamped ------------------------------


def test_underdamped_replicas_shape_and_metadata():
    B, d, T = 3, 2, 20
    F = _damped_harmonic(dim=d)
    x0 = jnp.asarray(np.random.default_rng(0).normal(size=(B, d)))

    proc = UnderdampedProcess(F, D=0.1)
    proc.initialize(x0)
    coll = proc.simulate(dt=0.01, Nsteps=T, key=random.PRNGKey(0))
    ds = coll.datasets[0]
    assert ds.X.shape == (T, B, d)
    assert ds.meta["num_particles"] == B
    assert ds.meta["pdepth"] == 0

    # explicit v0 must match the replica batch shape
    UnderdampedProcess(F, D=0.1).initialize(x0, v0=jnp.zeros_like(x0))
    with pytest.raises(ValueError):
        UnderdampedProcess(F, D=0.1).initialize(x0, v0=jnp.zeros((B + 1, d)))


def test_underdamped_single_replica_keeps_squeezed_output():
    proc = UnderdampedProcess(_damped_harmonic(), D=0.1)
    proc.initialize(jnp.array([[1.0, 0.0]]), v0=jnp.array([[0.0, 0.0]]))
    coll = proc.simulate(dt=0.01, Nsteps=10, key=random.PRNGKey(0))
    assert coll.datasets[0].X.shape == (10, 2)


# ----------------------------- End to end -------------------------------


def test_replica_ensemble_infers_harmonic_force():
    """A replica ensemble is inferred like a particle system and recovers F = -k x."""
    from SFI import OverdampedLangevinInference
    from SFI.bases.monomials import monomials_up_to

    B, d, k = 40, 2, 1.0
    proc = OverdampedProcess(_harmonic(k=k, dim=d), D=1.0)
    proc.initialize(jnp.asarray(np.random.default_rng(2).normal(size=(B, d))))
    coll = proc.simulate(
        dt=0.02, Nsteps=1500, key=random.PRNGKey(11), oversampling=4, prerun=100, compute_observables=False
    )
    assert coll.datasets[0].X.shape == (1500, B, d)

    basis = monomials_up_to(1, dim=d, rank="vector")
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    inf.infer_force_linear(basis)

    # F = -k x  <=>  coefficient -k on the diagonal linear terms, 0 elsewhere.
    labels = list(basis.labels)
    expected = np.array([-k if lab in ("x0·e0", "x1·e1") else 0.0 for lab in labels])
    coeffs = np.asarray(inf.force_coefficients).ravel()
    # ~60k samples at D = 1: coefficient noise ~0.04, finite-dt bias ~1%.
    np.testing.assert_allclose(coeffs, expected, atol=0.15)
