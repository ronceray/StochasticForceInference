# TODO: review this file
# tests/integrate/test_weights_and_reduction.py
import jax.numpy as jnp
import numpy as np

from SFI.integrate.api import integrate
from SFI.integrate.integrand import Integrand, Term, TimeOperand
from SFI.integrate.timeops import timeop
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


def _make_coll_simple(T=6, N=4, d=2, dt=0.5):
    X = np.cumsum(
        np.random.default_rng(0).normal(size=(T, N, d)).astype(np.float32), axis=0
    )
    ds = TrajectoryDataset.from_arrays(X=X, dt=dt)
    return TrajectoryCollection.from_dataset(ds)


def test_default_reduces_over_particles_sum():
    """
    If the program returns a per-particle vector y_i (leave the i axis present),
    the integrator reduces over i when reduce_over_particles=True (default).
    """
    coll = _make_coll_simple(T=5, N=3, d=2, dt=0.2)

    # timeop returning shape (N,). Keep i alive so the integrator reduces it.
    @timeop(name="onesN")
    def _ones(**streams):
        N = streams["X"].shape[0]  # row X is (N, d)
        return jnp.ones((N,), dtype=jnp.float32)

    _ones._requires = frozenset({"X"})  # type: ignore[attr-defined]

    # Important: pass the i-axis through the einsum ("i->i"). The integrator will then reduce i.
    prog = Integrand(
        times=[TimeOperand(_ones, alias="O")], terms=[Term(eq="i->i", ops=("O",))]
    )

    out = integrate(coll, prog, reduce="sum")  # default reduce_over_particles=True

    # Reference: Σ_t N * dt
    T = coll.datasets[0].X.shape[0] - 1
    N = coll.datasets[0].X.shape[1]
    dt = float(coll.datasets[0].dt)
    ref = N * dt * T
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6, atol=1e-6)


def test_reduce_over_particles_scalar_output_raises_then_ok_when_disabled():
    coll = _make_coll_simple(T=6, N=3, d=2, dt=0.2)

    @timeop(name="scalar1")
    def _scalar(**streams):
        return jnp.array(1.0, dtype=jnp.float32)  # no i-axis

    _scalar._requires = frozenset({"X"})  # type: ignore[attr-defined]

    prog = Integrand(
        times=[TimeOperand(_scalar, alias="S")], terms=[Term(eq="->", ops=("S",))]
    )

    import pytest

    with pytest.raises(ValueError):
        _ = integrate(coll, prog, reduce="sum", reduce_over_particles=True)

    out = integrate(coll, prog, reduce="sum", reduce_over_particles=False)
    T = (
        coll.datasets[0].T - 1
    )  # dt window enforced.
    dt = float(coll.datasets[0].dt)
    np.testing.assert_allclose(np.array(out), T * dt, rtol=1e-6, atol=1e-6)


def test_reduce_over_particles_scalar_output_raises_then_ok_when_disabled():
    coll = _make_coll_simple(T=6, N=3, d=2, dt=0.2)

    class ScalarProg:
        def require(self):
            return {"X"}

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.array(1.0, dtype=jnp.float32)

    import pytest

    with pytest.raises(ValueError):
        _ = integrate(coll, ScalarProg(), reduce="sum", reduce_over_particles=True)

    out = integrate(coll, ScalarProg(), reduce="sum", reduce_over_particles=False)
    ds = coll.datasets[0]
    T = ds.T - 1
    dt = float(ds.dt)
    np.testing.assert_allclose(np.array(out), T * dt, rtol=1e-6, atol=1e-6)


def test_masked_particles_zeroed_and_matches_Teff():
    T, N, d = 9, 5, 2
    X = np.cumsum(
        np.random.default_rng(0).normal(size=(T, N, d)).astype(np.float32), axis=0
    )
    mask = np.ones((T, N), dtype=bool)
    mask[2, :2] = False
    mask[5, 3:] = False
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, mask=mask)
    coll = TrajectoryCollection.from_dataset(ds)

    class OnesPerParticle:
        def require(self):
            return {"X"}

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.ones((row["X"].shape[0],), dtype=jnp.float32)

    out = integrate(coll, OnesPerParticle(), reduce="sum")

    # The integrator internally adds "__dt__" which tightens the mask window,
    # so Teff must be computed with the same requirement set.
    teff = ds.Teff({"X", "__dt__"})
    np.testing.assert_allclose(np.array(out), teff, rtol=1e-6, atol=1e-6)


def test_mean_of_ones_is_one_with_and_without_subsampling():
    coll = _make_coll_simple(T=11, N=4, d=2, dt=0.2)

    class OnesPerParticle:
        def require(self):
            return {"X"}

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.ones((row["X"].shape[0],), dtype=jnp.float32)

    out_full = integrate(coll, OnesPerParticle(), reduce="mean", subsampling=1)
    out_sub2 = integrate(coll, OnesPerParticle(), reduce="mean", subsampling=2)
    out_sub3 = integrate(coll, OnesPerParticle(), reduce="mean", subsampling=3)

    np.testing.assert_allclose(np.array(out_full), 1.0, rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.array(out_sub2), 1.0, rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.array(out_sub3), 1.0, rtol=1e-6, atol=0)


def test_subsampling_sum_equals_teff_on_kept_rows():
    coll = _make_coll_simple(T=12, N=3, d=2, dt=0.25)
    ds = coll.datasets[0]

    class OnesPerParticle:
        def require(self):
            return {"X"}

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.ones((row["X"].shape[0],), dtype=jnp.float32)

    out = integrate(coll, OnesPerParticle(), reduce="sum", subsampling=2)

    idx_all = np.array(ds.valid_indices({"X", "__dt__"}))
    teff_keep = ds.Teff({"X"}, subsampling=2)
    np.testing.assert_allclose(np.array(out), teff_keep, rtol=1e-6, atol=1e-6)
