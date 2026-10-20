# TODO: review this file
import jax.numpy as jnp
import pytest

from SFI.langevin.overdamped import OverdampedProcess
from SFI.statefunc import make_basis
from SFI.trajectory.collection import TrajectoryCollection


def harmonic_force(x, **kwargs):
    # simple linear spring: F(x) = -x
    return -x


def test_overdamped_simulate_time_metadata(jax_key):
    F = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F=F, D=1.0)
    x0 = jnp.array([1.0, 0.0])
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(x0)

    dt = 0.05
    Nsteps = 8
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    # one point per step
    assert ds.X.shape[0] == Nsteps
    assert ds.d == 2

    # Either dt or t must be present; if dt is scalar it should match the argument
    if ds.t is not None:
        t = jnp.asarray(ds.t)
        assert t.shape[0] == ds.T
    elif ds.dt is not None:
        dt_val = float(jnp.asarray(ds.dt))
        assert dt_val == pytest.approx(dt)
    else:
        pytest.fail("Dataset has neither t nor dt set")
