"""Tests for fd_velocity and the TrajectoryCollection array/velocity helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.utils.maths import fd_velocity


@pytest.mark.parametrize("scheme", ["central", "forward", "backward"])
def test_fd_velocity_linear_is_exact(scheme):
    # x(t) = 2 + 3t  ->  v == 3 everywhere, for every stencil (linear is exact)
    dt = 0.1
    t = np.arange(20) * dt
    x = (2.0 + 3.0 * t)[:, None]  # (T, 1)
    v = np.asarray(fd_velocity(x, dt, scheme=scheme))
    assert v.shape == x.shape
    np.testing.assert_allclose(v, 3.0, atol=1e-4)


def test_fd_velocity_preserves_shape_TNd():
    dt = 0.05
    x = jnp.asarray(np.random.RandomState(0).randn(15, 4, 3))  # (T, N, d)
    v = fd_velocity(x, dt)
    assert v.shape == x.shape


def test_fd_velocity_nonuniform_dt_linear():
    # non-uniform spacing, still linear in t -> central recovers the slope
    t = np.array([0.0, 0.1, 0.25, 0.4, 0.7, 1.1])
    x = (1.0 - 2.0 * t)[:, None]
    dt = np.diff(t)
    v = np.asarray(fd_velocity(x, dt, scheme="central"))
    assert v.shape == x.shape
    np.testing.assert_allclose(v, -2.0, atol=1e-4)


def test_fd_velocity_too_short_raises():
    with pytest.raises(ValueError):
        fd_velocity(np.zeros((1, 2)), 0.1)


def _tiny_collection(Nsteps=40):
    from SFI.bases import X
    from SFI.langevin import OverdampedProcess

    proc = OverdampedProcess(F=-1.0 * X(dim=2), D=0.5)
    proc.initialize(jnp.array([0.3, -0.2]))
    return proc.simulate(dt=0.05, Nsteps=Nsteps, key=__import__("jax").random.PRNGKey(1))


def test_collection_to_array_shape():
    coll = _tiny_collection()
    arr = coll.to_array()
    assert arr.ndim == 3 and arr.shape[-1] == 2
    assert arr.shape[0] == coll.datasets[0]._X3d().shape[0]


def test_collection_velocity_array_shape():
    coll = _tiny_collection()
    v = coll.velocity_array()
    t, Xarr, _ = coll.to_arrays()
    assert v.shape == Xarr.shape
