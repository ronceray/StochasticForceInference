# TODO: review this file
# tests/test_langevin.py

import jax.numpy as jnp
import pytest
from jax import random

from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.statefunc import make_basis
from SFI.trajectory.collection import TrajectoryCollection


@pytest.fixture
def jax_key():
    return random.PRNGKey(0)


# ----------------------------- Helpers ---------------------------------


def harmonic_force(x, **kwargs):
    # overdamped: F(x) = -x
    return -x


def harmonic_force_uv(x, v, **kwargs):
    # underdamped: F(x,v) = -x - v
    return -x - v


# ----------------------------- Overdamped -------------------------------


def test_overdamped_constant_diffusion_returns_collection(jax_key):
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    coll = proc.simulate(dt=0.01, Nsteps=20, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    assert ds.X.shape == (20, 2)
    assert bool(jnp.any(ds.X != ds.X[0]))


def test_overdamped_state_dependent_diffusion_smoketest(jax_key):
    def D_func(x, **kwargs):
        d = x.shape[-1]
        scale = jnp.sum(x**2) + 1.0
        return scale * jnp.eye(d)

    D_psf = make_basis(D_func, dim=2, rank=2, n_features=1).to_psf()
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()

    proc = OverdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([1.0])},
    )
    proc.initialize(jnp.array([0.5, -0.5]))

    coll = proc.simulate(dt=0.01, Nsteps=5, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    assert ds.X.shape == (5, 2)


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
    assert ds.X.shape[0] == Nsteps
    assert ds.d == 2

    if ds.t is not None:
        t = jnp.asarray(ds.t)
        assert t.shape[0] == ds.T
    elif ds.dt is not None:
        dt_val = float(jnp.asarray(ds.dt))
        assert dt_val == pytest.approx(dt)
    else:
        pytest.fail("Dataset has neither t nor dt set")


# ----------------------------- Underdamped ------------------------------


def test_underdamped_constant_diffusion_returns_collection_positions_only(jax_key):
    # IMPORTANT: make_basis does not infer needs_v here; set it explicitly.
    F_psf = make_basis(
        harmonic_force_uv, dim=2, rank=1, n_features=1, needs_v=True
    ).to_psf()

    proc = UnderdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})

    # v0 defaults to 0 (same shape as x0)
    proc.initialize(jnp.array([1.0, 0.0]))

    coll = proc.simulate(dt=0.01, Nsteps=20, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    assert ds.X.shape == (20, 2)
    assert bool(jnp.any(ds.X != ds.X[0]))


def test_underdamped_state_dependent_diffusion_x_only_smoketest(jax_key):
    def D_func(x, **kwargs):
        d = x.shape[-1]
        scale = jnp.sum(x**2) + 1.0
        return scale * jnp.eye(d)

    D_psf = make_basis(D_func, dim=2, rank=2, n_features=1).to_psf()
    F_psf = make_basis(
        harmonic_force_uv, dim=2, rank=1, n_features=1, needs_v=True
    ).to_psf()

    proc = UnderdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([1.0])},
    )
    proc.initialize(jnp.array([0.5, -0.5]))

    coll = proc.simulate(dt=0.01, Nsteps=5, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1
    assert coll.datasets[0].X.shape == (5, 2)


def test_underdamped_state_dependent_diffusion_xv_smoketest(jax_key):
    def D_func(x, v, **kwargs):
        d = x.shape[-1]
        scale = jnp.sum(x**2) + jnp.sum(v**2) + 1.0
        return scale * jnp.eye(d)

    # needs_v must be set explicitly for the diffusion too.
    D_psf = make_basis(D_func, dim=2, rank=2, n_features=1, needs_v=True).to_psf()
    F_psf = make_basis(
        harmonic_force_uv, dim=2, rank=1, n_features=1, needs_v=True
    ).to_psf()

    proc = UnderdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([1.0])},
    )
    proc.initialize(jnp.array([0.25, -0.25]))

    coll = proc.simulate(dt=0.01, Nsteps=5, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1
    assert coll.datasets[0].X.shape == (5, 2)


def test_underdamped_simulate_time_metadata(jax_key):
    F = make_basis(
        harmonic_force_uv, dim=2, rank=1, n_features=1, needs_v=True
    ).to_psf()

    proc = UnderdampedProcess(F=F, D=1.0)
    x0 = jnp.array([1.0, 0.0])
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(x0)  # v0 defaults to 0

    dt = 0.05
    Nsteps = 8
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=jax_key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    assert ds.X.shape[0] == Nsteps
    assert ds.d == 2

    if ds.t is not None:
        t = jnp.asarray(ds.t)
        assert t.shape[0] == ds.T
    elif ds.dt is not None:
        dt_val = float(jnp.asarray(ds.dt))
        assert dt_val == pytest.approx(dt)
    else:
        pytest.fail("Dataset has neither t nor dt set")


def test_underdamped_v0_is_used_in_deterministic_limit(jax_key):
    # If F(x,v)=0 and D=0, then x_{n+1} = x_n + dt*v0 exactly.
    def zero_force(x, v, **kwargs):
        return jnp.zeros_like(x)

    F_psf = make_basis(zero_force, dim=2, rank=1, n_features=1, needs_v=True).to_psf()

    proc = UnderdampedProcess(F_psf, D=0.0)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})

    x0 = jnp.array([1.0, 2.0])
    v0 = jnp.array([0.3, -0.1])
    proc.initialize(x0, v0=v0)

    dt = 0.2
    Nsteps = 4
    coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=jax_key)

    ds = coll.datasets[0]
    dt_arr = jnp.asarray(dt, dtype=x0.dtype)
    increments = jnp.broadcast_to(dt_arr * v0, (Nsteps,) + v0.shape)
    expected = x0 + jnp.cumsum(increments, axis=0)

    assert jnp.allclose(ds.X, expected, atol=1e-6, rtol=1e-6)


def test_underdamped_rejects_force_without_velocity_argument():
    # needs_v=False => must be rejected by UnderdampedProcess
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = UnderdampedProcess(F_psf, D=1.0)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})

    with pytest.raises(ValueError):
        proc.initialize(jnp.array([1.0, 0.0]))
