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


# ----------------- Underdamped observables (information/entropy) -----------------
#
# S accumulates the Stratonovich log path-probability ratio with the true
# force, Sum (dv - F_even ddt) . D^-1 . F_odd, using trapezoid (midpoint)
# evaluation of F_even/F_odd over each substep; I is the Ito information
# functional 1/4 Sum dv . D^-1 . F(x_t, v_t).


def _simulate_ud_trap(eps, *, k=1.0, gamma=1.0, D0=0.5, dt=0.01, Nsteps=20_000, seed=0):
    def trap_curl_friction(x, v, **kwargs):
        rot = jnp.stack([-x[..., 1], x[..., 0]], axis=-1)
        return -k * x + eps * rot - gamma * v

    F_psf = make_basis(
        trap_curl_friction, dim=2, rank=1, n_features=1, needs_v=True
    ).to_psf()
    proc = UnderdampedProcess(F_psf, D=D0)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([0.5, 0.0]))
    return proc.simulate(
        dt=dt, Nsteps=Nsteps, key=random.PRNGKey(seed),
        oversampling=8, prerun=500, compute_observables=True,
    )


def test_underdamped_observables_equilibrium_null():
    # Equilibrium (no curl): E[S] = 0 by the Ito/Stratonovich cancellation
    # <F_odd D^-1 F_odd> + <div_v F_odd> = gamma*d - gamma*d.  A naive Ito
    # (pre-point) evaluation would instead give ~ gamma*d*tau = 400 here.
    coll = _simulate_ud_trap(0.0, seed=1)
    obs = coll.datasets[0].meta["observables"]
    assert jnp.isfinite(obs["entropy"]) and jnp.isfinite(obs["information"])
    tau = 20_000 * 0.01
    fluct = (2.0 * 1.0 * 2 * tau) ** 0.5  # sqrt(2 * gamma * d * tau)
    assert abs(obs["entropy"]) < 5.0 * fluct, obs
    assert obs["information"] > 0.0


def _ud_lyapunov_sigma(k, gamma, D0, eps):
    """Exact EP rate of the linear UD trap+curl via the phase-space OU formula.

    With odd (velocity) variables the irreversible drift is the PARITY-AWARE
    split M = 0.5*(A + E A E) + D_z S^-1 (E = diag(I, -I)); the all-even
    formula A - D S^-1 over-counts by including the reversible Hamiltonian
    part.  Cross-checked against Sekimoto heat: sigma = (gamma<|v|^2> - d*D)/T.
    """
    import numpy as np
    import scipy.linalg

    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    A_drift = np.block([
        [np.zeros((2, 2)), np.eye(2)],
        [-k * np.eye(2) + eps * J, -gamma * np.eye(2)],
    ])
    D_z = np.zeros((4, 4))
    D_z[2:, 2:] = D0 * np.eye(2)
    S_cov = scipy.linalg.solve_lyapunov(A_drift, -2.0 * D_z)
    E = np.diag([1.0, 1.0, -1.0, -1.0])
    M = 0.5 * (A_drift + E @ A_drift @ E) + D_z @ np.linalg.inv(S_cov)
    sigma = float(np.trace(M.T @ np.linalg.pinv(D_z) @ M @ S_cov))
    # Sekimoto cross-check: heat rate / T with T = D/gamma
    v2 = float(np.trace(S_cov[2:, 2:]))
    sigma_heat = (gamma * v2 - 2 * D0) / (D0 / gamma)
    assert abs(sigma - sigma_heat) < 1e-8 * max(1.0, abs(sigma))
    return sigma, v2


def test_underdamped_observables_driven_positive():
    # Driven trap (curl eps): entropy production is positive and matches the
    # exact parity-aware Lyapunov rate.  The fluctuation scale of the log
    # path-ratio is set by the ODD quadratic <F- D^-1 F-> = gamma^2<|v|^2>/D
    # (the two large cancelling pieces fluctuate), not by sigma itself.
    k, gamma, D0, eps = 1.0, 1.0, 0.5, 0.5
    sigma_exact, v2 = _ud_lyapunov_sigma(k, gamma, D0, eps)
    assert sigma_exact > 0

    coll = _simulate_ud_trap(eps, k=k, gamma=gamma, D0=D0, seed=2)
    obs = coll.datasets[0].meta["observables"]
    tau = 20_000 * 0.01
    rate = obs["entropy"] / tau
    fluct = (2.0 * gamma**2 * v2 / D0 / tau) ** 0.5
    assert rate > 0.0
    assert abs(rate - sigma_exact) < 4.0 * fluct + 0.1 * sigma_exact, (
        rate,
        sigma_exact,
        fluct,
    )
