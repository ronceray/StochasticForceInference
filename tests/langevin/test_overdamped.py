# TODO: review this file
import jax.numpy as jnp
from jax import random

from SFI.langevin import OverdampedProcess
from SFI.statefunc import make_basis
from SFI.trajectory.collection import TrajectoryCollection


def harmonic_force(x, **kwargs):
    # simple linear spring: F(x) = -x
    return -x


def test_constant_diffusion_returns_collection():
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(0)
    coll = proc.simulate(dt=0.01, Nsteps=20, key=key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    # one recorded point per step
    assert ds.X.shape == (20, 2)
    # basic sanity: trajectory should move
    assert jnp.any(ds.X != ds.X[0])


def test_state_dependent_diffusion_smoketest():
    def D_func(x, **kwargs):
        d = x.shape[-1]
        scale = jnp.sum(x**2) + 1.0
        return scale * jnp.eye(d)

    # PSFs so we can set coefficients explicitly
    D_psf = make_basis(D_func, dim=2, rank=2, n_features=1).to_psf()
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()

    proc = OverdampedProcess(F_psf, D=D_psf)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D={"coeff": jnp.array([1.0])},
    )
    proc.initialize(jnp.array([0.5, -0.5]))

    key = random.PRNGKey(2)
    coll = proc.simulate(dt=0.01, Nsteps=5, key=key)

    assert isinstance(coll, TrajectoryCollection)
    assert len(coll.datasets) == 1

    ds = coll.datasets[0]
    assert ds.X.shape == (5, 2)


# ----- Tests for Basis objects passed directly (auto-promoted to PSF) -----


def test_basis_force_with_array_theta():
    """Passing a Basis as F and a raw array for theta_F should work."""
    basis = make_basis(harmonic_force, dim=2, rank=1, n_features=1)
    proc = OverdampedProcess(basis, D=0.1)
    proc.set_params(theta_F=jnp.array([1.0]))
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(10)
    coll = proc.simulate(dt=0.01, Nsteps=10, key=key)
    assert isinstance(coll, TrajectoryCollection)
    assert coll.datasets[0].X.shape == (10, 2)


def test_basis_force_with_dict_theta():
    """Passing a Basis as F and a dict for theta_F should work."""
    basis = make_basis(harmonic_force, dim=2, rank=1, n_features=1)
    proc = OverdampedProcess(basis, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(11)
    coll = proc.simulate(dt=0.01, Nsteps=10, key=key)
    assert isinstance(coll, TrajectoryCollection)
    assert coll.datasets[0].X.shape == (10, 2)


def test_basis_diffusion():
    """Passing a Basis as D should work (auto-promoted to PSF)."""
    def D_func(x, **kwargs):
        d = x.shape[-1]
        scale = jnp.sum(x**2) + 1.0
        return scale * jnp.eye(d)

    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    D_basis = make_basis(D_func, dim=2, rank=2, n_features=1)

    proc = OverdampedProcess(F_psf, D=D_basis)
    proc.set_params(
        theta_F={"coeff": jnp.array([1.0])},
        theta_D=jnp.array([1.0]),
    )
    proc.initialize(jnp.array([0.5, -0.5]))

    key = random.PRNGKey(12)
    coll = proc.simulate(dt=0.01, Nsteps=5, key=key)
    assert isinstance(coll, TrajectoryCollection)
    assert coll.datasets[0].X.shape == (5, 2)


# ----- Heun integrator tests -----


def test_heun_constant_diffusion():
    """Heun method with scalar constant diffusion produces valid trajectory."""
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(20)
    coll = proc.simulate(dt=0.01, Nsteps=20, key=key, method="heun")

    assert isinstance(coll, TrajectoryCollection)
    ds = coll.datasets[0]
    assert ds.X.shape == (20, 2)
    assert jnp.all(jnp.isfinite(ds.X))
    assert jnp.any(ds.X != ds.X[0])


def test_heun_matrix_diffusion():
    """Heun method with constant matrix diffusion."""
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    D_mat = jnp.array([[0.2, 0.05], [0.05, 0.1]])
    proc = OverdampedProcess(F_psf, D=D_mat)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(21)
    coll = proc.simulate(dt=0.01, Nsteps=10, key=key, method="heun")

    ds = coll.datasets[0]
    assert ds.X.shape == (10, 2)
    assert jnp.all(jnp.isfinite(ds.X))


def test_heun_state_dependent_diffusion():
    """Heun method with state-dependent diffusion."""
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

    key = random.PRNGKey(22)
    coll = proc.simulate(dt=0.01, Nsteps=5, key=key, method="heun")

    ds = coll.datasets[0]
    assert ds.X.shape == (5, 2)
    assert jnp.all(jnp.isfinite(ds.X))


def test_heun_metadata_records_integrator():
    """Metadata should record the integrator type."""
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    key = random.PRNGKey(30)
    coll_heun = proc.simulate(dt=0.01, Nsteps=5, key=key, method="heun")
    assert coll_heun.datasets[0].meta["integrator"] == "heun"

    proc.initialize(jnp.array([1.0, 0.0]))
    coll_euler = proc.simulate(dt=0.01, Nsteps=5, key=key, method="euler")
    assert coll_euler.datasets[0].meta["integrator"] == "euler"


def test_heun_invalid_method_raises():
    """Unknown method names should raise ValueError."""
    F_psf = make_basis(harmonic_force, dim=2, rank=1, n_features=1).to_psf()
    proc = OverdampedProcess(F_psf, D=0.1)
    proc.set_params(theta_F={"coeff": jnp.array([1.0])})
    proc.initialize(jnp.array([1.0, 0.0]))

    import pytest
    key = random.PRNGKey(31)
    with pytest.raises(ValueError, match="Unknown method"):
        proc.simulate(dt=0.01, Nsteps=5, key=key, method="rk4")


def test_heun_ergodic_ou_variance():
    """Heun should give a more accurate stationary variance than Euler at the same dt.

    For the 1-d OU process  dx = -k x dt + sqrt(2D) dW  the stationary
    distribution is N(0, D/k).  We run a single long trajectory with
    oversampling=1 (so algorithm bias is visible) at a fairly coarse dt
    and compare the empirical variance against the exact value D/k.

    Heun (weak order 2) should reproduce the stationary variance more
    accurately than Euler (weak order 1) at the same step size.
    """
    from SFI.statefunc import make_sf

    k = 2.0
    D_val = 0.5
    exact_var = D_val / k  # = 0.25
    dt = 0.1  # deliberately coarse to expose algorithm bias
    Nsteps = 5000

    F_sf = make_sf(lambda x, **kw: -k * x, dim=1, rank=1)

    errors = {}
    for method in ["euler", "heun"]:
        proc = OverdampedProcess(F_sf, D=D_val)
        proc.initialize(jnp.array([0.0]))
        key = random.PRNGKey(42)
        coll = proc.simulate(
            dt=dt, Nsteps=Nsteps, key=key,
            oversampling=1, method=method,
            compute_observables=False,
        )
        X = coll.datasets[0].X[:, 0]
        # drop first 500 steps as burn-in
        X_eq = X[500:]
        empirical_var = float(jnp.var(X_eq))
        errors[method] = abs(empirical_var - exact_var)

    # Heun should be closer to the exact variance than Euler
    assert errors["heun"] < errors["euler"], (
        f"Heun variance error ({errors['heun']:.6f}) should be smaller "
        f"than Euler ({errors['euler']:.6f})"
    )
