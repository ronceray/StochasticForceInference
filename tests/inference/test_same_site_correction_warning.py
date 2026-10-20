"""The same-particle Stratonovich correction warns on lattice-field (SPDE stencil) bases.

The correction contracts the instantaneous diffusion with each site's own
Jacobian block, which misses the cross-site terms when the noise couples
neighbouring sites (conserved or spatially correlated noise).  The engine warns
when ``cross_particle="same-particle"`` meets a stencil basis; the default
all-pairs correction, Ito moments and non-lattice bases are silent.
"""

import warnings

import jax.numpy as jnp
import pytest
from jax import random

from SFI import OverdampedLangevinInference
from SFI.bases import field_component, monomials_up_to, unit_vector_basis
from SFI.bases.spde import Laplacian, square_grid_extras
from SFI.langevin import OverdampedProcess
from SFI.langevin.noise import WhiteNoise

_MSG = "lattice-field"


def _lattice_fit_inputs(Nx=16, dx=0.5):
    phi = field_component(0, n_fields=1)
    basis = (Laplacian(ndim=1, bc="pbc")(phi) * unit_vector_basis(dim=1)) & (phi * unit_vector_basis(dim=1))
    proc = OverdampedProcess(basis, D=WhiteNoise(sigma=0.5, n_fields=1))
    proc.set_params(theta_F=jnp.array([1.0, -0.5]))
    proc.set_extras(extras_global=square_grid_extras(grid_shape=(Nx,), dx=dx))
    proc.initialize(0.1 * jnp.sin(2 * jnp.pi * jnp.arange(Nx) / Nx)[:, None])
    coll = proc.simulate(dt=0.01, Nsteps=200, key=random.PRNGKey(0))
    return coll, basis


def _silent(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    assert not [w for w in rec if _MSG in str(w.message)]


def test_same_particle_strato_on_lattice_basis_warns():
    coll, basis = _lattice_fit_inputs()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    with pytest.warns(UserWarning, match=_MSG):
        inf.infer_force_linear(basis, preset="robust", cross_particle="same-particle")
    assert inf.metadata["force_noise_correction"] == "same-particle"


def test_all_pairs_strato_on_lattice_basis_is_silent():
    coll, basis = _lattice_fit_inputs()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    _silent(lambda: inf.infer_force_linear(basis, preset="robust"))
    assert inf.metadata["force_noise_correction"] == "all-pairs"


def test_ito_on_lattice_basis_is_silent():
    coll, basis = _lattice_fit_inputs()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    _silent(lambda: inf.infer_force_linear(basis, M_mode="Ito", cross_particle="same-particle"))
    assert "force_noise_correction" not in inf.metadata


def test_strato_on_pointwise_basis_is_silent():
    coll, _ = _lattice_fit_inputs()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    _silent(lambda: inf.infer_force_linear(monomials_up_to(1, dim=1, rank="vector"), preset="robust"))


def test_program_only_has_no_side_effects():
    """Row-scope planning (``program_only``) records nothing: the correction's
    metadata and warning belong to its evaluation (the ULI path wrote
    ``force_noise_correction`` while planning)."""
    import jax.numpy as jnp

    from SFI import UnderdampedLangevinInference

    coll, basis = _lattice_fit_inputs()
    ud_basis = monomials_up_to(1, dim=1, include_v=True, rank="vector")
    for cls, method, B in ((OverdampedLangevinInference, "_strato_grad_moments", basis),
                           (UnderdampedLangevinInference, "_uli_grad_moments", ud_basis)):
        for cross in ("same-particle", "all-pairs"):
            inf = cls(coll)
            inf.force_basis = B
            setattr(inf, "__force_cross_particle__", cross)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                prog = getattr(inf, method)(A_inv=jnp.eye(1), program_only=True)
            assert prog.require()  # a program, not a value
            assert inf.metadata == {}, (cls.__name__, cross, inf.metadata)
            assert not [w for w in rec if _MSG in str(w.message)]
