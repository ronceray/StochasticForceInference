"""Lattice conserved noise: force corrections on stencil bases and the bond intensity (experimental)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI import OverdampedLangevinInference, TrajectoryCollection
from SFI.bases.spde import square_grid_extras
from SFI.inference.graph_diffusion import estimate_graph_noise
from SFI.langevin import OverdampedProcess
from SFI.langevin.noise import LatticeConservedNoise
from SFI.statefunc.layout import GridLayout, ScalarSector

L, DT, M = 12, 0.01, 0.09
TRUTH = np.array([-0.05, 0.5, -0.1])  # phi, lap phi, lap^2 phi


def _basis():
    layout = GridLayout(U=ScalarSector([0]), dim=1, ndim=2, bc="pbc")
    U = layout.U
    return layout.embed(rank=1, U=U & layout.lap(U) & layout.biharmonic(U))


@pytest.fixture(scope="module")
def model_b():
    """Linear Model B with lattice conserved noise, observed with localisation noise."""
    B = _basis()
    noise = LatticeConservedNoise(np.sqrt(M), grid_shape=(L, L), dx=1.0)
    ex = square_grid_extras(grid_shape=(L, L), dx=1.0)  # the noise carries its bond tables
    proc = OverdampedProcess(B, D=noise, theta_F=jnp.asarray(TRUTH, jnp.float32))
    proc.set_extras(extras_global=ex)
    proc.initialize(jnp.asarray(0.3 * np.random.default_rng(0).standard_normal((L * L, 1)), jnp.float32))
    X = np.asarray(proc.simulate(dt=DT, Nsteps=6000, key=jax.random.PRNGKey(1), oversampling=10,
                                 prerun=300).datasets[0].X)
    X = X + 0.03 * np.random.default_rng(2).standard_normal(X.shape).astype(np.float32)
    coll = TrajectoryCollection.from_arrays(X=X, dt=DT, extras_global=ex)
    return coll, B, noise


def _fit(inf, B, **kw):
    inf.compute_diffusion_constant(method="noisy")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inf.infer_force_linear(B, preset="robust", **kw)
    return np.asarray(inf.force_coefficients), any("lattice-field" in str(x.message) for x in w)


@pytest.mark.slow
def test_all_pairs_correction_on_conserved_noise_with_localisation_noise(model_b):
    coll, B, _ = model_b
    c_all, warned_all = _fit(OverdampedLangevinInference(coll), B)  # default: all-pairs correction
    c_site, warned_site = _fit(OverdampedLangevinInference(coll), B, cross_particle="same-particle")
    ito = OverdampedLangevinInference(coll)
    ito.compute_diffusion_constant(method="MSD")
    ito.infer_force_linear(B, preset="clean")
    c_ito = np.asarray(ito.force_coefficients)

    assert np.all(np.abs(c_all - TRUTH) < [0.06, 0.08, 0.02])  # ~1.5 seed std (3-coefficient fit, 12x12, T=6000)
    assert warned_site and not warned_all
    assert c_site[0] < -1.0  # per-site correction: O(1) bias (missing cross-site D_ij d_j b_i)
    assert c_ito[0] < -0.5  # Ito: localisation noise enters the drift of the phi term


def test_bond_intensity_is_the_mobility(model_b):
    coll, _, noise = model_b
    noisy = estimate_graph_noise(coll, noise, stencil="noisy").theta[0]
    msd = estimate_graph_noise(coll, noise, stencil="MSD").theta[0]
    # The noisy stencil's finite-dt bias is ~ -(3/2) lambda dt per mode (lambda up to ~10 on this lattice):
    # about -9% at dt = 0.01, independent of the localisation noise; MSD absorbs Lambda/dt instead.
    assert 0.85 * M < noisy < M
    assert msd > 1.1 * M
