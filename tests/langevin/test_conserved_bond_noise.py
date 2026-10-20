"""Lattice conserved noise as bond sources of a GraphNoise (experimental)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases import field_component, unit_vector_basis
from SFI.bases.spde import Laplacian, conserved_noise_pbc, square_grid_extras
from SFI.langevin import OverdampedProcess
from SFI.statefunc.nodes.interactions.prepare import prepare_structural_extras_for_expr
from SFI.langevin.graph_noise import (
    assemble_blocks,
    conserved_bond_family,
    LatticeConservedNoise,
    dense_D,
    family_amplitudes,
    lattice_bonds,
    noise_graph,
)


def _dense(noise, x, ex):
    N = x.shape[0]
    g = noise_graph(noise.families, ex, N)
    D = 0.0
    for f, th, tb, mk, se in zip(noise.families, noise.theta, g.tabs, g.msks, g.slot_edge):
        G = family_amplitudes(f, x, jnp.asarray(tb), jnp.asarray(mk), ex)
        Dg, De = assemble_blocks(G, jnp.asarray(tb), jnp.asarray(mk), jnp.asarray(se), N, g.E)
        D = D + th * np.asarray(dense_D(Dg, De, g.edges), float)
    return D


def _laplacian_matrix(grid_shape, dx, bc):
    """The SFI Laplacian stencil as a (P, P) matrix (applied to unit vectors)."""
    P = int(np.prod(grid_shape))
    lap = Laplacian(ndim=len(grid_shape), bc=bc)(field_component(0, n_fields=1))
    ex = dict(square_grid_extras(grid_shape=grid_shape, dx=dx))
    prepare_structural_extras_for_expr(lap, ex)
    cols = jax.vmap(lambda e: lap(e[:, None], extras=ex).reshape(P))(jnp.eye(P))
    return np.asarray(cols, float).T


@pytest.mark.parametrize("grid_shape,dx", [((7,), 0.5), ((4, 5), 0.7), ((2, 3), (0.5, 0.8)), ((3, 2, 2), 1.0)])
@pytest.mark.parametrize("bc", ["pbc", "noflux"])
def test_covariance_is_the_lattice_laplacian(grid_shape, dx, bc):
    M = 0.3
    P = int(np.prod(grid_shape))
    noise = LatticeConservedNoise(np.sqrt(M), grid_shape=grid_shape, dx=dx, bc=bc)
    D = _dense(noise, jnp.zeros((P, 1)), noise.extras)
    dV = float(np.prod(np.broadcast_to(np.asarray(dx, float), (len(grid_shape),))))
    np.testing.assert_allclose(D, -M / dV * _laplacian_matrix(grid_shape, dx, bc), atol=1e-5)
    np.testing.assert_allclose(D.sum(axis=1), 0.0, atol=1e-5)  # conservation: rows sum to zero


def test_isotropic_graph_laplacian_formula():
    shape, dx, M = (5, 4), 0.5, 0.2
    fam, ex = conserved_bond_family(shape, dx, bc="noflux")
    bonds = lattice_bonds(shape, bc="noflux")
    P = 20
    Lg = np.zeros((P, P))
    for p, q in bonds:
        Lg[p, p] += 1; Lg[q, q] += 1; Lg[p, q] -= 1; Lg[q, p] -= 1
    noise = LatticeConservedNoise(np.sqrt(M), grid_shape=shape, dx=dx, bc="noflux")
    np.testing.assert_allclose(_dense(noise, jnp.zeros((P, 1)), noise.extras), M / (dx**2 * dx**2) * Lg, atol=1e-5)
    assert len(bonds) == 4 * 4 + 5 * 3  # open grid: no bond crosses the boundary


def test_field_selection_and_mobility():
    """Two fields, conserved noise on field 1 only, mobility m(xbar) = 1 + xbar_1^2 at the bond mean."""
    shape, dx, M = (6,), 1.0, 0.4
    noise = LatticeConservedNoise(np.sqrt(M), grid_shape=shape, dx=dx, bc="pbc", n_fields=2, fields=[1],
                                     mobility=lambda xb: 1.0 + xb[1] ** 2)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((6, 2)), jnp.float32)
    D = _dense(noise, x, noise.extras).reshape(6, 2, 6, 2)
    assert np.abs(D[:, 0]).max() == 0 and np.abs(D[:, :, :, 0]).max() == 0
    ref = np.zeros((6, 6))
    xn = np.asarray(x, float)
    for p, q in lattice_bonds(shape, bc="pbc"):
        w = M * (1.0 + (0.5 * (xn[p, 1] + xn[q, 1])) ** 2)
        ref[p, p] += w; ref[q, q] += w; ref[p, q] -= w; ref[q, p] -= w
    np.testing.assert_allclose(D[:, 1, :, 1], ref, rtol=1e-5, atol=1e-6)


def test_anisotropic_dx_needs_one_family_per_axis():
    with pytest.raises(ValueError):
        conserved_bond_family((4, 4), (0.5, 1.0))
    noise = LatticeConservedNoise(np.sqrt(0.1), grid_shape=(4, 4), dx=(0.5, 1.0))
    assert len(noise.families) == 2 and set(noise.extras) == {f"bond/axis{a}/{k}" for a in (0, 1) for k in ("indptr", "indices")}


@pytest.mark.parametrize("bc", ["pbc", "noflux"])
def test_mass_is_conserved(bc):
    """Bond noise + a conservative drift: sum(phi) stays constant to rounding (pbc and no-flux)."""
    shape = (8, 8)
    noise = LatticeConservedNoise(np.sqrt(0.5), grid_shape=shape, dx=1.0, bc=bc, mobility=lambda xb: 1.0 + xb[0] ** 2)
    ex = square_grid_extras(grid_shape=shape, dx=1.0)  # the noise adds its own bond tables
    lap = Laplacian(ndim=2, bc=bc)(field_component(0, n_fields=1)) * unit_vector_basis(dim=1)
    proc = OverdampedProcess(lap, D=noise, theta_F=jnp.array([0.5]))
    proc.set_extras(extras_global=ex)
    x0 = jnp.asarray(np.random.default_rng(3).standard_normal((64, 1)), jnp.float32)
    proc.initialize(x0)
    X = np.asarray(proc.simulate(dt=0.01, Nsteps=300, key=jax.random.PRNGKey(0), oversampling=2).datasets[0].X)
    mass = X.sum(axis=(1, 2)).astype(float)
    assert np.abs(X[-1] - X[0]).max() > 0.5  # the field moved
    assert np.abs(mass - float(x0.sum())).max() < 1e-3  # float32 rounding over 600 steps of 64 sites


def test_structure_factor_matches_spectral_noise_at_large_scales():
    """Per-mode noise power: bond form 2 M/dV * lattice dispersion, spectral 2 M/dV * |k|^2 (constant M, pbc)."""
    L, dx, M = 16, 1.0, 0.3
    noise = LatticeConservedNoise(np.sqrt(M), grid_shape=(L, L), dx=dx)
    ex = noise.extras
    spec = conserved_noise_pbc(sigma=np.sqrt(M), grid_shape=(L, L), dx=dx)
    x = jnp.zeros((L * L, 1))
    keys = jax.random.split(jax.random.PRNGKey(5), 4000)

    def power(sampler):
        S = jax.vmap(lambda k: sampler(k, x, ex))(keys).reshape(-1, L, L)
        return np.asarray(jnp.mean(jnp.abs(jnp.fft.fft2(S)) ** 2, axis=0), float) / (L * L)

    k = 2 * np.pi * np.fft.fftfreq(L, d=dx)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    lat = (4 / dx**2) * (np.sin(k[:, None] * dx / 2) ** 2 + np.sin(k[None, :] * dx / 2) ** 2)
    S_bond, S_spec = power(noise.sample), power(spec.sample)
    nz = k2 > 0
    # each matches its own dispersion (4000 samples: ~2-3% per mode)
    np.testing.assert_allclose(S_bond[nz] / (2 * M * lat[nz]), 1.0, atol=0.12)
    np.testing.assert_allclose(S_spec[nz] / (2 * M * k2[nz]), 1.0, atol=0.12)
    assert S_bond[0, 0] < 1e-8 and S_spec[0, 0] < 1e-8
    # and the two agree on the long-wavelength modes
    small = nz & (k2 <= 2 * (2 * np.pi / L) ** 2 + 1e-9)
    np.testing.assert_allclose(S_bond[small] / S_spec[small], 1.0, atol=0.12)
