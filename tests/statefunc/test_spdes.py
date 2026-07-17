# TODO: review this file
"""Tests for SPDE composable operators in statefunc context.

Tests that the Laplacian operator (composable API) matches a manual
cross-stencil reference implementation, for both PBC and no-flux BC.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases.linear import field_component
from SFI.bases.spde import Laplacian, square_grid_extras
from SFI.statefunc.nodes.interactions.prepare import (
    prepare_structural_extras_for_expr,
)


def _manual_laplacian_cross(x, grid_shape, dx, bc):
    x = np.asarray(x)  # (P, dim)
    grid_shape = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape)
    P, dim = x.shape

    if np.isscalar(dx):
        dx = (float(dx),) * ndim
    else:
        dx = tuple(float(v) for v in dx)
    inv_dx2 = np.array([1.0 / (h * h) for h in dx], dtype=np.float64)

    coords = np.stack(np.unravel_index(np.arange(P), grid_shape, order="C"), axis=1)

    out = np.zeros((P, dim), dtype=np.float64)
    for ax, n in enumerate(grid_shape):
        plus = coords.copy()
        plus[:, ax] += 1
        minus = coords.copy()
        minus[:, ax] -= 1

        if bc == "pbc":
            plus[:, ax] %= n
            minus[:, ax] %= n
        elif bc in ("noflux", "drop"):
            inb_plus = (0 <= plus[:, ax]) & (plus[:, ax] < n)
            inb_minus = (0 <= minus[:, ax]) & (minus[:, ax] < n)
            plus[~inb_plus] = coords[~inb_plus]
            minus[~inb_minus] = coords[~inb_minus]
        else:
            raise ValueError(bc)

        p_plus = np.ravel_multi_index(plus.T, grid_shape, order="C")
        p_minus = np.ravel_multi_index(minus.T, grid_shape, order="C")

        out += (x[p_plus] + x[p_minus] - 2.0 * x) * inv_dx2[ax]

    return out


def _make_laplacian_with_extras(dim, grid_shape, dx, bc):
    """Build the composable Laplacian basis and its prepared extras."""
    ndim = len(grid_shape)
    Lap = Laplacian(ndim=ndim, bc=bc)

    # Concatenate Lap(phi_i) for each field → rank-0, n_features = dim
    phis = [field_component(i, n_fields=dim) for i in range(dim)]
    basis = Lap(phis[0])
    for i in range(1, dim):
        basis = basis & Lap(phis[i])

    extras = square_grid_extras(grid_shape=grid_shape, dx=dx)
    prepare_structural_extras_for_expr(basis, extras)
    return basis, extras


@pytest.mark.parametrize("bc", ["noflux", "pbc"])
def test_laplacian_matches_manual(bc):
    grid_shape = (4, 3)
    P = int(np.prod(grid_shape))
    dim = 2

    p = np.arange(P, dtype=np.float64)
    x0 = np.sin(0.3 * p)
    x1 = np.cos(0.17 * p + 0.1)
    x = np.stack([x0, x1], axis=1).astype(np.float32)

    L, extras = _make_laplacian_with_extras(dim, grid_shape, dx=(1.0, 2.0), bc=bc)
    y = L(jnp.asarray(x), extras=extras)
    assert y.shape == (P, dim)

    y_ref = _manual_laplacian_cross(x, grid_shape, dx=(1.0, 2.0), bc=bc)
    np.testing.assert_allclose(np.asarray(y), y_ref, rtol=1e-4, atol=1e-4)


def test_laplacian_freezes_masked_focal_sites():
    grid_shape = (5,)
    P = int(np.prod(grid_shape))
    x = jnp.arange(P, dtype=jnp.float32)[:, None]
    mask = np.ones((P,), dtype=bool)
    mask[2] = False

    Lap = Laplacian(ndim=1, bc="noflux")
    phi = field_component(0, n_fields=1)
    basis = Lap(phi)
    extras = square_grid_extras(grid_shape=grid_shape, dx=1.0)
    prepare_structural_extras_for_expr(basis, extras)
    y = basis(x, mask=jnp.asarray(mask), extras=extras)
    assert float(np.asarray(y)[2, 0]) == 0.0
