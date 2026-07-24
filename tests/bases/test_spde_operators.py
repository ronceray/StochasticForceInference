# TODO: review this file
"""Tests for SPDE composable differential operators (stencil-based).

Each operator is tested on analytic fields whose exact derivatives are known,
on periodic and no-flux Cartesian grids.  We check:

1. Correct output shape.
2. Numerical accuracy  (L-inf relative error < expected FD accuracy).
3. Conservation (zero spatial sum on PBC grids).
4. Boundary-condition correctness (PBC and noflux).
5. 1D, 2D, 3D, and non-square grids.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases.linear import field_component
from SFI.bases.spde import (
    AdvectionBy,
    Biharmonic,
    Curl,
    Divergence,
    Gradient,
    Laplacian,
    LaplacianOfGradientSquared,
    SkewGrad,
    SymGrad,
    minkowski_sum_offsets,
    square_grid_extras,
    vector_field,
)
from SFI.statefunc.nodes.interactions.prepare import (
    prepare_structural_extras_for_expr,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_extras(grid_shape, dx, basis):
    """Build extras dict with box descriptors + prepared stencil tables."""
    extras = square_grid_extras(grid_shape=grid_shape, dx=dx)
    prepare_structural_extras_for_expr(basis, extras)
    return extras


def _grid_coords_1d(Nx, dx):
    return jnp.arange(Nx, dtype=jnp.float32) * dx


def _grid_coords_2d(Nx, Ny, dx):
    if isinstance(dx, (tuple, list)):
        dx_x, dx_y = dx
    else:
        dx_x = dx_y = dx
    xs = jnp.arange(Nx) * dx_x
    ys = jnp.arange(Ny) * dx_y
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)


def _grid_coords_3d(Nx, Ny, Nz, dx):
    if isinstance(dx, (tuple, list)):
        dx_x, dx_y, dx_z = dx
    else:
        dx_x = dx_y = dx_z = dx
    xs = jnp.arange(Nx) * dx_x
    ys = jnp.arange(Ny) * dx_y
    zs = jnp.arange(Nz) * dx_z
    xx, yy, zz = jnp.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def _manual_laplacian_cross(x, grid_shape, dx, bc):
    """Reference Laplacian implementation for validation."""
    x = np.asarray(x)  # (P, dim)
    grid_shape = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape)
    P, dim = x.shape

    if np.isscalar(dx):
        dx = (float(dx),) * ndim
    elif not isinstance(dx, tuple):
        dx = tuple(float(v) for v in dx)
    inv_dx2 = np.array([1.0 / (h * h) for h in dx], dtype=np.float64)

    coords = np.stack(
        np.unravel_index(np.arange(P), grid_shape, order="C"), axis=1
    )

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


# ============================================================================
# Laplacian (composable)
# ============================================================================


class TestLaplacian:
    """Tests for the composable Laplacian operator."""

    def test_accuracy_2d_pbc(self):
        """Laplacian of sin field on 2D PBC grid."""
        Nx, Ny, dx = 32, 32, 0.2
        kx, ky = 2 * jnp.pi / (Nx * dx), 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        X = phi_vals[:, None]

        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        exact = -(kx**2 + ky**2) * phi_vals
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.02, f"Laplacian error: {err}"

    def test_conservation_pbc(self):
        """Laplacian has zero spatial sum on PBC grids."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert jnp.abs(jnp.sum(result)) < 1e-5, "Laplacian sum != 0 on PBC"

    def test_nonlinear_composition(self):
        """Laplacian of phi^3 produces correct shape and conservation."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi**3)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 1)
        # Conservation: nabla^2(phi^3) sums to zero on PBC
        assert jnp.abs(jnp.sum(result)) < 1e-4

    @pytest.mark.parametrize("bc", ["noflux", "pbc"])
    def test_matches_manual(self, bc):
        """Composable Laplacian matches manual cross-stencil reference."""
        grid_shape = (4, 3)
        P = int(np.prod(grid_shape))
        dim = 2

        p = np.arange(P, dtype=np.float64)
        x0 = np.sin(0.3 * p)
        x1 = np.cos(0.17 * p + 0.1)
        x = np.stack([x0, x1], axis=1).astype(np.float32)

        Lap = Laplacian(ndim=2, bc=bc)
        phi0 = field_component(0, n_fields=2)
        phi1 = field_component(1, n_fields=2)
        # Concatenate scalar-rank Laplacians → (P, 2) with features = fields
        basis = Lap(phi0) & Lap(phi1)
        extras = _make_extras(grid_shape, dx=(1.0, 2.0), basis=basis)
        y = basis(jnp.asarray(x), extras=extras)
        assert y.shape == (P, dim)

        y_ref = _manual_laplacian_cross(x, grid_shape, dx=(1.0, 2.0), bc=bc)
        np.testing.assert_allclose(
            np.asarray(y), y_ref, rtol=1e-4, atol=1e-4
        )

    def test_noflux_accuracy(self):
        """Noflux BC: constant field → zero, linear field → correct stencil."""
        Nx, Ny = 8, 8
        Lap = Laplacian(ndim=2, bc="noflux")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx, Ny), 1.0, basis)

        # Constant field -> zero
        result = basis(jnp.ones((64, 1)), extras=extras)
        assert jnp.allclose(result, 0.0, atol=1e-6)

        # Linear field: compare with manual reference
        coords = _grid_coords_2d(Nx, Ny, 1.0)
        X = coords[:, 0:1]
        result = basis(X, extras=extras)
        ref = _manual_laplacian_cross(
            np.asarray(X), (Nx, Ny), dx=1.0, bc="noflux"
        )
        np.testing.assert_allclose(
            np.asarray(result), ref, rtol=1e-5, atol=1e-5
        )

    def test_masked_focal_sites(self):
        """Masked sites produce zero output."""
        grid_shape = (5,)
        P = int(np.prod(grid_shape))
        x = jnp.arange(P, dtype=jnp.float32)[:, None]
        mask = np.ones((P,), dtype=bool)
        mask[2] = False

        Lap = Laplacian(ndim=1, bc="noflux")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras(grid_shape, 1.0, basis)
        y = basis(x, mask=jnp.asarray(mask), extras=extras)
        assert float(np.asarray(y)[2, 0]) == 0.0

    def test_1d(self):
        """Laplacian works correctly on 1D periodic grid."""
        Nx, dx = 64, 0.1
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_1d(Nx, dx)
        X = jnp.sin(kx * coords)[:, None]

        Lap = Laplacian(ndim=1, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx,), dx, basis)
        result = basis(X, extras=extras)

        exact = -(kx**2) * jnp.sin(kx * coords)
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.01, f"1D Laplacian error: {err}"
        assert jnp.abs(jnp.sum(result)) < 1e-4

    def test_3d(self):
        """Laplacian on 3D periodic grid."""
        Nx, Ny, Nz, dx = 8, 8, 8, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_3d(Nx, Ny, Nz, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        Lap = Laplacian(ndim=3, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx, Ny, Nz), dx, basis)
        result = basis(X, extras=extras)

        exact = -(kx**2) * jnp.sin(kx * coords[:, 0])
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.1, f"3D Laplacian error: {err}"

    def test_nonsquare_grid(self):
        """Laplacian on non-square 2D grid with non-uniform dx."""
        Nx, Ny = 16, 32
        dx = (0.5, 0.25)
        kx = 2 * jnp.pi / (Nx * dx[0])
        ky = 2 * jnp.pi / (Ny * dx[1])
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        X = phi_vals[:, None]

        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Lap(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        exact = -(kx**2 + ky**2) * phi_vals
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.02, f"Non-square Laplacian error: {err}"


# ============================================================================
# Gradient (composable)
# ============================================================================


class TestGradient:
    """Tests for the composable Gradient operator."""

    def test_shape_and_accuracy_2d(self):
        Nx, Ny, dx = 32, 32, 0.2
        kx, ky = 2 * jnp.pi / (Nx * dx), 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        X = phi_vals[:, None]

        Grad = Gradient(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Grad(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 2)

        exact_dx = kx * jnp.cos(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        exact_dy = ky * jnp.sin(kx * coords[:, 0]) * jnp.cos(ky * coords[:, 1])

        err_x = jnp.max(jnp.abs(result[:, 0] - exact_dx)) / jnp.max(
            jnp.abs(exact_dx)
        )
        err_y = jnp.max(jnp.abs(result[:, 1] - exact_dy)) / jnp.max(
            jnp.abs(exact_dy)
        )
        assert err_x < 0.02, f"d phi/dx error: {err_x}"
        assert err_y < 0.02, f"d phi/dy error: {err_y}"

    def test_noflux_constant(self):
        Grad = Gradient(ndim=2, bc="noflux")
        phi = field_component(0, n_fields=1)
        basis = Grad(phi)
        extras = _make_extras((8, 8), 1.0, basis)
        result = basis(jnp.ones((64, 1)), extras=extras)
        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_noflux_linear_field(self):
        """Noflux gradient of linear field: interior = slope."""
        Nx, Ny = 16, 16
        dx = 0.5
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = coords[:, 0:1]

        Grad = Gradient(ndim=2, bc="noflux")
        phi = field_component(0, n_fields=1)
        basis = Grad(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 2)

        gi = coords[:, 0] / dx
        interior = (gi > 0) & (gi < Nx - 1)
        np.testing.assert_allclose(
            np.asarray(result[interior, 0]), 1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(result[interior, 1]), 0.0, atol=1e-5
        )

    def test_1d(self):
        """Gradient in 1D."""
        Nx, dx = 64, 0.1
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_1d(Nx, dx)
        X = jnp.sin(kx * coords)[:, None]

        Grad = Gradient(ndim=1, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Grad(phi)
        extras = _make_extras((Nx,), dx, basis)
        result = basis(X, extras=extras)
        assert result.shape == (Nx, 1)

        exact = kx * jnp.cos(kx * coords)
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.005, f"1D Gradient error: {err}"

    def test_nonlinear_composition(self):
        """Gradient of phi^2."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0])
        X = phi_vals[:, None]

        Grad = Gradient(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Grad(phi**2)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        exact_dx = kx * jnp.sin(2 * kx * coords[:, 0])
        err = jnp.max(jnp.abs(result[:, 0] - exact_dx)) / jnp.max(
            jnp.abs(exact_dx)
        )
        assert err < 0.05, f"Gradient(phi^2) error: {err}"


# ============================================================================
# Biharmonic (composable)
# ============================================================================


class TestBiharmonic:
    """Tests for the composable Biharmonic operator."""

    def test_accuracy_2d(self):
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.cos(kx * coords[:, 0]) * jnp.cos(ky * coords[:, 1])
        X = phi_vals[:, None]

        Bih = Biharmonic(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Bih(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 1)

        exact = (kx**4 + 2 * kx**2 * ky**2 + ky**4) * phi_vals
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.05, f"Biharmonic error: {err}"

    def test_conservation_pbc(self):
        """Biharmonic has zero spatial sum on PBC grids."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        Bih = Biharmonic(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Bih(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert jnp.abs(jnp.sum(result)) < 1e-2

    def test_ndim_check(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            Biharmonic(ndim=1, bc="pbc")

    def test_3d(self):
        """Biharmonic on 3D PBC grid."""
        Nx, Ny, Nz, dx = 8, 8, 8, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_3d(Nx, Ny, Nz, dx)
        phi_vals = jnp.cos(kx * coords[:, 0])
        X = phi_vals[:, None]

        Bih = Biharmonic(ndim=3, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Bih(phi)
        extras = _make_extras((Nx, Ny, Nz), dx, basis)
        result = basis(X, extras=extras)

        exact = kx**4 * phi_vals
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.1, f"3D Biharmonic error: {err}"

    def test_nonlinear_composition(self):
        """Biharmonic of phi^2 produces correct shape and conservation."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        Bih = Biharmonic(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = Bih(phi**2)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 1)
        assert jnp.abs(jnp.sum(result)) < 1e-3


# ============================================================================
# LaplacianOfGradientSquared (composable)
# ============================================================================


class TestLaplacianOfGradientSquared:
    """Tests for the composable Lap(|Grad f|^2) operator (AMB+ term)."""

    def test_shape_2d(self):
        Nx, Ny, dx = 16, 16, 0.5
        X = jnp.ones((Nx * Ny, 1))

        LGS = LaplacianOfGradientSquared(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = LGS(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 1)
        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_conservation_pbc(self):
        """Lap(|Grad phi|^2) sums to zero on PBC."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        LGS = LaplacianOfGradientSquared(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = LGS(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert jnp.abs(jnp.sum(result)) < 1e-4

    def test_ndim_check(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            LaplacianOfGradientSquared(ndim=1, bc="pbc")

    def test_nontrivial_result(self):
        """Non-constant field produces non-zero LGS output."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        LGS = LaplacianOfGradientSquared(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        basis = LGS(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert jnp.max(jnp.abs(result)) > 0.1


# ============================================================================
# Algebra and composition
# ============================================================================


class TestOperatorAlgebra:
    """Test that operators compose correctly with basis algebra."""

    def test_laplacian_times_unit_vector(self):
        """Lap(phi) * eU produces rank-1 output with correct shape."""
        from SFI.bases.constants import unit_vector_basis

        Nx, Ny, dx = 8, 8, 1.0
        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=2)
        eU = unit_vector_basis(dim=2, axes=[0])
        basis = Lap(phi) * eU
        extras = _make_extras((Nx, Ny), dx, basis)

        X = jnp.ones((Nx * Ny, 2))
        result = basis(X, extras=extras)
        # rank-1 basis: (P, dim, n_features) = (64, 2, 1)
        assert result.shape == (Nx * Ny, 2, 1)

    def test_concatenation(self):
        """Lap(U)*eU & Lap(V)*eV produces combined basis."""
        from SFI.bases.constants import unit_vector_basis

        Nx, Ny, dx = 8, 8, 1.0
        Lap = Laplacian(ndim=2, bc="pbc")
        phi0 = field_component(0, n_fields=2)
        phi1 = field_component(1, n_fields=2)
        eU = unit_vector_basis(dim=2, axes=[0])
        eV = unit_vector_basis(dim=2, axes=[1])
        basis = Lap(phi0) * eU & Lap(phi1) * eV
        extras = _make_extras((Nx, Ny), dx, basis)

        X = jnp.ones((Nx * Ny, 2))
        result = basis(X, extras=extras)
        # rank-1 basis: (P, dim, n_features) = (64, 2, 2)
        assert result.shape == (Nx * Ny, 2, 2)

    def test_stencil_composition_works(self):
        """Lap(Lap(phi)) now works via stencil fusion."""
        Lap = Laplacian(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        lap_phi = Lap(phi)
        # Should NOT raise — composition is supported.
        basis = Lap(lap_phi)
        assert basis.n_features == 1


# ============================================================================
# Error handling
# ============================================================================


class TestErrors:
    """Test error handling for invalid inputs."""

    def test_non_stateexpr_input(self):
        Lap = Laplacian(ndim=2, bc="pbc")
        with pytest.raises(TypeError, match="StateExpr"):
            Lap(42)

    def test_repr(self):
        lap = Laplacian(ndim=2, bc="pbc")
        assert "Laplacian" in repr(lap)
        assert "ndim=2" in repr(lap)


# ============================================================================
# Stencil composition (Minkowski-sum fusion)
# ============================================================================


class TestMinkowskiSum:
    """Test the minkowski_sum_offsets utility directly."""

    def test_cross_cross_2d(self):
        """Cross ⊕ cross = biharmonic-shaped 13-point stencil."""
        from SFI.statefunc.nodes.interactions.stencils import (
            square_biharmonic_offsets,
            square_cross_offsets,
        )

        cross = square_cross_offsets(2, include_center=True)
        fused, idx_maps = minkowski_sum_offsets(cross, cross)

        bih = square_biharmonic_offsets(2, include_center=True)

        fused_set = {tuple(int(v) for v in r) for r in np.asarray(fused)}
        bih_set = {tuple(int(v) for v in r) for r in np.asarray(bih)}
        assert fused_set == bih_set

    def test_index_maps_shape(self):
        from SFI.statefunc.nodes.interactions.stencils import (
            square_cross_offsets,
        )

        cross = square_cross_offsets(2, include_center=True)
        fused, idx_maps = minkowski_sum_offsets(cross, cross)
        assert idx_maps.shape == (5, 5)

    def test_center_maps_to_center(self):
        from SFI.statefunc.nodes.interactions.stencils import (
            square_cross_offsets,
        )

        cross = square_cross_offsets(2, include_center=True)
        fused, idx_maps = minkowski_sum_offsets(cross, cross)
        # Center of outer (slot 0) + center of inner (slot 0) = (0,0)
        fused_np = np.asarray(fused)
        center_idx = idx_maps[0, 0]
        assert tuple(int(v) for v in fused_np[center_idx]) == (0, 0)


class TestComposition:
    """Test that stencil operators compose correctly via fusion."""

    def test_lap_lap_matches_biharmonic(self):
        """Lap(Lap(phi)) produces same result as dedicated Biharmonic."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.cos(kx * coords[:, 0]) * jnp.cos(ky * coords[:, 1])
        X = phi_vals[:, None]

        phi = field_component(0, n_fields=1)

        # Via composition
        Lap = Laplacian(ndim=2, bc="pbc")
        basis_comp = Lap(Lap(phi))
        extras_comp = _make_extras((Nx, Ny), dx, basis_comp)
        result_comp = basis_comp(X, extras=extras_comp)

        # Via dedicated Biharmonic
        Bih = Biharmonic(ndim=2, bc="pbc")
        basis_bih = Bih(phi)
        extras_bih = _make_extras((Nx, Ny), dx, basis_bih)
        result_bih = basis_bih(X, extras=extras_bih)

        # Float32 roundoff from different operation order (~1e-3).
        np.testing.assert_allclose(
            np.asarray(result_comp), np.asarray(result_bih),
            rtol=5e-4, atol=2e-3,
        )

    def test_div_grad_matches_laplacian(self):
        """Div(Grad(phi)) ≈ Laplacian(phi) (same continuous limit)."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        X = phi_vals[:, None]

        phi = field_component(0, n_fields=1)
        Grad = Gradient(ndim=2, bc="pbc")
        Div = Divergence(ndim=2, bc="pbc")

        # Composed — uses radius-2 FD stencil (different from direct Lap).
        basis_comp = Div(Grad(phi))
        extras_comp = _make_extras((Nx, Ny), dx, basis_comp)
        result_comp = basis_comp(X, extras=extras_comp)

        # Compare to analytic answer -(kx²+ky²)*sin(kx*x)*sin(ky*y).
        exact = -(kx**2 + ky**2) * phi_vals
        err = jnp.max(jnp.abs(result_comp[:, 0] - exact)) / jnp.max(
            jnp.abs(exact)
        )
        # Tolerance higher than direct Laplacian because composed FD
        # has different truncation error.
        assert err < 0.10, f"Div(Grad) vs exact error: {err}"

    def test_curl_grad_vanishes(self):
        """Curl(Grad(phi)) = 0 (identity of vector calculus)."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.cos(ky * coords[:, 1])
        X = phi_vals[:, None]

        phi = field_component(0, n_fields=1)
        Grad = Gradient(ndim=2, bc="pbc")
        C = Curl(ndim=2, bc="pbc")

        basis = C(Grad(phi))
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert jnp.max(jnp.abs(result)) < 1e-5

    def test_conservation_composed(self):
        """Lap(Lap(phi)) conserves on PBC."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        phi = field_component(0, n_fields=1)
        Lap = Laplacian(ndim=2, bc="pbc")
        basis = Lap(Lap(phi))
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert jnp.abs(jnp.sum(result)) < 1e-2

    def test_triple_composition(self):
        """Lap(Lap(Lap(phi))) works (three-level fusion)."""
        Nx, Ny, dx = 16, 16, 0.5
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)
        X = jnp.sin(kx * coords[:, 0])[:, None]

        phi = field_component(0, n_fields=1)
        Lap = Laplacian(ndim=2, bc="pbc")
        basis = Lap(Lap(Lap(phi)))
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 1)
        # nabla^6 of sin is (-1)^3 k^6 sin = -k^6 sin
        exact = -(kx**6) * jnp.sin(kx * coords[:, 0])
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.3, f"Triple Lap error: {err}"

    def test_bc_mismatch_raises(self):
        """Composing operators with different BCs raises ValueError."""
        Lap_pbc = Laplacian(ndim=2, bc="pbc")
        Lap_nf = Laplacian(ndim=2, bc="noflux")
        phi = field_component(0, n_fields=1)
        lap_phi = Lap_pbc(phi)
        with pytest.raises(ValueError, match="bc mismatch"):
            Lap_nf(lap_phi)


# ============================================================================
# Divergence
# ============================================================================


class TestDivergence:
    """Tests for the composable Divergence operator."""

    def test_accuracy_2d(self):
        """Divergence of an analytic vector field."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)

        # v = (sin(kx*x), sin(ky*y))
        # div v = kx*cos(kx*x) + ky*cos(ky*y)
        vx = jnp.sin(kx * coords[:, 0])
        vy = jnp.sin(ky * coords[:, 1])
        X = jnp.stack([vx, vy], axis=-1)

        v = vector_field(n_fields=2)
        Div = Divergence(ndim=2, bc="pbc")
        basis = Div(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 1)
        exact = kx * jnp.cos(kx * coords[:, 0]) + ky * jnp.cos(ky * coords[:, 1])
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.02, f"Divergence error: {err}"

    def test_incompressible_field(self):
        """Divergence of curl-like field is zero."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)

        # v = (-sin(ky*y), sin(kx*x)): div = 0
        vx = -jnp.sin(ky * coords[:, 1])
        vy = jnp.sin(kx * coords[:, 0])
        X = jnp.stack([vx, vy], axis=-1)

        v = vector_field(n_fields=2)
        Div = Divergence(ndim=2, bc="pbc")
        basis = Div(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)
        assert jnp.max(jnp.abs(result)) < 1e-5

    def test_wrong_features_raises(self):
        """Divergence requires n_features == ndim."""
        Div = Divergence(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        with pytest.raises(ValueError, match="n_features"):
            Div(phi)


# ============================================================================
# Curl
# ============================================================================


class TestCurl:
    """Tests for the composable Curl operator."""

    def test_accuracy_2d(self):
        """Scalar curl in 2D."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)

        # v = (sin(ky*y), sin(kx*x))
        # curl = d(sin(kx*x))/dx - d(sin(ky*y))/dy
        #      = kx*cos(kx*x) - ky*cos(ky*y)
        vx = jnp.sin(ky * coords[:, 1])
        vy = jnp.sin(kx * coords[:, 0])
        X = jnp.stack([vx, vy], axis=-1)

        v = vector_field(n_fields=2)
        C = Curl(ndim=2, bc="pbc")
        basis = C(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 1)
        exact = kx * jnp.cos(kx * coords[:, 0]) - ky * jnp.cos(ky * coords[:, 1])
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.02, f"Curl error: {err}"

    def test_ndim_check(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            Curl(ndim=1, bc="pbc")
        with pytest.raises(ValueError, match="ndim 2 or 3"):
            Curl(ndim=4, bc="pbc")

    def test_wrong_features_raises(self):
        C = Curl(ndim=2, bc="pbc")
        phi = field_component(0, n_fields=1)
        with pytest.raises(ValueError, match="n_features"):
            C(phi)


# ============================================================================
# SymGrad
# ============================================================================


class TestSymGrad:
    """Tests for the composable SymGrad operator."""

    def test_shape_2d(self):
        """SymGrad produces 3 features in 2D."""
        Nx, Ny, dx = 16, 16, 0.5
        SG = SymGrad(ndim=2, bc="pbc")
        v = vector_field(n_fields=2)
        basis = SG(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        X = jnp.ones((Nx * Ny, 2))
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 3)
        # Constant field → zero symmetric gradient
        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_accuracy_2d(self):
        """SymGrad of linear field is constant."""
        Nx, Ny, dx = 16, 16, 0.5
        coords = _grid_coords_2d(Nx, Ny, dx)

        # v = (x, 2y) → S_00 = 1, S_11 = 2, S_01 = 0
        X = jnp.stack([coords[:, 0], 2.0 * coords[:, 1]], axis=-1)

        SG = SymGrad(ndim=2, bc="pbc")
        v = vector_field(n_fields=2)
        basis = SG(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        # Interior points (avoid boundary artifacts for PBC-wrapped linear field)
        gi = coords[:, 0] / dx
        gj = coords[:, 1] / dx
        interior = (gi > 1) & (gi < Nx - 2) & (gj > 1) & (gj < Ny - 2)
        np.testing.assert_allclose(
            np.asarray(result[interior, 0]), 1.0, atol=0.02,
        )
        np.testing.assert_allclose(
            np.asarray(result[interior, 2]), 2.0, atol=0.02,
        )

    def test_ndim_check(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            SymGrad(ndim=1, bc="pbc")


# ============================================================================
# SkewGrad
# ============================================================================


class TestSkewGrad:
    """Tests for the composable SkewGrad operator."""

    def test_shape_2d(self):
        """SkewGrad produces 1 feature in 2D."""
        Nx, Ny, dx = 16, 16, 0.5
        W = SkewGrad(ndim=2, bc="pbc")
        v = vector_field(n_fields=2)
        basis = W(v)
        extras = _make_extras((Nx, Ny), dx, basis)
        X = jnp.ones((Nx * Ny, 2))
        result = basis(X, extras=extras)
        assert result.shape == (Nx * Ny, 1)
        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_half_curl(self):
        """SkewGrad W_01 = (dv_0/dx_1 - dv_1/dx_0)/2 = -curl/2 in 2D."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        ky = 2 * jnp.pi / (Ny * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)

        vx = jnp.sin(ky * coords[:, 1])
        vy = jnp.sin(kx * coords[:, 0])
        X = jnp.stack([vx, vy], axis=-1)

        W = SkewGrad(ndim=2, bc="pbc")
        C = Curl(ndim=2, bc="pbc")
        v = vector_field(n_fields=2)

        basis_w = W(v)
        extras_w = _make_extras((Nx, Ny), dx, basis_w)
        result_w = basis_w(X, extras=extras_w)

        basis_c = C(v)
        extras_c = _make_extras((Nx, Ny), dx, basis_c)
        result_c = basis_c(X, extras=extras_c)

        # W_01 = (dv_0/dy - dv_1/dx) / 2 = -curl / 2
        np.testing.assert_allclose(
            np.asarray(result_w[:, 0]),
            -0.5 * np.asarray(result_c[:, 0]),
            atol=1e-5,
        )

    def test_ndim_check(self):
        with pytest.raises(ValueError, match="ndim >= 2"):
            SkewGrad(ndim=1, bc="pbc")


# ============================================================================
# AdvectionBy
# ============================================================================


class TestAdvectionBy:
    """Tests for the AdvectionBy factory."""

    def test_accuracy_2d(self):
        """(v·∇)phi with uniform velocity = d phi/dx."""
        Nx, Ny, dx = 32, 32, 0.2
        kx = 2 * jnp.pi / (Nx * dx)
        coords = _grid_coords_2d(Nx, Ny, dx)

        phi_vals = jnp.sin(kx * coords[:, 0])
        # Put phi as field[0], uniform velocity (1,0) as field[0..1]
        # For simplicity, use n_fields=1 and separate velocity expr
        # velocity field: v = (1, 0), constant
        # With n_fields=1, the velocity expression also uses dim=1.
        # That won't work. Use n_fields=2 so both v and phi live in same space.

        # v = (1, 0); phi = sin(kx*x) stored in field[0]
        # State: field[0] = sin(kx*x), field[1] = 0 (unused for phi)
        # Velocity: v = (field[0]_constant, field[1]_constant) is tricky.
        # Let's use a simpler test: v = (v_x, v_y), phi = sin(kx*x) * sin(ky*y)
        # (v·∇)phi = v_x * kx*cos(kx*x)*sin(ky*y) + v_y * ky*sin(kx*x)*cos(ky*y)

        # Use n_fields = 2, field = (v_x, v_y), but phi is a scalar function
        # Actually, v_expr and phi_expr must share the same dim.
        # Let's use 3 fields: (v_x, v_y, phi)
        n_fields = 3
        ky = 2 * jnp.pi / (Ny * dx)

        vx_vals = jnp.ones(Nx * Ny) * 2.0
        vy_vals = jnp.ones(Nx * Ny) * 3.0
        phi_vals = jnp.sin(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
        X = jnp.stack([vx_vals, vy_vals, phi_vals], axis=-1)

        v = field_component(0, n_fields=n_fields) & field_component(1, n_fields=n_fields)
        phi = field_component(2, n_fields=n_fields)

        Adv = AdvectionBy(v, ndim=2, bc="pbc")
        basis = Adv(phi)
        extras = _make_extras((Nx, Ny), dx, basis)
        result = basis(X, extras=extras)

        assert result.shape == (Nx * Ny, 1)
        exact = (
            2.0 * kx * jnp.cos(kx * coords[:, 0]) * jnp.sin(ky * coords[:, 1])
            + 3.0 * ky * jnp.sin(kx * coords[:, 0]) * jnp.cos(ky * coords[:, 1])
        )
        err = jnp.max(jnp.abs(result[:, 0] - exact)) / jnp.max(jnp.abs(exact))
        assert err < 0.05, f"AdvectionBy error: {err}"

    def test_not_composable(self):
        """AdvectionBy results cannot be further composed."""
        n_fields = 3
        v = field_component(0, n_fields=n_fields) & field_component(1, n_fields=n_fields)
        phi = field_component(2, n_fields=n_fields)

        Adv = AdvectionBy(v, ndim=2, bc="pbc")
        Lap = Laplacian(ndim=2, bc="pbc")

        adv_phi = Adv(phi)
        # Results don't carry stencil metadata → TypeError from particles_input
        with pytest.raises(ValueError):
            Lap(adv_phi)


# ============================================================================
# Visualization
# ============================================================================


class TestVisualize:
    """Test visualize_stencil."""

    def test_laplacian_2d(self):
        lap = Laplacian(ndim=2, bc="pbc")
        s = lap.visualize_stencil()
        assert "5-point" in s
        assert "●" in s   # center marker

    def test_biharmonic_2d(self):
        bih = Biharmonic(ndim=2, bc="pbc")
        s = bih.visualize_stencil()
        assert "13-point" in s

    def test_gradient_1d(self):
        """1D stencils fall back to offset listing."""
        grad = Gradient(ndim=1, bc="pbc")
        s = grad.visualize_stencil()
        assert "Offsets:" in s

    def test_divergence(self):
        div = Divergence(ndim=2, bc="pbc")
        s = div.visualize_stencil()
        assert "Divergence" in s


# ============================================================================
# vector_field helper
# ============================================================================


class TestVectorField:
    """Test the vector_field convenience function."""

    def test_basic(self):
        v = vector_field(n_fields=3)
        assert v.n_features == 3
        assert v.dim == 3

    def test_labels(self):
        v = vector_field(n_fields=2, labels=["vx", "vy"])
        assert v.labels == ("vx", "vy") or list(v.labels) == ["vx", "vy"]


class TestBoxExtrasValidation:
    """``square_grid_extras`` rejects non-positive geometry at construction.

    The grid spacing is validated here (concrete, JIT-safe) rather than inside
    the traced stencil functions.
    """

    @pytest.mark.parametrize("dx", [-1.0, 0.0, [1.0, -2.0]])
    def test_nonpositive_dx_raises(self, dx):
        with pytest.raises(ValueError, match="dx must be positive"):
            square_grid_extras((4, 4), dx=dx)

    def test_nonpositive_grid_shape_raises(self):
        with pytest.raises(ValueError, match="grid_shape"):
            square_grid_extras((4, 0), dx=1.0)

    def test_valid_geometry_ok(self):
        extras = square_grid_extras((4, 4), dx=0.5)
        assert "box/dx" in extras and "box/grid_shape" in extras
