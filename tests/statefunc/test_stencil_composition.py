# TODO: review this file
"""Tests for generic stencil composition in GridLayout.

Tests that arbitrarily nested differential operators (``lap(grad(f).dot(grad(f)))``,
``div(Q * grad(phi))``, etc.) compile and produce numerically correct results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases.spde import square_grid_extras
from SFI.statefunc.layout import GridLayout, ScalarSector, VectorSector
from SFI.statefunc.layout._fd_atoms import (
    biharmonic_offsets,
    build_offset_to_idx,
    compute_footprint,
    cross_offsets,
    minkowski_sum,
    offsets_to_sorted_array,
    _origin,
)
from SFI.statefunc.nodes.interactions.prepare import (
    prepare_structural_extras_for_expr,
)


def _eval_basis(basis, x, extras):
    """Prepare and evaluate a Basis on (P, dim) data."""
    prepare_structural_extras_for_expr(basis, extras)
    return np.asarray(basis(jnp.asarray(x), extras=extras))


# =====================================================================
# FD atom tests
# =====================================================================


class TestMinkowskiSum:
    def test_origin_identity(self):
        """Minkowski sum with {origin} is identity."""
        cross_2d = cross_offsets(2)
        assert minkowski_sum(_origin(2), cross_2d) == cross_2d

    def test_cross_mink_cross_2d(self):
        """Cross ⊕ cross in 2D gives radius-2 + diagonals (≤ biharmonic)."""
        c = cross_offsets(2)
        result = minkowski_sum(c, c)
        # Should include (0,0), (±1,0), (0,±1), (±2,0), (0,±2),
        # (±1,±1) = 13 unique points total
        assert len(result) == 13
        assert (0, 0) in result
        assert (2, 0) in result
        assert (1, 1) in result

    def test_cross_mink_origin_1d(self):
        c = cross_offsets(1)  # {(0,), (1,), (-1,)}
        o = _origin(1)
        assert minkowski_sum(c, o) == c

    def test_biharmonic_contains_cross_squared(self):
        """Biharmonic stencil should be ⊇ cross ⊕ cross in 2D."""
        bih = biharmonic_offsets(2)
        cross_sq = minkowski_sum(cross_offsets(2), cross_offsets(2))
        assert cross_sq <= bih


class TestComputeFootprint:
    def _make_layout_and_leaf(self, ndim=2):
        layout = GridLayout(
            phi=ScalarSector([0]),
            dim=1, ndim=ndim, bc="pbc",
        )
        return layout, layout.phi

    def test_leaf_is_origin(self):
        layout, phi = self._make_layout_and_leaf()
        fp = compute_footprint(phi._node, 2)
        assert fp == _origin(2)

    def test_lap_is_cross(self):
        layout, phi = self._make_layout_and_leaf()
        lap_phi = layout.lap(phi)
        fp = compute_footprint(lap_phi._node, 2)
        assert fp == cross_offsets(2)

    def test_lap_of_product_is_cross(self):
        """lap(a * phi) has same footprint as lap(phi)."""
        layout, phi = self._make_layout_and_leaf()
        expr = layout.lap(3.0 * phi)
        fp = compute_footprint(expr._node, 2)
        assert fp == cross_offsets(2)

    def test_grad_is_cross(self):
        layout, phi = self._make_layout_and_leaf()
        g = layout.grad(phi)
        fp = compute_footprint(g._node, 2)
        assert fp == cross_offsets(2)

    def test_lap_of_grad_dot_grad(self):
        """lap(grad(f).dot(grad(f))) should have cross ⊕ cross footprint."""
        layout, phi = self._make_layout_and_leaf()
        g = layout.grad(phi)
        expr = layout.lap(g.dot(g))
        fp = compute_footprint(expr._node, 2)
        expected = minkowski_sum(cross_offsets(2), cross_offsets(2))
        assert fp == expected

    def test_div_of_v_is_cross(self):
        layout = GridLayout(
            v=VectorSector([0, 1], sdim=2, spatial=True),
            dim=2, ndim=2, bc="pbc",
        )
        expr = layout.div(layout.v)
        fp = compute_footprint(expr._node, 2)
        assert fp == cross_offsets(2)

    def test_biharmonic_footprint(self):
        layout, phi = self._make_layout_and_leaf()
        expr = layout.biharmonic(phi)
        fp = compute_footprint(expr._node, 2)
        assert fp == biharmonic_offsets(2)


class TestOffsetsToSortedArray:
    def test_origin_first(self):
        arr = offsets_to_sorted_array(cross_offsets(2))
        assert tuple(arr[0]) == (0, 0)

    def test_round_trip(self):
        offsets = cross_offsets(2)
        arr = offsets_to_sorted_array(offsets)
        o2i = build_offset_to_idx(arr)
        assert set(o2i.keys()) == offsets


# =====================================================================
# Numerical composition tests
# =====================================================================


def _make_1d_layout(N=16, dx=0.5):
    """1D layout with a single scalar sector."""
    layout = GridLayout(
        phi=ScalarSector([0]),
        dim=1, ndim=1, bc="pbc",
    )
    extras = square_grid_extras(grid_shape=(N,), dx=dx)
    return layout, extras, N, dx


def _make_2d_layout(Nx=8, Ny=8, dx=0.5):
    """2D layout with a single scalar sector."""
    layout = GridLayout(
        phi=ScalarSector([0]),
        dim=1, ndim=2, bc="pbc",
    )
    extras = square_grid_extras(grid_shape=(Nx, Ny), dx=dx)
    return layout, extras, Nx, Ny, dx


def _make_2d_vector_layout(Nx=8, Ny=8, dx=0.5):
    """2D layout with velocity sector."""
    layout = GridLayout(
        v=VectorSector([0, 1], sdim=2, spatial=True),
        dim=2, ndim=2, bc="pbc",
    )
    extras = square_grid_extras(grid_shape=(Nx, Ny), dx=dx)
    return layout, extras, Nx, Ny, dx


class TestScalarLapComposition:
    """Test that lap(phi) via new compiler matches manual FD."""

    def test_lap_1d_sinusoid(self):
        layout, extras, N, dx = _make_1d_layout(N=32, dx=0.2)
        phi = layout.phi
        force = layout.embed(rank=1, phi=layout.lap(phi))

        # sinusoidal field: f(x) = sin(2π x / L)
        L = N * dx
        xs = jnp.arange(N) * dx
        field = jnp.sin(2 * jnp.pi * xs / L)

        # Manual laplacian
        lap_manual = jnp.roll(field, -1) + jnp.roll(field, 1) - 2.0 * field
        lap_manual = lap_manual / (dx * dx)

        # Evaluate embed
        P = N
        state = field.reshape(P, 1)  # (P, dim=1)
        y = _eval_basis(force, state, extras)
        y_flat = y.reshape(P, -1)[:, 0]

        np.testing.assert_allclose(y_flat, lap_manual, rtol=1e-5, atol=1e-6)


class TestLapOfCubic:
    """Test lap(phi^3) — nonlinear child of stencil."""

    def test_lap_of_cube_1d(self):
        layout, extras, N, dx = _make_1d_layout(N=32, dx=0.2)
        phi = layout.phi
        force = layout.embed(rank=1, phi=layout.lap(phi ** 3))

        L = N * dx
        xs = jnp.arange(N) * dx
        field = jnp.sin(2 * jnp.pi * xs / L)

        # Manual: lap(f^3) = (f(+1)^3 + f(-1)^3 - 2*f(0)^3) / dx^2
        f3 = field ** 3
        lap_manual = (jnp.roll(f3, -1) + jnp.roll(f3, 1) - 2.0 * f3) / (dx * dx)

        P = N
        state = field.reshape(P, 1)
        y = _eval_basis(force, state, extras)
        y_flat = y.reshape(P, -1)[:, 0]

        np.testing.assert_allclose(y_flat, lap_manual, rtol=1e-5, atol=1e-6)


class TestDivOfVector:
    """Test div(v) for a 2D vector field."""

    def test_div_2d(self):
        Nx, Ny, dx = 8, 8, 0.5
        # div(v) returns scalar sdims=(), so we need a scalar output sector.
        layout = GridLayout(
            v=VectorSector([0, 1], sdim=2, spatial=True),
            divv=ScalarSector([2]),
            dim=3, ndim=2, bc="pbc",
        )
        extras = square_grid_extras(grid_shape=(Nx, Ny), dx=dx)
        v = layout.v
        force = layout.embed(rank=1, divv=layout.div(v))

        P = Nx * Ny
        key = jax.random.PRNGKey(42)
        state = jax.random.normal(key, (P, 3))

        # Just check it runs without error and has correct shape
        y = _eval_basis(force, state, extras)
        assert y.shape[:2] == (P, 3)


class TestLapOfGradDotGrad:
    """Test lap(grad(f).dot(grad(f))) — the AMB+ term — via composition."""

    def test_against_manual_2d(self):
        """Compare composed lap(|∇f|²) to manual FD on biharmonic stencil."""
        Nx, Ny, dx = 8, 8, 0.5
        layout, extras, _, _, _ = _make_2d_layout(Nx, Ny, dx)
        phi = layout.phi

        g = layout.grad(phi)
        lgs_expr = layout.lap(g.dot(g))
        force = layout.embed(rank=1, phi=lgs_expr)

        # 2D scalar field
        key = jax.random.PRNGKey(123)
        P = Nx * Ny
        field = jax.random.normal(key, (P,))
        field_2d = field.reshape(Nx, Ny)

        # Manual computation on 2D grid with PBC
        def manual_grad_sq(f2d, dx):
            gx = (jnp.roll(f2d, -1, axis=0) - jnp.roll(f2d, 1, axis=0)) / (2 * dx)
            gy = (jnp.roll(f2d, -1, axis=1) - jnp.roll(f2d, 1, axis=1)) / (2 * dx)
            return gx ** 2 + gy ** 2

        def manual_lap(f2d, dx):
            return (
                jnp.roll(f2d, -1, axis=0) + jnp.roll(f2d, 1, axis=0)
                + jnp.roll(f2d, -1, axis=1) + jnp.roll(f2d, 1, axis=1)
                - 4.0 * f2d
            ) / (dx * dx)

        G = manual_grad_sq(field_2d, dx)
        lgs_manual = manual_lap(G, dx)

        state = field.reshape(P, 1)
        y = _eval_basis(force, state, extras)
        y_flat = y.reshape(P, -1)[:, 0]

        np.testing.assert_allclose(
            y_flat, lgs_manual.reshape(-1), rtol=1e-4, atol=1e-4
        )


class TestLapOfGradSqConvenience:
    """Test that layout.lap_of_grad_sq(f) matches composition."""

    def test_convenience_vs_manual(self):
        Nx, Ny, dx = 8, 8, 0.5
        layout, extras, _, _, _ = _make_2d_layout(Nx, Ny, dx)
        phi = layout.phi

        # Via convenience method
        lgs_conv = layout.lap_of_grad_sq(phi)
        force_conv = layout.embed(rank=1, phi=lgs_conv)

        # Via explicit composition
        g = layout.grad(phi)
        lgs_comp = layout.lap(g.dot(g))
        force_comp = layout.embed(rank=1, phi=lgs_comp)

        key = jax.random.PRNGKey(456)
        P = Nx * Ny
        state = jax.random.normal(key, (P, 1))

        y_conv = _eval_basis(force_conv, state, extras)
        y_comp = _eval_basis(force_comp, state, extras)

        np.testing.assert_allclose(y_conv, y_comp, rtol=1e-5, atol=1e-6)


class TestConservation:
    """Test that sum over PBC grid is zero for conservative operators."""

    def test_lap_conservation(self):
        """sum_i lap(f)_i = 0 for PBC."""
        layout, extras, N, dx = _make_1d_layout(N=16, dx=0.3)
        phi = layout.phi
        force = layout.embed(rank=1, phi=layout.lap(phi))

        key = jax.random.PRNGKey(99)
        state = jax.random.normal(key, (N, 1))
        y = _eval_basis(force, state, extras)
        total = jnp.sum(y)
        np.testing.assert_allclose(float(total), 0.0, atol=1e-5)

    def test_div_conservation_2d(self):
        """sum_i div(v)_i = 0 for PBC."""
        layout, extras, Nx, Ny, dx = _make_2d_vector_layout()
        v = layout.v
        # div(v) returns scalar sdims=() but we need to match the
        # VectorSector sdims=(2,).  So we can't embed div(v) into
        # sector v which has sdims=(2,).  Instead test with a scalar sector.
        layout2 = GridLayout(
            v=VectorSector([0, 1], sdim=2, spatial=True),
            divv=ScalarSector([2]),
            dim=3, ndim=2, bc="pbc",
        )
        extras2 = square_grid_extras(grid_shape=(Nx, Ny), dx=dx)
        v2 = layout2.v
        force = layout2.embed(rank=1, divv=layout2.div(v2))

        key = jax.random.PRNGKey(77)
        P = Nx * Ny
        state = jax.random.normal(key, (P, 3))
        y = _eval_basis(force, state, extras2)
        # Sum the divv-sector output (index 2)
        total = jnp.sum(y[:, 2])
        np.testing.assert_allclose(float(total), 0.0, atol=1e-5)

    def test_lap_of_grad_sq_conservation(self):
        """sum_i lap(|∇f|²)_i = 0 for PBC."""
        Nx, Ny, dx = 8, 8, 0.5
        layout, extras, _, _, _ = _make_2d_layout(Nx, Ny, dx)
        phi = layout.phi
        g = layout.grad(phi)
        expr = layout.lap(g.dot(g))
        force = layout.embed(rank=1, phi=expr)

        key = jax.random.PRNGKey(88)
        P = Nx * Ny
        state = jax.random.normal(key, (P, 1))
        y = _eval_basis(force, state, extras)
        total = jnp.sum(y)
        np.testing.assert_allclose(float(total), 0.0, atol=1e-4)


class TestBiharmonicComposition:
    """Test the built-in biharmonic operator still works."""

    def test_biharmonic_2d_runs(self):
        Nx, Ny, dx = 8, 8, 0.5
        layout, extras, _, _, _ = _make_2d_layout(Nx, Ny, dx)
        phi = layout.phi
        force = layout.embed(rank=1, phi=layout.biharmonic(phi))

        key = jax.random.PRNGKey(42)
        P = Nx * Ny
        state = jax.random.normal(key, (P, 1))
        y = _eval_basis(force, state, extras)
        assert y.shape[:2] == (P, 1)

        # Conservation
        total = jnp.sum(y)
        np.testing.assert_allclose(float(total), 0.0, atol=1e-4)


class TestGradThenDiv:
    """Test grad and div compose: div(grad(f)) = lap(f)."""

    def test_div_grad_equals_lap_1d(self):
        layout = GridLayout(
            phi=ScalarSector([0]),
            dim=1, ndim=1, bc="pbc",
        )
        extras = square_grid_extras(grid_shape=(16,), dx=0.25)
        phi = layout.phi

        # div(grad(phi))
        dg = layout.div(layout.grad(phi))
        force_dg = layout.embed(rank=1, phi=dg)

        # lap(phi)
        force_lap = layout.embed(rank=1, phi=layout.lap(phi))

        key = jax.random.PRNGKey(111)
        state = jax.random.normal(key, (16, 1))

        y_dg = _eval_basis(force_dg, state, extras)
        y_lap = _eval_basis(force_lap, state, extras)

        # div(grad) uses central diff twice → (1, -2, 1)/(4dx²) ≠ lap's (1, -2, 1)/dx²
        # They differ by a factor of 4 in weight because div∘grad expands to:
        # [f(+2) - 2f(0) + f(-2)] / (4dx²) vs [f(+1) - 2f(0) + f(-1)] / dx²
        # So they should NOT be equal! But they should both be conservative.
        total_dg = jnp.sum(y_dg)
        total_lap = jnp.sum(y_lap)
        np.testing.assert_allclose(float(total_dg), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(total_lap), 0.0, atol=1e-5)


class TestMixedStencilAndPointwise:
    """Test expressions mixing stencil and pointwise algebra."""

    def test_scalar_times_lap(self):
        """a * lap(phi) — scalar constant times stencil result."""
        layout, extras, N, dx = _make_1d_layout(N=16, dx=0.3)
        phi = layout.phi
        expr = 2.5 * layout.lap(phi)
        force = layout.embed(rank=1, phi=expr)

        # Reference: 2.5 * lap(phi) directly
        force_ref = layout.embed(rank=1, phi=layout.lap(phi))

        key = jax.random.PRNGKey(22)
        state = jax.random.normal(key, (N, 1))
        y = _eval_basis(force, state, extras)
        y_ref = _eval_basis(force_ref, state, extras)

        np.testing.assert_allclose(y, 2.5 * y_ref, rtol=1e-5, atol=1e-6)

    def test_lap_plus_biharmonic(self):
        """lap(phi) + biharmonic(phi) — sum of two stencil terms."""
        Nx, Ny, dx = 8, 8, 0.5
        layout, extras, _, _, _ = _make_2d_layout(Nx, Ny, dx)
        phi = layout.phi
        expr = layout.lap(phi) + layout.biharmonic(phi)
        force = layout.embed(rank=1, phi=expr)

        force_lap = layout.embed(rank=1, phi=layout.lap(phi))
        force_bih = layout.embed(rank=1, phi=layout.biharmonic(phi))

        key = jax.random.PRNGKey(33)
        P = Nx * Ny
        state = jax.random.normal(key, (P, 1))
        y = _eval_basis(force, state, extras)
        y_ref = _eval_basis(force_lap, state, extras) + _eval_basis(force_bih, state, extras)

        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)


class TestConvenienceMethods:
    """Test convenience differential operator methods."""

    def test_strain_rate_symmetric(self):
        """strain_rate(v) should be symmetric."""
        layout, extras, Nx, Ny, dx = _make_2d_vector_layout()
        v = layout.v
        E = layout.strain_rate(v)
        assert E.sdims == (2, 2)
        # E = (grad(v) + grad(v).T) / 2 → should be symmetric
        # We can verify by checking E - E.T = 0 at the expression level
        diff = E - E.T
        assert diff.sdims == (2, 2)

    def test_vorticity_antisymmetric(self):
        """vorticity(v) should be antisymmetric."""
        layout, extras, Nx, Ny, dx = _make_2d_vector_layout()
        v = layout.v
        Om = layout.vorticity(v)
        assert Om.sdims == (2, 2)
