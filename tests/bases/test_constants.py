# TODO: review this file
"""Tests for SFI.bases.constants structural bases."""
import jax.numpy as jnp
import pytest

from SFI.bases.constants import (
    constant_array,
    identity_matrix_basis,
    ones_basis,
    symmetric_matrix_basis,
    unit_vector_basis,
)


class TestOnesBasis:
    def test_shape(self):
        B = ones_basis(3)
        x = jnp.zeros((4, 3))
        y = B(x)
        assert y.shape == (4, 1)  # scalar, 1 feature

    def test_value(self):
        B = ones_basis(2)
        x = jnp.array([[5.0, -3.0]])
        y = B(x)
        assert jnp.allclose(y, 1.0)


class TestUnitVectorBasis:
    def test_shape_full(self):
        B = unit_vector_basis(3)
        x = jnp.zeros((2, 3))
        y = B(x)
        assert y.shape == (2, 3, 3)  # (T, dim, F=dim)

    def test_shape_subset(self):
        B = unit_vector_basis(4, axes=[0, 2])
        x = jnp.zeros((2, 4))
        y = B(x)
        assert y.shape == (2, 4, 2)  # 2 out of 4 axes

    def test_orthogonality(self):
        B = unit_vector_basis(3)
        x = jnp.zeros((1, 3))
        y = B(x)[0]  # (3, 3)
        # Each column should be a unit vector
        for i in range(3):
            assert jnp.allclose(jnp.sum(y[:, i] ** 2), 1.0)
            for j in range(i + 1, 3):
                assert jnp.allclose(jnp.dot(y[:, i], y[:, j]), 0.0)


