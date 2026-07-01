# TODO: review this file
"""Tests for rank kwarg lifting and .vectorize()/.tensorize() methods."""
import jax.numpy as jnp
import pytest

from SFI.bases import monomials_up_to, monomials_degree


class TestMonomialRankKwarg:
    def test_rank_scalar(self):
        B = monomials_up_to(2, dim=2, rank='scalar')
        x = jnp.ones((3, 2))
        y = B(x)
        assert B.rank == 0
        assert y.shape == (3, 6)  # C(2+2,2)=6

    def test_rank_vector(self):
        B = monomials_up_to(2, dim=2, rank='vector')
        x = jnp.ones((3, 2))
        y = B(x)
        assert B.rank == 1
        assert y.shape == (3, 2, 12)  # 6 scalar × 2 vector

    def test_rank_symmetric_matrix(self):
        B = monomials_up_to(2, dim=2, rank='symmetric_matrix')
        x = jnp.ones((3, 2))
        y = B(x)
        assert B.rank == 2
        assert y.shape == (3, 2, 2, 18)  # 6 × 3 (d(d+1)/2=3)

    def test_rank_identity_matrix(self):
        B = monomials_up_to(2, dim=2, rank='identity_matrix')
        x = jnp.ones((3, 2))
        y = B(x)
        assert B.rank == 2
        assert y.shape == (3, 2, 2, 6)  # 6 × 1

    def test_monomials_degree_rank_vector(self):
        B = monomials_degree(2, dim=3, rank='vector')
        x = jnp.ones((2, 3))
        y = B(x)
        assert B.rank == 1
        # degree-2 in dim=3: C(3+2-1,2) = 6 scalar features × 3 vector
        assert y.shape == (2, 3, 18)

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="Unknown rank"):
            monomials_up_to(2, dim=2, rank='spinor')


class TestVectorizeMethod:
    def test_basic(self):
        S = monomials_up_to(2, dim=2)
        B = S.vectorize(2)
        assert B.rank == 1
        x = jnp.ones((3, 2))
        y = B(x)
        assert y.shape == (3, 2, 12)

    def test_subset_axes(self):
        S = monomials_up_to(1, dim=3)  # 4 scalar features (1, x0, x1, x2)
        B = S.vectorize(3, axes=[0, 2])
        assert B.rank == 1
        x = jnp.ones((2, 3))
        y = B(x)
        assert y.shape == (2, 3, 8)  # 4 × 2 selected axes

    def test_requires_rank_zero(self):
        V = monomials_up_to(1, dim=2, rank='vector')
        with pytest.raises(TypeError, match="rank-0"):
            V.vectorize(2)


class TestTensorizeMethod:
    def test_symmetric(self):
        S = monomials_up_to(2, dim=2)
        B = S.tensorize(2, mode='symmetric')
        assert B.rank == 2
        x = jnp.ones((3, 2))
        y = B(x)
        assert y.shape == (3, 2, 2, 18)

    def test_identity(self):
        S = monomials_up_to(2, dim=2)
        B = S.tensorize(2, mode='identity')
        assert B.rank == 2
        x = jnp.ones((3, 2))
        y = B(x)
        assert y.shape == (3, 2, 2, 6)

    def test_invalid_mode(self):
        S = monomials_up_to(1, dim=2)
        with pytest.raises(ValueError, match="Unknown mode"):
            S.tensorize(2, mode='antisymmetric')
