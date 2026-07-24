# TODO: review this file
from __future__ import annotations

import jax.numpy as jnp
import pytest

from SFI.statefunc import Rank, make_basis


def _basis_vector(dim: int, nF: int = 1):
    def f(x):
        return x

    return make_basis(f, dim=dim, rank=Rank.VECTOR, n_features=nF)


def _basis_matrix(dim: int, nF: int = 1):
    def f(x):
        v = x
        return jnp.outer(v, v)

    return make_basis(f, dim=dim, rank=Rank.MATRIX, n_features=nF)


@pytest.mark.parametrize("dim,nF", [(1, 1), (3, 1), (4, 2)])
def test_leaf_vector_estimate_bytes_per_sample(dim, nF):
    B = _basis_vector(dim, nF=nF)
    # output elements per sample: dim^rank * nF with rank=1
    rank = 1
    elems = (dim**rank) * nF
    isz = jnp.dtype(jnp.float32).itemsize
    expect = elems * isz
    got = B.estimate_bytes_per_sample(dtype=jnp.float32)
    assert got == expect


@pytest.mark.parametrize("dim,nF", [(2, 1), (3, 1), (3, 5)])
def test_leaf_matrix_estimate_bytes_per_sample(dim, nF):
    M = _basis_matrix(dim, nF=nF)
    rank = 2
    elems = (dim**rank) * nF
    isz = jnp.dtype(jnp.float32).itemsize
    expect = elems * isz
    got = M.estimate_bytes_per_sample(dtype=jnp.float32)
    assert got == expect


def test_composite_add_sums_children_plus_output():
    # Two vector leaves, same shape. Composite op should conservatively count:
    # sum(children) + my output.
    dim, nF = 3, 1
    A = _basis_vector(dim, nF)
    B = _basis_vector(dim, nF)
    expr = A + B  # elementwise add node
    isz = jnp.dtype(jnp.float32).itemsize
    leaf_bytes = (dim**1) * nF * isz
    expect = leaf_bytes + leaf_bytes + leaf_bytes
    got = expr.estimate_bytes_per_sample(dtype=jnp.float32)
    assert got == expect
