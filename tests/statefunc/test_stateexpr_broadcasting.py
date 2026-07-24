# TODO: review this file
# tests/statefunc/test_stateexpr_broadcasting.py
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.statefunc import Rank, StateExpr, make_basis  # noqa: F401


# ---------------------------------------------------------------------
# Helpers to build small, concrete expressions
# ---------------------------------------------------------------------
def basis_vector(dim: int, nF: int = 1):
    """Return a Basis whose single-sample output is a vector in R^dim with nF features."""
    if nF == 1:

        def f(x):
            # shape (dim,), feature axis is auto-added by factory to (dim,1)
            return x
    else:

        def f(x):
            # (dim, nF): multiple features for uniform-across-features tests
            cols = [x, 2 * x] + [x * 0 + (k + 1) for k in range(max(0, nF - 2))]
            return jnp.stack(cols, axis=-1)

    return make_basis(
        f,
        dim=dim,
        rank=Rank.VECTOR,
        n_features=nF,
    )


def basis_matrix(dim: int, nF: int = 1):
    """Return a Basis with rank-2 output in R^{dim x dim}."""
    if nF == 1:

        def f(x):
            v = x
            return jnp.outer(v, v)  # (dim, dim)
    else:

        def f(x):
            v = x
            A = jnp.outer(v, v)  # feature 0
            B = jnp.eye(v.size)  # feature 1
            rest = [jnp.ones_like(B) * (k + 1) for k in range(max(0, nF - 2))]
            return jnp.stack([A, B, *rest], axis=-1)  # (dim, dim, nF)

    return make_basis(
        f,
        dim=dim,
        rank=Rank.MATRIX,
        n_features=nF,
    )


# ---------------------------------------------------------------------
# Basic scalar broadcasting
# ---------------------------------------------------------------------
@pytest.mark.parametrize("d,val", [(1, 2.0), (3, -1.5)])
def test_scalar_broadcast_vector(d, val):
    B = basis_vector(d)  # rank=1
    x = jnp.arange(d, dtype=jnp.float32) + 1  # 1..d
    Y = (B + val)(x)  # (dim, 1)
    Y_ref = x + val
    assert Y.shape == (d, 1)
    np.testing.assert_allclose(np.squeeze(Y, -1), np.asarray(Y_ref), rtol=0, atol=1e-6)


@pytest.mark.parametrize("d,val", [(2, 3.0), (4, 0.25)])
def test_scalar_broadcast_matrix(d, val):
    M = basis_matrix(d)  # rank=2
    x = jnp.arange(d, dtype=jnp.float32) + 1
    Y = (M * val)(x)  # (d, d, 1)
    Y_ref = jnp.outer(x, x) * val
    assert Y.shape == (d, d, 1)
    np.testing.assert_allclose(np.squeeze(Y, -1), np.asarray(Y_ref), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# Spatial broadcasting on rank axes
# ---------------------------------------------------------------------
@pytest.mark.parametrize("d", [2, 5])
def test_rank_broadcast_vector_add_length_d(d):
    B = basis_vector(d)  # (dim,1) per sample
    shift = jnp.arange(d, dtype=jnp.float32)  # (dim,)
    x = jnp.linspace(0.0, 1.0, d)
    Y = (B + shift)(x)  # (dim,1)
    Y_ref = x + shift
    np.testing.assert_allclose(np.squeeze(Y, -1), np.asarray(Y_ref), rtol=0, atol=1e-6)


@pytest.mark.parametrize("d", [3, 6])
def test_rank_broadcast_matrix_row_col_patterns(d):
    M = basis_matrix(d)  # (d,d,1)
    x = jnp.linspace(1.0, 2.0, d)

    # (d,1) broadcasts down columns
    col = jnp.linspace(0.0, 1.0, d).reshape(d, 1)
    Yc = (M + col)(x)
    Yc_ref = jnp.outer(x, x) + col
    np.testing.assert_allclose(
        np.squeeze(Yc, -1), np.asarray(Yc_ref), rtol=0, atol=1e-6
    )

    # (1,d) broadcasts across rows
    row = jnp.linspace(0.0, 1.0, d).reshape(1, d)
    Yr = (M + row)(x)
    Yr_ref = jnp.outer(x, x) + row
    np.testing.assert_allclose(
        np.squeeze(Yr, -1), np.asarray(Yr_ref), rtol=0, atol=1e-6
    )

    # Elementwise pattern (d,d)
    mask = jnp.eye(d)
    Ym = (M * mask)(x)
    Ym_ref = jnp.outer(x, x) * mask
    np.testing.assert_allclose(
        np.squeeze(Ym, -1), np.asarray(Ym_ref), rtol=0, atol=1e-6
    )


# ---------------------------------------------------------------------
# Left/right ops and np/jnp ufunc entry points
# ---------------------------------------------------------------------
@pytest.mark.parametrize("d", [4])
def test_reverse_ops_and_ufuncs(d):
    B = basis_vector(d)
    x = jnp.arange(d, dtype=jnp.float32)

    a = 2.0 + B
    b = B + 2.0
    np.testing.assert_allclose(a(x).squeeze(-1), b(x).squeeze(-1))

    # numpy/jax ufunc entry points should route to __array_priority__/__array_ufunc__
    c = B + jnp.ones(d)  # (d,)
    dres = (jnp.arange(d) + 1) * B
    np.testing.assert_allclose(c(x).squeeze(-1), (x + 1).astype(np.float32))
    np.testing.assert_allclose(
        dres(x).squeeze(-1), ((jnp.arange(d) + 1) * x).astype(jnp.float32)
    )


# ---------------------------------------------------------------------
# No-feature semantics for constants: rejections and “uniform across features”
# ---------------------------------------------------------------------
def test_feature_shaped_constant_is_rejected():
    # Build a multi-feature vector expression
    d, nF = 3, 4
    B = basis_vector(d, nF=nF)  # shape (d, nF)
    x = jnp.arange(d, dtype=jnp.float32) + 1
    c_feat = jnp.arange(nF, dtype=jnp.float32) + 1  # (nF,)

    with pytest.raises((TypeError, ValueError)):
        _ = (B + c_feat)(x)


def test_uniform_across_features_broadcast():
    d, nF = 4, 2
    B = basis_vector(d, nF=nF)  # feature 0 = x, feature 1 = 2x
    shift = jnp.arange(d, dtype=jnp.float32)  # (d,)
    x = jnp.linspace(0.0, 1.0, d)

    Y = (B + shift)(x)  # (d, nF)
    Y_ref0 = x + shift
    Y_ref1 = 2 * x + shift
    np.testing.assert_allclose(Y[:, 0], np.asarray(Y_ref0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(Y[:, 1], np.asarray(Y_ref1), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# Error cases: incompatible spatial shapes
# ---------------------------------------------------------------------
def test_incompatible_spatial_shape_raises():
    d = 3
    B = basis_vector(d)
    x = jnp.arange(d, dtype=jnp.float32)
    bad = jnp.ones((d, d))  # too many axes for rank=1
    with pytest.raises(TypeError):
        _ = (B + bad)(x)


def test_matrix_requires_rank_compatible_mask():
    d = 3
    M = basis_matrix(d)
    x = jnp.arange(d, dtype=jnp.float32) + 1
    bad = jnp.ones((d, d, d))  # 3 spatial axes vs rank=2
    with pytest.raises(TypeError):
        _ = (M * bad)(x)
