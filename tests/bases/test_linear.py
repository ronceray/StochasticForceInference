# TODO: review this file
import jax.numpy as jnp

from SFI.bases.linear import (
    V,
    X,
    linear_basis,
    v_coordinate,
    v_coordinates,
    x_coordinate,
    x_coordinates,
)


def test_X_vector_contract_no_batch():
    dim = 3
    basis = X(dim)
    x = jnp.arange(dim, dtype=jnp.float32)  # (dim,)
    y = basis(x)
    assert y.shape == (dim, 1)
    assert jnp.allclose(y[:, 0], x)


def test_V_vector_contract_no_batch():
    dim = 4
    basis = V(dim)
    x = jnp.zeros((dim,), dtype=jnp.float32)
    v = jnp.arange(dim, dtype=jnp.float32)
    y = basis(x, v=v)
    assert y.shape == (dim, 1)
    assert jnp.allclose(y[:, 0], v)


def test_x_coordinate_scalar_feature_auto_axis():
    dim = 5
    idx = 2
    basis = x_coordinate(idx, dim=dim)
    x = jnp.arange(dim, dtype=jnp.float32)
    y = basis(x)
    # BasisLeaf auto-inserts feature axis for scalar n_features=1
    assert y.shape == (1,)
    assert jnp.allclose(y[0], x[idx])


def test_x_coordinates_multiple():
    dim = 6
    inds = [0, 2, 5]
    basis = x_coordinates(inds, dim=dim)
    x = jnp.arange(dim, dtype=jnp.float32)
    y = basis(x)
    assert y.shape == (len(inds),)
    assert jnp.allclose(y, x[jnp.array(inds)])


def test_v_coordinate_and_v_coordinates():
    dim = 3
    idx = 1
    inds = [0, 2]
    bc = v_coordinate(idx, dim=dim)
    bcs = v_coordinates(inds, dim=dim)
    x = jnp.zeros((dim,), dtype=jnp.float32)
    v = jnp.arange(dim, dtype=jnp.float32)

    y1 = bc(x, v=v)
    y2 = bcs(x, v=v)

    assert y1.shape == (1,)
    assert y2.shape == (len(inds),)
    assert jnp.allclose(y1[0], v[idx])
    assert jnp.allclose(y2, v[jnp.array(inds)])


def test_linear_basis_degree1_shapes_x_only_and_xv_no_batch():
    dim = 2
    lb_x = linear_basis(dim, include_x=True, include_v=False)
    lb_xv = linear_basis(dim, include_x=True, include_v=True)

    x = jnp.array([[1.0, 2.0]])
    v = jnp.array([[10.0, 20.0]])

    yx = lb_x(x)
    yxv = lb_xv(x, v=v)

    assert yx.shape == (1, dim)
    assert yxv.shape == (1, 2 * dim)
