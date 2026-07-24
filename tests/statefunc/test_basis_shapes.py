# TODO: review this file
import jax.numpy as jnp

from SFI.statefunc import *

# 1) Basic shape/axis behavior (feature-last)


def test_scalar_single_feature_axis_added():
    f = lambda x, **_: x[0]  # rank-0, single feature implicit
    B = make_basis(f, dim=2, rank=0, n_features=1)
    x = jnp.ones((5, 2))
    y = B(x)
    assert B.n_features == 1
    assert y.shape == (5, 1)
    assert jnp.allclose(y, 1.0)


def test_vector_single_feature_axis_added():
    v = lambda x, **_: x  # returns vector of length dim
    B = make_basis(v, dim=3, rank=1, n_features=1)
    x = jnp.arange(9.0).reshape(3, 3)
    y = B(x)
    assert y.shape == (3, 3, 1)  # feature axis appended
    assert jnp.allclose(y[..., 0], x)


def test_multi_feature_explicit_axis():
    g = lambda x, **_: jnp.stack([x[..., 0], x[..., 1]], -1)  # two features
    B = make_basis(g, dim=2, rank=0, n_features=2, labels=["φ_x", "φ_y"])
    x = jnp.array([[1.0, 2.0]])
    y = B(x)
    assert B.n_features == 2
    assert y.shape == (1, 2)
    assert jnp.allclose(y, jnp.array([[1.0, 2.0]]))
