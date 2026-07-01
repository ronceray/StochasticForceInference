# TODO: review this file
import jax.numpy as jnp

from SFI.statefunc import *

φx = make_basis(lambda x, **_: x[..., 0], dim=2, rank=0, n_features=1, labels=["x"])
φy = make_basis(lambda x, **_: x[..., 1], dim=2, rank=0, n_features=1, labels=["y"])


def test_concat_and_sum():
    B = φx & φy  # 2 features
    S = φx + φy  # still 1 feature
    x = jnp.array([[1.0, 2.0]])
    assert B(x).shape == (1, 2)
    assert jnp.allclose(S(x), 3.0)


def test_elementwise_map_and_labels():
    Sq = φx.elementwisemap(lambda z: z**2, label_fn=lambda l: f"{l}²")
    x = jnp.array([[3.0, 0.0]])
    y = Sq(x)
    assert jnp.allclose(y, 9.0)
    assert Sq.labels == ("x²",)


def test_slice_and_concat_roundtrip():
    Bx = φx
    By = φy
    C = (Bx & By)[[1, 0]]  # reorder features
    x = jnp.array([[3.0, 4.0]])
    y = C(x)
    assert y.shape == (1, 2)
    assert jnp.allclose(y, jnp.array([[4.0, 3.0]]))
