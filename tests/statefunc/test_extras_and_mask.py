# TODO: review this file
import jax.numpy as jnp
import pytest

from SFI.statefunc import *


def test_masking_zeroes_masked_entries():
    B = make_basis(lambda x, **_: x[..., 0], dim=1, rank=Rank.SCALAR, n_features=1)
    x = jnp.arange(5.0).reshape(-1, 1)
    mask = jnp.array([1, 0, 1, 0, 1], dtype=bool)
    y = B(x, mask=mask)
    assert y.shape == (5, 1)
    assert jnp.allclose(y.squeeze(), jnp.array([0, 0, 2, 0, 4]))


def test_needs_v_enforces_shape():
    B = make_basis(
        lambda x, *, v: v[..., :1], dim=3, rank=Rank.SCALAR, n_features=1, needs_v=True
    )
    x = jnp.ones((2, 3))
    with pytest.raises(ValueError):
        _ = B(x, v=jnp.ones((2, 1)))  # wrong trailing shape
    y = B(x, v=x)
    assert y.shape == (2, 1)


def test_required_extras_union_through_concat():
    f1 = lambda x, *, extras: x[..., :1] + extras["a"]
    f2 = lambda x, *, extras: x[..., :1] + extras["b"]
    B1 = make_basis(f1, dim=1, rank=Rank.SCALAR, n_features=1, extras_keys=("a",))
    B2 = make_basis(f2, dim=1, rank=Rank.SCALAR, n_features=1, extras_keys=("b",))
    S = (B1 & B2).to_psf()
    with pytest.raises(KeyError):
        _ = S(
            jnp.ones((1, 1)), params={"coeff": jnp.array([1.0, 2.0])}, extras={"a": 1.0}
        )
    _ = S(
        jnp.ones((1, 1)),
        params={"coeff": jnp.array([1.0, 2.0])},
        extras={"a": 1.0, "b": 2.0},
    )
