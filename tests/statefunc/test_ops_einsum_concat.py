# TODO: review this file
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.statefunc import *


def _scalar_basis(dim):
    return make_basis(lambda x, **_: x[0], dim=dim, rank=Rank.SCALAR, n_features=1)


def vec_basis(dim=2, nF=1):
    # x: (..., dim) -> y: (..., dim, nF)
    return make_basis(
        lambda x, **_: jnp.repeat(x[..., None], nF, axis=-1),
        dim=dim,
        rank=Rank.VECTOR,
        n_features=nF,
    )


def scal_basis(dim=2, nF=1):
    return make_basis(
        lambda x, **_: jnp.ones((nF,)), dim=dim, rank=Rank.SCALAR, n_features=nF
    )


# Concatenation / arithmetic


def test_concat_features():
    B1 = _scalar_basis(2)
    B2 = _scalar_basis(2)
    B = B1 & B2
    x = jnp.zeros((2, 2))
    y = B(x)
    assert B.n_features == 2
    assert y.shape == (2, 2)


def test_add_and_mul():
    B = _scalar_basis(2)
    C = B + B
    D = B * B
    x = jnp.ones((3, 2))
    assert jnp.allclose(C(x), 2.0)
    assert jnp.allclose(D(x), 1.0)


# Rank ops / einsum


def test_outer_rank():
    v = make_basis(lambda x, **_: x, dim=2, rank=Rank.VECTOR, n_features=1)
    O = StateExpr.einsum("i,j->ij", v, v)
    x = jnp.array([[1.0, 2.0]])
    y = O(x)
    assert y.shape == (1, 2, 2, 1)
    assert jnp.allclose(y[..., 0], jnp.outer(x[0], x[0])[None, ...])


def test_outer_rank_and_value():
    v = vec_basis(dim=2, nF=1)
    O = StateExpr.einsum("i,j->ij", v, v)
    x = jnp.array([[1.0, 2.0]])
    y = O(x)
    assert y.shape == (1, 2, 2, 1)
    assert jnp.allclose(y[..., 0], jnp.outer(x[0], x[0])[None, ...])


def test_cartesian_product_of_features_two_operands():
    v1 = vec_basis(dim=3, nF=3)
    v2 = vec_basis(dim=3, nF=5)
    expr = StateExpr.einsum("i,i->", v1, v2)
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)  # batch=2
    y = expr(x)
    assert y.shape == (2, 3 * 5)


def test_scalar_times_vector_and_spaces_in_spec():
    s = scal_basis(dim=3, nF=4)
    v = vec_basis(dim=3, nF=2)
    expr = StateExpr.einsum(" , j -> j ", s, v)
    x = jnp.arange(3.0)[None, :]
    y = expr(x)
    assert y.shape == (1, 3, 8)


def test_full_reduction_scalar_output_keeps_batch_and_features():
    v = vec_basis(dim=2, nF=7)
    expr = StateExpr.einsum("i,i->", v, v)
    x = jnp.array([[3.0, 4.0], [0.0, 1.0]])
    y = expr(x)
    assert y.shape == (2, 49)


def test_dot_is_einsum_ii_to_scalar_spatial():
    v = vec_basis(dim=4, nF=1)
    d1 = v.dot(v)
    d2 = StateExpr.einsum("i,i->", v, v)
    x = jnp.arange(4.0)[None, :]
    assert jnp.allclose(d1(x), d2(x))


def test_matmul_builds_true_matrix_product():
    a = vec_basis(dim=2, nF=2)
    b = vec_basis(dim=2, nF=3)
    expr = a @ b
    x = jnp.array([[1.0, 2.0]])
    y = expr(x)
    # (vector@(vector)) -> scalar; features multiply
    assert y.shape == (1, 6)


# Spec validation via the classmethod


def test_forbid_ellipsis_anywhere_in_spec():
    v = vec_basis()
    with pytest.raises(ValueError):
        _ = StateExpr.einsum("...i,i->", v, v)
    with pytest.raises(ValueError):
        _ = StateExpr.einsum("i,...i->", v, v)
    with pytest.raises(ValueError):
        _ = StateExpr.einsum("i,i->...", v, v)


def test_operand_count_mismatch():
    v = vec_basis()
    with pytest.raises(ValueError, match="operands"):
        _ = StateExpr.einsum("i->", v, v)


def test_rhs_letters_must_be_subset_of_lhs():
    v = vec_basis()
    with pytest.raises(ValueError):
        _ = StateExpr.einsum("i,i->j", v, v)


def test_child_rank_must_equal_token_length():
    v = vec_basis()
    with pytest.raises(ValueError, match="rank mismatch"):
        _ = StateExpr.einsum("ij,i->j", v, v)


def test_uppercase_letters_rejected():
    v = vec_basis()
    with pytest.raises(ValueError):
        _ = StateExpr.einsum("I,i->", v, v)


def test_too_many_children_feature_suffix_limit():
    bases = [scal_basis(nF=1) for _ in range(27)]
    with pytest.raises(ValueError, match="Too many children"):
        spec = ",".join([""] * 27) + "->"
        _ = StateExpr.einsum(spec, *bases)


def test_einsum_needs_stateexpr():
    a = jnp.ones((3,), dtype=jnp.float32)
    b = jnp.ones((3,), dtype=jnp.float32)
    with pytest.raises(TypeError):
        _ = StateExpr.einsum("i,i->", a, b)


def test_einsum_array_operand_rank_inferred():
    v = vec_basis(dim=3, nF=1)
    a = jnp.array([1.0, 2.0, 3.0])
    x = jnp.array([[0.5, -1.0, 0.0]])
    out = StateExpr.einsum("i,i->", v, a)(x)
    ref = jnp.dot(x.squeeze(), a)[None, None]  # keep batch and features
    assert out.shape == (1, 1)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
