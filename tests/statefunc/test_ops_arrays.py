# TODO: review this file
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.statefunc import Rank, StateExpr, make_basis


def _vec_expr(dim, nF=1):
    # f(x)=x, vector-valued, nF features copied
    def f(x, *, extras):
        return jnp.repeat(x[..., None], nF, axis=-1)

    return make_basis(f, dim=dim, rank=Rank.VECTOR, n_features=nF)


def _scal_expr(dim, nF=1):
    return make_basis(
        lambda x, **_: jnp.ones((nF,)), dim=dim, rank=Rank.SCALAR, n_features=nF
    )


def _mat_from_outer(B):
    # rank-2 spatial tensor from outer product
    return StateExpr.einsum("i,j->ij", B, B)


# ---------- matmul semantics (true matrix multiply) ----------


def test_matmul_vector_vector_is_dot():
    d = 4
    B = _vec_expr(d, nF=1)
    x = jnp.arange(1.0, d + 1.0)
    y = (B @ B)(x).squeeze()  # scalar per feature
    ref = jnp.dot(x, x)
    assert y.shape == ()
    assert np.allclose(y, ref)


def test_matmul_matrix_vector_and_vector_matrix():
    d = 3
    B = _vec_expr(d, nF=1)
    M = _mat_from_outer(B)  # (..., d, d, 1)
    x = jnp.array([0.5, -1.0, 2.0])

    # M @ B -> vector
    y1 = (M @ B)(x).squeeze(-1)  # (..., d)
    ref1 = jnp.matmul(jnp.outer(x, x), x)
    np.testing.assert_allclose(y1, ref1, rtol=1e-6, atol=1e-6)

    # B @ M -> vector
    y2 = (B @ M)(x).squeeze(-1)
    ref2 = jnp.matmul(x, jnp.outer(x, x))
    np.testing.assert_allclose(y2, ref2, rtol=1e-6, atol=1e-6)


def test_matmul_matrix_matrix():
    d = 3
    B = _vec_expr(d, nF=1)
    M = _mat_from_outer(B)
    x = jnp.array([0.2, -0.3, 0.7])
    y = (M @ M)(x).squeeze(-1)
    ref = jnp.matmul(jnp.outer(x, x), jnp.outer(x, x))
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


def test_matmul_array_left_and_right():
    d = 3
    B = _vec_expr(d, nF=1)
    A = jnp.arange(1.0, d + 1.0)  # (d,)
    M = jnp.arange(1.0, d * d + 1.0).reshape(d, d)

    x = jnp.linspace(0.5, 1.0, d)

    # array @ B -> dot
    y1 = (A @ B)(x).squeeze()  # ()
    np.testing.assert_allclose(y1, jnp.dot(A, x), rtol=1e-6)

    # B @ array -> dot
    y2 = (B @ A)(x).squeeze()
    np.testing.assert_allclose(y2, jnp.dot(x, A), rtol=1e-6)

    # matrix @ B -> vector
    y3 = (M @ B)(x).squeeze(-1)
    np.testing.assert_allclose(y3, M @ x, rtol=1e-6)

    # B @ matrix -> vector
    y4 = (B @ M)(x).squeeze(-1)
    np.testing.assert_allclose(y4, x @ M, rtol=1e-6)


def test_matmul_invalid_with_scalar_raises():
    d = 3
    v = _vec_expr(d, nF=1)
    s = _scal_expr(d, nF=1)
    with pytest.raises(TypeError):
        _ = v @ s
    with pytest.raises(TypeError):
        _ = s @ v


# ---------- einsum with arrays and invariances ----------


@pytest.mark.parametrize("d", [2, 3])
def test_einsum_array_mix_outer(d):
    B = _vec_expr(d, nF=1)
    A = jnp.arange(1, d + 1.0, dtype=jnp.float32)
    x = jnp.linspace(0.5, 1.0, d, dtype=jnp.float32)

    E1 = StateExpr.einsum("i,j->ij", B, A)(x).squeeze(-1)
    E2 = StateExpr.einsum("j,i->ji", A, B)(x).squeeze(-1)
    ref = jnp.einsum("i,j->ij", x, A)
    np.testing.assert_allclose(E1, ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(E2, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("d", [3])
def test_einsum_array_mix_contraction(d):
    B = _vec_expr(d, nF=1)
    M = _mat_from_outer(B)
    a = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    x = jnp.array([0.2, -0.3, 0.7], dtype=jnp.float32)
    y = StateExpr.einsum("ij,j->i", M, a)(x).squeeze(-1)
    ref = jnp.einsum("ij,j->i", jnp.einsum("i,j->ij", x, x), a)
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


# ---------- dot / tensordot semantics ----------


def test_dot_defaults_and_axes():
    d = 4
    B = _vec_expr(d, nF=1)
    x = jnp.arange(1.0, d + 1.0, dtype=jnp.float32)
    a = jnp.array([2.0, -1.0, 0.5, 3.0], dtype=jnp.float32)

    # default: last(self) with first(other)
    s1 = B.dot(a)(x).squeeze(-1)
    np.testing.assert_allclose(s1, jnp.dot(x, a), rtol=1e-6, atol=1e-6)

    T = _mat_from_outer(B)
    # Frobenius inner product requires contracting 2 axes for rank-2
    s2 = T.dot(jnp.eye(d, dtype=jnp.float32), axes=2)(x).squeeze(-1)
    ref2 = jnp.einsum("ij,ij->", jnp.einsum("i,j->ij", x, x), jnp.eye(d))
    np.testing.assert_allclose(s2, ref2, rtol=1e-6, atol=1e-6)


def test_tensordot_tuple_axes_and_partial_contractions():
    d = 3
    B = _vec_expr(d, nF=1)
    M = _mat_from_outer(B)
    x = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    A = jnp.arange(1.0, d + 1.0).astype(jnp.float32)

    # vector · vector
    out1 = B.tensordot(A, axes=1)(x).squeeze(-1)
    ref1 = jnp.tensordot(x, A, axes=1)
    np.testing.assert_allclose(out1, ref1, rtol=1e-6, atol=1e-6)

    # contract first axis of M with first axis of identity -> leaves last axis
    out2 = M.dot(jnp.eye(d, dtype=jnp.float32), axes=((0,), (0,)))(x).squeeze(-1)
    ref2 = jnp.tensordot(jnp.outer(x, x), jnp.eye(d), axes=((0,), (0,)))
    np.testing.assert_allclose(out2, ref2, rtol=1e-6, atol=1e-6)


# ---------- errors ----------


def test_array_shape_mismatch_raises():
    d = 3
    B = _vec_expr(d, nF=1)
    bad = jnp.ones((d + 1,), dtype=jnp.float32)
    with pytest.raises(TypeError):
        _ = B @ bad
