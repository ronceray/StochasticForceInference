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


def test_slice_with_numpy_and_jax_index_arrays_is_warning_free():
    """Regression: ``B[np.asarray(inf.force_support)]`` stored the index array as a
    static field ("A JAX array is being set as static"); a NumPy integer scalar
    crashed (no len()); a NumPy boolean mask was read as integer indices."""
    import warnings

    import numpy as np

    from SFI.bases import monomials_up_to

    B = monomials_up_to(2, dim=2, rank="vector")
    x = jnp.array([[0.3, -0.7], [1.1, 0.2]])
    ref = B[[0, 3, 5]]
    y_ref = ref(x)
    mask = np.zeros(B.n_features, dtype=bool)
    mask[[0, 3, 5]] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for idx in (np.asarray([0, 3, 5]), jnp.asarray([0, 3, 5]), (0, 3, 5), mask, list(mask)):
            Bs = B[idx]
            assert Bs.n_features == 3
            assert list(Bs.labels) == list(ref.labels)
            assert jnp.allclose(Bs(x), y_ref)
            assert Bs.root.idx == (0, 3, 5) and all(type(i) is int for i in Bs.root.idx)
        B1 = B[np.int64(3)]
        assert B1.n_features == 1 and jnp.allclose(B1(x)[..., 0], B(x)[..., 3])
