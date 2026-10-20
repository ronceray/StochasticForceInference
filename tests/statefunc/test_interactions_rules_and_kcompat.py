# TODO: review this file
import jax.numpy as jnp
import pytest

from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.nodes.interactions import AutoPairs, FromExtrasPairsCSR


def _interactor_K2(d=2):
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]  # vector rank, one feature

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


def _interactor_K3(d=2):
    def f(Xk, *, extras=None):
        # simple: sum over participants (for shape only)
        return Xk.sum(axis=0)[..., None]

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=3)


def test_fixedK_mismatch_between_interactor_and_spec():
    # Interactor K=3 with a K=2 spec must fail at dispatch construction
    inter3 = _interactor_K3(d=2)
    with pytest.raises(ValueError, match="Fixed-K mismatch"):
        _ = inter3.dispatch(AutoPairs(), return_as="basis")


def test_from_extras_pairs_requires_keys():
    inter = _interactor_K2(d=2)
    expr = inter.dispatch(FromExtrasPairsCSR("indptr", "indices"), return_as="basis")
    x = jnp.zeros((3, 2))
    # Missing keys → KeyError during build() at call time
    with pytest.raises(KeyError):
        _ = expr(x, extras={})


def test_from_extras_pairs_happy_path():
    inter = _interactor_K2(d=2)

    # Tiny CSR for P=3 with i<j only
    P = 3
    rows = jnp.array([0, 0, 1], dtype=jnp.int32)
    cols = jnp.array([1, 2, 2], dtype=jnp.int32)
    deg = jnp.bincount(rows, length=P, minlength=P).astype(jnp.int32)
    indptr = jnp.pad(jnp.cumsum(deg), (1, 0))
    order = jnp.argsort(rows, stable=True)
    indices = cols[order]

    expr = inter.dispatch(FromExtrasPairsCSR("indptr", "indices"), return_as="basis")
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = expr(x, extras={"indptr": indptr, "indices": indices})
    assert y.shape == (P, 2, 1)


def _interactor_unit_scalar(d=2):
    # feature is ||Xj - Xi|| (scalar), m=1
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        r = jnp.linalg.norm(Xj - Xi)
        return jnp.array([r])  # (1,) → scalar rank, feature last

    return make_interactor(f, dim=d, rank=Rank.SCALAR, K=2)


def test_autopairs_symmetric_vs_asymmetric_global_sum():
    P, d = 4, 2
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    inter = _interactor_unit_scalar(d)

    # Asymmetric (i<j): each unordered edge once
    expr_asym = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="global",
        reducer="sum",
        return_as="basis",
    )
    y_asym = expr_asym(x)[..., 0]

    # Symmetric (i!=j directed): each unordered edge twice
    expr_sym = inter.dispatch(
        AutoPairs(symmetric=True, exclude_self=True),
        owners="global",
        reducer="sum",
        return_as="basis",
    )
    y_sym = expr_sym(x)[..., 0]

    assert jnp.allclose(y_sym, 2.0 * y_asym)
