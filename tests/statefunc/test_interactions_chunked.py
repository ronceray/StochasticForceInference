# TODO: review this file
import jax
import jax.numpy as jnp

from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.nodes.interactions import AutoPairs, PairsCSR


def _pair_dx_interactor(d=2):
    # f(Xk) = Xj - Xi  → vector rank, single feature (feature last)
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


def _self_only_pairs_csr(P):
    # Each i has only itself as neighbor; with exclude_self=True → no kept edges.
    indptr = jnp.arange(P + 1, dtype=jnp.int32)  # deg = 1 each
    indices = jnp.arange(P, dtype=jnp.int32)  # neighbor[i] = i
    return PairsCSR(indptr=indptr, indices=indices)


def test_chunked_equals_unchunked_focal_sum():
    P, d = 12, 2
    key = jax.random.key(0)
    x = jax.random.normal(key, (P, d))
    inter = _pair_dx_interactor(d)

    spec = AutoPairs(symmetric=False, exclude_self=True)
    expr_big = inter.dispatch(
        spec, owners="focal", reducer="sum", chunk_size=None, return_as="basis"
    )
    expr_sml = inter.dispatch(
        spec, owners="focal", reducer="sum", chunk_size=5, return_as="basis"
    )

    y_big = expr_big(x)
    y_sml = expr_sml(x)
    assert jnp.allclose(y_big, y_sml)


def test_chunked_global_mean_matches_unchunked():
    P, d = 10, 2
    key = jax.random.key(1)
    x = jax.random.normal(key, (P, d))
    inter = _pair_dx_interactor(d)

    spec = AutoPairs(symmetric=False, exclude_self=True)
    expr_big = inter.dispatch(
        spec, owners="global", reducer="mean", chunk_size=None, return_as="basis"
    )
    expr_sml = inter.dispatch(
        spec, owners="global", reducer="mean", chunk_size=3, return_as="basis"
    )

    y_big = expr_big(x)
    y_sml = expr_sml(x)
    assert jnp.allclose(y_big, y_sml)


def test_chunked_degree_normalization_focal_mean():
    P, d = 9, 2
    # Make x simple but not trivial
    x = jnp.array([[i, i % 3] for i in range(P)], dtype=jnp.float32)
    inter = _pair_dx_interactor(d)

    spec = AutoPairs(symmetric=False, exclude_self=True)
    expr_big = inter.dispatch(
        spec,
        owners="focal",
        reducer="mean",
        normalize_by_degree=True,
        chunk_size=None,
        return_as="basis",
    )
    expr_sml = inter.dispatch(
        spec,
        owners="focal",
        reducer="mean",
        normalize_by_degree=True,
        chunk_size=4,
        return_as="basis",
    )

    y_big = expr_big(x)
    y_sml = expr_sml(x)
    assert jnp.allclose(y_big, y_sml)


def test_all_edges_filtered_return_zero():
    P, d = 5, 2
    x = jnp.zeros((P, d))
    inter = _pair_dx_interactor(d)

    spec = _self_only_pairs_csr(P)
    # exclude_self=True means all contributions are masked out
    expr_focal = inter.dispatch(
        spec,
        owners="focal",
        reducer="sum",
        exclude_self=True,
        chunk_size=3,
        return_as="basis",
    )
    expr_glob = inter.dispatch(
        spec,
        owners="global",
        reducer="sum",
        exclude_self=True,
        chunk_size=3,
        return_as="basis",
    )

    yf = expr_focal(x)
    yg = expr_glob(x)

    assert yf.shape == (P, d, 1)
    assert yg.shape == (d, 1)
    assert jnp.allclose(yf, 0.0)
    assert jnp.allclose(yg, 0.0)
