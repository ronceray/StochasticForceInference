# TODO: review this file
import jax
import jax.numpy as jnp
import numpy as np

from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.nodes.interactions import AutoPairs, PairsCSR


def _pair_dx_interactor(d=2):
    # f(Xk) = Xj - Xi  → vector rank, single feature (feature last)
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]  # (*rank=(d,), m=1)

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


def _build_complete_pairs_csr(P, exclude_self=True, symmetric=False):
    # Builds CSR for a complete directed (or i<j) graph on P nodes.
    if symmetric:
        # directed complete graph (optionally excluding self)
        if exclude_self:
            deg = jnp.full((P,), P - 1, dtype=jnp.int32)
            indptr = jnp.pad(jnp.cumsum(deg), (1, 0))
            indices = jnp.concatenate(
                [
                    jnp.concatenate([jnp.arange(i), jnp.arange(i + 1, P)])
                    for i in range(P)
                ],
                axis=0,
            ).astype(jnp.int32)
        else:
            deg = jnp.full((P,), P, dtype=jnp.int32)
            indptr = jnp.pad(jnp.cumsum(deg), (1, 0))
            indices = jnp.tile(jnp.arange(P, dtype=jnp.int32), (P,)).reshape(-1)
        return PairsCSR(indptr=indptr, indices=indices)
    # asymmetric: keep only i<j as neighbors of i
    rows = []
    cols = []
    for i in range(P):
        for j in range(i + 1, P):
            rows.append(i)
            cols.append(j)
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    deg = jnp.bincount(rows, length=P, minlength=P).astype(jnp.int32)
    indptr = jnp.pad(jnp.cumsum(deg), (1, 0))
    order = jnp.argsort(rows, stable=True)
    indices = cols[order]
    return PairsCSR(indptr=indptr, indices=indices)


def test_focal_sum_agrees_with_naive():
    P, d = 4, 2
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    inter = _pair_dx_interactor(d)
    spec = _build_complete_pairs_csr(P, exclude_self=True, symmetric=False)

    expr = inter.dispatch(
        spec, owners="focal", reducer="sum", exclude_self=True, return_as="basis"
    )
    y = expr(x)  # (..., P, d, 1)
    assert y.shape == (P, d, 1)

    y_naive = jnp.zeros((P, d))
    for i in range(P):
        for j in range(i + 1, P):
            y_naive = y_naive.at[i].add(x[j] - x[i])
    assert jnp.allclose(y[..., 0], y_naive)


def test_all_owners_scatter_to_both_endpoints():
    P, d = 3, 2
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inter = _pair_dx_interactor(d)
    spec = _build_complete_pairs_csr(P, exclude_self=True, symmetric=False)

    expr = inter.dispatch(spec, owners="all", reducer="sum", return_as="basis")
    y = expr(x)

    y_naive = jnp.zeros((P, d))
    for i in range(P):
        for j in range(i + 1, P):
            dx = x[j] - x[i]
            y_naive = y_naive.at[i].add(dx)
            y_naive = y_naive.at[j].add(dx)
    assert jnp.allclose(y[..., 0], y_naive)


def test_custom_antisymmetric_weights():
    P, d = 3, 2
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inter = _pair_dx_interactor(d)
    spec = _build_complete_pairs_csr(P, exclude_self=True, symmetric=False)

    expr = inter.dispatch(
        spec,
        owners="custom",
        owner_weights=jnp.array([+1.0, -1.0]),
        reducer="sum",
        return_as="basis",
    )
    y = expr(x)

    y_naive = jnp.zeros((P, d))
    for i in range(P):
        for j in range(i + 1, P):
            dx = x[j] - x[i]
            y_naive = y_naive.at[i].add(+dx)
            y_naive = y_naive.at[j].add(-dx)
    assert jnp.allclose(y[..., 0], y_naive)


def test_global_sum_equals_sum_of_local_contributions():
    P, d = 4, 2
    x = jnp.arange(P * d, dtype=jnp.float32).reshape(P, d)
    inter = _pair_dx_interactor(d)
    spec = _build_complete_pairs_csr(P, exclude_self=True, symmetric=False)

    expr = inter.dispatch(spec, owners="global", reducer="sum", return_as="basis")
    y_global = expr(x)  # (..., d, 1)
    assert y_global.shape == (d, 1)

    acc = jnp.zeros((d,))
    for i in range(P):
        for j in range(i + 1, P):
            acc = acc + (x[j] - x[i])
    assert jnp.allclose(y_global[..., 0], acc)


def test_mean_with_degree_normalization_focal():
    P, d = 4, 2
    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    inter = _pair_dx_interactor(d)

    expr = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="mean",
        normalize_by_degree=True,
        return_as="basis",
    )
    y = expr(x)

    y_naive = jnp.zeros((P, d))
    deg = np.zeros((P,), dtype=int)
    for i in range(P):
        for j in range(i + 1, P):
            deg[i] += 1
            y_naive = y_naive.at[i].add(x[j] - x[i])
    for i in range(P):
        if deg[i] > 0:
            y_naive = y_naive.at[i].set(y_naive[i] / deg[i])

    assert jnp.allclose(y[..., 0], y_naive)


def test_chunking_keeps_results_identical():
    P, d = 10, 2
    key = jax.random.key(0)
    x = jax.random.normal(key, (P, d))
    inter = _pair_dx_interactor(d)

    expr_big = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=None,
        return_as="basis",
    )
    expr_sml = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=7,
        return_as="basis",
    )
    y_big = expr_big(x)
    y_sml = expr_sml(x)
    assert jnp.allclose(y_big, y_sml)
