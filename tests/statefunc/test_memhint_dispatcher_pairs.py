# TODO: review this file
from __future__ import annotations

import jax.numpy as jnp
import pytest

from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.memhint import SampleMeta
from SFI.statefunc.nodes.interactions import AutoPairs


def _pair_dx_interactor(d=2):
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]  # vector, m=1

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


@pytest.mark.parametrize("P,dim,chunk", [(8, 2, 5), (10, 3, 4)])
def test_dispatcher_hint_components_focal(P, dim, chunk):
    inter = _pair_dx_interactor(dim)
    expr = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=chunk,
        return_as="basis",
    )

    isz = jnp.dtype(jnp.float32).itemsize
    isz_i32 = jnp.dtype(jnp.int32).itemsize
    isz_bool = jnp.dtype(jnp.bool_).itemsize

    # M_est for asymmetric pairs with exclude_self reduces to P*(P-1)/2 unordered, but
    # dispatcher works on directed edges when symmetric=False? We sized chunk directly,
    # so per-sample scales with `chunk`.
    K = 2
    r = 1
    m = 1
    rb = dim**r
    out_per_edge = rb * m * isz

    gather_per_edge = K * dim * isz + (2 * isz_i32 + 2 * isz_bool)
    acc_bytes = P * rb * m * isz + P * isz_i32

    expect = chunk * (gather_per_edge + out_per_edge) + acc_bytes

    got = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P))
    assert got == expect


@pytest.mark.parametrize("P,dim,chunk", [(8, 2, 5), (10, 3, 4)])
def test_dispatcher_hint_components_global(P, dim, chunk):
    inter = _pair_dx_interactor(dim)
    expr = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="global",
        reducer="sum",
        chunk_size=chunk,
        return_as="basis",
    )

    isz = jnp.dtype(jnp.float32).itemsize
    isz_i32 = jnp.dtype(jnp.int32).itemsize
    isz_bool = jnp.dtype(jnp.bool_).itemsize

    K = 2
    r = 1
    m = 1
    rb = dim**r
    out_per_edge = rb * m * isz
    gather_per_edge = K * dim * isz + (2 * isz_i32 + 2 * isz_bool)
    acc_bytes = rb * m * isz  # global accumulator has no P

    expect = chunk * (gather_per_edge + out_per_edge) + acc_bytes

    got = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P))
    assert got == expect


def test_chunking_reduces_per_sample_bytes():
    P, dim = 12, 2
    inter = _pair_dx_interactor(dim)
    # Make two expressions that differ only by chunk_size
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
        chunk_size=4,
        return_as="basis",
    )

    # With chunk_size=None the estimator assumes unchunked pass and a larger working set.
    hb_big = expr_big.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P)
    )
    hb_sml = expr_sml.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P)
    )
    assert hb_big > hb_sml
