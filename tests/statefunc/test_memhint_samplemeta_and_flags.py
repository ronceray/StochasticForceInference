# TODO: review this file
from __future__ import annotations

import jax.numpy as jnp

from SFI.bases.linear import X
from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.memhint import SampleMeta
from SFI.statefunc.nodes.interactions import AutoPairs


def _pair_dx_interactor(d=2):
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]  # vector, m=1

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


def test_samplemeta_P_scales_broadcast_cost():
    dim = 3
    inter = _pair_dx_interactor(dim)
    expr_inter = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=3,
        return_as="basis",
    )
    expr_global = X(dim)
    expr = expr_inter + expr_global

    isz = jnp.dtype(jnp.float32).itemsize

    P1, P2 = 4, 9
    b1 = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P1))
    b2 = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P2))
    # broadcast overhead grows with P, so total must grow too
    assert b2 > b1

    # The total linear-in-P slope has 3 pieces:
    #   broadcast(pdepth 0→1) + op's output buffer (pdepth=1) + focal accumulator
    isz_i32 = jnp.dtype(jnp.int32).itemsize
    rb = dim**1
    broadcast_slab = rb * 1 * isz  # lifts the global child
    op_output_slab = rb * 1 * isz  # output of the add node
    acc_slope = rb * 1 * isz + isz_i32  # owners='focal' accumulator

    expected_delta = (P2 - P1) * (broadcast_slab + op_output_slab + acc_slope)
    assert (b2 - b1) == expected_delta


def test_dispatcher_mask_flag_increases_gather_bytes():
    P, dim = 6, 2
    inter = _pair_dx_interactor(dim)
    expr_no_mask = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=3,
        return_as="basis",
    )
    expr_with_mask = expr_no_mask  # same structure; we only toggle SampleMeta flag

    b0 = expr_no_mask.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P, has_mask=False)
    )
    b1 = expr_with_mask.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P, has_mask=True)
    )
    assert b1 > b0

    # Difference should be chunk_E * K * sizeof(bool)
    isz_bool = jnp.dtype(jnp.bool_).itemsize
    K = 2
    chunk = 3
    expect_delta = chunk * K * isz_bool
    assert (b1 - b0) == expect_delta
