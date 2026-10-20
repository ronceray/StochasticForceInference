# TODO: review this file
from __future__ import annotations

import jax.numpy as jnp
import pytest

from SFI.bases.linear import X
from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.memhint import SampleMeta
from SFI.statefunc.nodes.interactions import AutoPairs


def _pair_dx_interactor(d=2):
    def f(Xk, *, extras=None):
        Xi, Xj = Xk[0], Xk[1]
        return (Xj - Xi)[..., None]  # vector, m=1 (feature last)

    return make_interactor(f, dim=d, rank=Rank.VECTOR, K=2)


@pytest.mark.parametrize("P,dim,chunk", [(5, 2, 3), (9, 3, 4)])
def test_broadcast_overhead_when_mixing_particle_and_global(P, dim, chunk):
    # interactor (owners='focal') has pdepth=1. X(dim) has pdepth=0.
    inter = _pair_dx_interactor(dim)
    expr_inter = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=chunk,
        return_as="basis",
    )  # output per sample: (P, dim, 1)

    expr_global = X(dim)  # output per sample: (dim, 1), pdepth=0

    # Compose with addition: children have different pdepth → broadcast overhead applies.
    expr = expr_inter + expr_global

    isz = jnp.dtype(jnp.float32).itemsize
    # child bytes
    bytes_inter = expr_inter.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P)
    )
    bytes_global = expr_global.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P)
    )
    # output bytes of the composite op equals the "deeper" pdepth child, i.e. interactor
    out_bytes = (dim**1) * 1 * P * isz

    # broadcast overhead: global child (pdepth=0) to pdepth=1 costs (P^1 - 1) times its output slab
    b_over = (P - 1) * ((dim**1) * 1 * isz)

    expect = bytes_inter + bytes_global + b_over + out_bytes
    got = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P))
    assert got == expect


def test_no_broadcast_overhead_when_pdepth_equal():
    P, dim = 7, 3
    inter = _pair_dx_interactor(dim)
    expr1 = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=3,
        return_as="basis",
    )
    expr2 = inter.dispatch(
        AutoPairs(symmetric=False, exclude_self=True),
        owners="focal",
        reducer="sum",
        chunk_size=4,
        return_as="basis",
    )
    expr = expr1 + expr2
    got = expr.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P))
    # Manual expected: sum(children) + output (no broadcast term since both pdepth=1)
    isz = jnp.dtype(jnp.float32).itemsize
    child_bytes = expr1.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=P)
    ) + expr2.estimate_bytes_per_sample(dtype=jnp.float32, sample=SampleMeta(P=P))
    out_bytes = (dim**1) * 1 * P * isz
    expect = child_bytes + out_bytes
    assert got == expect
