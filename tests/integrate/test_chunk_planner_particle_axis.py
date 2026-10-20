"""The chunk planner must count the broadcast particle axis of pdepth=0 expressions.

Regression for the stacked-layout blow-up: ``X`` of shape ``(T, N, d)`` with a
``pdepth=0`` basis used to be estimated as if ``N == 1``, so the per-row byte
hint was ``N`` times too small, chunking was effectively disabled, and peak
memory scaled as ``T * N * F**2`` regardless of the memory budget.  The same
data split into ``N`` single-particle datasets stayed within budget.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.integrate.api import _build_plan, integrate
from SFI.integrate.integrand import ExprOperand, Integrand, Term
from SFI.integrate.timeops import stream
from SFI.statefunc.factory import make_basis
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


def _collection(T, N, d=2, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(scale=np.sqrt(dt), size=(T, N, d)), axis=0).astype(np.float32)
    ds = TrajectoryDataset.from_arrays(X=jnp.asarray(X), dt=dt)
    return TrajectoryCollection.from_dataset(ds)


def _vec_basis(d, nF):
    """pdepth=0 vector features, output (..., d, nF)."""

    def f(x, *, v=None, **kw):
        return jnp.stack([(k + 1) * x for k in range(nF)], axis=-1)

    return make_basis(f, dim=d, rank=1, n_features=nF)


def _gram_program(basis):
    BL = ExprOperand(expr=basis, x=stream("X"), alias="BL")
    BR = ExprOperand(expr=basis, x=stream("X"), alias="BR")
    return Integrand(exprs=[BL, BR], terms=[Term(eq="ima,imb->iab", ops=("BL", "BR"))])


def _plan(coll, prog, budget):
    return _build_plan(
        coll,
        prog,
        reduce="sum",
        reduce_over_particles=True,
        weight_by_dt=True,
        subsampling=1,
        chunk_target_bytes=int(budget),
        context=None,
    )


@pytest.mark.parametrize("N", [1, 8, 64])
def test_bytes_hint_covers_broadcast_gram_block(N):
    """The hint must at least cover what the batched kernel materializes per row."""
    d, nF = 2, 6
    prog = _gram_program(_vec_basis(d, nF))
    plan = _plan(_collection(T=16, N=N, d=d), prog, budget=1 << 30)

    itemsize = 4  # float32 data
    gram_block = N * nF * nF * itemsize  # (N, F, F) einsum output before the particle sum
    state_block = 2 * N * d * nF * itemsize  # BL and BR outputs, (N, d, F) each
    assert plan.bytes_hint is not None
    assert plan.bytes_hint >= gram_block + state_block


def test_bytes_hint_scales_with_particle_count():
    d, nF, T = 2, 6, 64
    prog = _gram_program(_vec_basis(d, nF))
    hint_1 = _plan(_collection(T, 1, d), prog, budget=1 << 30).bytes_hint
    hint_32 = _plan(_collection(T, 32, d), prog, budget=1 << 30).bytes_hint
    # Every per-row term (state buffers, einsum intermediate, node transients)
    # is linear in N for a pdepth=0 program.
    assert hint_32 >= 16 * hint_1


def test_stacked_layout_is_chunked_under_budget():
    """A budget sized for ~4 chunks of the true per-row cost must yield >= 4 chunks."""
    d, nF, T, N = 2, 6, 64, 32
    prog = _gram_program(_vec_basis(d, nF))
    hint_1 = _plan(_collection(T, 1, d), prog, budget=1 << 30).bytes_hint
    budget = N * hint_1 * T // 4  # what a correct N-aware estimate implies for 4 chunks

    plan = _plan(_collection(T, N, d), prog, budget=budget)
    assert plan.K_fixed is not None and plan.K_fixed <= T // 4
    assert len(plan.chunks) >= 4


def test_chunked_and_unchunked_stacked_gram_agree():
    """Splitting the stacked layout into chunks must not change the reduction."""
    d, nF, T, N = 2, 5, 40, 16
    coll = _collection(T, N, d)
    prog = _gram_program(_vec_basis(d, nF))

    one_chunk = integrate(coll, prog, reduce="sum", reduce_over_particles=True, chunk_target_bytes=1 << 30)
    hint = _plan(coll, prog, budget=1 << 30).bytes_hint
    many = integrate(coll, prog, reduce="sum", reduce_over_particles=True, chunk_target_bytes=int(hint * 7))
    assert len(_plan(coll, prog, budget=hint * 7).chunks) > 1
    np.testing.assert_allclose(np.asarray(many), np.asarray(one_chunk), rtol=1e-5, atol=1e-6)


def test_stacked_and_split_layouts_agree():
    """(T, N, d) stacked vs N single-particle datasets: identical Gram (up to summation order)."""
    d, nF, T, N = 2, 5, 40, 8
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.normal(scale=0.3, size=(T, N, d)), axis=0).astype(np.float32)
    dt = 0.1
    stacked = TrajectoryCollection.from_dataset(TrajectoryDataset.from_arrays(X=jnp.asarray(X), dt=dt))
    parts = [TrajectoryDataset.from_arrays(X=jnp.asarray(X[:, i]), dt=dt) for i in range(N)]
    split = TrajectoryCollection.from_dataset(parts[0]).concat(parts[1:])

    prog = _gram_program(_vec_basis(d, nF))
    g_stacked = integrate(stacked, prog, reduce="sum", reduce_over_particles=True)
    g_split = integrate(split, prog, reduce="sum", reduce_over_particles=True)
    np.testing.assert_allclose(np.asarray(g_stacked), np.asarray(g_split), rtol=1e-4, atol=1e-5)
