"""Chunk planner sizes time blocks from XLA's buffer assignment of the compiled kernel.

Static node hints count outputs and gathers only; they cannot see a kernel's internal working
set (here a large matrix product inside a basis function) nor the tangent batches of
forward-mode Jacobians, and under-estimated same-particle Jacobians of interaction bases ~100x.
The planner now compiles the kernel the runtime will execute and reads its memory analysis.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import SFI.integrate.api as api
from SFI.integrate import Integrand, Term, stream
from SFI.integrate.integrand import ExprOperand
from SFI.statefunc import make_basis
from SFI.trajectory import TrajectoryCollection

N_INT = 1200  # internal (N_INT, N_INT) float32 matrices: ~5.8 MB each, invisible to static hints


def _heavy_basis():
    def f(x):
        a = jnp.sin(x[0] * jnp.arange(N_INT, dtype=x.dtype))
        b = jnp.cos(x[1] * jnp.arange(N_INT, dtype=x.dtype))
        A = jnp.outer(a, b)
        feats = jnp.stack([jnp.trace(A @ A.T), jnp.sum(A @ A)]) * 1e-6
        return jnp.stack([feats, -feats])  # (dim, n_features)

    return make_basis(f, dim=2, rank=1, n_features=2)


def _program(B):
    return Integrand(exprs=[ExprOperand(B, x=stream("X"), alias="B")], terms=[Term(eq="idf->if", ops=("B",))])


def _collection(T=200):
    rng = np.random.default_rng(0)
    return TrajectoryCollection.from_arrays(X=rng.standard_normal((T, 2)).astype(np.float32), dt=0.1)


@pytest.fixture
def plans(monkeypatch):
    seen = []
    orig = api._build_plan

    def spy(*a, **k):
        plan = orig(*a, **k)
        seen.append(plan)
        return plan

    monkeypatch.setattr(api, "_build_plan", spy)
    return seen


def test_measured_blocks_respect_the_budget(plans):
    coll, prog = _collection(), _program(_heavy_basis())
    budget = 50 * 1024**2
    static = prog.estimate_bytes_per_sample(coll.peek_row(require=set(prog.require()) | {"__dt__"}))
    assert static * 200 < budget  # static sizing would put every row in one block
    val = api.integrate(coll, prog, reduce="sum", chunk_target_bytes=budget)
    plan = plans[-1]
    assert plan.bytes_hint > 20 * static
    assert plan.K_fixed * plan.bytes_hint <= budget + plan.bytes_hint
    assert 1 <= plan.K_fixed < 200
    # chunking does not change the result
    ref = api.integrate(coll, prog, reduce="sum", chunk_target_bytes=budget, measure_memory=False)
    assert plans[-1].K_fixed == 199  # all valid rows in one block
    np.testing.assert_allclose(np.asarray(val), np.asarray(ref), rtol=1e-4)


def test_falls_back_to_static_hint_without_memory_analysis(plans, monkeypatch):
    monkeypatch.setattr(api, "_xla_bytes", lambda fn, *args: None)
    coll, prog = _collection(), _program(_heavy_basis())
    api.integrate(coll, prog, reduce="sum", chunk_target_bytes=50 * 1024**2)
    assert plans[-1].K_fixed == 199


def test_explicit_bytes_per_sample_wins():
    coll, prog = _collection(), _program(_heavy_basis())
    plan, run = api.make_parametric_integrator(coll, prog, chunk_target_bytes=1000, bytes_per_sample=100,
                                               theta_probe={})
    assert plan.K_fixed == 10 and plan.bytes_hint == 100
