"""Tangent operands (directional derivatives) and per-row random keys in ``integrate``."""
import jax
import jax.numpy as jnp
import numpy as np

from SFI import TrajectoryCollection
from SFI.integrate import integrate, stream
from SFI.integrate.integrand import ExprOperand, Integrand, Term, TimeOperand
from SFI.integrate.timeops import normal_probe
from SFI.statefunc import make_basis, make_interactor

N, d, T = 6, 2, 30


def _setup():
    nb = np.array([[(i - 1) % N, (i + 1) % N] for i in range(N)], np.int32).reshape(-1)
    ex = {"r/indptr": jnp.asarray(np.arange(0, 2 * N + 1, 2, dtype=np.int32)), "r/indices": jnp.asarray(nb)}
    coup = make_interactor(lambda Xk: jnp.tanh(Xk[1] - Xk[0])[:, None] * (1 + Xk[0][0] ** 2), dim=d, rank=1, K=2,
                           n_features=1)
    B = make_basis(lambda x: jnp.sin(x)[:, None], dim=d, rank=1, n_features=1) & coup.dispatch_pairs_from_extras(
        indptr_key="r/indptr", indices_key="r/indices")
    X = np.random.default_rng(0).standard_normal((T, N, d)).astype(np.float32)
    return B, ex, X, TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=ex)


def _jac(B, ex, x):
    return np.asarray(jax.jacfwd(lambda xx: B(xx, extras=ex))(jnp.asarray(x)))  # (i, o, a, j, m)


def test_tangent_operand_is_the_jacobian_contraction():
    B, ex, X, coll = _setup()
    prog = Integrand(exprs=[ExprOperand(B, x=stream("X"), tx=stream("dX"), alias="J")],
                     times=[TimeOperand(stream("dX"), alias="u")], terms=[Term(eq="ina,in->ia", ops=("J", "u"))])
    ref = sum(np.einsum("inajm,jm,in->a", _jac(B, ex, X[t]), X[t + 1] - X[t], X[t + 1] - X[t]) for t in range(T - 1))
    for batch in (True, False):
        val = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, batch=batch))
        np.testing.assert_allclose(val, ref, rtol=1e-5)


def test_velocity_tangent():
    vb = make_basis(lambda x, v: (x * v ** 2)[:, None], dim=d, rank=1, n_features=1, needs_v=True)
    _, _, X, coll = _setup()
    from SFI.integrate.timeops import velocity

    prog = Integrand(exprs=[ExprOperand(vb, x=stream("X"), v=velocity("dX", "dt"), tv=stream("X"), alias="J")],
                     terms=[Term(eq="ina->ia", ops=("J",))])
    val = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False))
    v = (X[1:] - X[:-1]) / 0.1
    ref = np.sum(X[:-1] * 2 * v * X[:-1])
    np.testing.assert_allclose(val.sum(), ref, rtol=1e-4)


def test_seeded_probes_are_reproducible_chunk_invariant_and_unbiased():
    B, ex, X, coll = _setup()
    prog = Integrand(exprs=[ExprOperand(B, x=stream("X"), tx=normal_probe("X"), alias="Jz")],
                     times=[TimeOperand(normal_probe("X"), alias="z")], terms=[Term(eq="ina,in->ia", ops=("Jz", "z"))])
    a = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=3))
    b = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=3, chunk_target_bytes=20000))
    c = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=4))
    e = np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=3, batch=False))
    np.testing.assert_allclose(a, b, rtol=1e-5)
    np.testing.assert_allclose(a, e, rtol=1e-5)
    assert np.abs(a - c).max() > 1e-3
    # Hutchinson: E[z . J z] = tr J, averaged over many seeds
    tr = sum(np.einsum("inain->a", _jac(B, ex, X[t])) for t in range(T - 1))
    est = np.mean([np.asarray(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=s)) for s in range(40)], 0)
    np.testing.assert_allclose(est, tr, rtol=0.05)


def test_independent_probe_indices():
    _, _, _, coll = _setup()
    prog = Integrand(times=[TimeOperand(normal_probe("X", index=0), alias="z0"),
                            TimeOperand(normal_probe("X", index=1), alias="z1")],
                     terms=[Term(eq="in,in->i", ops=("z0", "z1"))])
    val = float(np.sum(integrate(coll, prog, reduce="sum", weight_by_dt=False, seed=0)))
    n = (T - 1) * N * d
    assert abs(val) < 4 * np.sqrt(n)  # independent: mean 0, sd sqrt(n)
