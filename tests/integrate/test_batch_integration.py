# TODO: review this file
# tests/integrate/test_batch_integration.py
"""
Compare batch=True and batch=False integration paths for numerical equivalence.

Every test calls integrate() twice (once with batch=False, once with batch=True)
and asserts the results match within floating-point tolerance.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.integrate.api import integrate, make_parametric_integrator
from SFI.integrate.integrand import (
    ConstOperand,
    ExprOperand,
    Integrand,
    Term,
    TimeOperand,
)
from SFI.integrate.timeops import stream, velocity
from SFI.statefunc.factory import make_basis
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


# ---------- helpers ----------

def _make_collection(T=12, N=5, d=2, dt=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dW = rng.normal(scale=np.sqrt(dt), size=(T, N, d)).astype(np.float32)
    X = np.cumsum(dW, axis=0).astype(np.float32)
    ds = TrajectoryDataset.from_arrays(X=jnp.array(X), dt=dt)
    return TrajectoryCollection.from_dataset(ds), jnp.array(X), dt


def _scalar_basis(dim, nF):
    """Scalar features: (..., nF)."""

    def f(x, **kw):
        base = x[..., 0]
        feats = [base + 0.1 * (k + 1) for k in range(nF)]
        return jnp.stack(feats, axis=-1)

    return make_basis(f, dim=dim, rank=0, n_features=nF)


def _vec_basis(dim, nF):
    """Vector features: (..., d, nF)."""

    def f(x, *, v=None, **kw):
        feats = [(k + 1) * x for k in range(nF)]
        return jnp.stack(feats, axis=-1)

    return make_basis(f, dim=dim, rank=1, n_features=nF)


# ---------- tests ----------


def test_scalar_gram_sum():
    """Scalar Gram G_ab = sum_t dt * sum_i BL_ia * BR_ib."""
    coll, X, dt = _make_collection(T=10, N=4, d=2, dt=0.25)
    d = X.shape[-1]
    nF = 3

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(coll, prog, reduce="sum", reduce_over_particles=False)
    out = integrate(coll, prog, reduce="sum", reduce_over_particles=False, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_scalar_gram_mean():
    """Mean reduction should also match."""
    coll, X, dt = _make_collection(T=10, N=4, d=2, dt=0.25)
    d = X.shape[-1]
    nF = 3

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(coll, prog, reduce="mean", reduce_over_particles=False)
    out = integrate(coll, prog, reduce="mean", reduce_over_particles=False, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_vector_with_velocity_and_constant():
    """V_im * A_mn * B_ina -> ia, with reduce_over_particles=True."""
    coll, X, dt = _make_collection(T=15, N=3, d=2, dt=0.1, seed=42)
    d = X.shape[-1]
    nF = 4

    B = ExprOperand(expr=_vec_basis(d, nF), x=stream("X"), alias="B")
    V = TimeOperand(velocity("dX", "dt"), alias="V")
    A = ConstOperand(jnp.eye(d), alias="A")

    prog = Integrand(
        exprs=[B],
        times=[V],
        consts=[A],
        terms=[Term(eq="im,mn,ina->ia", ops=("V", "A", "B"))],
    )

    ref = integrate(coll, prog, reduce="sum", reduce_over_particles=True)
    out = integrate(coll, prog, reduce="sum", reduce_over_particles=True, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_trapezoid_gram():
    """Shifted Gram: 0.5*(G_rect + G_shift)."""
    coll, X, dt = _make_collection(T=9, N=3, d=2, dt=0.25, seed=1)
    d = X.shape[-1]
    nF = 4

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR0 = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR0")
    BRP = ExprOperand(
        expr=_scalar_basis(d, nF), x=stream("X_plus"), alias="BRP"
    )

    G_rect = Integrand(
        exprs=[BL, BR0], terms=[Term(eq="ia,ib->ab", ops=("BL", "BR0"))]
    )
    G_shift = Integrand(
        exprs=[BL, BRP], terms=[Term(eq="ia,ib->ab", ops=("BL", "BRP"))]
    )
    G_trap = 0.5 * G_rect + 0.5 * G_shift

    ref = integrate(coll, G_trap, reduce="mean", reduce_over_particles=False)
    out = integrate(
        coll, G_trap, reduce="mean", reduce_over_particles=False, batch=True
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_variable_dt():
    """Variable-dt dataset."""
    T, N, d = 8, 4, 2
    rng = np.random.default_rng(1)
    t = np.cumsum(np.abs(rng.normal(size=T))).astype(np.float32)
    X = np.cumsum(
        np.random.default_rng(2).normal(size=(T, N, d)).astype(np.float32), axis=0
    )
    ds = TrajectoryDataset.from_arrays(X=X, t=t)
    coll = TrajectoryCollection.from_dataset(ds)

    nF = 3
    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(coll, prog, reduce="sum", reduce_over_particles=False)
    out = integrate(coll, prog, reduce="sum", reduce_over_particles=False, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_two_datasets():
    """Concatenation of two datasets with different N."""
    d = 2
    X1 = np.cumsum(
        np.random.default_rng(0).normal(size=(7, 3, d)).astype(np.float32), axis=0
    )
    X2 = np.cumsum(
        np.random.default_rng(3).normal(size=(5, 2, d)).astype(np.float32), axis=0
    )
    ds1 = TrajectoryDataset.from_arrays(X=X1, dt=0.2)
    ds2 = TrajectoryDataset.from_arrays(X=X2, dt=0.3)
    coll = TrajectoryCollection.from_dataset(ds1).concat([ds2])

    nF = 3
    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(coll, prog, reduce="sum", reduce_over_particles=False)
    out = integrate(coll, prog, reduce="sum", reduce_over_particles=False, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_chunking_batch():
    """Force small chunks to exercise multi-chunk batch path."""
    coll, X, dt = _make_collection(T=30, N=6, d=2, dt=0.1)
    d = X.shape[-1]
    nF = 4

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(
        coll, prog, reduce="sum", reduce_over_particles=False,
        chunk_target_bytes=128,
    )
    out = integrate(
        coll, prog, reduce="sum", reduce_over_particles=False,
        chunk_target_bytes=128, batch=True,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_particle_reduction_with_mask():
    """Particle reduction with a partially-masked dataset."""
    T, N, d = 10, 5, 2
    rng = np.random.default_rng(7)
    X = np.cumsum(
        rng.normal(size=(T, N, d)).astype(np.float32), axis=0
    )
    mask = rng.choice([True, False], size=(T, N), p=[0.8, 0.2])
    mask[0, :] = True  # ensure at least one fully-valid time step
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, mask=mask)
    coll = TrajectoryCollection.from_dataset(ds)

    nF = 3
    B = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="B")
    # Keep particle axis: "ia->ia" (identity), let kernel reduce over i
    prog = Integrand(
        exprs=[B],
        terms=[Term(eq="ia->ia", ops=("B",))],
    )

    ref = integrate(coll, prog, reduce="sum", reduce_over_particles=True)
    out = integrate(coll, prog, reduce="sum", reduce_over_particles=True, batch=True)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_parametric_integrator_batch():
    """make_parametric_integrator with batch=True."""
    coll, X, dt = _make_collection(T=10, N=3, d=2, dt=0.2, seed=5)
    d = X.shape[-1]
    nF = 3

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    _, run_ref = make_parametric_integrator(
        coll, prog, reduce="sum", reduce_over_particles=False
    )
    _, run_batch = make_parametric_integrator(
        coll, prog, reduce="sum", reduce_over_particles=False, batch=True,
    )

    ref = run_ref(None)
    out = run_batch(None)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_subsampling_batch():
    """Subsampling should give the same result in both paths."""
    coll, X, dt = _make_collection(T=20, N=4, d=2, dt=0.1)
    d = X.shape[-1]
    nF = 3

    BL = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BL")
    BR = ExprOperand(expr=_scalar_basis(d, nF), x=stream("X"), alias="BR")
    prog = Integrand(
        exprs=[BL, BR],
        terms=[Term(eq="ia,ib->ab", ops=("BL", "BR"))],
    )

    ref = integrate(
        coll, prog, reduce="sum", reduce_over_particles=False, subsampling=3,
    )
    out = integrate(
        coll, prog, reduce="sum", reduce_over_particles=False, subsampling=3,
        batch=True,
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
