"""Dynamic-mask semantics for state expressions (interactions).

``dynamic_mask`` marks increments that must not enter the fit while the
particle's *position* stays known (``TrajectoryDataset`` docstring): such a
particle must remain visible to its neighbours' interactions.  The integrator
therefore hands state expressions the static ``mask_state`` and uses the fit
mask ``mask_out`` only to exclude rows from the reduction.  Regression for a
bug where ``mask_out`` was forwarded as the state mask, so a dynamically
masked particle vanished from every neighbour's interaction slots.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.bases.pairs import gaussian_kernels, radial_pair_basis
from SFI.integrate import Integrand, Term, stream
from SFI.integrate.api import integrate
from SFI.integrate.integrand import ExprOperand, TimeOperand
from SFI.trajectory import TrajectoryCollection, TrajectoryDataset

T, N, D = 24, 5, 2


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0, (T, N, D)).astype(np.float32)


def _basis():
    return radial_pair_basis(gaussian_kernels([0.8, 1.6]), dim=D).dispatch_pairs()


def _program(B):
    """Sum_i b_i(x) . dX_i: the force-fit projection (pair forces alone would cancel)."""
    return Integrand(exprs=[ExprOperand(B, x=stream("X"), alias="B")],
                     times=[TimeOperand(stream("dX"), alias="dX")],
                     terms=[Term(eq="idf,id->if", ops=("B", "dX"))])


def _reference(X, B, fit_mask, state_mask, dt):
    """Brute force: per frame, evaluate B on the particles visible at that frame, then keep fit rows."""
    out = 0.0
    for t in range(T - 1):  # valid rows need t+1 (dX)
        vis = state_mask[t] & state_mask[t + 1]
        idx = np.where(vis)[0]
        y = np.zeros((N, D, 2))
        y[idx] = np.asarray(B(jnp.asarray(X[t, idx])))
        rows = fit_mask[t] & state_mask[t + 1]
        dX = X[t + 1] - X[t]
        out = out + dt * np.einsum("idf,id->f", y[rows], dX[rows])
    return out


@pytest.mark.parametrize("chunk_target_bytes", [64, 1 << 26])
def test_dynamically_masked_particle_stays_a_neighbour(chunk_target_bytes):
    X = _positions()
    B = _basis()
    dyn = np.ones((T, N), bool)
    dyn[5:15, 1] = False  # particle 1's increments excluded, position known
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, dynamic_mask=dyn)
    val = np.asarray(integrate(TrajectoryCollection.from_dataset(ds), _program(B), reduce="sum",
                               chunk_target_bytes=chunk_target_bytes))
    ref = _reference(X, B, fit_mask=dyn, state_mask=np.ones((T, N), bool), dt=0.1)
    np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-6)
    # the buggy semantics (particle 1 removed from its neighbours' slots) differs
    wrong = _reference(X, B, fit_mask=dyn, state_mask=dyn, dt=0.1)
    assert np.max(np.abs(wrong - ref)) > 1e-3


def test_statically_masked_particle_is_not_a_neighbour():
    X = _positions(1)
    B = _basis()
    stat = np.ones((T, N), bool)
    stat[8:12, 2] = False  # position unknown (NaN, as from_columns stores it)
    X[~stat] = np.nan
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, mask=stat)
    val = np.asarray(integrate(TrajectoryCollection.from_dataset(ds), _program(B), reduce="sum"))
    ref = _reference(X, B, fit_mask=stat, state_mask=stat, dt=0.1)
    assert np.all(np.isfinite(val))
    np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-6)


def test_producers_emit_state_mask():
    X = _positions(2)
    stat = np.ones((T, N), bool)
    stat[10, 3] = False
    dyn = stat.copy()
    dyn[4, 0] = False
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, mask=stat, dynamic_mask=dyn)
    require = {"X", "dX", "__dt__"}
    row = ds.make_producer(require, include_dt=True)(jnp.asarray(4))
    assert bool(row["mask_state"][0]) and not bool(row["mask_out"][0])
    rows = ds.make_batch_producer(require, include_dt=True)(jnp.arange(2, 12))
    ms, mo = np.asarray(rows["mask_state"]), np.asarray(rows["mask_out"])
    assert ms[2, 0] and not mo[2, 0]  # t = 4: dynamically masked, still visible
    assert not ms[7, 3] and not mo[7, 3]  # t = 9: position unknown at t + 1 (dX)
    assert not ms[8, 3] and not mo[8, 3]  # t = 10: position unknown


def test_no_dynamic_mask_unchanged():
    X = _positions(3)
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1)
    row = ds.make_producer({"X", "__dt__"}, include_dt=True)(jnp.asarray(3))
    assert row["mask_state"] is row["mask_out"]


def test_pair_fit_with_entering_particle_is_finite():
    """Canonical from_columns path: a particle entering mid-run (NaN before) must not poison pair fits."""
    from SFI import OverdampedLangevinInference

    rng = np.random.default_rng(0)
    Tn, Nn = 300, 4
    X = np.cumsum(0.05 * rng.standard_normal((Tn, Nn, 2)), 0) + rng.uniform(0, 2, (1, Nn, 2))
    rows = [(i, t) for t in range(Tn) for i in range(Nn) if not (i == 3 and t < 100)]
    pid = np.array([r[0] for r in rows])
    tid = np.array([r[1] for r in rows])
    xs = np.array([X[t, i] for i, t in rows])
    coll = TrajectoryCollection.from_columns(pid, tid, xs, dt=0.01)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(_basis(), preset="clean")
    assert np.all(np.isfinite(np.asarray(inf.force_coefficients)))


def _group_interactor():
    """Per-slot hyperedge kernel: each member's displacement from the group centroid (depends on who is present)."""
    from SFI.statefunc import make_interactor
    from SFI.statefunc.nodes.interactions import FromExtrasHyperCSR

    def kernel(Xk, mask, extras):
        w = mask.astype(Xk.dtype)[:, None]
        c = (Xk * w).sum(0) / jnp.maximum(w.sum(), 1.0)
        return (extras["grp/gain"] * (Xk - c) * w)[..., None]  # (K, d, 1)

    # a declared (non-structural) extra makes the raw integrate() stream extras, as the Voronoi kernels do
    inter = make_interactor(kernel, dim=D, rank=1, Kmax=4, n_features=1, per_slot=True, extras_keys=("grp/gain",))
    return inter.dispatch(FromExtrasHyperCSR("grp/indptr", "grp/indices"), owners="all")


def _groups():
    rows = [[0, 1, 2], [1, 3, 4], [0, 4], [2, 3, 1, 0]]
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int32)
    return {"grp/indptr": jnp.asarray(indptr), "grp/indices": jnp.asarray(np.concatenate(rows).astype(np.int32)),
            "grp/gain": jnp.asarray(1.5, jnp.float32)}


@pytest.mark.parametrize("chunk_target_bytes", [64, 1 << 26])
def test_dynamically_masked_member_stays_in_its_hyperedges(chunk_target_bytes):
    X = _positions(4)
    B = _group_interactor()
    ex = _groups()
    dyn = np.ones((T, N), bool)
    dyn[3:20, 1] = False  # member of three groups; increments excluded, position known
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, dynamic_mask=dyn, extras_global=ex)
    val = np.asarray(integrate(TrajectoryCollection.from_dataset(ds), _program(B), reduce="sum",
                               chunk_target_bytes=chunk_target_bytes))
    ref = 0.0
    for t in range(T - 1):
        y = np.asarray(B(jnp.asarray(X[t]), extras=ex))  # every member present
        dX = X[t + 1] - X[t]
        ref = ref + 0.1 * np.einsum("idf,id->f", y[dyn[t]], dX[dyn[t]])
    np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-6)
