"""Hyperedge dispatch: variable-size (HyperCSR) schedules, per-slot outputs,
slot validity, spec-carried extras, padding, and the same-particle Jacobian.

The reference model is a polygon energy: each hyperedge is an ordered
polygon of particles and E = sum_c (A_c - A0)^2 / 2 + Gamma * L_c, so the
dispatched per-slot force must equal -grad E computed by autodiff of a plain
loop over polygons.  This is the vertex-model use case.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.langevin import OverdampedProcess
from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.nodes.interactions import (
    AutoPairs,
    FromExtrasHyperCSR,
    FromExtrasHyperFixed,
    HyperCSR,
    HyperFixed,
    PairsCSR,
)
from SFI.utils.neighbors import pad_neighbor_csr

A0, GAMMA = 0.8, 0.3

# Ragged polygons over P=9 particles (sizes 3..5); particle 8 belongs to none.
POLYS = [[0, 1, 2], [1, 3, 4, 2], [3, 5, 6, 7, 4], [5, 0, 6]]
P, D = 9, 2


def _positions(seed=0):
    return jax.random.uniform(jax.random.PRNGKey(seed), (P, D), minval=-1.0, maxval=1.0)


def _csr(polys):
    indptr = np.concatenate([[0], np.cumsum([len(c) for c in polys])]).astype(np.int32)
    return jnp.asarray(indptr), jnp.asarray(np.concatenate(polys).astype(np.int32))


def _padded(polys, K):
    """(M, K) table padded with each row's first participant + slot mask."""
    tab = np.zeros((len(polys), K), np.int32)
    msk = np.zeros((len(polys), K), bool)
    for i, c in enumerate(polys):
        tab[i, : len(c)], tab[i, len(c) :], msk[i, : len(c)] = c, c[0], True
    return jnp.asarray(tab), jnp.asarray(msk)


def _area_perim(Xk, mask):
    K = Xk.shape[0]
    n = jnp.sum(mask).astype(jnp.int32)
    s = jnp.arange(K)
    nxt = jnp.where(s + 1 < n, s + 1, 0)
    u, un = Xk, Xk[nxt]
    cross = u[:, 0] * un[:, 1] - un[:, 0] * u[:, 1]
    A = 0.5 * jnp.sum(jnp.where(mask, cross, 0.0))
    seg = jnp.sqrt(jnp.sum((un - u) ** 2, -1) + 1e-30)
    return A, jnp.sum(jnp.where(mask, seg, 0.0))


def _cell_energy(Xk, mask):
    A, L = _area_perim(Xk, mask)
    return 0.5 * (A - A0) ** 2 + GAMMA * L


def _per_slot_force(Xk, mask):
    """(K, d, 1): force on every slot of one polygon."""
    return (-jax.grad(_cell_energy)(Xk, mask))[..., None]


def _reference_force(X, polys=POLYS):
    def E(X):
        return sum(_cell_energy(X[jnp.array(c)], jnp.ones(len(c), bool)) for c in polys)

    return -jax.grad(E)(X)


def _per_slot_interactor(**kw):
    return make_interactor(lambda Xk, mask: _per_slot_force(Xk, mask), dim=D, rank=Rank.VECTOR, per_slot=True, **kw)


# ---------------------------------------------------------------------------
# HyperCSR: JIT, fixed vs variable arity, padding, overflow
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arity", [{"K": 6}, {"Kmax": 6}])
def test_hypercsr_per_slot_matches_energy_gradient_under_jit(arity):
    X = _positions()
    indptr, indices = _csr(POLYS)
    B = _per_slot_interactor(**arity).dispatch(HyperCSR(indptr, indices), owners="all")
    F = jax.jit(lambda x: B(x))(X)[..., 0]
    np.testing.assert_allclose(F, _reference_force(X), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(F[8], 0.0)  # particle in no polygon


def test_hypercsr_matches_padded_hyperfixed():
    X = _positions(1)
    inter = _per_slot_interactor(K=6)
    indptr, indices = _csr(POLYS)
    tab, msk = _padded(POLYS, 6)
    F_csr = inter.dispatch(HyperCSR(indptr, indices), owners="all")(X)
    F_fix = inter.dispatch(HyperFixed(hyper=tab, slot_mask=msk), owners="all")(X)
    np.testing.assert_allclose(F_csr, F_fix, rtol=1e-6, atol=1e-7)


def test_hypercsr_ignores_entries_past_indptr_end():
    X = _positions(2)
    indptr, indices = _csr(POLYS)
    padded = jnp.concatenate([indices, jnp.array([0, 8, 8], jnp.int32)])
    inter = _per_slot_interactor(K=6)
    F = inter.dispatch(HyperCSR(indptr, indices), owners="all")(X)
    F_pad = jax.jit(lambda x: inter.dispatch(HyperCSR(indptr, padded), owners="all")(x))(X)
    np.testing.assert_allclose(F_pad, F, rtol=1e-6, atol=1e-7)


def test_hypercsr_overflow_raises_instead_of_truncating():
    X = _positions()
    indptr, indices = _csr(POLYS)  # largest polygon has 5 members
    B = _per_slot_interactor(K=4).dispatch(HyperCSR(indptr, indices), owners="all")
    with pytest.raises(ValueError, match="more than K=4"):
        B(X)
    # Traced schedule (from extras, under jit): checked at run time.
    B_ex = _per_slot_interactor(K=4).dispatch(FromExtrasHyperCSR("ip", "ix"), owners="all")
    with pytest.raises(Exception, match="more than K=4"):
        jax.block_until_ready(jax.jit(lambda x, e: B_ex(x, extras=e))(X, {"ip": indptr, "ix": indices}))


def test_empty_hyperedge_contributes_nothing():
    X = _positions(3)
    polys = [POLYS[0], [], POLYS[1]]
    indptr = jnp.array([0, 3, 3, 7], jnp.int32)
    indices = jnp.array(POLYS[0] + POLYS[1], jnp.int32)
    for owners in ("all", "focal"):
        inter = _per_slot_interactor(K=6)
        F = inter.dispatch(HyperCSR(indptr, indices), owners=owners)(X)
        F_ref = inter.dispatch(HyperCSR(*_csr([POLYS[0], POLYS[1]])), owners=owners)(X)
        np.testing.assert_allclose(F, F_ref, rtol=1e-6, atol=1e-7)
    assert len(polys) == 3


def test_arity_checks():
    inter_var = make_interactor(lambda Xk: Xk[0][:, None], dim=D, rank=Rank.VECTOR, Kmax=5)
    with pytest.raises(ValueError, match="Arity mismatch"):
        inter_var.dispatch(PairsCSR(jnp.array([0, 1]), jnp.array([0])))
    inter_a = make_interactor(lambda Xk: Xk[0][:, None], dim=D, rank=Rank.VECTOR, Kmax=5)
    inter_b = make_interactor(lambda Xk: Xk[1][:, None], dim=D, rank=Rank.VECTOR, Kmax=4)
    with pytest.raises(ValueError, match="Kmax"):
        (inter_a & inter_b).dispatch(HyperCSR(*_csr(POLYS)))


# ---------------------------------------------------------------------------
# Per-slot outputs: owners, composition, pair specs
# ---------------------------------------------------------------------------
def test_per_slot_focal_on_rotated_table_equals_all():
    """One hyperedge per (polygon, member), rotated so the member is focal."""
    X = _positions(4)
    rotated = [c[s:] + c[:s] for c in POLYS for s in range(len(c))]
    inter = _per_slot_interactor(K=6)
    F_all = inter.dispatch(HyperCSR(*_csr(POLYS)), owners="all")(X)
    F_focal = inter.dispatch(HyperCSR(*_csr(rotated)), owners="focal")(X)
    np.testing.assert_allclose(F_focal, F_all, rtol=1e-5, atol=1e-6)


def test_per_slot_custom_and_global():
    X = _positions(5)
    inter = _per_slot_interactor(K=6)
    spec = HyperCSR(*_csr(POLYS))
    F_all = inter.dispatch(spec, owners="all")(X)
    w = jnp.full((6,), 2.0)
    F_custom = inter.dispatch(spec, owners="custom", owner_weights=w)(X)
    np.testing.assert_allclose(F_custom, 2.0 * F_all, rtol=1e-6, atol=1e-7)
    total = inter.dispatch(spec, owners="global", reducer="sum")(X)
    np.testing.assert_allclose(total, F_all.sum(axis=0), rtol=1e-5, atol=1e-6)
    n_slots = sum(len(c) for c in POLYS)
    mean = inter.dispatch(spec, owners="global", reducer="mean")(X)
    np.testing.assert_allclose(mean, F_all.sum(axis=0) / n_slots, rtol=1e-5, atol=1e-6)


def test_per_slot_composition_and_pairs_are_rejected():
    inter = _per_slot_interactor(K=2)
    with pytest.raises(NotImplementedError, match="dispatched directly"):
        (inter & inter).dispatch(HyperCSR(*_csr([[0, 1]])))
    with pytest.raises(NotImplementedError, match="not pair specs"):
        inter.dispatch(AutoPairs())


def test_per_slot_output_shape_is_checked():
    bad = make_interactor(lambda Xk: Xk[..., None], dim=D, rank=Rank.SCALAR, K=3, per_slot=True)
    with pytest.raises(ValueError, match="output shape"):
        bad.dispatch(HyperCSR(*_csr([[0, 1, 2]])), owners="all")(_positions())


# ---------------------------------------------------------------------------
# Slot validity for per-hyperedge outputs (broadcast to slots)
# ---------------------------------------------------------------------------
def _centroid_interactor(K):
    def f(Xk, mask):
        w = mask.astype(Xk.dtype)
        return (jnp.sum(Xk * w[:, None], 0) / jnp.sum(w))[:, None]

    return make_interactor(f, dim=D, rank=Rank.VECTOR, K=K)


def _centroid_reference(X, polys, owners):
    out = np.zeros((P, D))
    for c in polys:
        cen = np.asarray(X)[c].mean(0)
        for p in c if owners == "all" else c[:1]:
            out[p] += cen
    return out


@pytest.mark.parametrize("owners", ["all", "focal"])
def test_broadcast_outputs_skip_padded_slots(owners):
    X = _positions(6)
    B = _centroid_interactor(6).dispatch(HyperCSR(*_csr(POLYS)), owners=owners)
    np.testing.assert_allclose(B(X)[..., 0], _centroid_reference(X, POLYS, owners), rtol=1e-5, atol=1e-6)


def test_global_reductions_count_only_real_hyperedges():
    X = _positions(7)
    cents = np.stack([np.asarray(X)[c].mean(0) for c in POLYS])
    tab, msk = _padded(POLYS + [[8]], 6)
    msk = msk.at[-1].set(False)  # a padding row (focal slot masked)
    inter = _centroid_interactor(6)
    spec = HyperFixed(hyper=tab, slot_mask=msk)
    for red, ref in (("sum", cents.sum(0)), ("mean", cents.mean(0)), ("max", cents.max(0))):
        got = inter.dispatch(spec, owners="global", reducer=red)(X)[..., 0]
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Chunking and batching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("owners", ["all", "focal", "global"])
def test_chunked_matches_unchunked(owners):
    X = _positions(8)
    rotated = [c[s:] + c[:s] for c in POLYS for s in range(len(c))]  # M=15, not a multiple of 4
    spec = HyperCSR(*_csr(rotated))
    inter = _per_slot_interactor(K=6)
    full = inter.dispatch(spec, owners=owners)(X)
    chunked = inter.dispatch(spec, owners=owners, chunk_size=4)(X)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-6)


def test_batched_states_with_shared_and_per_sample_schedules():
    Xb = jnp.stack([_positions(9), _positions(10)])
    inter = _per_slot_interactor(K=6)
    shared = inter.dispatch(HyperCSR(*_csr(POLYS)), owners="all")(Xb)
    for b in range(2):
        np.testing.assert_allclose(shared[b, ..., 0], _reference_force(Xb[b]), rtol=1e-5, atol=1e-6)

    # Per-sample topology: sample 1 drops the last polygon (indices padded to a common length).
    ip0, ix0 = _csr(POLYS)
    ip1 = jnp.concatenate([_csr(POLYS[:3])[0], _csr(POLYS[:3])[0][-1:]])
    ix1 = jnp.concatenate([_csr(POLYS[:3])[1], jnp.zeros(3, jnp.int32)])
    spec = HyperCSR(jnp.stack([ip0, ip1]), jnp.stack([ix0, ix1]))
    per_sample = inter.dispatch(spec, owners="all")(Xb)
    np.testing.assert_allclose(per_sample[0, ..., 0], _reference_force(Xb[0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(per_sample[1, ..., 0], _reference_force(Xb[1], POLYS[:3]), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Spec-carried extras (per-hyperedge / per-slot data, slot ids)
# ---------------------------------------------------------------------------
def test_from_extras_hyperfixed_edge_and_slot_extras():
    X = _positions(11)
    tab, msk = _padded(POLYS, 6)
    M = len(POLYS)
    pol = jax.random.normal(jax.random.PRNGKey(1), (M, D))  # one vector per polygon
    wgt = jnp.where(msk, jnp.arange(6, dtype=jnp.float32)[None, :] + 1.0, 0.0)  # one weight per slot

    def f(Xk, mask, extras):
        return (extras["cell/pol"][None, :] * extras["cell/w"][:, None])[..., None]  # (K, d, 1)

    inter = make_interactor(f, dim=D, rank=Rank.VECTOR, K=6, per_slot=True, spec_extras=("cell/pol", "cell/w"))
    rule = FromExtrasHyperFixed("cell/hyper", "cell/mask", edge_extras_keys=("cell/pol",), slot_extras_keys=("cell/w",))
    extras = {"cell/hyper": tab, "cell/mask": msk, "cell/pol": pol, "cell/w": wgt}
    for chunk in (None, 3):
        B = inter.dispatch(rule, owners="all", chunk_size=chunk)
        got = jax.jit(lambda x, e: B(x, extras=e))(X, extras)[..., 0]
        ref = np.zeros((P, D))
        for m, c in enumerate(POLYS):
            for s, p in enumerate(c):
                ref[p] += np.asarray(pol[m]) * (s + 1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_from_extras_hypercsr_slot_ids_and_extras():
    X = _positions(12)
    indptr, indices = _csr(POLYS)
    sid = jnp.arange(indices.shape[0], dtype=jnp.int32)  # global slot code
    edge_val = jnp.arange(len(POLYS), dtype=jnp.float32) + 10.0

    def f(Xk, mask, extras):
        val = extras["slot_id"].astype(Xk.dtype) + extras["e"]
        return jnp.where(mask, val, 0.0)[:, None, None] * jnp.ones((1, D, 1))

    inter = make_interactor(f, dim=D, rank=Rank.VECTOR, K=6, per_slot=True, spec_extras=("slot_id", "e"))
    rule = FromExtrasHyperCSR("ip", "ix", key_slot_ids="sid", edge_extras_keys=("e",))
    B = inter.dispatch(rule, owners="all")
    got = jax.jit(lambda x, e: B(x, extras=e))(X, {"ip": indptr, "ix": indices, "sid": sid, "e": edge_val})
    ref = np.zeros(P)
    k = 0
    for m, c in enumerate(POLYS):
        for p in c:
            ref[p] += k + 10.0 + m
            k += 1
    np.testing.assert_allclose(got[:, 0, 0], ref, rtol=1e-6)


def test_spec_extras_validation():
    with pytest.raises(ValueError, match="spec_extras"):
        make_interactor(lambda Xk: Xk[0][:, None], dim=D, rank=Rank.VECTOR, K=3, spec_extras=("a",))
    inter = make_interactor(
        lambda Xk, extras: Xk[0][:, None] * extras["a"], dim=D, rank=Rank.VECTOR, K=3, spec_extras=("a",)
    )
    with pytest.raises(KeyError, match="spec_extras"):
        inter.dispatch(HyperCSR(*_csr([[0, 1, 2]])))(_positions())


# ---------------------------------------------------------------------------
# PairsCSR padding (pad_neighbor_csr) never leaks into the output
# ---------------------------------------------------------------------------
def _pair_interactor():
    def f(Xk):
        dx = Xk[1] - Xk[0]
        return (-jnp.sqrt(jnp.sum(dx**2) + 1e-12) * jnp.ones(D))[:, None]  # negative outputs

    return make_interactor(f, dim=D, rank=Rank.VECTOR, K=2)


@pytest.mark.parametrize(
    "owners,reducer,chunk",
    [
        ("all", "sum", None),
        ("custom", "sum", None),
        ("global", "sum", None),
        ("global", "max", None),
        ("all", "sum", 4),
    ],
)
def test_padded_pairs_csr_matches_unpadded(owners, reducer, chunk):
    X = _positions(13)
    indptr = np.array([0, 2, 3, 5, 5, 6, 6, 6, 6, 6], np.int32)
    indices = np.array([1, 2, 3, 4, 5, 7], np.int32)
    ip_p, ix_p = pad_neighbor_csr(indptr, indices, max_nnz=11)
    kw = dict(owners=owners, reducer=reducer, chunk_size=chunk)
    if owners == "custom":
        kw["owner_weights"] = jnp.array([1.0, -0.5])
    ref = _pair_interactor().dispatch(PairsCSR(jnp.asarray(indptr), jnp.asarray(indices)), **kw)(X)
    got = _pair_interactor().dispatch(PairsCSR(jnp.asarray(ip_p), jnp.asarray(ix_p)), **kw)(X)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)
    if reducer == "max":
        assert float(jnp.max(got)) < 0.0  # masked entries must not inject zeros into a max


# ---------------------------------------------------------------------------
# Same-particle Jacobian (used by the parametric core) vs brute-force AD
# ---------------------------------------------------------------------------
def _brute_force_spj(B, X):
    def block(p):
        return jax.jacfwd(lambda xp: B(X.at[p].set(xp))[p])(X[p])  # (*rank, F, d)

    J = jnp.stack([block(p) for p in range(X.shape[0])])
    return jnp.moveaxis(J, -1, 1)  # (P, d, *rank, F)


@pytest.mark.parametrize(
    "case",
    ["per_slot_all", "per_slot_focal_rotated", "broadcast_all", "broadcast_focal", "hyperfixed_repeats"],
)
def test_same_particle_jacobian_matches_brute_force(case):
    X = _positions(14)
    rotated = [c[s:] + c[:s] for c in POLYS for s in range(len(c))]
    if case == "per_slot_all":
        B = _per_slot_interactor(K=6).dispatch(HyperCSR(*_csr(POLYS)), owners="all")
    elif case == "per_slot_focal_rotated":
        B = _per_slot_interactor(K=6).dispatch(HyperCSR(*_csr(rotated)), owners="focal")
    elif case == "broadcast_all":
        B = _centroid_interactor(6).dispatch(HyperCSR(*_csr(POLYS)), owners="all")
    elif case == "broadcast_focal":
        B = _centroid_interactor(6).dispatch(HyperCSR(*_csr(rotated)), owners="focal")
    else:
        # Kernel that reads the padded (repeated-focal) slots, as no-flux stencils do.
        def f(Xk):
            return jnp.sum((Xk - Xk[0]) ** 2 + Xk**2, axis=0)[:, None]

        tab, _ = _padded(POLYS, 6)
        B = make_interactor(f, dim=D, rank=Rank.VECTOR, K=6).dispatch(HyperFixed(hyper=tab), owners="focal")
    J = B.d_x(same_particle=True)(X)
    np.testing.assert_allclose(J, _brute_force_spj(B, X), rtol=1e-4, atol=1e-5)


def test_padded_pairs_jacobian_matches_unpadded():
    X = _positions(15)
    indptr = np.array([0, 2, 3, 5, 5, 6, 6, 6, 6, 6], np.int32)
    indices = np.array([1, 2, 3, 4, 5, 7], np.int32)
    ip_p, ix_p = pad_neighbor_csr(indptr, indices, max_nnz=11)
    B = _pair_interactor().dispatch(PairsCSR(jnp.asarray(indptr), jnp.asarray(indices)), owners="all")
    B_pad = _pair_interactor().dispatch(PairsCSR(jnp.asarray(ip_p), jnp.asarray(ix_p)), owners="all")
    np.testing.assert_allclose(B_pad.d_x(same_particle=True)(X), B.d_x(same_particle=True)(X), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# End to end: a time-varying topology from extras inside the simulator scan
# ---------------------------------------------------------------------------
def test_simulation_with_hypercsr_topology_from_extras():
    X0 = _positions(16)
    indptr, indices = _csr(POLYS)
    padded = jnp.concatenate([indices, jnp.zeros(4, jnp.int32)])
    B = _per_slot_interactor(K=6).dispatch(FromExtrasHyperCSR("topo/indptr", "topo/indices"), owners="all")
    topo = {"topo/indptr": indptr, "topo/indices": padded}
    proc = OverdampedProcess(F=B, D=1e-4, theta_F=jnp.array([1.0]), extras_global=topo)
    proc.initialize(X0)
    coll = proc.simulate(dt=1e-3, Nsteps=20, key=jax.random.PRNGKey(0), oversampling=2, compute_observables=False)
    X = np.asarray(coll.to_array())
    assert X.shape == (20, P, D) and np.all(np.isfinite(X))
    # Particle 8 belongs to no polygon: pure diffusion, tiny displacement.
    assert np.max(np.abs(X[:, 8] - np.asarray(X0[8]))) < 0.05
