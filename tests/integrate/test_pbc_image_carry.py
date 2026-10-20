"""Periodic pair features of one program row share one image.

A pair crossing the minimum-image cut ``|Δx| = L/2`` between frames ``t`` and
``t+1`` flips the sign of ``φ(r) r̂_ij`` if the image is re-picked at ``t+1``.
A program evaluating pair features at several frames (Stratonovich
trapezoid, shifted Gram, ULI) puts every pair on the image of one reference
frame (reserved ``pbc_ref`` extra) that precedes every increment of the row:
the earliest pair frame when it does, else the latest such frame the row
already reads, else the frame before the stencil.  Single-state programs keep
the minimum image; free-space bases never read a reference.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from SFI import TrajectoryCollection
from SFI.bases.pairs import pbc_displacement, power_kernels, radial_pair_basis
from SFI.integrate import integrate, stream
from SFI.integrate.api import integrate_rows
from SFI.integrate import velocity
from SFI.integrate.integrand import ExprOperand, Integrand, Term, TimeOperand
from SFI.statefunc import make_interactor

L = 10.0
BOX = np.array([L, L])


def _positions():
    """Three particles, unwrapped; particle 1 crosses the x-cut of particle 0 twice."""
    T = 8
    X = np.zeros((T, 3, 2))
    X[:, 0] = [1.0, 1.0]
    X[:, 1, 0] = 1.0 + np.array([4.2, 4.6, 4.9, 5.2, 5.4, 5.1, 4.8, 4.5])
    X[:, 1, 1] = 1.3
    X[:, 2] = [3.0, 6.0] + 0.1 * np.arange(T)[:, None]
    return X


def _expected(X, t, offset, ref_offset=None):
    """Σ_j r̂_ij at frame t + offset, image of frame t + ref_offset (own frame if None)."""
    N = X.shape[1]
    xe = X[t + offset]
    xr = xe if ref_offset is None else X[t + ref_offset]
    y = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = (xe[j] - xe[i]) - L * np.round((xr[j] - xr[i]) / L)
                y[i] += dx / np.linalg.norm(dx)
    return y[..., None]  # (N, d, F=1)


def _handwritten(box):
    """User interactor following the documented ``pbc_ref`` recipe."""

    def local(XK, *, extras):
        pr = extras.get("pbc_ref")
        ref = None if pr is None else (pr[1], pr[0])
        dx = pbc_displacement(XK[1], XK[0], extras["box"] if box == "extras" else box, ref=ref)
        return (dx / jnp.sqrt(jnp.sum(dx**2)))[:, None]

    return make_interactor(local, dim=2, rank=1, K=2, extras_keys=("box",) if box == "extras" else (),
                           particle_extras=("pbc_ref",))


def _basis(box, dispatch):
    if dispatch == "handwritten":
        inter, dispatch = _handwritten(box), "dense"
    else:
        inter = radial_pair_basis(power_kernels([0]), dim=2, box=box)
    if dispatch == "dense":
        return inter.dispatch_pairs(exclude_self=True, owners="focal", reducer="sum", return_as="basis")
    return inter.dispatch_pairs_from_extras(indptr_key="nb/indptr", indices_key="nb/indices",
                                            exclude_self=True, owners="focal", reducer="sum")


def _collection(X, box, dispatch):
    ex = {}
    if box == "extras":
        ex["box"] = jnp.asarray(BOX)
    if dispatch == "csr":
        N = X.shape[1]
        ex["nb/indptr"] = jnp.arange(0, N * (N - 1) + 1, N - 1, dtype=jnp.int32)
        ex["nb/indices"] = jnp.asarray([j for i in range(N) for j in range(N) if j != i], jnp.int32)
    return TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=ex or None)


def test_pbc_displacement_reference_image():
    box = jnp.array([10.0])
    # Pair at +5.2 whose reference sits at +4.9: continuation, not the -4.8 minimum image.
    dx = pbc_displacement(jnp.array([5.2]), jnp.array([0.0]), box, ref=(jnp.array([4.9]), jnp.array([0.0])))
    assert jnp.allclose(dx, 5.2, atol=1e-6)
    assert jnp.allclose(pbc_displacement(jnp.array([5.2]), jnp.array([0.0]), box), -4.8, atol=1e-6)


@pytest.mark.parametrize("dispatch", ["dense", "csr", "handwritten"])
@pytest.mark.parametrize("box", ["extras", "static"])
def test_x_plus_carries_anchor_image(box, dispatch):
    X = _positions()
    box_arg = jnp.asarray(BOX) if box == "static" else "extras"
    B = _basis(box_arg, dispatch)
    coll = _collection(X, box, dispatch)
    ex = dict(coll.datasets[0].extras_global or {})

    V = (X[1:] - X[:-1]) / 0.1  # V[t] = dX_t / dt
    rows = lambda prog: integrate_rows(coll, prog, reduce_over_particles=False, weight_by_dt=False)[0]  # noqa: E731
    B0 = ExprOperand(expr=B, x=stream("X"), alias="B0")
    Bp = ExprOperand(expr=B, x=stream("X_plus"), alias="Bp")
    Bm = ExprOperand(expr=B, x=stream("X_minus"), alias="Bm")
    Vop = TimeOperand(velocity("dX", "dt"), alias="V")
    assert B0.has_pbc_pairs

    # Trapezoid (Stratonovich) stencil: frame t enters the increment, image from t-1.
    prog = Integrand(exprs=[B0, Bp], times=[Vop],
                     terms=[Term(eq="ina,in->ia", ops=("B0", "V")), Term(eq="ina,in->ia", ops=("Bp", "V"))])
    assert prog.pbc_reference == "X_minus" and "X_minus" in prog.require()
    t_idx, Y = rows(prog)
    want = np.stack([np.einsum("ina,in->ia", _expected(X, t, 0, -1) + _expected(X, t, 1, -1), V[t]) for t in t_idx])
    np.testing.assert_allclose(np.asarray(Y), want, atol=1e-4)
    tot = integrate(coll, prog, reduce="sum", reduce_over_particles=False, weight_by_dt=False)
    np.testing.assert_allclose(np.asarray(tot), want.sum(0), atol=1e-3)
    # Not vacuous: re-picking the image at each frame differs at the crossings.
    naive = np.stack([np.einsum("ina,in->ia", _expected(X, t, 0) + _expected(X, t, 1), V[t]) for t in t_idx])
    assert np.abs(naive - want).max() > 1.0
    np.testing.assert_allclose(np.asarray(B(jnp.asarray(X), extras=ex))[t_idx + 1],
                               np.stack([_expected(X, t, 1) for t in t_idx]), atol=1e-5)

    # Shifted-Gram stencil (no increment): image of the earliest frame, t-1.
    gram = Integrand(exprs=[Bm, B0], terms=[Term(eq="ina,inb->iab", ops=("Bm", "B0"))])
    assert gram.pbc_reference == "X_minus"
    t_idx, Y = rows(gram)
    want = np.stack([np.einsum("ina,inb->iab", _expected(X, t, -1), _expected(X, t, 0, -1)) for t in t_idx])
    np.testing.assert_allclose(np.asarray(Y), want, atol=1e-5)

    # Single-frame program (Itô-shift stencil): minimum image of its own frame.
    prog1 = Integrand(exprs=[Bm], times=[Vop], terms=[Term(eq="ina,in->ia", ops=("Bm", "V"))])
    assert prog1.pbc_reference is None
    t_idx, Y = rows(prog1)
    want1 = np.stack([np.einsum("ina,in->ia", _expected(X, t, -1), V[t]) for t in t_idx])
    np.testing.assert_allclose(np.asarray(Y), want1, atol=1e-4)


def test_free_space_pairs_read_no_reference():
    B = _basis(None, "dense")
    B0 = ExprOperand(expr=B, x=stream("X"), alias="B0")
    Bp = ExprOperand(expr=B, x=stream("X_plus"), alias="Bp")
    assert not B0.has_pbc_pairs
    prog = Integrand(exprs=[B0, Bp], times=[TimeOperand(velocity("dX", "dt"), alias="V")],
                     terms=[Term(eq="ina,in->ia", ops=("B0", "V")), Term(eq="ina,in->ia", ops=("Bp", "V"))])
    assert prog.pbc_reference is None and "X_minus" not in prog.require()


def test_reference_reuses_scope_frame_before_the_increment():
    B = _basis("extras", "dense")
    B0 = ExprOperand(expr=B, x=stream("X"), alias="B0")
    Bp = ExprOperand(expr=B, x=stream("X_plus"), alias="Bp")
    prog = Integrand(exprs=[B0, Bp], times=[TimeOperand(velocity("dX", "dt"), alias="V")],
                     terms=[Term(eq="ina,in->ia", ops=("B0", "V")), Term(eq="ina,in->ia", ops=("Bp", "V"))])
    # Frames after the increment's start depend on it: never a reference.
    assert prog.pbc_reference_in({1, 2}) == "X_minus"
    assert prog.pbc_reference_in({-2, -1, 2}) == "X_minus"
    assert prog.pbc_reference_in({-2}) == "X_minusminus"


@pytest.mark.parametrize("batch", [True, False])
def test_parametric_integrator_keeps_its_build_scope(batch):
    """A plan built inside ``shared_rows`` evaluates its program in that scope
    wherever ``run`` is first traced: the image reference is part of the plan.

    Regression: the reference was resolved at trace time, so tracing outside the
    block asked for a stream the plan never gathered (``KeyError: 'X_minus'``).
    """
    import jax

    from SFI.integrate import shared_rows
    from SFI.integrate.api import make_minibatch_parametric_integrator, make_parametric_integrator

    X = _positions()
    coll = _collection(X, "extras", "dense")
    B = _basis("extras", "dense")
    B0 = ExprOperand(expr=B, x=stream("X"), alias="B0")
    Bp = ExprOperand(expr=B, x=stream("X_plus"), alias="Bp")
    prog = Integrand(exprs=[B0, Bp], times=[TimeOperand(velocity("dX", "dt"), alias="V")],
                     terms=[Term(eq="ina,in->ia", ops=("B0", "V")), Term(eq="ina,in->ia", ops=("Bp", "V"))])
    with shared_rows({-2}):
        assert prog.pbc_reference == "X_minusminus"
        want = np.asarray(integrate(coll, prog, reduce="sum", batch=batch))
        _, run = make_parametric_integrator(coll, prog, reduce="sum", batch=batch)
        _, _, run_mb = make_minibatch_parametric_integrator(coll, prog, batch_size=10**6, reduce="sum", batch=batch)
    assert prog.pbc_reference == "X_minus"  # the program alone, outside the block
    np.testing.assert_allclose(np.asarray(jax.jit(run)(None)), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(run_mb(None, jax.random.PRNGKey(0))), want, rtol=1e-5, atol=1e-5)
    # Traced inside another scope, it still evaluates in its own.
    with shared_rows({-1}):
        _, run_free = make_parametric_integrator(coll, prog, reduce="sum", batch=batch)
    with shared_rows({-2}):
        free = np.asarray(jax.jit(run_free)(None))
    np.testing.assert_allclose(free, np.asarray(integrate(coll, prog, reduce="sum", batch=batch)), rtol=1e-5, atol=1e-5)
