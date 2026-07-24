import pytest
import jax.numpy as jnp
import numpy as np

from SFI.integrate.api import integrate
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


def _make_dataset_from_X(X: jnp.ndarray, dt: float, extras_time=None):
    """
    Build a real TrajectoryDataset directly to ensure scalar dt is honored.
    """
    ds = TrajectoryDataset.from_arrays(X=X, dt=dt)
    return ds


def _vec_features_basis(dim: int, nF: int):
    """
    Real Basis via make_basis: returns vector-valued features with shape (..., dim, nF),
    i.e. rank=1 → rank block (dim) before features.
    """

    def f(x, *, v=None, **kw):
        feats = [(k + 1) * x for k in range(nF)]  # each (..., d)
        F = jnp.stack(feats, axis=-1)  # (..., d, nF)
        return F

    return make_basis(f, dim=dim, rank=1, n_features=nF)


def _scalar_features_basis(dim: int, nF: int):
    """Basis that returns scalar features with shape (..., nF)."""

    def f(x, **kw):
        base = x[..., 0]
        feats = [base + 0.1 * (k + 1) for k in range(nF)]
        return jnp.stack(feats, axis=-1)  # (..., nF)

    return make_basis(f, dim=dim, rank=0, n_features=nF)


def _make_collection(T=12, N=5, d=2, dt=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dW = rng.normal(scale=np.sqrt(dt), size=(T, N, d)).astype(np.float32)
    X = np.cumsum(dW, axis=0).astype(np.float32)
    ds = _make_dataset_from_X(jnp.array(X), dt)
    return TrajectoryCollection.from_dataset(ds), jnp.array(X), dt


def test_gram_rectangle_shift_trapezoid_mean_real_stateexpr():
    """
    Gram-like terms with scalar features:
      G_rect  = < BL(X_t)  ⊗ BR(X_t)   >
      G_shift = < BL(X_t)  ⊗ BR(X_{t+dt}) >
      G_trap  = 0.5 * (rect + shift)
    Einsum 'ia, ib -> ab' already sums over i inside the term, so we set
    reduce_over_particles=False.
    """
    N = 3
    coll, X, dt = _make_collection(T=9, N=N, d=2, dt=0.25, seed=1)
    d = X.shape[-1]
    nF = 4

    BL = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X"), alias="BL")
    BR0 = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X"), alias="BR0")
    BRP = ExprOperand(
        expr=_scalar_features_basis(d, nF), x=stream("X_plus"), alias="BRP"
    )

    G_rect = Integrand(exprs=[BL, BR0], terms=[Term(eq="ia,ib->ab", ops=("BL", "BR0"))])
    G_shift = Integrand(
        exprs=[BL, BRP], terms=[Term(eq="ia,ib->ab", ops=("BL", "BRP"))]
    )
    G_trap = 0.5 * G_rect + 0.5 * G_shift

    out = integrate(coll, G_trap, reduce="mean", reduce_over_particles=False)

    # Manual trapezoid mean
    ds = coll.datasets[0]
    idx = np.array(ds.valid_indices({"X", "X_plus"}))
    acc = 0.0
    Teff = 0.0
    for t in idx:
        BLx = _scalar_features_basis(d, nF)(X[t])  # (N, nF_a)
        BRx = _scalar_features_basis(d, nF)(X[t])  # (N, nF_b)
        BRp = _scalar_features_basis(d, nF)(X[t + 1]) if t < X.shape[0] - 1 else BRx
        g_rect = jnp.einsum("ia,ib->ab", BLx, BRx)
        g_shift = jnp.einsum("ia,ib->ab", BLx, BRp)
        g_trap = 0.5 * (g_rect + g_shift)
        acc = acc + g_trap * dt
        Teff += dt * N
    ref = acc / Teff
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_memory_hint_estimation_positive_and_drives_chunking():
    coll, X, dt = _make_collection(T=30, N=6, d=2, dt=0.1)
    d = X.shape[-1]
    nF = 5
    B = ExprOperand(expr=_vec_features_basis(d, nF), x=stream("X"), alias="B")
    V = TimeOperand(velocity("dX", "dt"), alias="V")
    A = ConstOperand(jnp.eye(d), alias="A")

    prog = Integrand(
        exprs=[B],
        times=[V],
        consts=[A],
        terms=[Term(eq="im,mn,ina->ia", ops=("V", "A", "B"))],
    )

    sample = coll.peek_row(require=prog.require())
    hint = prog.estimate_bytes_per_sample(sample)
    assert hint is not None and int(hint) > 0

    # Force tiny chunks; exercise the iterator returning multiple slices
    chunks = list(
        coll.iter_slices(
            require=prog.require(), bytes_hint=hint, chunk_target_bytes=128
        )
    )
    assert len(chunks) >= 2


def test_variable_dt_vector_sum_matches_manual():
    T, N, d = 8, 4, 2
    rng = np.random.default_rng(1)
    t = np.cumsum(np.abs(rng.normal(size=T))).astype(np.float32)
    X = np.cumsum(
        np.random.default_rng(2).normal(size=(T, N, d)).astype(np.float32), axis=0
    )
    ds = TrajectoryDataset.from_arrays(X=X, t=t)
    coll = TrajectoryCollection.from_dataset(ds)

    class OnesPerParticle:
        def require(self):
            return {"X"}  # integrate() adds "__dt__"

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.ones((row["X"].shape[0],), dtype=jnp.float32)

    out = integrate(coll, OnesPerParticle(), reduce="sum")

    idx = np.array(ds.valid_indices({"X", "__dt__"}))
    dt = t[idx + 1] - t[idx]
    ref = np.sum(dt * N)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6, atol=1e-6)


def test_concat_two_datasets_sum_of_ones():
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

    class OnesPerParticle:
        def require(self):
            return {"X"}

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.ones((row["X"].shape[0],), dtype=jnp.float32)

    out = integrate(coll, OnesPerParticle(), reduce="sum")

    ref = (ds1.N * (ds1.T - 1) * ds1.dt) + (ds2.N * (ds2.T - 1) * ds2.dt)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6, atol=1e-6)


def test_no_valid_indices_returns_zero():
    T, N, d = 1, 3, 2
    X = np.zeros((T, N, d), dtype=np.float32)
    with pytest.warns(UserWarning, match="Very short trajectory"):
        ds = TrajectoryDataset.from_arrays(X=X, dt=0.5)
    coll = TrajectoryCollection.from_dataset(ds)

    class NeedsXplusScalar:
        def require(self):
            return {"X_plus"}  # forces empty window

        def estimate_bytes_per_sample(self, sample):
            return None

        def __call__(self, **row):
            return jnp.array(1.0, dtype=jnp.float32)

    out = integrate(coll, NeedsXplusScalar(), reduce="sum", reduce_over_particles=False)
    np.testing.assert_allclose(np.array(out), 0.0, rtol=0, atol=0)


def test_uli_force_gram_no_valid_frames_raises_clearly():
    """A Gram integration that yields no valid frames must raise a clear error.

    ``integrate`` legitimately returns the scalar ``0.0`` for an empty plan
    (see :func:`test_no_valid_indices_returns_zero`), but ``__G_matrix__`` then
    feeds that scalar to ``jnp.swapaxes(G, -1, -2)``, which used to detonate with
    a cryptic ``IndexError: index -2 is out of bounds for axis 0 with size 0``.
    The Gram builder must instead surface an actionable message naming the
    no-valid-frames condition.
    """
    import warnings

    from SFI.inference.underdamped import UnderdampedLangevinInference

    # T=1: no forward displacement, so the force-Gram integrand (which needs
    # dX/dt) has zero valid rows for every particle.
    X = np.zeros((1, 3, 2), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = TrajectoryDataset.from_arrays(X=jnp.asarray(X), dt=0.1)
    coll = TrajectoryCollection.from_dataset(ds)

    inf = UnderdampedLangevinInference(coll)
    inf.force_basis = _vec_features_basis(dim=2, nF=2)
    inf.A_inv = jnp.eye(2)
    setattr(inf, "__force_G_mode__", "rectangle")

    with pytest.raises(ValueError, match="no valid frames"):
        inf._force_G_matrix()


def test_build_plan_does_not_mask_real_row_errors(monkeypatch):
    """A genuine row-construction failure must propagate, not become an empty plan.

    ``_build_plan`` probes one row via ``peek_row`` and treats a ``ValueError``
    as "no usable window" → empty plan → scalar result. But ``peek_row`` also
    raises ``ValueError`` when materialising a row genuinely fails (e.g. a
    malformed extra). When the datasets *do* have a valid time window, such an
    error must surface, not be silently swallowed (which previously produced the
    cryptic scalar-Gram → swapaxes crash).
    """
    from SFI.trajectory.collection import TrajectoryCollection

    coll, X, dt = _make_collection(T=12, N=4, d=2)
    basis = _vec_features_basis(dim=2, nF=3)
    BL = ExprOperand(expr=basis, x=stream("X"), alias="BL")
    prog = Integrand(exprs=[BL], terms=[Term(eq="ima->a", ops=("BL",))])

    # Sanity: this collection genuinely has a usable window.
    assert any(ds.valid_indices({"X"}).size > 0 for ds in coll.datasets)

    def boom(*args, **kwargs):
        raise ValueError("synthetic row-construction failure")

    monkeypatch.setattr(TrajectoryCollection, "peek_row", boom)

    with pytest.raises(ValueError, match="synthetic row-construction failure"):
        integrate(coll, prog, reduce="sum")


def test_add_empty_ops_raises():
    from SFI.integrate.timeops import add
    import pytest
    with pytest.raises(ValueError, match="at least one"):
        add([])


def test_integrand_duplicate_alias_raises():
    d, nF = 2, 3
    B1 = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X"), alias="B")
    B2 = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X"), alias="B")
    with pytest.raises(ValueError, match="Duplicate alias"):
        Integrand(exprs=[B1, B2])


def test_integrand_add_duplicate_alias_raises():
    d, nF = 2, 3
    B1 = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X"), alias="B")
    B2 = ExprOperand(expr=_scalar_features_basis(d, nF), x=stream("X_plus"), alias="B")
    g1 = Integrand(exprs=[B1], terms=[Term(eq="ia->a", ops=("B",))])
    g2 = Integrand(exprs=[B2], terms=[Term(eq="ia->a", ops=("B",))])
    with pytest.raises(ValueError, match="alias.*different"):
        _ = g1 + g2
