"""Spike: a transient structural overlay replacing the on-dataset ``_cache/`` lifecycle.

``build_structural_overlay(expr, dataset)`` builds the dispatcher-owned stencil /
CSR tables host-side into a *throwaway* mapping, keyed by the structural
descriptor (grid shape / bc / ...), without ever writing them onto
``dataset.extras_global``. This lets inference feed the structural arrays into the
JIT eval without persisting them on the dataset — so a stale table cannot survive
a later transform.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from SFI import OverdampedLangevinInference, TrajectoryCollection
from SFI.bases.linear import field_component
from SFI.bases.spde import Laplacian, square_grid_extras
from SFI.langevin import OverdampedProcess
from SFI.statefunc import Rank, make_interactor
from SFI.statefunc.layout import GridLayout, ScalarSector
from SFI.statefunc.nodes.interactions import FromExtrasPairsCSR
from SFI.statefunc.nodes.interactions.prepare import (
    build_structural_overlay,
    is_cache_key,
    prepare_structural_extras_for_expr,
)


def _laplacian_field_dataset(grid_shape=(8, 8), dx=1.0, T=32, seed=0):
    """A small grid field dataset + a Laplacian basis that prepares a stencil."""
    P = int(np.prod(grid_shape))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P, 1)).astype(np.float32)
    extras = square_grid_extras(grid_shape=grid_shape, dx=dx)
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=extras)
    basis = Laplacian(ndim=len(grid_shape), bc="pbc")(field_component(0, n_fields=1))
    return coll.datasets[0], basis


def test_overlay_carries_stencil_without_touching_dataset():
    ds, basis = _laplacian_field_dataset()

    overlay = build_structural_overlay(basis, ds)

    # The overlay carries the prepared structural arrays...
    assert any(is_cache_key(k) for k in overlay), "overlay should hold _cache/ stencil arrays"
    # ...but the dataset's own extras are never polluted with them.
    assert not any(
        is_cache_key(k) for k in (ds.extras_global or {})
    ), "build_structural_overlay must not write _cache/ keys onto the dataset"


def _hyper_table(overlay):
    """The (P, K) stencil neighbour table from an overlay (P = number of grid cells)."""
    key = next(k for k in overlay if k.endswith("/hyper"))
    return np.asarray(overlay[key])


def test_overlay_rebuilds_and_does_not_reuse_a_stale_cache():
    """The headline bug: a dataset carrying a stale ``_cache/`` (e.g. a forgotten
    purge after a grid change) must NOT silently yield wrong forces.

    The old host-prep path skips rebuilding when a ``_cache/`` is already present,
    so a stale stencil survives. ``build_structural_overlay`` rebuilds from the
    descriptors, so it always matches the dataset's current grid.
    """
    _, basis = _laplacian_field_dataset(grid_shape=(4, 4))
    stale = build_structural_overlay(basis, _laplacian_field_dataset(grid_shape=(8, 8))[0])

    # A (4, 4) dataset poisoned with a stale (8, 8) stencil cache.
    small_ds, _ = _laplacian_field_dataset(grid_shape=(4, 4))
    poisoned = replace(small_ds, extras_global={**small_ds.extras_global, **stale})

    # Document the latent bug: the old host-prep path sees the cache and skips,
    # so the stale (P=64) stencil survives on a 16-cell grid.
    created = prepare_structural_extras_for_expr(basis, dict(poisoned.extras_global))
    assert created == {}, "old path silently keeps the stale cache (the bug)"

    # The overlay ignores the stale cache and rebuilds for the real (4, 4) grid.
    overlay = build_structural_overlay(basis, poisoned)
    assert _hyper_table(overlay).shape[0] == 16, "overlay must rebuild for the current grid"
    assert _hyper_table(stale).shape[0] == 64, "sanity: the stale cache was the (8, 8) stencil"


# ---------------------------------------------------------------------------
# SPDE force-inference path: behaviour-preservation + no-dataset-mutation.
# ---------------------------------------------------------------------------


def _spde_force_setup(grid=(6, 6), T=40, seed=0):
    """A tiny single-field SPDE force-inference setup (exercises the _cache/ stencil)."""
    P = int(np.prod(grid))
    layout = GridLayout(U=ScalarSector([0]), dim=1, ndim=2, bc="pbc")
    U = layout.U
    generic = layout.const(1) & U & layout.lap(U)  # 3 candidate features incl. ∇²U
    basis = layout.embed(rank=1, U=generic)

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P, 1)).astype(np.float32)
    extras = square_grid_extras(grid_shape=grid, dx=1.0)
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=extras)
    return coll, basis


# Golden values captured from the current (pre-refactor) inference path.
_GOLDEN_G = np.array(
    [
        [4.759860e00, -6.855008e-02, 4.097821e-09],
        [-9.118840e-02, 2.373528e00, -9.474977e00],
        [-2.104789e-08, -9.474977e00, 4.713657e01],
    ]
)
_GOLDEN_COEFFS = np.array([-0.18745829, -19.621288, -0.03835477])


def test_spde_force_inference_matches_golden():
    """Behaviour anchor: the SPDE Gram matrix / coefficients must not drift."""
    coll, basis = _spde_force_setup()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="trapeze")

    np.testing.assert_allclose(np.asarray(inf.force_G_full), _GOLDEN_G, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(inf.force_coefficients).ravel(), _GOLDEN_COEFFS, rtol=2e-4, atol=1e-3
    )


def test_force_inference_does_not_replace_self_data():
    """The refactor: ``infer_force_linear`` must thread structural arrays transiently,
    never round-tripping ``self.data`` through a prepare/purge of ``_cache/`` keys."""
    coll, basis = _spde_force_setup()
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")

    data_before = inf.data
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="trapeze")

    assert inf.data is data_before, "infer_force_linear must not replace self.data"
    assert not any(
        is_cache_key(k) for ds in inf.data.datasets for k in (ds.extras_global or {})
    ), "no _cache/ keys should ever be left on self.data"


def test_pooled_two_dataset_spde_fit_keeps_data_clean():
    """Per-dataset (pooled) fit: structural threading must work across datasets and
    still leave the pooled collection pristine."""
    coll1, basis = _spde_force_setup(seed=0)
    coll2, _ = _spde_force_setup(seed=1)
    pooled = coll1.concat([coll2], weights="pool")

    inf = OverdampedLangevinInference(pooled)
    inf.compute_diffusion_constant(method="WeakNoise")

    data_before = inf.data
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="trapeze")

    assert inf.data is data_before
    assert not any(is_cache_key(k) for ds in inf.data.datasets for k in (ds.extras_global or {}))
    assert np.all(np.isfinite(np.asarray(inf.force_coefficients)))


def test_stencil_built_once_per_inference_not_per_frame(monkeypatch):
    """Performance guard: the (expensive) stencil host-build runs a handful of times
    per inference — not once per frame. This is the memoisation the on-dataset
    ``_cache/`` provided, now preserved by the single transient build."""
    import SFI.statefunc.nodes.interactions.stencils as st

    n_builds = {"count": 0}
    original = st.hyperfixed_square_stencil

    def _counting(*args, **kwargs):
        n_builds["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(st, "hyperfixed_square_stencil", _counting)

    T = 40
    coll, basis = _spde_force_setup(T=T)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="WeakNoise")
    inf.infer_force_linear(basis, M_mode="Ito", G_mode="trapeze")

    # One composed stencil, built once (not ~T times for T frames).
    assert n_builds["count"] < T // 4, f"stencil rebuilt too often: {n_builds['count']}"


def test_dynamic_csr_overlay_is_empty_and_preserves_user_csr():
    """The dynamic-neighbour (nonreciprocal) path: a user-supplied CSR
    (``FromExtrasPairsCSR``) is genuine input, not host-built scaffolding — the
    overlay is empty and the user's CSR is left untouched."""

    def pair_fn(Xk, *, extras=None):
        return (Xk[1] - Xk[0])[..., None]

    inter = make_interactor(pair_fn, dim=2, rank=Rank.VECTOR, K=2)
    expr = inter.dispatch(FromExtrasPairsCSR("indptr", "indices"), return_as="basis")

    P = 3
    indptr = jnp.array([0, 1, 2, 2], dtype=jnp.int32)
    indices = jnp.array([1, 2], dtype=jnp.int32)
    X = np.random.default_rng(0).standard_normal((6, P, 2)).astype(np.float32)
    coll = TrajectoryCollection.from_arrays(
        X=X, dt=0.1, extras_global={"indptr": indptr, "indices": indices}
    )
    ds = coll.datasets[0]

    overlay = build_structural_overlay(expr, ds)

    assert overlay == {}, "user-provided CSR needs no host-built structural arrays"
    assert "indptr" in (ds.extras_global or {}) and "indices" in (ds.extras_global or {})


def _spde_sim_process(grid=(6, 6), seed=0):
    """A tiny SPDE process whose force is a grid Laplacian (φ̇ = θ·∇²φ)."""
    P = int(np.prod(grid))
    layout = GridLayout(U=ScalarSector([0]), dim=1, ndim=2, bc="pbc")
    U = layout.U
    basis = layout.embed(rank=1, U=layout.lap(U))
    proc = OverdampedProcess(basis, D=0.05)
    proc.set_params(theta_F=jnp.array([0.1], dtype=jnp.float32))
    proc.set_extras(extras_global=square_grid_extras(grid_shape=grid, dx=1.0))
    X0 = jnp.asarray(np.random.default_rng(seed).standard_normal((P, 1)), dtype=jnp.float32)
    proc.initialize(X0)
    return proc


def test_simulation_does_not_persist_cache_on_process_extras():
    """A stencil simulation must not leave ``_cache/`` scaffolding on the process's
    user-facing extras (it is kept in a private build-once store instead)."""
    import jax

    proc = _spde_sim_process()
    coll = proc.simulate(dt=0.01, Nsteps=20, key=jax.random.PRNGKey(0))

    assert not any(
        is_cache_key(k) for k in (proc.extras_global or {})
    ), "process extras_global must not accumulate _cache/ scaffolding"
    assert not any(is_cache_key(k) for ds in coll.datasets for k in (ds.extras_global or {}))
    assert np.all(np.isfinite(np.asarray(coll.datasets[0].X))), "stencil force must apply"
