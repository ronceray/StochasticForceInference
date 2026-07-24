# TODO: review this file
import numpy as np
import pytest

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TimeSeriesExtra, TrajectoryDataset, time_series_extra


def has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import yaml  # noqa: F401  # for schema metadata header

        return True
    except Exception:
        return False


def has_h5py():
    try:
        import h5py  # noqa: F401
        import yaml  # noqa: F401  # for header

        return True
    except Exception:
        return False


ALL_FORMATS = [
    "csv",
    pytest.param(
        "parquet",
        marks=pytest.mark.skipif(
            not has_pyarrow(), reason="pyarrow/yaml not installed"
        ),
    ),
    pytest.param(
        "h5",
        marks=pytest.mark.skipif(not has_h5py(), reason="h5py/yaml not installed"),
    ),
]


def _make_collection(T=5, N=3, d=2, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(
        rng.normal(size=(T, N, d)).astype(np.float32),
        axis=0,
    )
    mask = np.ones((T, N), dtype=bool)

    t_vec = np.arange(T, dtype=float) * dt
    P_mass = np.linspace(1.0, 2.0, N)
    TP_signal = np.arange(T * N, dtype=float).reshape(T, N, 1)

    extras_global = {
        "t": time_series_extra(t_vec),
        "box": np.array([1.0, 1.0]),
    }
    extras_local = {
        "mass": P_mass,
        "signal": time_series_extra(TP_signal),
    }

    ds = TrajectoryDataset.from_arrays(
        X=X,
        t=t_vec,
        mask=mask,
        extras_global=extras_global,
        extras_local=extras_local,
        meta={"unit": "arb"},
    )
    return TrajectoryCollection.from_dataset(ds)


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_collection_singlefile_roundtrip(tmp_path, fmt):
    coll = _make_collection()
    suffix = {"csv": ".csv", "parquet": ".parquet", "h5": ".h5"}[fmt]
    path = tmp_path / f"traj{suffix}"

    coll.save(path, format=fmt)
    coll2 = TrajectoryCollection.load(path)

    assert len(coll2.datasets) == 1
    ds = coll.datasets[0]
    ds2 = coll2.datasets[0]

    # core data
    assert ds2.T == ds.T
    assert ds2.N == ds.N
    assert ds2.d == ds.d
    np.testing.assert_allclose(np.asarray(ds2._X3d()), np.asarray(ds._X3d()))
    np.testing.assert_array_equal(np.asarray(ds2._M2d()), np.asarray(ds._M2d()))

    # extras/global
    assert set(ds2.extras_global.keys()) == set(ds.extras_global.keys())
    assert "t" in ds2.extras_global
    assert isinstance(ds2.extras_global["t"], TimeSeriesExtra)
    np.testing.assert_allclose(
        np.asarray(ds2.extras_global["t"].data),
        np.asarray(ds.extras_global["t"].data),
    )

    # extras/local
    assert set(ds2.extras_local.keys()) == set(ds.extras_local.keys())
    np.testing.assert_allclose(
        np.asarray(ds2.extras_local["mass"]),
        np.asarray(ds.extras_local["mass"]),
    )

    # Time-series extra: allow IO to drop a trailing singleton dim
    assert isinstance(ds2.extras_local["signal"], TimeSeriesExtra)
    sig2 = np.asarray(ds2.extras_local["signal"].data)
    sig = np.asarray(ds.extras_local["signal"].data)

    # At least (T, N, ...) must match
    assert sig2.shape[:2] == (ds.T, ds.N)

    # If original has a trailing dim 1, squeeze it for comparison
    if sig.ndim == 3 and sig.shape[2] == 1:
        sig = sig[..., 0]

    np.testing.assert_allclose(sig2, sig)


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_collection_directory_multi_roundtrip(tmp_path, fmt):
    coll = _make_collection(T=7, seed=1)
    coll2 = _make_collection(T=9, seed=2)

    from SFI.trajectory.collection import TrajectoryCollection as TC

    multi = TC(
        datasets=[coll.datasets[0], coll2.datasets[0]],
        weights=None,
    )

    dstdir = tmp_path / "outdir"
    multi.save(dstdir, format=fmt)
    loaded = TC.load(dstdir)

    assert len(loaded.datasets) == 2
    assert loaded.datasets[0].T == coll.datasets[0].T
    assert loaded.datasets[1].T == coll2.datasets[0].T


# ---------------------------------------------------------------------------
# Issue #15: callable / FunctionExtra extras_global must not crash on load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_callable_extras_global_warns_and_loads(tmp_path, fmt):
    """Save with a callable in extras_global must warn and produce a loadable file.

    Regression test for GitHub issue #15: previously yaml.dump() serialised
    callables with !!python/object tags which yaml.safe_load() rejected on load.
    """
    import warnings

    from SFI.trajectory.dataset import function_extra

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 1, 2)).astype(np.float32)

    plain_callable = lambda t_idx, context=None: np.float32(t_idx) * 0.1
    fe = function_extra(lambda x: x ** 2)

    coll = TrajectoryCollection.from_arrays(
        X=X,
        dt=0.1,
        extras_global={
            "box": np.array([1.0, 1.0]),         # non-callable: should survive
            "gen": plain_callable,                # plain callable: skip with warning
            "adhesion": fe,                       # FunctionExtra: skip with warning
        },
    )

    suffix = {"csv": ".csv", "parquet": ".parquet", "h5": ".h5"}[fmt]
    path = tmp_path / f"traj{suffix}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coll.save(path, format=fmt)

    # Both callables should have triggered a UserWarning
    warn_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("gen" in m for m in warn_msgs), f"Expected warning for 'gen', got: {warn_msgs}"
    assert any("adhesion" in m for m in warn_msgs), f"Expected warning for 'adhesion', got: {warn_msgs}"

    # Load must succeed without ConstructorError or any exception
    coll2 = TrajectoryCollection.load(path)
    ds2 = coll2.datasets[0]

    # Non-callable extra 'box' must survive the round-trip
    assert "box" in ds2.extras_global
    np.testing.assert_allclose(
        np.asarray(ds2.extras_global["box"]), np.array([1.0, 1.0])
    )

    # Callable extras should have been omitted (not present after load)
    assert "gen" not in ds2.extras_global
    assert "adhesion" not in ds2.extras_global


# ---------------------------------------------------------------------------
# Issue #14: auto-relabeling and particle column compression
# ---------------------------------------------------------------------------

from SFI.trajectory.io import (
    assemble_X_from_columns,
    _greedy_compress_particles,
)


def test_relabel_sparse_ids():
    """Problem 1: sparse post-screening IDs are compressed to range(N).

    Particles with IDs {0, 5, 100} must produce X of shape (T, 3, d),
    not (T, 101, d). The id_map should record the original IDs.
    """
    rng = np.random.default_rng(42)
    # particle 0 at t=0,1; particle 5 at t=0,1; particle 100 at t=0,1
    particle_idx = np.array([0, 0, 5, 5, 100, 100], dtype=int)
    time_idx = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    state_vectors = rng.normal(size=(6, 2))

    X, mask, id_map = assemble_X_from_columns(
        particle_idx, time_idx, state_vectors, relabel=True
    )

    assert X.shape == (2, 3, 2), f"Expected (2,3,2), got {X.shape}"
    assert mask.shape == (2, 3)
    assert mask.all(), "All entries should be present"
    np.testing.assert_array_equal(id_map, [0, 5, 100])


def test_relabel_already_compact():
    """If IDs are already 0..N-1, relabeling is a no-op; id_map is None (trivial map)."""
    rng = np.random.default_rng(0)
    particle_idx = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    time_idx = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    state_vectors = rng.normal(size=(6, 2))

    X, mask, id_map = assemble_X_from_columns(
        particle_idx, time_idx, state_vectors, relabel=True
    )

    assert X.shape == (2, 3, 2)
    # Already-compact IDs → no informative map to record
    assert id_map is None


def test_greedy_compress_nonoverlapping():
    """Three non-overlapping particles (gap ≥ 2) should collapse into one column.

    Particle 0: t ∈ {0,1,2}; particle 1: t ∈ {4,5,6}; particle 2: t ∈ {8,9,10}
    → all share column 0, in temporal order.
    """
    particle_idx = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int
    )  # compact IDs
    time_idx = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10], dtype=int)

    new_idx, column_origins, t_first, t_last = _greedy_compress_particles(
        particle_idx, time_idx
    )

    assert len(column_origins) == 1, f"Expected 1 column, got {len(column_origins)}"
    assert column_origins[0] == [0, 1, 2], f"Wrong origins: {column_origins[0]}"
    assert (new_idx == 0).all()


def test_greedy_compress_overlapping():
    """Two sets of particles: some overlap, some don't.

    particle 0: t ∈ {0,1,2}  \\ overlap → separate columns
    particle 3: t ∈ {0,1,2}  /
    particle 1: t ∈ {4,5,6}  → fits after particle 0 (gap ≥ 2) in column 0
    particle 2: t ∈ {8,9,10} → fits after particle 1 in column 0
    """
    particle_idx = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int
    )
    time_idx = np.array(
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 0, 1, 2], dtype=int
    )

    new_idx, column_origins, t_first, t_last = _greedy_compress_particles(
        particle_idx, time_idx
    )

    assert len(column_origins) == 2, f"Expected 2 columns, got {len(column_origins)}"
    # Column 0 should pack particles 0, 1, 2 (non-overlapping)
    assert column_origins[0] == [0, 1, 2], f"col 0: {column_origins[0]}"
    # Column 1 should contain particle 3 (overlaps 0)
    assert column_origins[1] == [3], f"col 1: {column_origins[1]}"


def test_greedy_compress_gap_exactly_one():
    """Gap of exactly 1 frame is NOT enough (requires ≥ 2-frame gap)."""
    # particle 0: t ∈ {0,1}; particle 1: t ∈ {2,3}  → gap of 1 → separate columns
    particle_idx = np.array([0, 0, 1, 1], dtype=int)
    time_idx = np.array([0, 1, 2, 3], dtype=int)

    _new, column_origins, _tf, _tl = _greedy_compress_particles(
        particle_idx, time_idx
    )

    assert len(column_origins) == 2, (
        "Gap of 1 is not enough; expected 2 columns"
    )


def test_greedy_compress_gap_exactly_two():
    """Gap of exactly 2 frames IS enough (condition: t_last[prev] <= t_first[new] - 2)."""
    # particle 0: t ∈ {0,1}; particle 1: t ∈ {3,4}  → gap of 2 frames → share column
    particle_idx = np.array([0, 0, 1, 1], dtype=int)
    time_idx = np.array([0, 1, 3, 4], dtype=int)

    _new, column_origins, _tf, _tl = _greedy_compress_particles(
        particle_idx, time_idx
    )

    assert len(column_origins) == 1, (
        "Gap of 2 should be enough; expected 1 column"
    )


def test_assemble_compress_mask_gap():
    """After compression, the gap frames must have mask=False (no spurious increments)."""
    # particle 0: t=0,1; particle 1: t=3,4 → share compressed column 0
    # frames t=2 must be mask=False for column 0
    rng = np.random.default_rng(7)
    particle_idx = np.array([0, 0, 1, 1], dtype=int)
    time_idx = np.array([0, 1, 3, 4], dtype=int)
    state_vectors = rng.normal(size=(4, 2))

    X, mask, id_map = assemble_X_from_columns(
        particle_idx, time_idx, state_vectors,
        relabel=True, compress_particles=True
    )

    assert X.shape == (5, 1, 2), f"Expected (5,1,2), got {X.shape}"
    assert mask[0, 0] and mask[1, 0]   # particle 0 present
    assert not mask[2, 0]              # gap — must be False
    assert mask[3, 0] and mask[4, 0]   # particle 1 present
    assert isinstance(id_map, dict)
    assert id_map["column_origins"] == [[0, 1]]


def test_from_columns_compress_end_to_end():
    """End-to-end test: from_columns with compress_particles=True.

    Tests that:
    - X.N is compressed
    - meta['particle_column_map'] is recorded
    - extras_local['original_particle_id'] is NOT set (replaced by column_map)
    - Per-particle P_ extras are reindexed as TimeSeriesExtra
    - TP_ extras are reindexed correctly
    """
    from SFI.trajectory.dataset import TimeSeriesExtra, time_series_extra

    rng = np.random.default_rng(99)
    # 3 particles, non-overlapping: 0 at t=0,1; 1 at t=3,4; 2 at t=6,7
    particle_idx = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    time_idx = np.array([0, 1, 3, 4, 6, 7], dtype=int)
    d = 2
    state_vectors = rng.normal(size=(6, d))

    # Per-particle constant extra (shape N=3)
    mass = np.array([1.0, 2.0, 3.0])
    # TP extra (T=8, N=3, 1)
    T_raw = 8
    N_raw = 3
    signal_data = rng.normal(size=(T_raw, N_raw, 1))
    # Build sparse tables: only actually present (t, pid) rows
    extras_local = {
        "mass": mass,
        "signal": time_series_extra(signal_data),
    }

    coll = TrajectoryCollection.from_columns(
        particle_idx=particle_idx,
        time_idx=time_idx,
        state_vectors=state_vectors,
        extras_local=extras_local,
        dt=0.1,
        relabel=True,
        compress_particles=True,
    )

    ds = coll.datasets[0]

    # All 3 particles collapse to 1 column
    assert ds.N == 1, f"Expected N=1 after compression, got {ds.N}"
    assert ds.T == T_raw

    # Mapping stored in meta
    assert "particle_column_map" in ds.meta
    assert ds.meta["particle_column_map"] == [[0, 1, 2]]

    # original_particle_id should NOT be in extras_local (compress path uses meta instead)
    assert "original_particle_id" not in ds.extras_local

    # mass extra: promoted to TimeSeriesExtra of shape (T, 1)
    assert "mass" in ds.extras_local
    mass_extra = ds.extras_local["mass"]
    assert isinstance(mass_extra, TimeSeriesExtra)
    mass_arr = np.asarray(mass_extra.data)
    assert mass_arr.shape[:2] == (T_raw, 1)
    # At t=0,1 → particle 0 had mass 1.0
    np.testing.assert_allclose(mass_arr[0, 0], 1.0)
    np.testing.assert_allclose(mass_arr[1, 0], 1.0)
    # At t=3,4 → particle 1 had mass 2.0
    np.testing.assert_allclose(mass_arr[3, 0], 2.0)
    np.testing.assert_allclose(mass_arr[4, 0], 2.0)
    # At t=6,7 → particle 2 had mass 3.0
    np.testing.assert_allclose(mass_arr[6, 0], 3.0)
    np.testing.assert_allclose(mass_arr[7, 0], 3.0)

    # signal extra: reindexed TimeSeriesExtra
    assert "signal" in ds.extras_local
    sig_extra = ds.extras_local["signal"]
    assert isinstance(sig_extra, TimeSeriesExtra)
    sig_arr = np.asarray(sig_extra.data)
    assert sig_arr.shape[:2] == (T_raw, 1)
    np.testing.assert_allclose(sig_arr[0, 0], signal_data[0, 0])
    np.testing.assert_allclose(sig_arr[3, 0], signal_data[3, 1])
    np.testing.assert_allclose(sig_arr[6, 0], signal_data[6, 2])


def test_from_columns_relabel_stores_original_id():
    """Without compression, relabel=True stores original_particle_id in extras_local."""
    rng = np.random.default_rng(11)
    particle_idx = np.array([0, 0, 0, 0, 5, 5, 5, 5, 100, 100, 100, 100], dtype=int)
    time_idx = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int)
    state_vectors = rng.normal(size=(12, 2))

    coll = TrajectoryCollection.from_columns(
        particle_idx=particle_idx,
        time_idx=time_idx,
        state_vectors=state_vectors,
        dt=0.1,
        relabel=True,
        compress_particles=False,
    )

    ds = coll.datasets[0]
    assert ds.N == 3
    assert "original_particle_id" in ds.extras_local
    np.testing.assert_array_equal(ds.extras_local["original_particle_id"], [0, 5, 100])


def _write_messy_csv(path, T=4, N=2, seed=3):
    """CSV with non-canonical column order plus a junk numeric column."""
    rng = np.random.default_rng(seed)
    lines = ["# dt: 0.05", "t,x,y,quality,particle"]
    for t in range(T):
        for n in range(N):
            lines.append(f"{t},{rng.normal():.6f},{rng.normal():.6f},0.99,{n}")
    path.write_text("\n".join(lines) + "\n")


def test_load_csv_named_columns_excludes_junk(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "messy.csv"
    _write_messy_csv(p)
    coll = TrajectoryCollection.load(
        p, particle_column="particle", time_column="t", state_columns=("x", "y")
    )
    ds = coll.datasets[0]
    assert np.asarray(ds._X3d()).shape == (4, 2, 2)
    assert float(ds.dt) == pytest.approx(0.05)


def test_load_csv_named_columns_missing_raises(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "messy.csv"
    _write_messy_csv(p)
    with pytest.raises(ValueError, match="not found"):
        TrajectoryCollection.load(p, particle_column="nope", time_column="t")


def test_load_csv_positional_defaults_unchanged(tmp_path):
    """Backcompat: canonical column order still loads with the int defaults."""
    pytest.importorskip("pandas")
    p = tmp_path / "canonical.csv"
    rng = np.random.default_rng(4)
    lines = ["# dt: 0.1", "particle_id,frame,x,y"]
    for t in range(3):
        for n in range(2):
            lines.append(f"{n},{t},{rng.normal():.6f},{rng.normal():.6f}")
    p.write_text("\n".join(lines) + "\n")
    coll = TrajectoryCollection.load(p)
    assert np.asarray(coll.datasets[0]._X3d()).shape == (3, 2, 2)
