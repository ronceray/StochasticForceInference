"""TrajectoryCollection.from_dataframe and named-column loading."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from SFI.trajectory import TimeSeriesExtra, TrajectoryCollection


def _tracks_df(T=6, N=2, seed=0, particle_name="track_id", time_name="frame"):
    """A messy tracking table: shuffled columns + a junk column."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(T):
        for n in range(N):
            rows.append(
                {
                    "junk": "detector_7",
                    "y": float(rng.normal()),
                    particle_name: n,
                    "x": float(rng.normal()),
                    time_name: t,
                }
            )
    return pd.DataFrame(rows)


def test_explicit_columns_round_trip():
    df = _tracks_df()
    coll = TrajectoryCollection.from_dataframe(
        df, particle="track_id", time="frame", coords=("x", "y"), dt=0.1
    )
    ds = coll.datasets[0]
    X = np.asarray(ds._X3d())
    assert X.shape == (6, 2, 2)
    # column order respected: coords=("x","y") even though df has y before x
    df0 = df[(df.track_id == 0) & (df.frame == 0)].iloc[0]
    assert np.allclose(X[0, 0], [df0.x, df0.y])
    assert float(ds.dt) == pytest.approx(0.1)


def test_autodetection_with_explicit_coords():
    df = _tracks_df(particle_name="particle", time_name="frame")
    coll = TrajectoryCollection.from_dataframe(df, coords=("x", "y"), dt=0.1)
    assert np.asarray(coll.datasets[0]._X3d()).shape == (6, 2, 2)


def test_default_coords_on_clean_table():
    df = _tracks_df().drop(columns=["junk"])
    coll = TrajectoryCollection.from_dataframe(df, particle="track_id", time="frame", dt=0.1)
    # remaining columns (y, x) become coordinates in dataframe order
    assert np.asarray(coll.datasets[0]._X3d()).shape == (6, 2, 2)


def test_autodetect_requires_unambiguous_time():
    df = _tracks_df(time_name="frame")
    df["time"] = df["frame"]  # two time candidates
    with pytest.raises(ValueError, match="ambiguous time"):
        TrajectoryCollection.from_dataframe(df, particle="track_id", coords=("x", "y"))


def test_missing_time_raises():
    df = _tracks_df().rename(columns={"frame": "zzz"})
    with pytest.raises(ValueError, match="no time column"):
        TrajectoryCollection.from_dataframe(df, particle="track_id", coords=("x", "y"))


def test_single_trajectory_without_particle_column():
    df = _tracks_df(N=1).drop(columns=["track_id"])
    coll = TrajectoryCollection.from_dataframe(df, time="frame", coords=("x", "y"), dt=0.1)
    assert np.asarray(coll.datasets[0]._X3d()).shape == (6, 1, 2)


def test_float_time_column_becomes_time_axis():
    df = _tracks_df()
    df["frame"] = df["frame"].astype(float) * 0.05  # physical times
    coll = TrajectoryCollection.from_dataframe(
        df, particle="track_id", time="frame", coords=("x", "y")
    )
    ds = coll.datasets[0]
    t = np.asarray(ds.t)
    assert t.shape == (6,)
    assert np.allclose(t, np.arange(6) * 0.05)


def test_prefixed_extras_and_user_merge():
    df = _tracks_df()
    df["TG_temp"] = 300.0 + df["frame"].astype(float)
    coll = TrajectoryCollection.from_dataframe(
        df,
        particle="track_id",
        time="frame",
        coords=("x", "y"),
        dt=0.1,
        extras_global={"box": np.array([1.0, 1.0])},
    )
    eg = coll.datasets[0].extras_global
    assert isinstance(eg["temp"], TimeSeriesExtra)
    assert np.allclose(np.asarray(eg["temp"].data), 300.0 + np.arange(6))
    assert np.allclose(np.asarray(eg["box"]), [1.0, 1.0])


def test_unknown_explicit_column_raises():
    df = _tracks_df()
    with pytest.raises(ValueError, match="not found"):
        TrajectoryCollection.from_dataframe(df, particle="nope", time="frame", coords=("x", "y"))
    with pytest.raises(ValueError, match="coords columns not found"):
        TrajectoryCollection.from_dataframe(df, particle="track_id", time="frame", coords=("x", "z"))
