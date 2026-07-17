"""TrajectoryDataset/Collection.split_time correctness."""

import numpy as np
import pytest

from SFI.trajectory import TrajectoryCollection
from SFI.trajectory.dataset import (
    FunctionExtra,
    TimeSeriesExtra,
    TrajectoryDataset,
    time_series_extra,
)


def _dataset(T=10, N=2, d=2, with_t=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, N, d))
    mask = np.ones((T, N), dtype=bool)
    mask[3, 1] = False
    kw = dict(t=np.arange(T) * 0.1) if with_t else dict(dt=0.1)
    return TrajectoryDataset.from_arrays(
        X=X,
        mask=mask,
        extras_global={
            "box": np.array([1.0, 1.0]),
            "drive": time_series_extra(np.arange(T, dtype=float)),
            "fext": FunctionExtra(lambda x: x),
        },
        extras_local={"signal": time_series_extra(rng.normal(size=(T, N)))},
        meta={"unit": "arb"},
        **kw,
    )


def test_split_shapes_and_values():
    ds = _dataset()
    train, test = ds.split_time(0.7)
    Xtr, Xte = np.asarray(train.X), np.asarray(test.X)
    assert Xtr.shape[0] == 7 and Xte.shape[0] == 3
    np.testing.assert_allclose(np.concatenate([Xtr, Xte]), np.asarray(ds.X))
    # masks sliced too
    assert not bool(np.asarray(train.mask)[3, 1])


def test_split_absolute_time_kept():
    ds = _dataset(with_t=True)
    train, test = ds.split_time(0.7)
    np.testing.assert_allclose(np.asarray(test.t), np.arange(7, 10) * 0.1)


def test_split_extras_contract():
    ds = _dataset()
    train, test = ds.split_time(0.7)
    # TimeSeriesExtra sliced
    assert isinstance(test.extras_global["drive"], TimeSeriesExtra)
    np.testing.assert_allclose(np.asarray(test.extras_global["drive"].data), [7.0, 8.0, 9.0])
    np.testing.assert_allclose(
        np.asarray(train.extras_local["signal"].data),
        np.asarray(ds.extras_local["signal"].data)[:7],
    )
    # FunctionExtra and statics pass through
    assert isinstance(test.extras_global["fext"], FunctionExtra)
    np.testing.assert_allclose(np.asarray(test.extras_global["box"]), [1.0, 1.0])


def test_split_callable_extra_offset():
    ds = TrajectoryDataset.from_arrays(
        X=np.zeros((10, 1, 1)),
        dt=0.1,
        extras_global={"gen": lambda t_idx, context=None: np.asarray(t_idx, dtype=float)},
    )
    train, test = ds.split_time(0.7)
    assert float(train.extras_global["gen"](0)) == 0.0
    # the test half keeps seeing its original absolute indices
    assert float(test.extras_global["gen"](0)) == 7.0


def test_split_gap_and_meta():
    ds = _dataset(T=12)
    train, test = ds.split_time(0.5, gap=2)
    assert np.asarray(train.X).shape[0] == 6
    assert np.asarray(test.X).shape[0] == 4
    assert train.meta["split_time"]["role"] == "train"
    assert test.meta["split_time"] == {
        "role": "test",
        "start": 8,
        "stop": 12,
        "fraction": 0.5,
        "gap": 2,
    }


def test_split_too_short_raises():
    ds = _dataset(T=4)
    with pytest.raises(ValueError, match="at least 2 frames"):
        ds.split_time(0.9)
    with pytest.raises(ValueError, match="fraction"):
        ds.split_time(1.5)


def test_collection_split_and_weights():
    ds1, ds2 = _dataset(T=10, seed=1), _dataset(T=20, seed=2)
    coll = TrajectoryCollection.from_dataset(ds1).concat(
        [TrajectoryCollection.from_dataset(ds2)]
    )
    train, test = coll.split_time(0.8)
    assert len(train.datasets) == 2 and len(test.datasets) == 2
    assert np.asarray(train.datasets[0].X).shape[0] == 8
    assert np.asarray(test.datasets[1].X).shape[0] == 4
    # 'pool' reweight gives unnormalised unit weights (one per dataset).
    np.testing.assert_allclose(np.asarray(train.weights), 1.0, atol=1e-6)

    train_k, _ = coll.split_time(0.8, reweight="keep")
    np.testing.assert_allclose(
        np.asarray(train_k.weights), np.asarray(coll.weights), atol=1e-6
    )
