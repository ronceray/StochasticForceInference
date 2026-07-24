"""Stable per-dataset identity (uuid) and registry-derived dataset_index."""

from __future__ import annotations

import numpy as np

from SFI import TrajectoryCollection


def _coll(seed, T=60):
    rng = np.random.default_rng(seed)
    return TrajectoryCollection.from_arrays(X=rng.standard_normal((T, 1, 1)), dt=0.1)


def test_dataset_has_stable_uuid():
    ds = _coll(0).datasets[0]
    assert isinstance(ds.uuid, str) and ds.uuid


def test_uuid_survives_degradation():
    coll = _coll(1)
    u = coll.datasets[0].uuid
    assert coll.degrade(downsample=2).datasets[0].uuid == u


def test_dataset_index_is_registry_derived():
    pooled = _coll(2).concat([_coll(3)], weights="pool")
    assert pooled.datasets[0].uuid != pooled.datasets[1].uuid
    assert pooled.dataset_index(0) == 0
    assert pooled.dataset_index(1) == 1
