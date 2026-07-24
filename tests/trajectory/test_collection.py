# TODO: review this file
import jax.numpy as jnp
import pytest

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset


def make_ds(T, N=1, d=1, dt=0.5):
    X = jnp.zeros((T, N, d), dtype=jnp.float32)
    mask = jnp.ones((T, N), dtype=bool)
    return TrajectoryDataset.from_arrays(X=X, dt=dt, t=None, mask=mask)


def test_weights_pool_per_dataset_and_iter_slices_chunking():
    ds1 = make_ds(10)
    ds2 = make_ds(6)
    coll = TrajectoryCollection(
        [ds1, ds2],
        jnp.ones((2,), dtype=jnp.float32),
    ).with_weights("pool")

    # "pool": equal unit multipliers (unnormalised)
    w = coll.weights
    assert w.shape == (2,)
    assert float(w[0]) == pytest.approx(float(w[1]))

    # "per_dataset": multiplier proportional to 1/Teff, up-weighting the
    # smaller dataset so each contributes equally.
    cpd = TrajectoryCollection(
        [ds1, ds2],
        jnp.ones((2,), dtype=jnp.float32),
    ).with_weights("per_dataset", required={"X"})
    assert float(cpd.weights[0] / cpd.weights[1]) == pytest.approx(6 / 10)

    # iter_slices: bytes_hint controls chunking
    require = {"X", "__dt__"}
    bytes_hint = 16
    chunk_target_bytes = 64
    chunks = list(
        coll.iter_slices(
            require=require,
            bytes_hint=bytes_hint,
            chunk_target_bytes=chunk_target_bytes,
        )
    )
    sizes = [int(p["t_idx"].shape[0]) for p in chunks]
    # At least a couple of full chunks exist
    assert sizes.count(4) >= 2

    # All valid indices across both datasets must be covered
    idx1 = ds1.valid_indices(require)
    idx2 = ds2.valid_indices(require)
    assert sum(sizes) == int(idx1.shape[0] + idx2.shape[0])


def test_peek_row_structure():
    ds = make_ds(5, N=2, d=3)
    coll = TrajectoryCollection.from_dataset(ds)
    row = coll.peek_row(require={"X", "mask", "__dt__"})
    assert {"X", "mask", "dt", "N_total", "N_active"}.issubset(row.keys())
    assert row["X"].shape == (2, 3)
    assert row["mask"].shape == (2,)
