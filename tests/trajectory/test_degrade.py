# TODO: review this file
import jax.numpy as jnp
import numpy as np

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset
from SFI.trajectory.degrade import degrade_collection, degrade_dataset


def make_ds(T=8, N=3, d=2, dt=0.1):
    t = jnp.arange(T, dtype=float) * dt
    X = jnp.stack([t, t], axis=-1)  # (T,2), same on both coords
    X = X[:, None, :].repeat(N, axis=1)  # (T,N,2)
    mask = jnp.ones((T, N), dtype=bool)
    eg = {"t": np.asarray(t)}
    el = {"signal": np.arange(T * N, dtype=float).reshape(T, N, 1)}
    return TrajectoryDataset.from_arrays(
        X=X,
        t=t,
        mask=mask,
        extras_global=eg,
        extras_local=el,
        meta={"dt": dt},
    )


def test_degrade_dataset_downsample_and_blur_extras():
    ds = make_ds(T=9, N=4, d=2, dt=0.2)
    # downsample=2, motion_blur=1 => keep times 0,2,4,6,...; window size 2
    ds2 = degrade_dataset(
        ds,
        downsample=2,
        motion_blur=1,
        noise=None,
        ROI=None,
        seed=0,
    )

    t = np.asarray(ds.t)
    keep = np.arange(0, max(ds.T - 1, 0), 2)
    t_expected = np.array(
        [np.mean(t[k : k + 2]) for k in keep],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(ds2.t), t_expected)

    # extras_local: time-dependent signal downsampled in time
    sig2 = np.asarray(ds2.extras_local["signal"])
    assert sig2.shape[0] == len(t_expected)

    # Mask unchanged when ROI/loss/noise are neutral in this test
    assert np.all(np.asarray(ds2._M2d()))


def test_degrade_dataset_roi_and_loss_masks():
    ds = make_ds(T=8, N=5, d=2, dt=0.1)
    # ROI: tight square; plus random data loss
    ds2 = degrade_dataset(
        ds,
        downsample=1,
        motion_blur=0,
        ROI=0.01,
        data_loss_fraction=0.25,
        seed=42,
    )
    M = np.asarray(ds2._M2d())
    assert M.shape == (ds.T, ds.N)
    # Some entries must be False due to ROI or loss
    assert M.sum() < ds.T * ds.N


def test_degrade_collection_reweights_and_preserves_length():
    d1 = make_ds(T=10, N=3, d=2, dt=0.1)
    d2 = make_ds(T=12, N=3, d=2, dt=0.1)
    coll = TrajectoryCollection.from_dataset(d1).concat([d2])

    out = degrade_collection(coll, downsample=2, motion_blur=1)
    assert len(out.datasets) == 2

    # Weights are unnormalised multipliers; the default 'pool' gives unit weights.
    w = np.asarray(out.weights)
    assert np.allclose(w, 1.0)
