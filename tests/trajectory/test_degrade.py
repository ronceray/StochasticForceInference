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


# --------------------------------------------------------------------------- #
# Extras resampling rule (regression: static CSR tables were cut to the kept
# frames and cast to float64; explicit TimeSeriesExtra were not resampled).
# --------------------------------------------------------------------------- #


def _ring_csr(N):
    import jax.numpy as jnp

    nbrs = np.array([[(i - 1) % N, (i + 1) % N] for i in range(N)], np.int32).reshape(-1)
    return {
        "ring/indptr": jnp.asarray(np.arange(0, 2 * N + 1, 2, dtype=np.int32)),
        "ring/indices": jnp.asarray(nbrs),
    }


def test_degrade_never_resamples_integer_structural_extras():
    from SFI.trajectory import TrajectoryCollection as TC

    T, N, d = 11, 11, 2  # indptr has length N + 1 = 12 >= T
    X = np.random.default_rng(0).normal(size=(T, N, d))
    ex = _ring_csr(N)
    coll = TC.from_arrays(X=X, dt=0.1, extras_global=ex, extras_local={"species": np.arange(N) % 2})
    out = coll.degrade(downsample=2).datasets[0]
    assert out.T == 6
    for k, v in ex.items():
        got = out.extras_global[k]
        assert np.asarray(got).dtype == np.asarray(v).dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(v))
    np.testing.assert_array_equal(np.asarray(out.extras_local["species"]), np.arange(N) % 2)


def test_degrade_resamples_time_series_extras_like_x():
    import jax.numpy as jnp

    from SFI.trajectory import TimeSeriesExtra, time_series_extra

    T, N, d = 13, 4, 2
    rng = np.random.default_rng(1)
    X = rng.normal(size=(T, N, d))
    h = np.linspace(0.0, 1.0, T)
    labels = np.arange(T, dtype=np.int32) % 3
    plain32 = (np.arange(T) ** 2).astype(np.float32)
    longer = np.linspace(0.0, 1.0, T + 1)  # leading axis != T: static
    loc_ts = rng.normal(size=(T, N))
    ds = TrajectoryDataset.from_arrays(
        X=X,
        dt=0.1,
        extras_global={"h": time_series_extra(h), "lab": time_series_extra(labels), "p32": plain32, "long": longer},
        extras_local={"s": time_series_extra(loc_ts), "radius": np.full(N, 0.5)},
    )
    ds2 = degrade_dataset(ds, downsample=3, motion_blur=1)
    keep = np.arange(0, T - 1, 3)
    win = lambda a: np.stack([np.mean(a[k : k + 2], axis=0) for k in keep])  # noqa: E731

    eg, el = ds2.extras_global, ds2.extras_local
    assert isinstance(eg["h"], TimeSeriesExtra) and isinstance(el["s"], TimeSeriesExtra)
    np.testing.assert_allclose(np.asarray(eg["h"].data), win(h), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(el["s"].data), win(loc_ts), rtol=1e-5, atol=1e-6)
    # Integer time series: taken at the window start, dtype kept.
    np.testing.assert_array_equal(np.asarray(eg["lab"].data), labels[keep])
    assert np.asarray(eg["lab"].data).dtype == np.int32
    # Plain floating array of leading length exactly T: legacy resampling, dtype kept.
    assert np.asarray(eg["p32"]).dtype == np.float32
    np.testing.assert_allclose(np.asarray(eg["p32"]), win(plain32.astype(float)), rtol=1e-6)
    # Static arrays untouched.
    np.testing.assert_array_equal(np.asarray(eg["long"]), longer)
    np.testing.assert_array_equal(np.asarray(el["radius"]), np.full(N, 0.5))
    # The degraded dataset slices its time series consistently at every frame.
    ex = ds2.build_extras(jnp.arange(ds2.T))
    assert np.asarray(ex["h"]).shape == (ds2.T,)


def test_degrade_reindexes_callable_extras():
    import jax.numpy as jnp

    T, N, d = 10, 2, 1
    X = np.zeros((T, N, d))
    gen = lambda t_idx, context=None: 1.0 * jnp.asarray(t_idx)  # noqa: E731
    ds = TrajectoryDataset.from_arrays(X=X, dt=0.1, extras_global={"g": gen})
    ds2 = degrade_dataset(ds, downsample=2)
    ex = ds2.build_extras(jnp.arange(ds2.T))
    np.testing.assert_array_equal(np.asarray(ex["g"]), np.arange(0, T, 2))


def test_degrade_then_fit_on_user_csr():
    """Original crash: the next fit failed with 'Indexer must have integer type'."""
    import jax.numpy as jnp

    from SFI import OverdampedLangevinInference
    from SFI.statefunc import make_basis, make_interactor

    N, d = 20, 2  # T = 21 frames = len(indptr): the old rule resampled the CSR
    rng = np.random.default_rng(2)
    X = np.cumsum(0.1 * rng.standard_normal((N + 1, N, d)), axis=0)
    ex = _ring_csr(N)
    inter = make_interactor(lambda Xk: (Xk[1] - Xk[0])[:, None], dim=d, rank=1, K=2, n_features=1)
    B = make_basis(lambda x: x[:, None], dim=d, rank=1, n_features=1) & inter.dispatch_pairs_from_extras(
        indptr_key="ring/indptr", indices_key="ring/indices"
    )
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.01, extras_global=ex).degrade(downsample=2)
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant(method="MSD")
    inf.infer_force_linear(B)
    assert np.all(np.isfinite(np.asarray(inf.force_coefficients)))
    assert np.asarray(coll.datasets[0].extras_global["ring/indptr"]).dtype == jnp.int32
