# TODO: review this file
import jax.numpy as jnp

from SFI.trajectory.dataset import TrajectoryDataset, time_series_extra


def make_ds(T=6, N=3, d=2, *, use_t=True):
    X = jnp.arange(T * N * d, dtype=jnp.float32).reshape(T, N, d)
    mask = jnp.ones((T, N), dtype=bool)
    if use_t:
        t = jnp.linspace(0.0, 0.5 * (T - 1), T, dtype=jnp.float32)
        dt = None
    else:
        t = None
        dt = 0.5
    eg = {
        "scale": jnp.array(2.0),
        "bias_t": time_series_extra(jnp.arange(T, dtype=jnp.float32)),
    }
    el = {"perp": time_series_extra(jnp.arange(T * N, dtype=jnp.float32).reshape(T, N))}
    return TrajectoryDataset.from_arrays(
        X=X, t=t, dt=dt, mask=mask, extras_global=eg, extras_local=el
    )


def test_valid_indices_offsets_dx_and_dt_window():
    ds = make_ds(T=5, N=1, d=1, use_t=True)
    # dX implies (0,+1) window -> valid 0..3
    idx_dx = ds.valid_indices({"dX"})
    assert jnp.all(idx_dx == jnp.array([0, 1, 2, 3], dtype=jnp.int32))
    # "__dt__" should imply (0,+1) as well (required by integrate mean path)
    idx_dtwin = ds.valid_indices({"__dt__"})
    assert jnp.all(idx_dtwin == jnp.array([0, 1, 2, 3], dtype=jnp.int32))


def test_valid_indices_subsampling():
    ds = make_ds(T=10, N=1, d=1)
    idx = ds.valid_indices({"X_plus"}, subsampling=3)  # needs t+1 valid => up to 8
    assert jnp.all(idx == jnp.array([0, 3, 6], dtype=jnp.int32))


def test_producer_single_row_shapes_and_extras():
    ds = make_ds(T=6, N=2, d=3)
    require = {"X", "dX", "mask", "extras"}
    prod = ds.make_producer(require, include_dt=True)
    t = jnp.array(2, dtype=jnp.int32)
    row = prod(t)
    # streams
    assert row["X"].shape == (ds.N, ds.d)
    assert row["dX"].shape == (ds.N, ds.d)
    assert row["mask"].shape == (ds.N,)
    # dt present
    assert "dt" in row and row["dt"].shape == ()
    # counts
    assert row["N_total"].shape == ()
    assert row["N_active"].shape == ()
    # extras: static + time-series
    ex = row["extras"]
    assert jnp.shape(ex["scale"]) == ()
    assert jnp.shape(ex["bias_t"]) == ()
    assert jnp.shape(ex["perp"]) == (ds.N,)
