# TODO: review this file
import jax.numpy as jnp
import numpy as np

from SFI.integrate.api import integrate
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset, time_series_extra


class DummyProgram:
    """Consumes X, mask, extras. Returns per-particle value."""

    def require(self):
        # integrator will add "__dt__"
        return {"X", "mask", "extras"}

    def estimate_bytes_per_sample(self, sample):
        # conservative small constant
        return 1024

    def __call__(self, **row):
        X = row["X"]  # (N,d)
        # simple per-particle value: sum over d, scaled and biased by extras
        scale = row["extras"]["scale"]  # scalar
        bias = row["extras"]["bias_t"]  # scalar time-series
        y = jnp.sum(X, axis=-1) * scale + bias  # (N,)
        return y


def make_ds(T=6, N=3, d=2, *, use_t=True):
    X = jnp.arange(T * N * d, dtype=jnp.float32).reshape(T, N, d)
    mask = jnp.ones((T, N), dtype=bool)
    # knock out one particle at one neighbor time to exercise mask tightening
    mask = mask.at[1, 0].set(False)  # particle 0 invalid at t=1
    if use_t:
        t = jnp.linspace(0.0, 0.5 * (T - 1), T, dtype=jnp.float32)
        dt = None
    else:
        t = None
        dt = 0.5
    eg = {
        "scale": jnp.array(2.0, dtype=jnp.float32),
        "bias_t": time_series_extra(jnp.linspace(0.0, 1.0, T, dtype=jnp.float32)),
    }
    return TrajectoryDataset.from_arrays(
        X=X, t=t, dt=dt, mask=mask, extras_global=eg, extras_local={}, meta={}
    )


def brute_force_integral(ds, program, reduce="sum"):
    # emulate integrate() on CPU in Python
    require = set(program.require()) | {"__dt__"}
    idx = np.array(ds.valid_indices(require), dtype=np.int32)
    if idx.size == 0:
        return 0.0
    out = 0.0
    denom = 0.0
    prod = ds.make_producer(require, include_dt=True)
    for t in idx:
        row = prod(jnp.array(int(t)))
        y = program(**row)  # (N,)
        m = row["mask_out"] if "mask_out" in row else jnp.ones_like(y, bool)
        y_masked = jnp.where(m, y, 0.0)
        y_part = jnp.sum(y_masked)  # reduce over particles
        out += float(y_part * row["dt"])
        denom += float(row["dt"] * row["N_active"])
    if reduce == "sum":
        return out
    if denom <= 0:
        raise ValueError("non-positive exposure")
    return out / denom


def test_integrate_sum_and_mean_match_bruteforce():
    ds = make_ds(T=6, N=3, d=2, use_t=True)
    coll = TrajectoryCollection.from_dataset(ds)
    prog = DummyProgram()

    # sum
    val_sum = integrate(coll, prog, reduce="sum", chunk_target_bytes=64, subsampling=1)
    ref_sum = brute_force_integral(ds, prog, reduce="sum")
    np.testing.assert_allclose(
        np.asarray(val_sum), np.asarray(ref_sum), rtol=1e-6, atol=1e-6
    )

    # mean
    val_mean = integrate(
        coll, prog, reduce="mean", chunk_target_bytes=48, subsampling=1
    )
    ref_mean = brute_force_integral(ds, prog, reduce="mean")
    np.testing.assert_allclose(
        np.asarray(val_mean), np.asarray(ref_mean), rtol=1e-6, atol=1e-6
    )


def test_padding_last_chunk_is_neutral():
    ds = make_ds(T=7, N=2, d=1, use_t=False)  # constant dt path
    coll = TrajectoryCollection.from_dataset(ds)
    prog = DummyProgram()
    # bytes_hint small → small K_fixed → many chunks and a padded tail
    val = integrate(coll, prog, reduce="sum", chunk_target_bytes=32, subsampling=1)
    ref = brute_force_integral(ds, prog, reduce="sum")
    np.testing.assert_allclose(np.asarray(val), np.asarray(ref), rtol=1e-6, atol=1e-6)
