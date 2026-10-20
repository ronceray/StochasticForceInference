"""Compiled-run cache: array extras are traced arguments of one cached executable.

``simulate`` used to close the extras over a fresh jitted step, so every
neighbour-list rebuild in :func:`simulate_chunked` recompiled the run.  The run
is now compiled once per configuration and reused while the extras keep their
shapes; these tests pin that down (compile count, per-call overhead, bitwise
trajectories against the uncached path) and check that the cache never serves
stale values.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from SFI.bases.pairs import pair_direction, parametric_radial_kernel
from SFI.langevin import OverdampedProcess, UnderdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.statefunc import make_sf
from SFI.utils.neighbors import make_neighbor_extras

N, L, CUTOFF, SKIN = 40, 12.0, 2.5, 1.0
BOX = np.array([L, L])

# Few-frame runs are the point here, not inference-ready data.
pytestmark = pytest.mark.filterwarnings("ignore:Very short trajectory")


def _pair_force():
    kw = dict(dim=2, box="extras", spatial_dims=slice(0, 2))
    k = parametric_radial_kernel(lambda r, p: -p["eps"] * jnp.exp(-r), params={"eps": ()}, **kw)
    return (k * pair_direction(**kw)).dispatch_pairs_from_extras(
        indptr_key="indptr", indices_key="indices", return_as="psf", drop_features=True
    )


def _x0(seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.0, L, size=(N, 2)))


def _pair_process(x0):
    extras = {"box": jnp.asarray(BOX), **make_neighbor_extras(np.asarray(x0), CUTOFF + SKIN, BOX)}
    proc = OverdampedProcess(_pair_force(), D=0.05, extras_global=extras)
    proc.set_params(theta_F={"eps": 1.0})
    proc.initialize(x0)
    return proc


def _chunked(proc, *, jit_compile=True, n_rebuilds=6):
    return simulate_chunked(
        proc,
        dt=0.01,
        Nsteps=3 * n_rebuilds,
        key=random.PRNGKey(7),
        cutoff=CUTOFF,
        box=BOX,
        skin=SKIN,
        rebuild_every=3,
        oversampling=2,
        nnz_safety=2.0,
        jit_compile=jit_compile,
    )


def test_chunked_rebuilds_reuse_one_compiled_run():
    proc = _pair_process(_x0())
    _chunked(proc)
    assert len(proc._run_cache) == 1
    (fn,) = proc._run_cache.values()
    assert fn._cache_size() == 1  # traced once: every rebuild hit the same executable


def test_chunked_cached_trajectory_is_bitwise_the_uncached_one():
    X_cached = _chunked(_pair_process(_x0())).to_array(axis="time")
    X_plain = _chunked(_pair_process(_x0()), jit_compile=False).to_array(axis="time")
    np.testing.assert_array_equal(np.asarray(X_cached), np.asarray(X_plain))


def test_rebuild_overhead_after_first_compile():
    """A same-shaped extras update costs a dispatch, not a compile (was ~1 s per rebuild)."""
    x0 = _x0()
    proc = _pair_process(x0)
    padded = {k: proc.extras_global[k] for k in ("indptr", "indices")}
    proc.simulate(0.01, Nsteps=2, key=random.PRNGKey(0), oversampling=1, compute_observables=False)
    times = []
    for i in range(8):
        # A fresh array object each time, as a neighbour rebuild produces.
        proc.set_extras(
            extras_global={**proc.extras_global, **{k: jnp.array(np.asarray(v)) for k, v in padded.items()}}
        )
        t = time.perf_counter()
        coll = proc.simulate(0.01, Nsteps=2, key=random.PRNGKey(i), oversampling=1, compute_observables=False)
        np.asarray(coll.datasets[0].X)
        times.append(time.perf_counter() - t)
    assert np.median(times) < 0.05, times


def test_updated_array_extras_are_not_stale():
    """Same shape, new values: the cached run must see the new values."""
    F = make_sf(lambda x, extras: jnp.broadcast_to(extras["f"], x.shape), dim=2, rank=1, extras_keys=("f",))
    proc = OverdampedProcess(F, D=1e-24, extras_global={"f": jnp.array([1.0, 0.0])})
    proc.initialize(jnp.zeros(2))
    X1 = proc.simulate(0.1, Nsteps=1, key=random.PRNGKey(0), oversampling=1).datasets[0].X
    proc.set_extras(extras_global={"f": jnp.array([0.0, -2.0])})
    X2 = proc.simulate(0.1, Nsteps=1, key=random.PRNGKey(0), oversampling=1).datasets[0].X
    np.testing.assert_allclose(np.asarray(X1[-1]), [0.1, 0.0], atol=1e-9)
    np.testing.assert_allclose(np.asarray(X2[-1]), [0.1, -0.2], atol=1e-9)
    assert len(proc._run_cache) == 1


def test_reinitialize_drops_runs_bound_to_old_parameters():
    proc = _pair_process(_x0())
    proc.simulate(0.01, Nsteps=2, key=random.PRNGKey(0), oversampling=1, compute_observables=False)
    proc.set_params(theta_F={"eps": 0.0})
    proc.initialize(_x0())
    assert not proc._run_cache
    X = proc.simulate(0.01, Nsteps=2, key=random.PRNGKey(0), oversampling=1, compute_observables=False)
    # eps = 0: pure diffusion, so the run matches a force-free process bitwise.
    free = OverdampedProcess(make_sf(lambda x, **kw: jnp.zeros_like(x), dim=2, rank=1), D=0.05)
    free.initialize(_x0())
    X0 = free.simulate(0.01, Nsteps=2, key=random.PRNGKey(0), oversampling=1, compute_observables=False)
    np.testing.assert_allclose(np.asarray(X.datasets[0].X), np.asarray(X0.datasets[0].X), atol=1e-9)


@pytest.mark.parametrize("compute_observables", [False, True])
def test_underdamped_cached_run_matches_uncached(compute_observables):
    def make():
        F = make_sf(lambda x, v, extras: -extras["k"] * x - v, dim=2, rank=1, needs_v=True, extras_keys=("k",))
        proc = UnderdampedProcess(F, D=0.2, extras_global={"k": jnp.array(1.0)})
        proc.initialize(jnp.array([1.0, 0.0]), v0=jnp.zeros(2))
        return proc

    runs = {}
    for jc in (True, False):
        proc = make()
        out = []
        for i, k in enumerate((1.0, 2.0, 3.0)):
            proc.set_extras(extras_global={"k": jnp.array(k)})
            c = proc.simulate(
                0.01,
                Nsteps=5,
                key=random.PRNGKey(i),
                oversampling=2,
                prerun=1 if i == 0 else 0,
                jit_compile=jc,
                compute_observables=compute_observables,
            )
            out.append((np.asarray(c.datasets[0].X), c.datasets[0].meta.get("observables")))
        runs[jc] = out
        if jc:
            assert sum(fn._cache_size() for fn in proc._run_cache.values()) == 2  # prerun / no-prerun configs
    for (Xa, oa), (Xb, ob) in zip(runs[True], runs[False]):
        np.testing.assert_array_equal(Xa, Xb)
        assert oa == ob


# --------------------------------------------------------------------------- #
# Containers, NumPy scalars and key extras
# --------------------------------------------------------------------------- #
def _const_force_process(extras, fn):
    F = make_sf(fn, dim=2, rank=1, extras_keys=tuple(extras))
    proc = OverdampedProcess(F, D=1e-24, extras_global=extras)
    proc.initialize(jnp.zeros(2))
    return proc


def _one_step(proc):
    """Displacement over one step of the (noise-free) deterministic force."""
    x_before = np.asarray(proc._x)
    x_after = np.asarray(proc.simulate(0.1, Nsteps=1, key=random.PRNGKey(0), oversampling=1).datasets[0].X[-1])
    return x_after - x_before


def test_mutated_container_extras_are_not_stale():
    cfg = {"f": jnp.array([1.0, 0.0])}
    proc = _const_force_process({"cfg": cfg}, lambda x, extras: jnp.broadcast_to(extras["cfg"]["f"], x.shape))
    np.testing.assert_allclose(_one_step(proc), [0.1, 0.0], atol=1e-6)
    cfg["f"] = jnp.array([0.0, -2.0])  # edited in place
    proc.set_extras(extras_global={"cfg": cfg})
    np.testing.assert_allclose(_one_step(proc), [0.0, -0.2], atol=1e-6)

    lst = [1.0, 0.0]
    proc2 = _const_force_process({"f": lst}, lambda x, extras: jnp.broadcast_to(jnp.asarray(extras["f"]), x.shape))
    np.testing.assert_allclose(_one_step(proc2), [0.1, 0.0], atol=1e-6)
    lst[1] = -2.0
    proc2.set_extras(extras_global={"f": lst})
    np.testing.assert_allclose(_one_step(proc2), [0.1, -0.2], atol=1e-6)


def test_numpy_scalar_extras_are_keyed_by_value():
    proc = _const_force_process({"k": np.int64(2)}, lambda x, extras: jnp.ones_like(x) * extras["k"])
    for _ in range(3):
        proc.set_extras(extras_global={"k": np.int64(2)})
        _one_step(proc)
    assert len(proc._run_cache) == 1


def test_prng_key_extra_is_accepted():
    proc = _const_force_process(
        {"rk": __import__("jax").random.key(0)},
        lambda x, extras: jnp.zeros_like(x) + 0.0 * __import__("jax").random.normal(extras["rk"], x.shape),
    )
    assert np.all(np.isfinite(_one_step(proc)))


def test_run_cache_is_bounded():
    from SFI.langevin.base import _RUN_CACHE_SIZE

    proc = _const_force_process({}, lambda x: -x)
    for i in range(_RUN_CACHE_SIZE + 4):
        proc.simulate(0.01 * (1 + i), Nsteps=2, key=random.PRNGKey(i), oversampling=1)
    assert len(proc._run_cache) == _RUN_CACHE_SIZE
