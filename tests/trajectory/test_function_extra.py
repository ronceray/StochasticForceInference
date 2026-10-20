# TODO: review this file
"""Tests for FunctionExtra — passing JAX-compatible callables through extras."""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.trajectory.dataset import (
    FunctionExtra,
    TrajectoryDataset,
    function_extra,
    time_series_extra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds(T=6, N=2, d=2):
    X = jnp.ones((T, N, d), dtype=jnp.float32)
    mask = jnp.ones((T, N), dtype=bool)
    return TrajectoryDataset.from_arrays(X=X, dt=0.1, mask=mask)


# ---------------------------------------------------------------------------
# Unit tests: FunctionExtra / function_extra
# ---------------------------------------------------------------------------

def test_function_extra_wraps_callable():
    fn = lambda x: x ** 2
    fe = FunctionExtra(fn)
    assert fe.func is fn


def test_function_extra_constructor_rejects_non_callable():
    with pytest.raises(TypeError):
        function_extra(42)


def test_function_extra_constructor_accepts_callable():
    fn = lambda x: x + 1
    fe = function_extra(fn)
    assert fe.func is fn


# ---------------------------------------------------------------------------
# build_extras: FunctionExtra is passed through, plain callable is invoked
# ---------------------------------------------------------------------------

def test_build_extras_passes_function_extra_through():
    fn = lambda x: jnp.sum(x ** 2)
    ds = _make_ds()
    ds2 = TrajectoryDataset.from_arrays(
        X=ds.X, dt=0.1, mask=ds.mask,
        extras_global={"field": FunctionExtra(fn)},
    )
    extras = ds2.build_extras(jnp.int32(0))
    # Should be the unwrapped function, NOT its return value
    assert extras["field"] is fn
    # And it should still be callable
    assert jnp.isclose(extras["field"](jnp.array([1.0, 2.0])), 5.0)


def test_build_extras_still_invokes_plain_callable():
    """Plain callables should still be invoked as time-dependent generators."""
    fn = lambda t_idx, context=None: jnp.array(t_idx * 10, dtype=jnp.float32)
    ds = _make_ds()
    ds2 = TrajectoryDataset.from_arrays(
        X=ds.X, dt=0.1, mask=ds.mask,
        extras_global={"gen": fn},
    )
    extras = ds2.build_extras(jnp.int32(3))
    # Plain callable should be invoked: 3 * 10 = 30
    assert jnp.isclose(extras["gen"], 30.0)


# ---------------------------------------------------------------------------
# Basis integration: basis can call a FunctionExtra inside JIT
# ---------------------------------------------------------------------------

def test_basis_calls_function_extra():
    """A basis function that calls extras['field'](x) should work."""
    from SFI.statefunc import Rank, make_basis

    field_fn = lambda x: jnp.sum(x ** 2, axis=-1, keepdims=True)

    B = make_basis(
        lambda x, *, extras: extras["field"](x),
        dim=2,
        rank=Rank.SCALAR,
        n_features=1,
        extras_keys=("field",),
    )
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = B(x, extras={"field": field_fn})
    expected = jnp.array([[5.0], [25.0]])
    assert jnp.allclose(y, expected)


def test_basis_calls_function_extra_under_jit():
    """Same test under explicit jax.jit."""
    import jax
    from SFI.statefunc import Rank, make_basis

    field_fn = lambda x: jnp.sum(x ** 2, axis=-1, keepdims=True)

    B = make_basis(
        lambda x, *, extras: extras["field"](x),
        dim=2,
        rank=Rank.SCALAR,
        n_features=1,
        extras_keys=("field",),
    )
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    @jax.jit
    def eval_b(x):
        return B(x, extras={"field": field_fn})

    y = eval_b(x)
    expected = jnp.array([[5.0], [25.0]])
    assert jnp.allclose(y, expected)


# ---------------------------------------------------------------------------
# Degrade path: FunctionExtra survives degradation
# ---------------------------------------------------------------------------

def test_degrade_preserves_function_extra():
    from SFI.trajectory.degrade import degrade_dataset

    fn = lambda x: x ** 2
    ds = TrajectoryDataset.from_arrays(
        X=jnp.ones((10, 2, 2), dtype=jnp.float32),
        dt=0.1,
        mask=jnp.ones((10, 2), dtype=bool),
        extras_global={"field": FunctionExtra(fn), "scale": jnp.array(2.0)},
    )
    ds2 = degrade_dataset(ds, downsample=2)
    # The function should survive degradation (passed through)
    extras = ds2.build_extras(jnp.int32(0))
    assert extras["field"] is fn
    # Array extras should still work
    assert "scale" in extras
