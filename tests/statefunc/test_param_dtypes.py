"""Parameter dtypes: a float spec keeps a wider float value (no float32 rounding under x64)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.statefunc.params import ParamSpec, ParamSuite


@pytest.fixture
def x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _suite():
    return ParamSuite([ParamSpec("w", (2,)), ParamSpec("n", (), dtype=jnp.int32)])


def test_float64_values_are_not_rounded_to_float32(x64):
    suite = _suite()
    theta = jnp.array([0.1, 1.0 + 1e-12, 3.0], dtype=jnp.float64)
    params = suite.materialize(theta)
    assert params["w"].dtype == jnp.float64
    assert float(params["w"][1]) == 1.0 + 1e-12          # float32 would round this to 1.0
    assert params["n"].dtype == jnp.int32                 # non-float specs keep their dtype
    back = suite.vectorize(params)
    assert back.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(back), np.asarray(theta))
    coerced = suite.coerce({"w": jnp.array([0.1, 0.2], dtype=jnp.float64), "n": 3})
    assert coerced["w"].dtype == jnp.float64 and coerced["n"].dtype == jnp.int32


def test_float32_values_and_python_scalars_keep_the_spec_dtype():
    suite = _suite()
    params = suite.materialize(jnp.array([0.1, 0.2, 3.0], dtype=jnp.float32))
    assert params["w"].dtype == jnp.float32
    coerced = suite.coerce({"w": [1, 2], "n": 3})       # ints → the float spec dtype
    assert coerced["w"].dtype == jnp.float32
