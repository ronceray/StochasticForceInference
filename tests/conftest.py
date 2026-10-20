"""Shared pytest fixtures and JAX test configuration.

JAX is pinned to the CPU backend (no CUDA probing) and preallocation is
disabled so the suite runs deterministically on machines without a GPU.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no CUDA probing
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402  (must follow the JAX_PLATFORMS env setup above)
import pytest  # noqa: E402


@pytest.fixture
def jax_key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# float64 isolation
# ---------------------------------------------------------------------------
# Several modules need jax float64 and switch it on with a module-scoped
# fixture.  Switched on at import time instead, it would run at collection
# and put every other module in float64, with different random realizations:
# precision-sensitive tests then fail only in a full-suite run.  Each module
# starts from the session's setting and leaves it unchanged.

_X64_AT_START = bool(jax.config.jax_enable_x64)


def pytest_collection_finish(session):
    if bool(jax.config.jax_enable_x64) != _X64_AT_START:
        jax.config.update("jax_enable_x64", _X64_AT_START)
        session.config.issue_config_time_warning(
            pytest.PytestWarning(
                "a test module changed jax_enable_x64 at import time; reset for the run "
                "(use a module-scoped fixture that restores it)"
            ),
            stacklevel=2,
        )


@pytest.fixture(autouse=True, scope="module")
def _isolate_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)
