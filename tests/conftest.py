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
