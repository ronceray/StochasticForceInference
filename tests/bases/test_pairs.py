# TODO: review this file
"""Tests for SFI.bases.pairs generic pair-interaction toolkit."""
import jax.numpy as jnp

from SFI.bases.pairs import (
    pbc_displacement,
    wrap_angle,
)


# ── PBC utilities ──────────────────────────────────────────────────

class TestPBCDisplacement:
    def test_no_wrapping(self):
        dx = pbc_displacement(jnp.array([0.3]), jnp.array([0.1]), jnp.array([1.0]))
        assert jnp.allclose(dx, 0.2, atol=1e-6)

    def test_wrapping(self):
        dx = pbc_displacement(jnp.array([0.1]), jnp.array([0.9]), jnp.array([1.0]))
        assert jnp.allclose(dx, 0.2, atol=1e-6)

    def test_2d(self):
        xj = jnp.array([0.1, 0.9])
        xi = jnp.array([0.9, 0.1])
        box = jnp.array([1.0, 1.0])
        dx = pbc_displacement(xj, xi, box)
        assert jnp.allclose(dx, jnp.array([0.2, -0.2]), atol=1e-6)


class TestWrapAngle:
    def test_in_range(self):
        assert jnp.allclose(wrap_angle(1.0), 1.0, atol=1e-6)

    def test_positive_wrap(self):
        assert jnp.allclose(wrap_angle(4.0), 4.0 - 2 * jnp.pi, atol=1e-5)

    def test_negative_wrap(self):
        assert jnp.allclose(wrap_angle(-4.0), -4.0 + 2 * jnp.pi, atol=1e-5)

