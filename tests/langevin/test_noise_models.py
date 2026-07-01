# TODO: review this file
"""Tests for noise models: WhiteNoise, ConservedNoise, CompositeNoise.

Covers:
- Conservation (spatial sum = 0) for ConservedNoise
- Variance scaling with sigma and grid resolution
- CompositeNoise field-index coverage
- effective_D_per_site consistency
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from SFI.langevin.noise import ConservedNoise, WhiteNoise, CompositeNoise


class TestConservedNoise:
    """Tests for the ConservedNoise model."""

    def test_spatial_sum_zero(self):
        """Conserved noise has zero spatial sum at every sample."""
        cn = ConservedNoise(sigma=1.0, grid_shape=(32, 32), dx=1.0, n_fields=1)
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.zeros((32 * 32, 1))

        for i in range(5):
            key, subkey = jax.random.split(key)
            sample = cn.sample(subkey, x_dummy, {})
            assert sample.shape == (32 * 32, 1)
            spatial_sum = float(jnp.sum(sample))
            assert abs(spatial_sum) < 1e-4, f"Sum = {spatial_sum}"

    def test_spatial_sum_zero_multifield(self):
        """Conservation holds per-field for multi-field systems."""
        cn = ConservedNoise(sigma=0.5, grid_shape=(16, 16), dx=0.5, n_fields=2)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.zeros((256, 2))
        sample = cn.sample(key, x_dummy, {})
        assert sample.shape == (256, 2)
        for f in range(2):
            s = float(jnp.sum(sample[:, f]))
            assert abs(s) < 1e-5, f"Field {f} sum = {s}"

    def test_variance_scales_with_sigma(self):
        """Variance scales as sigma^2."""
        key = jax.random.PRNGKey(1)
        grid_shape = (64, 64)
        x_dummy = jnp.zeros((64 * 64, 1))

        vars_ = {}
        for sigma in [0.5, 1.0, 2.0]:
            cn = ConservedNoise(
                sigma=sigma, grid_shape=grid_shape, dx=1.0, n_fields=1
            )
            samples = []
            for i in range(20):
                key, subkey = jax.random.split(key)
                samples.append(cn.sample(subkey, x_dummy, {}))
            stacked = jnp.stack(samples, axis=0)
            vars_[sigma] = float(jnp.var(stacked))

        # var(2*sigma) / var(sigma) ≈ 4
        ratio = vars_[2.0] / vars_[1.0]
        assert 2.5 < ratio < 6.0, f"Variance ratio = {ratio}"

    def test_effective_D_per_site(self):
        """effective_D_per_site returns correct shape and positive values."""
        cn = ConservedNoise(
            sigma=0.3, grid_shape=(16, 16), dx=1.0, n_fields=2
        )
        D = cn.effective_D_per_site({})
        assert D.shape == (2, 2)
        assert float(D[0, 0]) > 0
        assert float(D[1, 1]) > 0
        np.testing.assert_allclose(float(D[0, 1]), 0.0)

    def test_1d_conservation(self):
        """Conservation works in 1D."""
        cn = ConservedNoise(sigma=1.0, grid_shape=(128,), dx=0.5, n_fields=1)
        key = jax.random.PRNGKey(7)
        x_dummy = jnp.zeros((128, 1))
        sample = cn.sample(key, x_dummy, {})
        assert sample.shape == (128, 1)
        assert abs(float(jnp.sum(sample))) < 1e-3

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ConservedNoise(sigma=-1.0, grid_shape=(8, 8), dx=1.0)

    def test_repr(self):
        cn = ConservedNoise(sigma=0.3, grid_shape=(8, 8), dx=1.0, n_fields=1)
        r = repr(cn)
        assert "ConservedNoise" in r
        assert "0.3" in r


class TestCompositeNoise:
    """Tests for the CompositeNoise model."""

    def test_basic_composite(self):
        """CompositeNoise mixes conserved + white correctly."""
        cn = ConservedNoise(
            sigma=0.5, grid_shape=(16, 16), dx=1.0, n_fields=1
        )
        wn = WhiteNoise(sigma=0.1, n_fields=1)
        comp = CompositeNoise(
            components=[(cn, [0]), (wn, [1])], n_fields=2
        )
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.zeros((256, 2))
        sample = comp.sample(key, x_dummy, {})
        assert sample.shape == (256, 2)

        # Conserved field should sum to zero
        s0 = float(jnp.sum(sample[:, 0]))
        assert abs(s0) < 1e-4, f"Conserved field sum = {s0}"

    def test_overlapping_indices_raises(self):
        wn1 = WhiteNoise(sigma=0.1, n_fields=1)
        wn2 = WhiteNoise(sigma=0.2, n_fields=1)
        with pytest.raises(ValueError, match="Overlapping"):
            CompositeNoise(
                components=[(wn1, [0]), (wn2, [0])], n_fields=1
            )

    def test_missing_indices_raises(self):
        wn = WhiteNoise(sigma=0.1, n_fields=1)
        with pytest.raises(ValueError, match="cover"):
            CompositeNoise(components=[(wn, [0])], n_fields=2)
