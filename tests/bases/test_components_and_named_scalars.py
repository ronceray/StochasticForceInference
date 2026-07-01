# TODO: review this file
"""Tests for the compositional primitives:

- ``x_components`` / ``v_components`` / ``unit_axes`` / ``frame``
- ``named_scalar`` / ``named_scalars`` (rank-0 PSFs with named params)
- ``ParamSpec.default`` plumbing through PSF and the simulator
- Auto-labels for ``+`` / ``-`` / parametric ``*`` compositions
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from SFI.bases import (
    frame,
    named_scalar,
    named_scalars,
    unit_axes,
    v_components,
    x_components,
)
from SFI.bases.linear import linear_basis  # noqa: F401  (used in compat check)
from SFI.langevin import OverdampedProcess
from SFI.statefunc import Basis, PSF


# ---------------------------------------------------------------------------
# x_components / v_components / unit_axes basic shape and label contracts
# ---------------------------------------------------------------------------
def test_x_components_default_labels_3d():
    x, y, z = x_components(3)
    assert tuple(x.labels) == ("x",)
    assert tuple(y.labels) == ("y",)
    assert tuple(z.labels) == ("z",)
    s = jnp.array([4.0, 5.0, 6.0])
    assert float(x(s).squeeze()) == 4.0
    assert float(z(s).squeeze()) == 6.0


def test_x_components_high_dim_auto_labels():
    comps = x_components(7)
    labels = [tuple(c.labels)[0] for c in comps]
    # dim > len(default human labels) -> indexed names for all axes
    assert labels == ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]


def test_x_components_custom_labels():
    a, b, c = x_components(3, labels=["a", "b", "c"])
    assert tuple(a.labels) == ("a",)
    assert tuple(b.labels) == ("b",)
    with pytest.raises(ValueError):
        x_components(3, labels=["a", "b"])


def test_v_components_3d():
    vx, vy, vz = v_components(3)
    assert tuple(vx.labels) == ("vx",)
    s = jnp.zeros(3)
    v = jnp.array([7.0, 8.0, 9.0])
    assert float(vy(s, v=v).squeeze()) == 8.0


def test_unit_axes_values_and_labels():
    ex, ey, ez = unit_axes(3)
    assert tuple(ex.labels) == ("ex",)
    s = jnp.zeros(3)
    # rank=1 vector with 1 feature -> shape (3, 1)
    assert ex(s).shape == (3, 1)
    assert jnp.allclose(ex(s).squeeze(-1), jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(ey(s).squeeze(-1), jnp.array([0.0, 1.0, 0.0]))
    assert jnp.allclose(ez(s).squeeze(-1), jnp.array([0.0, 0.0, 1.0]))


def test_frame_overdamped_3d():
    one, x, y, z, ex, ey, ez = frame(3)
    s = jnp.array([1.0, 2.0, 3.0])
    assert float(one(s).squeeze()) == 1.0
    assert float(y(s).squeeze()) == 2.0
    assert ex(s).shape == (3, 1)


def test_frame_underdamped_3d():
    bundle = frame(3, velocity=True)
    assert len(bundle) == 1 + 3 + 3 + 3
    one, x, y, z, vx, vy, vz, ex, ey, ez = bundle
    s = jnp.array([0.0, 0.0, 0.0])
    v = jnp.array([10.0, 11.0, 12.0])
    assert float(vy(s, v=v).squeeze()) == 11.0


# ---------------------------------------------------------------------------
# named_scalar / named_scalars
# ---------------------------------------------------------------------------
def test_named_scalar_default_evaluation():
    sigma = named_scalar("sigma", default=20.0)
    assert isinstance(sigma, PSF)
    val = sigma(jnp.zeros(3))           # uses default
    assert float(val) == 20.0
    val2 = sigma(jnp.zeros(3), params={"sigma": 30.0})
    assert float(val2) == 30.0


def test_named_scalar_no_default_requires_params():
    sigma = named_scalar("sigma")
    with pytest.raises(ValueError):
        sigma(jnp.zeros(3))             # no default and no params


def test_named_scalars_positional_no_defaults():
    sigma, rho, beta = named_scalars("sigma", "rho", "beta")
    with pytest.raises(ValueError):
        sigma(jnp.zeros(3))


def test_named_scalars_kwargs_with_defaults_preserve_order():
    sigma, rho, beta = named_scalars(sigma=10.0, rho=28.0, beta=8 / 3)
    assert float(sigma(jnp.zeros(3))) == 10.0
    assert float(rho(jnp.zeros(3))) == 28.0
    assert abs(float(beta(jnp.zeros(3))) - 8 / 3) < 1e-6


def test_named_scalars_rejects_mixed_args():
    with pytest.raises(TypeError):
        named_scalars("a", b=1.0)


# ---------------------------------------------------------------------------
# Compositional algebra: auto-labels and Basis -> PSF promotion
# ---------------------------------------------------------------------------
def test_add_sub_auto_labels():
    x, y, z = x_components(3)
    assert tuple((y - x).labels) == ("y-x",)
    assert tuple((y + x).labels) == ("y+x",)


def test_basis_psf_promotion_via_mul():
    x, y, _ = x_components(3)
    sigma, = named_scalars("sigma")
    expr = x * sigma                      # Basis * PSF should promote to PSF
    assert isinstance(expr, PSF)
    expr2 = sigma * x                     # PSF * Basis (was already fine)
    assert isinstance(expr2, PSF)


def test_lorenz_composition_matches_handwritten():
    x, y, z = x_components(3)
    ex, ey, ez = unit_axes(3)
    sigma, rho, beta = named_scalars(sigma=10.0, rho=28.0, beta=8 / 3)
    F = (sigma * (y - x)) * ex \
        + (x * (rho - z) - y) * ey \
        + (x * y - beta * z) * ez

    state = jnp.array([1.5, -0.5, 2.0])
    val = F(state)
    expected = jnp.array(
        [
            10.0 * (-0.5 - 1.5),
            1.5 * (28.0 - 2.0) - (-0.5),
            1.5 * (-0.5) - (8 / 3) * 2.0,
        ]
    )
    assert jnp.allclose(val, expected, atol=1e-6)


def test_lorenz_simulator_uses_psf_defaults():
    """OverdampedProcess(F) with no theta_F should pull defaults from the PSF
    template (named_scalars with defaults)."""
    x, y, z = x_components(3)
    ex, ey, ez = unit_axes(3)
    sigma, rho, beta = named_scalars(sigma=10.0, rho=28.0, beta=8 / 3)
    F = (sigma * (y - x)) * ex \
        + (x * (rho - z) - y) * ey \
        + (x * y - beta * z) * ez

    proc = OverdampedProcess(F, D=0.1 * jnp.eye(3))
    proc.initialize(jnp.array([1.0, 2.0, 3.0]))
    # Should not raise; defaults were honored.


def test_psf_bind_no_args_uses_defaults():
    sigma = named_scalar("sigma", default=42.0)
    sf = sigma.bind()                     # no args
    assert float(sf(jnp.zeros(3))) == 42.0


def test_psf_bind_no_args_without_defaults_raises():
    sigma = named_scalar("sigma")
    with pytest.raises(ValueError):
        sigma.bind()
