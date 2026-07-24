# TODO: review this file
from __future__ import annotations

import jax.numpy as jnp

from SFI.statefunc import Rank, make_basis
from SFI.statefunc.memhint import SampleMeta


def _quad_basis(dim):
    return make_basis(
        lambda x, **_: x[0] ** 2,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=1,
        needs_v=False,
        labels=["x2"],
    )


def test_derivative_inflates_memory_hint_in_grad_mode():
    basis = _quad_basis(1)  # scalar rank, nF=1
    # d_x() builds a derivative expression internally. We compare forward vs grad.
    expr = basis  # treat as a plain expression first
    forward_bytes = expr.estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=None), mode="forward"
    )
    grad_bytes = expr.d_x().estimate_bytes_per_sample(
        dtype=jnp.float32, sample=SampleMeta(P=None)
    )

    # Our DerivativeNode bumps by ~2x in grad mode; leaf is unaffected but op default applies.
    assert grad_bytes >= forward_bytes
    # It's okay if fusion makes it looser, but the conservative factor is exactly 2 for the op.
    assert grad_bytes == 2 * forward_bytes
