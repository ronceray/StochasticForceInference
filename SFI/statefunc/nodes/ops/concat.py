# SFI/statefunc/nodes/ops/concat.py
"""Concatenation operator node."""

import jax.numpy as jnp

from ..base import BaseOpNode


# ──────────────────────────────────────────────────────────────────────────────
#  ConcatNode – feature-axis concatenation
# ──────────────────────────────────────────────────────────────────────────────
class ConcatNode(BaseOpNode):
    """
    Concatenate the **feature axes** of several sub-bases.

    All children must share the same *spatial* contract
    ``(rank, dim, pdepth)``; the resulting node inherits that contract
    while its ``n_features`` is the sum of the children's features.
    """

    CONTRACT_MODE = "concat"

    # combine tensors
    def _op(self, outs, *, params):
        return jnp.concatenate(outs, axis=-1)

    def with_children(self, new_children):
        return ConcatNode(*new_children)
