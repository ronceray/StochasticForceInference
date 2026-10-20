# SFI/statefunc/nodes/ops/einsum.py
"""Einsum contraction operator node."""

import itertools

import equinox as eqx
import jax.numpy as jnp

from ..base import BaseNode, BaseOpNode

# helper letters for generating spatial tokens (implementation detail)
_LETTERS = "mnopqrstuvwxyz"  # 14 letters ⇒ rank ≤ 14 each is fine


def _rank_letters(rank: int, offset: int = 0) -> str:
    """Return `rank` distinct lower-case letters starting at `offset`."""
    if rank + offset > len(_LETTERS):
        raise ValueError("Rank exceeds available einsum letters")
    return _LETTERS[offset : offset + rank]


# ──────────────────────────────────────────────────────────────────────────────
#  EinsumNode  –  generic contraction over spatial (rank) axes
# ──────────────────────────────────────────────────────────────────────────────
class EinsumNode(BaseOpNode):
    """
    Generic einsum contraction of *N* sub-bases along their **rank axes**.

    Notes
    -----
    * Specs are spatial-only (no ellipses); batch is implicit.
    * Validation happens in `contract.merge_contract`. This node only implements
      the execution and internal axis wiring.

    Parameters
    ----------
    *children : BaseNode
    spec      : str
        Einstein notation **with LOWERCASE letters only** - one comma-separated
        operand per child, followed by '->' and the RHS letters. **No ellipses**:
        batch/particle axes are implicit and auto-injected.

        Examples (spatial-only specs):
            "m,n->mn"     vector x vector outer
            "m,m->"       dot / trace
            ",n->n"       scalar x vector
            "mn,n->m"     tensor x vector
            "m,n,p->mnp"  3-way outer product
            "mn,mp->np"   contraction on first index
    """

    CONTRACT_MODE = "einsum"
    _BASIS_LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    spec: str = eqx.field(static=True)
    _einsum: str = eqx.field(static=True, repr=False)

    # ------------------------------------------------------------------
    def __init__(self, *children: BaseNode, spec: str):
        # spec is assumed validated by contract
        object.__setattr__(self, "spec", spec.replace(" ", ""))
        super().__init__(*children)

        lhs, rhs = self.spec.split("->")
        lhs_ops = lhs.split(",")

        # Build internal einsum by:
        #  - prefixing '...' on every operand and RHS for batch
        #  - appending unique per-child feature letters to operands
        #  - appending the concatenated feature letters on RHS
        if len(children) > len(self._BASIS_LETTERS_UPPER):
            raise ValueError("Too many children for auto basis letters (limit 26)")

        embl_ops = [f"...{op}{self._BASIS_LETTERS_UPPER[i]}" for i, op in enumerate(lhs_ops)]
        rhs_with_batch = f"...{rhs}{self._BASIS_LETTERS_UPPER[: len(children)]}"
        einsum_full = ",".join(embl_ops) + "->" + rhs_with_batch
        object.__setattr__(self, "_einsum", einsum_full)

    def with_children(self, new_children):
        return EinsumNode(*new_children, spec=self.spec)

    # ------------------------------------------------------------------
    def _op(self, outs, *, params):
        y = jnp.einsum(self._einsum, *outs)
        # collapse the per-child feature/basis axes into one
        if len(self.children) > 1:
            y = y.reshape(y.shape[: -len(self.children)] + (-1,))
        return y

    # ------------------------------------------------------------------
    # flatten = Cartesian product of leaf callables --------------------
    def flatten(self):
        funcs_lists, label_lists, desc_lists = zip(*(ch.flatten() for ch in self.children))

        funcs, labels, descs = [], [], []
        for idx_combo in itertools.product(*[range(len(lst)) for lst in funcs_lists]):

            def _prod(X, *, v=None, mask=None, extras=None, params=None, _idx=idx_combo):
                parts = [
                    funcs_lists[k][_idx[k]](X, v=v, mask=mask, extras=extras, params=params)[..., None]
                    for k in range(len(_idx))
                ]
                res = jnp.einsum(self._einsum, *parts)
                return res.squeeze(tuple(range(-len(_idx), 0)))  # drop len-1 axes

            label = "·".join(
                (f"({label_lists[k][idx]})" if any(c in label_lists[k][idx] for c in "+-·") else label_lists[k][idx])
                for k, idx in enumerate(idx_combo)
            )
            desc = tuple(desc_lists[k][idx] for k, idx in enumerate(idx_combo))

            funcs.append(_prod)
            labels.append(label)
            descs.append(desc)

        return funcs, labels, descs

    # ------------------------------------------------------------------
    def _tree_id(self):
        return hash((self.spec, tuple(ch._tree_id() for ch in self.children)))
