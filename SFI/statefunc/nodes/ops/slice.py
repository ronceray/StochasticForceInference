# SFI/statefunc/nodes/ops/slice.py
"""Slice features along the feature axis."""

from typing import Sequence

import equinox as eqx

from ..base import BaseNode, BaseOpNode
from ..contract import _ContractMixin


# ---------------------------------------------------------------------
# Helper node – slice the **feature axis**
# ---------------------------------------------------------------------
class SliceFeaturesNode(BaseOpNode):
    """Select or reorder *feature* entries of a child basis.

    The node simply indexes the last axis (feature axis) of its child
    while preserving every spatial axis.  Supported *idx* forms are

    * ``int``               - keep a single feature → ``n_features = 1``
    * ``slice``             - standard Python slice
    * ``Sequence[int]``     - fancy indexing/ordering
    * ``Sequence[bool]``    - boolean mask (length must equal ``n_features``)

    Notes
    -----
    - Rank, dim, pdepth & needs_v are *identical* to the child - only
      ``n_features`` changes.
    - No parameters or extras are consumed, so the node is deterministic.
    """

    idx: int | slice | Sequence[int] | Sequence[bool] = eqx.field(static=True)

    # ---------------- constructor ----------------
    def __init__(self, child: BaseNode, idx):
        object.__setattr__(self, "idx", idx)
        super().__init__(child)
        object.__setattr__(self, "param_suite", child.param_suite)

    # ------------- static contract merge ----------
    def _merge_static(self, children):
        ch = children[0]
        n_out: int
        if isinstance(self.idx, int):
            n_out = 1
        elif isinstance(self.idx, slice):
            n_out = len(range(*self.idx.indices(ch.n_features)))
        else:  # fancy or boolean indexing
            n_out = len(self.idx)
        return _ContractMixin.inherit_contract(ch, n_features=n_out)

    # ---------------- combine ---------------------
    def _op(self, outs, *, params):
        y = outs[0]
        if isinstance(self.idx, int):
            # Keep the feature axis (size 1), including for negative indices.
            return y[..., [self.idx]]
        return y[..., self.idx]

    # ---------------- flatten (label-aware) --------
    def flatten(self):
        """Select the labels/funcs/descriptors matching the sliced features."""
        child_funcs, child_labels, child_descs = self.children[0].flatten()

        # Resolve idx to a concrete list of integer indices
        n_child = len(child_funcs)
        if isinstance(self.idx, int):
            indices = [self.idx % n_child]
        elif isinstance(self.idx, slice):
            indices = list(range(*self.idx.indices(n_child)))
        else:
            seq = list(self.idx)
            if seq and isinstance(seq[0], bool):
                # boolean mask
                indices = [i for i, b in enumerate(seq) if b]
            else:
                indices = [int(i) % n_child for i in seq]

        funcs = [child_funcs[i] for i in indices]
        labels = [child_labels[i] for i in indices]
        descs = [child_descs[i] for i in indices]
        return funcs, labels, descs
