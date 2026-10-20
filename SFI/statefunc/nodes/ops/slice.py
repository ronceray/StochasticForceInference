# SFI/statefunc/nodes/ops/slice.py
"""Slice features along the feature axis."""

from typing import Sequence

import equinox as eqx
import numpy as np

from ..base import BaseNode, BaseOpNode
from ..contract import _ContractMixin


def _normalize_feature_index(idx):
    """Coerce a feature selection to hashable, host-side Python values.

    ``idx`` is a static (compile-time) field, so array-likes — NumPy / JAX
    index arrays such as ``np.asarray(inf.force_support)`` — are converted
    to a tuple of Python ints (boolean masks to the positions of their
    ``True`` entries) and NumPy integer scalars to ``int``.  Storing an array
    as a static field triggers equinox's "JAX array is being set as static"
    warning; a NumPy integer scalar has no ``len`` and a NumPy boolean mask
    was read as integer positions.
    """
    if isinstance(idx, slice):
        return idx
    if isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_)):
        return int(idx)
    arr = np.asarray(idx)
    if arr.ndim == 0:
        if np.issubdtype(arr.dtype, np.integer):
            return int(arr)
        raise TypeError(f"Feature index must be an int, slice, or 1-D int/bool sequence; got {idx!r}.")
    if arr.ndim != 1:
        raise IndexError(f"Feature index must be 1-D; got shape {arr.shape}.")
    if arr.dtype == np.bool_:
        return tuple(int(i) for i in np.flatnonzero(arr))
    if arr.size == 0:
        return ()
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Feature index array must have an integer or boolean dtype; got {arr.dtype}.")
    return tuple(int(i) for i in arr)


# ---------------------------------------------------------------------
# Helper node – slice the **feature axis**
# ---------------------------------------------------------------------
class SliceFeaturesNode(BaseOpNode):
    """Select or reorder *feature* entries of a child basis.

    The node simply indexes the last axis (feature axis) of its child
    while preserving every spatial axis.  Supported *idx* forms are

    * ``int``               - keep a single feature → ``n_features = 1``
    * ``slice``             - standard Python slice
    * ``Sequence[int]``     - fancy indexing/ordering (lists, tuples, NumPy
      or JAX integer arrays)
    * ``Sequence[bool]``    - boolean mask (length must equal ``n_features``)

    Array-like selections are stored as a tuple of Python ints (boolean
    masks as the positions of their ``True`` entries): the index is a static
    field and must be a plain host-side value.

    Notes
    -----
    - Rank, dim, pdepth & needs_v are *identical* to the child - only
      ``n_features`` changes.
    - No parameters or extras are consumed, so the node is deterministic.
    """

    idx: int | slice | Sequence[int] | Sequence[bool] = eqx.field(static=True)

    # ---------------- constructor ----------------
    def __init__(self, child: BaseNode, idx):
        object.__setattr__(self, "idx", _normalize_feature_index(idx))
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

    def with_children(self, new_children):
        # Preserve the feature selection so StateExpr.specialize() can fold a
        # ``dataset_index`` leaf living under a feature slice.
        return SliceFeaturesNode(new_children[0], self.idx)

    # ---------------- combine ---------------------
    def _op(self, outs, *, params):
        y = outs[0]
        if isinstance(self.idx, int):
            # Keep the feature axis (size 1), including for negative indices.
            return y[..., [self.idx]]
        if isinstance(self.idx, slice):
            return y[..., self.idx]
        return y[..., list(self.idx)]

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
