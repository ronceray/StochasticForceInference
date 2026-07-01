# SFI/statefunc/nodes/ops/reshape_rank.py
"""Lossless reshape between spatial (rank) axes and the feature axis."""

from functools import reduce
from operator import mul

import equinox as eqx
import jax.numpy as jnp

from ..base import BaseNode, BaseOpNode
from ..contract import Rank, _ContractMixin


def _rank_axis_sizes(node: _ContractMixin) -> tuple[int, ...]:
    """Return per-axis sizes for rank axes: sdims if present, else (dim,)^rank."""
    if node.sdims is not None:
        return node.sdims
    if node.dim is None:
        raise ValueError("ReshapeRankNode requires known axis sizes (dim or sdims); got dim=None, sdims=None.")
    return (node.dim,) * int(node.rank)


class ReshapeRankNode(BaseOpNode):
    """Lossless reshape that moves data between rank (spatial) axes and features.

    Given a child with output layout::

        batch · (dim,)^source_rank · n_features

    this node reshapes to::

        batch · (dim,)^target_rank · n_features'

    where  ``n_features' = n_features x dim^(source_rank - target_rank)``
    (folding when *target_rank < source_rank*) or
    ``n_features' = n_features / dim^(target_rank - source_rank)``
    (unfolding when *target_rank > source_rank*).

    The reshape uses C-order so that folding followed by unfolding back
    to the original rank is the identity — the pair is invertible.

    Parameters
    ----------
    child : BaseNode
        Source node (must have known ``dim``).
    target_rank : int or Rank
        Desired output rank.

    Notes
    -----
    * Folding merges the *innermost* spatial axes into the feature axis.
    * Unfolding splits the feature axis to create new *innermost* spatial axes.
    * The child's ``dim`` must be known (not ``None``).
    * For unfolding, ``n_features`` must be divisible by ``dim^Δrank``.
    """

    _target_rank: Rank = eqx.field(static=True)

    def __init__(self, child: BaseNode, *, target_rank: int | Rank):
        object.__setattr__(self, "_target_rank", Rank(int(target_rank)))

        if child.dim is None and child.sdims is None:
            raise ValueError("ReshapeRankNode requires a child with known dim or sdims; got dim=None, sdims=None.")
        src = int(child.rank)
        tgt = int(self._target_rank)

        if tgt > src:
            # Unfolding: need to split the feature axis. Determine how many
            # elements need to be carved out. For uniform dim this is dim^(tgt-src).
            # For sdims, the target rank axes beyond the source's sdims are unknown
            # at this point — we require uniform dim for unfolding (sdims=None or
            # all dims equal).
            if child.sdims is not None:
                raise ValueError(
                    "ReshapeRankNode: unfolding with non-uniform sdims is not "
                    "supported. Use explicit reshape or einsum instead."
                )
            assert child.dim is not None  # guaranteed: sdims is None and guard on L62 ensures one is non-None
            factor = child.dim ** (tgt - src)
            if child.n_features % factor != 0:
                raise ValueError(
                    f"Cannot unfold {child.n_features} features into rank-{tgt} "
                    f"with dim={child.dim}: {child.n_features} is not divisible "
                    f"by dim^{tgt - src} = {factor}."
                )

        super().__init__(child)

    # ------------------------------------------------------------------
    def _merge_static(self, children):
        ch = children[0]
        src = int(ch.rank)
        tgt = int(self._target_rank)
        axis_sizes = _rank_axis_sizes(ch)

        if tgt < src:
            # Folding: innermost (src − tgt) rank axes → features
            factor = reduce(mul, axis_sizes[tgt:], 1)
            n_out = ch.n_features * factor
            out_sdims = axis_sizes[:tgt] if ch.sdims is not None else None
        elif tgt > src:
            # Unfolding: features → new innermost rank axes (uniform dim only)
            dim = ch.dim
            factor = dim ** (tgt - src)
            n_out = ch.n_features // factor
            # Uniform unfolding: new axes are all dim-sized
            out_sdims = axis_sizes + (dim,) * (tgt - src) if ch.sdims is not None else None
        else:
            n_out = ch.n_features
            out_sdims = ch.sdims

        return _ContractMixin.inherit_contract(ch, rank=self._target_rank, n_features=n_out, sdims=out_sdims)

    # ------------------------------------------------------------------
    def _op(self, outs, *, params):
        y = outs[0]
        src = int(self.children[0].rank)
        tgt = int(self._target_rank)

        if tgt == src:
            return y

        if tgt < src:
            # Fold innermost (src − tgt) spatial axes + feature axis into one.
            n_fold = src - tgt
            new_shape = y.shape[: -(n_fold + 1)] + (-1,)
            return jnp.reshape(y, new_shape)
        else:
            # Unfold feature axis into (tgt − src) new spatial axes + features.
            n_unfold = tgt - src
            dim = self.children[0].dim
            n_feat_out = self.n_features
            new_shape = y.shape[:-1] + (dim,) * n_unfold + (n_feat_out,)
            return jnp.reshape(y, new_shape)

    # ------------------------------------------------------------------
    def flatten(self):
        child_funcs, child_labels, child_descs = self.children[0].flatten()

        n_out = self.n_features

        def _slice_feature(j):
            return lambda x, *, v=None, mask=None, extras=None, params=None, _j=j: (
                self(x, v=v, mask=mask, extras=extras, params=params)[..., _j]
            )

        src = int(self.children[0].rank)
        tgt = int(self._target_rank)
        tag = "fold" if tgt < src else "unfold"

        funcs = [_slice_feature(j) for j in range(n_out)]
        labels = [f"{tag}_{j}" for j in range(n_out)]
        descs = [{f"{tag}_of": [d for d in child_descs], "index": j} for j in range(n_out)]
        return funcs, labels, descs

    # ------------------------------------------------------------------
    def _tree_id(self):
        return hash(("ReshapeRankNode", int(self._target_rank), self.children[0]._tree_id()))
