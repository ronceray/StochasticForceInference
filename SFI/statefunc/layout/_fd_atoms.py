"""Finite-difference offset geometry for stencil composition.

This module provides pure-Python (no JAX) utilities to compute the
**footprint** of an arbitrary StructuredExpr tree: the union of all
grid offsets that the compiled local function will need to read.

The key idea: each differential operator (``lap``, ``grad``, ``div``, …)
has a *template offset set* (the stencil it reads relative to its
evaluation point).  Nesting operators produces a **Minkowski sum** of
their template offsets — exactly the set of points needed by the fused
stencil.

This module is layout-engine–agnostic: it only understands the IR node
types from :mod:`~SFI.statefunc.structexpr`.

.. note::

   A future PINN layout would *not* use this module at all — it would
   compile ``_DiffOpNode`` via autodiff instead of FD stencils.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..structexpr import _StructNode

# =====================================================================
# Offset sets (frozenset of int tuples)
# =====================================================================

OffsetSet = frozenset[tuple[int, ...]]


def _origin(ndim: int) -> OffsetSet:
    """Single-point footprint at the grid-site origin."""
    return frozenset({(0,) * ndim})


def cross_offsets(ndim: int) -> OffsetSet:
    """Cross stencil: center + ±e_a for all axes.  K = 1 + 2·ndim."""
    pts: list[tuple[int, ...]] = [(0,) * ndim]
    for a in range(ndim):
        e = [0] * ndim
        e[a] = 1
        pts.append(tuple(e))
        e[a] = -1
        pts.append(tuple(e))
    return frozenset(pts)


def biharmonic_offsets(ndim: int) -> OffsetSet:
    """Biharmonic stencil: radius-2 + cross-1 + diagonal corners.

    Matches the ordering contract of
    :func:`~SFI.statefunc.nodes.interactions.stencils.square_biharmonic_offsets`
    but returned as *set* (ordering doesn't matter for footprint computation).
    """
    pts: list[tuple[int, ...]] = [(0,) * ndim]
    # radius-2 along each axis
    for a in range(ndim):
        e = [0] * ndim
        e[a] = 2
        pts.append(tuple(e))
        e[a] = -2
        pts.append(tuple(e))
    # radius-1 cross
    for a in range(ndim):
        e = [0] * ndim
        e[a] = 1
        pts.append(tuple(e))
        e[a] = -1
        pts.append(tuple(e))
    # diagonal corners
    for pair in itertools.combinations(range(ndim), 2):
        for signs in itertools.product([1, -1], repeat=2):
            e = [0] * ndim
            for ax, s in zip(pair, signs):
                e[ax] = s
            pts.append(tuple(e))
    return frozenset(pts)


def op_template_offsets(op_name: str, ndim: int) -> OffsetSet:
    """Template offset set for a named differential operator."""
    if op_name in ("lap", "grad", "div"):
        return cross_offsets(ndim)
    elif op_name == "biharmonic":
        return biharmonic_offsets(ndim)
    else:
        raise ValueError(f"Unknown diff op: {op_name!r}")


# =====================================================================
# Minkowski sum
# =====================================================================


def minkowski_sum(a: OffsetSet, b: OffsetSet) -> OffsetSet:
    """Minkowski sum of two offset sets.

    Returns ``{oa + ob  for oa in a  for ob in b}``  (element-wise
    addition of tuples, deduplicating).
    """
    result: set[tuple[int, ...]] = set()
    for oa in a:
        for ob in b:
            result.add(tuple(ia + ib for ia, ib in zip(oa, ob)))
    return frozenset(result)


# =====================================================================
# Footprint computation (bottom-up tree walk)
# =====================================================================


def compute_footprint(node: _StructNode, ndim: int) -> OffsetSet:
    """Compute the set of grid offsets read by *node*.

    A pure-pointwise node returns ``{origin}`` (reads only the local
    site).  Each ``_DiffOpNode`` inflates the footprint by Minkowski-
    summing its template offsets with the child's footprint.

    Binary / einsum / stack nodes return the union of their children's
    footprints (all children are evaluated at the same base offset).
    """
    from ..structexpr import (
        _BinaryOp,
        _ConcatOp,
        _ConstNode,
        _DiffOpNode,
        _EinsumOp,
        _SectorLeaf,
        _SliceOp,
        _StackOp,
        _UnaryOp,
    )

    origin = _origin(ndim)

    if isinstance(node, (_SectorLeaf, _ConstNode)):
        return origin

    if isinstance(node, _BinaryOp):
        return compute_footprint(node.left, ndim) | compute_footprint(node.right, ndim)

    if isinstance(node, _UnaryOp):
        return compute_footprint(node.child, ndim)

    if isinstance(node, (_EinsumOp, _StackOp, _ConcatOp)):
        fp = origin
        for c in node.children:
            fp = fp | compute_footprint(c, ndim)
        return fp

    if isinstance(node, _SliceOp):
        return compute_footprint(node.child, ndim)

    if isinstance(node, _DiffOpNode):
        child_fp = compute_footprint(node.child, ndim)
        template = op_template_offsets(node.op_name, ndim)
        return minkowski_sum(template, child_fp)

    # Fallback: unknown node → origin (conservative)
    return origin


# =====================================================================
# Sorted offset array (for dispatch)
# =====================================================================


def offsets_to_sorted_array(offsets: OffsetSet) -> np.ndarray:
    """Convert offset set to a deterministic ``(K, ndim)`` int32 array.

    The origin ``(0, …, 0)`` is always placed first (index 0).
    Remaining offsets are in lexicographic order.  This convention is
    important: the mask pre-processing in ``wrap_as_local_fn`` expects
    the center at index 0.
    """
    if not offsets:
        raise ValueError("Empty offset set")
    ndim = len(next(iter(offsets)))
    origin = (0,) * ndim
    rest = sorted(o for o in offsets if o != origin)
    sorted_list = [origin] + rest if origin in offsets else rest
    return np.array(sorted_list, dtype=np.int32)


def build_offset_to_idx(offsets: np.ndarray) -> dict[tuple[int, ...], int]:
    """Build a ``{tuple(offset): stencil_index}`` lookup dict."""
    return {tuple(int(x) for x in offsets[i]): i for i in range(len(offsets))}
