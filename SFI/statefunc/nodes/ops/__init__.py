# SFI/statefunc/nodes/ops/__init__.py
"""
Operator node classes.

These are the building blocks for composite state expressions.
Most users will never need them directly — they’re primarily
for developers extending `SFI.statefunc`.
"""

from .concat import ConcatNode
from .derivative import DerivativeNode
from .einsum import EinsumNode
from .linear import CoeffNode, DenseNode
from .mapn import MapNNode
from .reshape_rank import ReshapeRankNode
from .slice import SliceFeaturesNode

__all__ = [
    "ConcatNode",
    "MapNNode",
    "EinsumNode",
    "DenseNode",
    "CoeffNode",
    "SliceFeaturesNode",
    "DerivativeNode",
    "ReshapeRankNode",
]
