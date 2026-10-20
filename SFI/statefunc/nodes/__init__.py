# SFI/statefunc/nodes/__init__.py
"""
Developer-facing node classes.

Most users should import from `SFI.statefunc` (Basis/PSF/SF).
This module exposes the node types for advanced composition.
"""

from .base import BaseNode, BaseOpNode
from .contract import Rank
from .leaf import InteractionLeaf, SimpleLeaf
from .ops import (
    CoeffNode,
    ConcatNode,
    DenseNode,
    DerivativeNode,
    EinsumNode,
    MapNNode,
    ReshapeRankNode,
    SliceFeaturesNode,
)

__all__ = [
    "BaseNode",
    "BaseOpNode",
    "SimpleLeaf",
    "ConcatNode",
    "MapNNode",
    "EinsumNode",
    "DenseNode",
    "CoeffNode",
    "SliceFeaturesNode",
    "DerivativeNode",
    "ReshapeRankNode",
    "Rank",
]


from .interactions import (
    AutoPairs,
    FromExtrasPairsCSR,
    HyperCSR,
    HyperFixed,
    InteractionDispatcher,
    PairsCSR,
)

__all__ += [
    "InteractionDispatcher",
    "PairsCSR",
    "HyperFixed",
    "HyperCSR",
    "AutoPairs",
    "FromExtrasPairsCSR",
    "InteractionLeaf",
]
