"""Structured-dimensions layout sub-package.

Re-exports for backward compatibility with ``from SFI.statefunc.layout import ...``.
"""

from ._base import (
    IdentityLayout,
    StateLayout,
    _BaseLayout,
)
from ._grid import GridLayout
from ._sectors import (
    ScalarSector,
    Sector,
    SymTensorSector,
    TensorSector,
    VectorSector,
)

__all__ = [
    "ScalarSector",
    "VectorSector",
    "SymTensorSector",
    "TensorSector",
    "Sector",
    "StateLayout",
    "IdentityLayout",
    "GridLayout",
]
