"""Sector types for the structured dimensions layer.

Defines :class:`ScalarSector`, :class:`VectorSector`,
:class:`SymTensorSector`, :class:`TensorSector` and the convenience
type alias :data:`Sector`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# =====================================================================
# Sector types
# =====================================================================


@dataclass(frozen=True, slots=True)
class ScalarSector:
    """A single scalar field occupying one data index.

    ``sdims = ()``, ``n_data = 1``.
    """

    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if len(self.indices) != 1:
            raise ValueError(f"ScalarSector requires exactly 1 index, got {len(self.indices)}: {self.indices}")

    @property
    def sdims(self) -> tuple[int, ...]:
        return ()

    @property
    def n_data(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class VectorSector:
    """A vector field with *sdim* components.

    ``sdims = (sdim,)``, ``n_data = sdim``.

    Parameters
    ----------
    spatial : bool
        When ``True``, declares that this sector's components correspond
        to spatial coordinates.  Layouts with ``ndim`` validate that
        ``sdim == ndim``.
    """

    indices: tuple[int, ...]
    sdim: int
    spatial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if self.sdim < 1:
            raise ValueError(f"sdim must be >= 1, got {self.sdim}")
        if len(self.indices) != self.sdim:
            raise ValueError(f"VectorSector(sdim={self.sdim}) requires {self.sdim} indices, got {len(self.indices)}")

    @property
    def sdims(self) -> tuple[int, ...]:
        return (self.sdim,)

    @property
    def n_data(self) -> int:
        return self.sdim


@dataclass(frozen=True, slots=True)
class SymTensorSector:
    """A symmetric tensor field, Voigt-packed in data space.

    ``sdims = (sdim, sdim)``.

    When ``traceless=False`` (default), ``n_data = sdim * (sdim + 1) // 2``
    (full Voigt packing).  When ``traceless=True``,
    ``n_data = sdim * (sdim + 1) // 2 - 1`` (the trace degree of freedom
    is removed: the diagonal is constrained so that
    :math:`Q_{00} + Q_{11} + \\ldots = 0`).

    Parameters
    ----------
    traceless : bool
        When ``True``, the tensor is both symmetric *and* traceless.
        Only the independent components are stored in data space;
        the last diagonal entry is reconstructed as minus the sum
        of the other diagonal entries.
    """

    indices: tuple[int, ...]
    sdim: int
    traceless: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        expected = self._n_independent
        if len(self.indices) != expected:
            kind = "traceless symmetric" if self.traceless else "symmetric"
            raise ValueError(
                f"SymTensorSector(sdim={self.sdim}, traceless={self.traceless}) "
                f"requires {expected} indices ({kind} packing), "
                f"got {len(self.indices)}"
            )

    @property
    def _n_independent(self) -> int:
        """Number of independent components stored in data space."""
        n_voigt = self.sdim * (self.sdim + 1) // 2
        return n_voigt - 1 if self.traceless else n_voigt

    @property
    def sdims(self) -> tuple[int, ...]:
        return (self.sdim, self.sdim)

    @property
    def n_data(self) -> int:
        return self._n_independent

    @property
    def voigt_pairs(self) -> list[tuple[int, int]]:
        """Upper-triangle ``(i, j)`` pairs matching the index order.

        When ``traceless=True`` the last diagonal pair
        ``(sdim-1, sdim-1)`` is omitted (it is reconstructed from the
        other diagonal entries).
        """
        pairs: list[tuple[int, int]] = []
        for i in range(self.sdim):
            for j in range(i, self.sdim):
                pairs.append((i, j))
        if self.traceless:
            # Remove the last diagonal entry (sdim-1, sdim-1)
            pairs = [p for p in pairs if p != (self.sdim - 1, self.sdim - 1)]
        return pairs


@dataclass(frozen=True, slots=True)
class TensorSector:
    """A general tensor field with arbitrary ``sdims``.

    ``n_data = prod(sdims)``.
    """

    indices: tuple[int, ...]
    sdims: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        object.__setattr__(self, "sdims", tuple(self.sdims))
        expected = math.prod(self.sdims)
        if len(self.indices) != expected:
            raise ValueError(f"TensorSector(sdims={self.sdims}) requires {expected} indices, got {len(self.indices)}")

    @property
    def n_data(self) -> int:
        return math.prod(self.sdims)


# Convenience type alias
Sector = ScalarSector | VectorSector | SymTensorSector | TensorSector
