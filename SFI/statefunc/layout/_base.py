"""Layout protocol, base class, and IdentityLayout.

Defines :class:`StateLayout` (protocol), :class:`_BaseLayout` (implementation
base), and :class:`IdentityLayout` (trivial single-sector layout).
"""

from __future__ import annotations

import itertools
from typing import Any, Protocol, runtime_checkable

from ..structexpr import StructuredExpr, _ConstNode, _SectorLeaf
from ._sectors import Sector, VectorSector

# =====================================================================
# Layout instance counter  (unique IDs for layout-compatibility checks)
# =====================================================================

_layout_counter = itertools.count()


def _next_layout_id() -> int:
    return next(_layout_counter)


# =====================================================================
# Layout protocol
# =====================================================================


@runtime_checkable
class StateLayout(Protocol):
    """Protocol for all layouts (Grid, Particle, Identity, …)."""

    @property
    def dim(self) -> int:
        """Total data width  (``x.shape[-1]``)."""
        ...

    def unpack(self) -> dict[str, StructuredExpr]:
        """Return named symbolic field leaves."""
        ...

    def embed(self, rank: int = 1, **named_fields: StructuredExpr) -> Any:
        """Compile inner expressions into outer ``StateExpr``."""
        ...


# =====================================================================
# _BaseLayout  (shared logic for all concrete layouts)
# =====================================================================


class _BaseLayout:
    """Common base for Layout implementations.

    Handles sector storage, index validation, and field-expression
    creation.  Subclasses add engine-specific operators and ``embed()``.
    """

    def __init__(self, *, dim: int, **sectors: Sector) -> None:
        self._dim = dim
        self._layout_id = _next_layout_id()
        self._sectors: dict[str, Sector] = dict(sectors)
        self._validate_indices()
        self._fields = self._build_fields()

    # --- public interface ---------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sectors(self) -> dict[str, Sector]:
        """Read-only mapping ``name → Sector``."""
        return dict(self._sectors)

    def unpack(self) -> dict[str, StructuredExpr]:
        """Return a dict of named symbolic field leaves."""
        return dict(self._fields)

    # --- attribute access for field names -----------------------------

    def __getattr__(self, name: str) -> StructuredExpr:
        # Guard against recursion during __init__
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            fields = object.__getattribute__(self, "_fields")
        except AttributeError:
            raise AttributeError(name) from None
        if name in fields:
            return fields[name]
        raise AttributeError(f"'{type(self).__name__}' has no field '{name}'. Available fields: {', '.join(fields)}")

    # --- validation ---------------------------------------------------

    def _validate_indices(self) -> None:
        """No overlap; all indices in ``range(dim)``."""
        seen: dict[int, str] = {}
        for name, sector in self._sectors.items():
            for idx in sector.indices:
                if not (0 <= idx < self._dim):
                    raise ValueError(f"Sector '{name}': index {idx} out of range for dim={self._dim}")
                if idx in seen:
                    raise ValueError(f"Index {idx} appears in both sector '{seen[idx]}' and sector '{name}'")
                seen[idx] = name

    # --- internal -----------------------------------------------------

    def _build_fields(self) -> dict[str, StructuredExpr]:
        fields: dict[str, StructuredExpr] = {}
        for name, sector in self._sectors.items():
            fields[name] = StructuredExpr(
                sdims=sector.sdims,
                n_features=1,
                param_suite=None,
                labels=(name,),
                _layout_id=self._layout_id,
                _node=_SectorLeaf(
                    sector_name=name,
                    indices=sector.indices,
                    sdims=sector.sdims,
                ),
            )
        return fields

    def const(
        self,
        value: float | int = 1,
        label: str | None = None,
    ) -> StructuredExpr:
        """Scalar constant compatible with this layout.

        Parameters
        ----------
        value : float or int
            The constant value (default ``1``).
        label : str, optional
            Human-readable label.  Defaults to ``str(int(value))`` for
            integer-valued numbers, ``str(value)`` otherwise.
        """
        if label is None:
            if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
                label = str(int(value))
            else:
                label = f"{value:g}"
        return StructuredExpr(
            sdims=(),
            n_features=1,
            param_suite=None,
            labels=(label,),
            _layout_id=self._layout_id,
            _node=_ConstNode(value=value, sdims=()),
        )

    def __repr__(self) -> str:
        parts = [f"dim={self._dim}"]
        for name, sector in self._sectors.items():
            parts.append(f"{name}={sector!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# =====================================================================
# IdentityLayout  (trivial: one vector sector spanning all of dim)
# =====================================================================


class IdentityLayout(_BaseLayout):
    """Trivial layout with a single ``state`` field spanning all of *dim*.

    Example::

        layout = IdentityLayout(dim=3)
        x = layout.state  # StructuredExpr(sdims=(3,), n_features=1)
    """

    def __init__(self, dim: int) -> None:
        super().__init__(
            dim=dim,
            state=VectorSector(indices=tuple(range(dim)), sdim=dim),
        )

    def embed(self, rank: int = 1, **named_fields: StructuredExpr) -> Any:
        raise NotImplementedError("IdentityLayout.embed() is not yet implemented.")
