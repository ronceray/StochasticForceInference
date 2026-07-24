from typing import Any, Literal, Optional

from .basis import Basis
from .nodes.base import BaseNode
from .nodes.interactions.dispatcher import InteractionDispatcher
from .nodes.interactions.specs import (
    AutoPairs,
    FromExtrasPairsCSR,
    HyperCSR,
    HyperFixed,
    PairsCSR,
    SpecRule,
)
from .psf import PSF
from .stateexpr import StateExpr


# Helper: does the subtree contain any parameters?
def _tree_is_parametric(root: BaseNode) -> bool:
    def _walk(n: Any) -> bool:
        if getattr(n, "param_suite", None) is not None:
            return True
        for ch in getattr(n, "children", ()):
            if _walk(ch):
                return True
        for attr in ("child", "inner"):
            ch = getattr(n, attr, None)
            if ch is not None and _walk(ch):
                return True
        return False

    return _walk(root)


class Interactor(StateExpr):
    """
    Local interaction expression (pre-dispatch).

    - `root` must be a local graph built from InteractionLeaf(s):
        particles_input=True, pdepth=0.

    Compose as usual: `inter = make_interactor(...); inter2 = (inter & inter)...`
    Then call `.dispatch(...)` exactly once to obtain a **Basis** or a **PSF**.
    """

    # sugar: keep the same fluent ops API as StateExpr (inherited)

    # ---- Dispatch API ---------------------------------------------------------
    def dispatch(
        self,
        spec: PairsCSR | HyperFixed | HyperCSR | SpecRule,
        *,
        owners: Literal["focal", "all", "custom", "global"] = "focal",
        focal_index: int = 0,
        owner_weights=None,
        reducer: Literal["sum", "mean", "max"] = "sum",
        normalize_by_degree: bool = False,
        exclude_self: bool = True,
        chunk_size: Optional[int] = None,
        return_as: Literal["auto", "basis", "psf"] = "auto",
        drop_features: Optional[bool] = None,
    ):
        disp = InteractionDispatcher(
            self.root,
            spec=spec,
            owners=owners,
            focal_index=focal_index,
            owner_weights=owner_weights,
            reducer=reducer,
            normalize_by_degree=normalize_by_degree,
            exclude_self=exclude_self,
            chunk_size=chunk_size,
        )
        # Decide wrapper kind
        kind = return_as
        if kind == "auto":
            kind = "psf" if _tree_is_parametric(self.root) else "basis"
        if kind == "basis":
            return Basis(disp)
        elif kind == "psf":
            return PSF(disp, drop_features=drop_features)
        else:
            raise ValueError(f"Unknown return_as={return_as!r}")

    # common sugars for pairs
    def dispatch_pairs(self, *, symmetric=True, exclude_self=True, **kwargs):
        return self.dispatch(AutoPairs(symmetric=symmetric, exclude_self=exclude_self), **kwargs)

    def dispatch_pairs_from_extras(self, *, indptr_key: str, indices_key: str, **kwargs):
        return self.dispatch(FromExtrasPairsCSR(indptr_key, indices_key), **kwargs)
