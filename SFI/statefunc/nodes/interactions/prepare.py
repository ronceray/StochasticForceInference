# SFI/statefunc/nodes/interactions/prepare.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset

# -----------------------------------------------------------------------------
# Cache-only extras convention
# -----------------------------------------------------------------------------
CACHE_PREFIX = "_cache/"
"""
Prefix reserved for **auto-generated** extras.

Any key under ``_cache/``:
- may be large (CSR tables, hyper tables, slot masks, slot geometry...),
- is **derivable** from smaller context (box geometry, offsets, bc, etc.),
- must be safe to drop when the dataset/collection context changes
  (degradation, regridding, cropping, etc.).

Rule of thumb
-------------
If an extra can be regenerated without losing information, it belongs under
``_cache/``.
"""


def is_cache_key(k: str, *, prefix: str = CACHE_PREFIX) -> bool:
    """Return True iff `k` is a cache-only key."""
    return str(k).startswith(prefix)


def purge_cache_extras(
    extras: Optional[Mapping[str, Any]],
    *,
    prefix: str = CACHE_PREFIX,
) -> Optional[Dict[str, Any]]:
    """
    Drop all cache-only keys from an extras mapping.

    This is the universal "invalidate derived structural extras" operation.
    """
    if extras is None:
        return None
    if not any(is_cache_key(k, prefix=prefix) for k in extras.keys()):
        return dict(extras)
    return {k: v for k, v in extras.items() if not is_cache_key(k, prefix=prefix)}


# -----------------------------------------------------------------------------
# Host-side structural preparation hooks
# -----------------------------------------------------------------------------
def _iter_nodes(root: Any) -> Iterable[Any]:
    """DFS over node trees using a conventional `children` attribute."""
    stack = [root]
    seen: set[int] = set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        for ch in getattr(n, "children", ()) or ():
            stack.append(ch)


def prepare_structural_extras_for_expr(expr: Any, extras: MutableMapping[str, Any]) -> Dict[str, Any]:
    """
    Host-side: call `spec.prepare_extras(extras)` for any dispatcher-owned spec in `expr`.

    Contract
    --------
    - This is **host-only** (must run outside JIT tracing).
    - Any returned keys **must** start with ``_cache/``.
      This makes "what is purgable" explicit and prevents accidentally
      smuggling structural arrays into user-facing extras.

    Returns
    -------
    dict
        Newly created cache extras (also inserted into `extras` in-place).
    """
    root = expr.root
    created: Dict[str, Any] = {}

    for node in _iter_nodes(root):
        spec = getattr(node, "spec", None)
        if spec is None:
            continue

        fn = getattr(spec, "prepare_extras", None)
        if fn is None:
            continue

        out = fn(extras)
        if not out:
            continue

        bad = [k for k in out.keys() if not is_cache_key(k)]
        if bad:
            raise ValueError(f"prepare_extras() must only return keys under {CACHE_PREFIX!r}. Offending keys: {bad}")

        extras.update(out)
        created.update(out)

    return created


def build_structural_overlay(expr: Any, dataset: TrajectoryDataset) -> Dict[str, Any]:
    """
    Build an expression's dispatcher-owned structural arrays into a *throwaway* overlay.

    Returns just the ``_cache/`` arrays the specs in `expr` need (CSR / stencil
    tables), built host-side from the dataset's **descriptors** (box geometry,
    user-supplied CSR keys, ...) — **never written onto the dataset**. The build
    always starts from the descriptors, dropping any pre-existing ``_cache/`` keys,
    so the overlay cannot reuse a stale structural table: a changed descriptor
    yields a freshly-built array.

    This is the persistence-free counterpart to
    :func:`prepare_collection_for_expr`. It runs host-side (outside JIT) and is
    meant to be merged into the extras seen by a single evaluation, then discarded.
    """
    descriptors: Dict[str, Any] = {
        k: v
        for src in (dataset.extras_global or {}, dataset.extras_local or {})
        for k, v in src.items()
        if not is_cache_key(k)
    }
    return prepare_structural_extras_for_expr(expr, descriptors)


def prepare_collection_for_expr(coll: TrajectoryCollection, *exprs: Any) -> TrajectoryCollection:
    """
    Return a *transient* collection whose datasets carry the structural arrays
    required by `exprs`, freshly built from each dataset's descriptors.

    The structural arrays live under ``_cache/`` in ``extras_global``. Any
    pre-existing ``_cache/`` keys are dropped and the union of every expression's
    overlay is rebuilt, so a stale table can never survive. The input `coll` is
    never mutated; the returned collection is meant to be used for a single
    evaluation and then discarded (see
    :meth:`~SFI.inference.base.BaseLangevinInference._structural_scope`).
    """
    new_datasets: list[TrajectoryDataset] = []
    for ds in coll.datasets:
        overlay: Dict[str, Any] = {}
        for expr in exprs:
            if expr is not None:
                overlay.update(build_structural_overlay(expr, ds))
        if overlay:
            eg = {k: v for k, v in (ds.extras_global or {}).items() if not is_cache_key(k)}
            eg.update(overlay)
            ds2 = replace(ds, extras_global=eg)
        else:
            ds2 = ds
        new_datasets.append(ds2)

    return TrajectoryCollection(datasets=new_datasets, weights=coll.weights)
