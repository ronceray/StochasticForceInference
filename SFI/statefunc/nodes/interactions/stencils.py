"""
Stencil helpers for grid-based SPDEs.

This module provides *spec constructors* for fixed-K regular-grid stencils.

Why a separate module?
---------------------
The :mod:`~SFI.statefunc.nodes.interactions.specs` module defines generic
containers (:class:`~SFI.statefunc.nodes.interactions.specs.HyperFixed`,
:class:`~SFI.statefunc.nodes.interactions.specs.HyperCSR`, ...). Stencils are
domain-specific *ways of producing* such specs (e.g. finite differences on a
Cartesian lattice), so keeping them here avoids mixing generic API with
application-level utilities.

Conventions
-----------
We assume that a field discretized on a regular grid is represented as a
particle system with:

* particle axis = flattened grid index ``p = 0..P-1``;
* state dimension ``dim`` = number of field components per grid site.

This matches the StateExpr/dispatcher convention used across SFI.

Cache-only extras
----------------
When stencil hyper tables / slot masks / slot geometry are materialized as extras
(e.g. via a `prepare_extras()` hook), they must be stored under the ``_cache/``
prefix so they can be purged safely after any context-changing operation.

Box polymorphism strategy
-------------------------
A stencil-based operator (e.g. Laplacian) is created **once** with boundary
conditions and offset layout fixed. The dataset provides only:

* ``{box}/grid_shape`` : integer array-like, shape (ndim,)
* ``{box}/dx``         : float array-like, shape (ndim,) or scalar

The neighbor lists (HyperFixed spec) are then built *on demand* from grid_shape
(and cached by :class:`~SFI.statefunc.nodes.interactions.specs.CachedRule`).

This avoids having to precompute and store large structural arrays in dataset
extras, while keeping the basis / PSF object reusable across grid sizes.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .specs import FromExtrasHyperFixed, HyperFixed, SpecRule

# =============================================================================
# Small utilities (grid <-> flat index)
# =============================================================================


def _as_int_tuple(x) -> tuple[int, ...]:
    """Convert a grid shape-like object to a Python tuple of ints.

    This is used for cache keys and NumPy-side construction of stencil specs.
    """
    if isinstance(x, tuple):
        return tuple(int(v) for v in x)
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"grid_shape must be 1D (got shape {arr.shape})")
    return tuple(int(v) for v in arr.tolist())


def _as_float_array(dx: float | Sequence[float], *, ndim: int) -> np.ndarray:
    """Return dx as a NumPy array of shape (ndim,)."""
    if np.isscalar(dx):
        return np.full((ndim,), float(dx), dtype=np.float64)
    arr = np.asarray(dx, dtype=np.float64)
    if arr.shape != (ndim,):
        raise ValueError(f"dx must be scalar or shape ({ndim},) (got {arr.shape})")
    return arr


def _coords_from_flat(p: np.ndarray, grid_shape: tuple[int, ...], *, order: str) -> np.ndarray:
    """Map flat indices to coordinates.

    Parameters
    ----------
    p
        Flat indices, shape (...,).
    grid_shape
        Grid shape (n0, n1, ..., n_{ndim-1}).
    order
        Flattening order, either "C" (row-major) or "F" (column-major).

    Returns
    -------
    coords
        Integer coordinates, shape (..., ndim).
    """
    if order not in ("C", "F"):
        raise ValueError(f"order must be 'C' or 'F' (got {order!r})")
    coords = np.stack(np.unravel_index(p, grid_shape, order=order), axis=-1)
    return coords.astype(np.int32)


def _flat_from_coords(coords: np.ndarray, grid_shape: tuple[int, ...], *, order: str) -> np.ndarray:
    """Map integer coordinates to flat indices.

    coords must be shape (..., ndim), with 0 <= coords[...,ax] < grid_shape[ax].
    """
    if order == "C":
        strides = np.cumprod((1,) + grid_shape[::-1])[:-1][::-1]
    elif order == "F":
        strides = np.cumprod((1,) + grid_shape)[:-1]
    else:
        raise ValueError(f"order must be 'C' or 'F' (got {order!r})")
    strides = np.asarray(strides, dtype=np.int64)  # (ndim,)
    return (coords.astype(np.int64) * strides).sum(axis=-1).astype(np.int32)


# =============================================================================
# Offset constructors and weights (Cartesian square grid)
# =============================================================================


def square_axis_offsets(
    ndim: int,
    axis: int,
    *,
    scheme: Literal["central", "forward", "backward"] = "central",
    include_center: bool = True,
) -> Array:
    """Offsets for a 1D finite-difference stencil along one grid axis.

    Parameters
    ----------
    ndim
        Grid dimension.
    axis
        Which grid axis to differentiate along (0 ≤ axis < ndim).
    scheme
        One of: "central" (±e_axis), "forward" (+e_axis), "backward" (-e_axis).
    include_center
        Whether to include the zero offset as the first slot.

    Returns
    -------
    offsets
        Integer offsets shaped (K, ndim).
    """
    ndim = int(ndim)
    axis = int(axis)
    if not (0 <= axis < ndim):
        raise ValueError(f"axis must be in [0, {ndim}) (got {axis})")

    offs: list[tuple[int, ...]] = []
    if include_center:
        offs.append((0,) * ndim)

    e = [0] * ndim
    if scheme == "central":
        e[axis] = 1
        offs.append(tuple(e))
        e[axis] = -1
        offs.append(tuple(e))
    elif scheme == "forward":
        e[axis] = 1
        offs.append(tuple(e))
    elif scheme == "backward":
        e[axis] = -1
        offs.append(tuple(e))
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")

    return jnp.asarray(offs, dtype=jnp.int32)


def square_cross_offsets(ndim: int, *, include_center: bool = True) -> Array:
    """Offsets for the standard cross stencil (center + nearest neighbors).

    Ordering convention (important!)
    -------------------------------
    Returned offsets are:

        [0, +e0, -e0, +e1, -e1, ..., +e_{ndim-1}, -e_{ndim-1}]

    This ordering is assumed by :func:`square_weights_laplacian_cross` and by
    the SPDE Laplacian helper in :mod:`SFI.bases.spde`.

    Parameters
    ----------
    ndim
        Grid dimension.
    include_center
        Whether to include the center offset as the first slot.

    Returns
    -------
    offsets
        Integer offsets shaped (K, ndim), with K = 1 + 2*ndim if include_center.
    """
    ndim = int(ndim)
    offs: list[tuple[int, ...]] = []
    if include_center:
        offs.append((0,) * ndim)
    for ax in range(ndim):
        e = [0] * ndim
        e[ax] = 1
        offs.append(tuple(e))
        e[ax] = -1
        offs.append(tuple(e))
    return jnp.asarray(offs, dtype=jnp.int32)


def square_weights_laplacian_cross(dx: float | Sequence[float], *, ndim: int) -> Array:
    """Weights for the cross Laplacian stencil.

    This matches the ordering returned by :func:`square_cross_offsets`.

    For anisotropic dx:
        Δu ≈ Σ_ax (u_{+ax} - 2 u_0 + u_{-ax}) / dx_ax^2

    Parameters
    ----------
    dx
        Grid spacing (scalar or per-axis sequence).
    ndim
        Grid dimension.

    Returns
    -------
    w
        Float weights shaped (K,), with K = 1 + 2*ndim.
    """
    dxv = _as_float_array(dx, ndim=int(ndim))
    inv = 1.0 / (dxv * dxv)  # (ndim,)

    # Slot 0: center coefficient = -2 * sum(inv_dx2)
    w0 = -2.0 * inv.sum()

    # Slots: (+e0, -e0, +e1, -e1, ...)
    w = [w0]
    for ax in range(int(ndim)):
        w.append(inv[ax])
        w.append(inv[ax])
    return jnp.asarray(w, dtype=jnp.float32)


def square_biharmonic_offsets(ndim: int, *, include_center: bool = True) -> Array:
    r"""Offsets for the biharmonic :math:`\nabla^4` stencil on a Cartesian grid.

    In 2D, this is the standard 13-point stencil
    (extended cross radius-2 **plus** diagonal corners ``(±1, ±1)``).

    Ordering convention
    -------------------
    Returned offsets are:

        [center,
         +2e0, -2e0, +2e1, -2e1, ...,          # extended along axes (radius 2)
         +e0, -e0, +e1, -e1, ...,               # cross (radius 1)
         +e0+e1, +e0-e1, -e0+e1, -e0-e1, ...]  # diagonal corners

    Parameters
    ----------
    ndim
        Grid dimension (2 or 3).
    include_center
        Whether to include the center offset as the first slot.

    Returns
    -------
    offsets
        Integer offsets shaped ``(K, ndim)``.
    """
    ndim = int(ndim)
    offs: list[tuple[int, ...]] = []

    if include_center:
        offs.append((0,) * ndim)

    # --- radius-2 along each axis ---
    for ax in range(ndim):
        e = [0] * ndim
        e[ax] = 2
        offs.append(tuple(e))
        e[ax] = -2
        offs.append(tuple(e))

    # --- radius-1 cross ---
    for ax in range(ndim):
        e = [0] * ndim
        e[ax] = 1
        offs.append(tuple(e))
        e[ax] = -1
        offs.append(tuple(e))

    # --- diagonal corners: all combinations of ±1 in pairs of axes ---
    import itertools

    for ax_pair in itertools.combinations(range(ndim), 2):
        for signs in itertools.product([1, -1], repeat=2):
            e = [0] * ndim
            for a, s in zip(ax_pair, signs):
                e[a] = s
            offs.append(tuple(e))

    return jnp.asarray(offs, dtype=jnp.int32)


# =============================================================================
# HyperFixed construction for square stencils
# =============================================================================


def hyperfixed_square_stencil(
    *,
    grid_shape: Sequence[int],
    offsets: Array,
    bc: Literal["pbc", "noflux", "drop"] = "noflux",
    order: Literal["C", "F"] = "C",
    include_slot_extras_offset: bool = True,
) -> tuple[Array, Array, Optional[Array]]:
    """Build a fixed-K stencil as HyperFixed tables for a Cartesian grid.

    Each grid site becomes one hyperedge of arity K, whose participants are
    the focal site and its neighbors at the provided integer `offsets`.

    Parameters
    ----------
    grid_shape
        Spatial grid shape, length ndim.
    offsets
        Integer offsets of shape (K, ndim).
    bc
        Boundary condition handling:

        - ``"pbc"``: periodic wrap
        - ``"noflux"``: out-of-bounds neighbors are replaced by focal index
          (Neumann-like for cross stencils)
        - ``"drop"``: out-of-bounds slots are deactivated via slot_mask
          (neighbors replaced by focal index but masked out)
    order
        Flattening order used to map coords <-> flat indices.
    include_slot_extras_offset
        If True, also returns `slot_extras_offset` = integer offsets broadcast to
        shape (P, K, ndim). This is useful for direction-aware operators
        (grad/div/elastic contractions), and for non-square lattices you can
        later replace this with real edge vectors as extras.

    Returns
    -------
    hyper
        Int array, shape (P, K), neighbor indices for each focal site.
    slot_mask
        Bool array, shape (P, K), True for active slots.
    slot_extras_offset
        Optional int array, shape (P, K, ndim), the integer offset vectors.
    """
    gshape = _as_int_tuple(grid_shape)
    ndim = len(gshape)
    offsets_np = np.asarray(offsets, dtype=np.int32)
    if offsets_np.ndim != 2 or offsets_np.shape[1] != ndim:
        raise ValueError(f"offsets must have shape (K,{ndim}) (got {offsets_np.shape})")
    K = int(offsets_np.shape[0])

    P = int(np.prod(gshape))
    p = np.arange(P, dtype=np.int32)
    coords = _coords_from_flat(p, gshape, order=order)  # (P, ndim)

    # Neighbor coordinates for each slot: (P, K, ndim)
    nbr = coords[:, None, :] + offsets_np[None, :, :]

    # In-bounds mask per slot: (P, K)
    inb = np.ones((P, K), dtype=bool)
    for ax, n in enumerate(gshape):
        inb &= (nbr[..., ax] >= 0) & (nbr[..., ax] < n)

    if bc == "pbc":
        for ax, n in enumerate(gshape):
            nbr[..., ax] %= n
        slot_mask = np.ones((P, K), dtype=bool)
    elif bc in ("noflux", "drop"):
        # Replace any out-of-bounds slot by focal coordinate.
        #
        # - For "noflux": keeping focal indices is the *actual* boundary behavior.
        # - For "drop": focal indices are placeholders; the slot will be deactivated
        #   by slot_mask.
        #
        # IMPORTANT: boolean indexing must match shapes; use np.where to avoid
        # shape pitfalls on (P,K,ndim) arrays.
        nbr = np.where(inb[..., None], nbr, coords[:, None, :])
        if bc == "noflux":
            slot_mask = np.ones((P, K), dtype=bool)
        else:
            slot_mask = inb
    else:
        raise ValueError(f"Unknown bc={bc!r}")

    hyper = _flat_from_coords(nbr, gshape, order=order).astype(np.int32)  # (P, K)

    if include_slot_extras_offset:
        slot_extras_offset = np.broadcast_to(offsets_np[None, :, :], (P, K, ndim)).astype(np.int32)
    else:
        slot_extras_offset = None

    return (
        jnp.asarray(hyper, dtype=jnp.int32),
        jnp.asarray(slot_mask),
        (None if slot_extras_offset is None else jnp.asarray(slot_extras_offset, dtype=jnp.int32)),
    )


class HyperFixedSquareStencilFromBox(SpecRule):
    """Build a HyperFixed stencil from *box* extras: `grid_shape` (and optionally `dx`).

    This is the preferred entry point for *box-polymorphic* SPDE operators.

    Only the *small* box descriptors are required as dataset extras, while the
    large hyper tables are built on demand and typically cached by wrapping
    this rule in :class:`~SFI.statefunc.nodes.interactions.specs.CachedRule`.

    Parameters
    ----------
    offsets
        Integer offsets, shape (K, ndim). The slot order must match the operator.
    bc, order
        Boundary condition and flattening order.
    key_grid_shape
        Extras key for the grid shape (length ndim).
    include_slot_extras_offset
        Whether to store integer offset vectors in `spec.slot_extras["offset"]`.
        This is useful for future direction-aware operators (grad/div, elasticity),
        and for non-square lattices you can later substitute real edge vectors.
    """

    offsets: tuple = eqx.field(static=True)  # tuple-of-tuples of ints
    bc: Literal["pbc", "noflux", "drop"] = eqx.field(static=True)
    order: Literal["C", "F"] = eqx.field(static=True)
    key_grid_shape: str = eqx.field(static=True)
    include_slot_extras_offset: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        offsets: Array,
        bc: Literal["pbc", "noflux", "drop"] = "noflux",
        order: Literal["C", "F"] = "C",
        key_grid_shape: str,
        include_slot_extras_offset: bool = True,
    ):
        _arr = np.asarray(offsets, dtype=np.int32)
        object.__setattr__(self, "offsets", tuple(tuple(int(x) for x in row) for row in _arr))
        object.__setattr__(self, "bc", bc)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "key_grid_shape", str(key_grid_shape))
        object.__setattr__(self, "include_slot_extras_offset", bool(include_slot_extras_offset))

    def arity(self):
        K = len(self.offsets)
        return ("fixed", K)

    def required_extras(self) -> tuple[str, ...]:
        # These are small globals and should be forwarded.
        return (self.key_grid_shape,)

    def structural_extras(self) -> tuple[str, ...]:
        # We do not consume large structural arrays from extras here.
        return ()

    def cache_key(self, extras) -> tuple:
        # CachedRule will combine (P, cache_key). We include full grid_shape to
        # avoid collisions for different factorizations with same P.
        gshape = _as_int_tuple(extras[self.key_grid_shape])
        return (gshape, self.bc, self.order, len(self.offsets))

    def build(self, x, *, v=None, mask=None, extras=None) -> HyperFixed:
        if extras is None:
            raise KeyError("HyperFixedSquareStencilFromBox: extras is required")
        gshape = _as_int_tuple(extras[self.key_grid_shape])
        hyper, slot_mask, slot_offs = hyperfixed_square_stencil(
            grid_shape=gshape,
            offsets=np.array(self.offsets, dtype=np.int32),
            bc=self.bc,
            order=self.order,
            include_slot_extras_offset=self.include_slot_extras_offset,
        )

        slot_extras = None
        if slot_offs is not None:
            slot_extras = {"offset": slot_offs}

        return HyperFixed(hyper=hyper, slot_mask=slot_mask, slot_extras=slot_extras)


# =============================================================================
# Convenience: add box extras
# =============================================================================


def box_extras(
    *,
    grid_shape: Sequence[int],
    dx: float | Sequence[float] = 1.0,
    prefix: str = "box",
) -> Dict[str, Array]:
    """Return the minimal extras dictionary required by box-polymorphic stencils.

    This is intended for `TrajectoryCollection.extras_global.update(...)`.

    Keys
    ----
    - ``{prefix}/grid_shape``: int32 array, shape (ndim,)
    - ``{prefix}/dx``        : float32 array, shape (ndim,)

    Notes
    -----
    - We store dx as a length-ndim vector even when input is scalar, so that
      downstream operators can assume a stable shape.
    - These are *globals* (not structural) and should flow through the dispatcher.
    """
    gshape = _as_int_tuple(grid_shape)
    if any(n <= 0 for n in gshape):
        raise ValueError(f"grid_shape entries must be positive; got {gshape}")
    dxv = _as_float_array(dx, ndim=len(gshape))
    if np.any(dxv <= 0.0):
        raise ValueError(f"dx must be positive; got {dxv}")
    return {
        f"{prefix}/grid_shape": jnp.asarray(gshape, dtype=jnp.int32),
        f"{prefix}/dx": jnp.asarray(dxv, dtype=jnp.float32),
    }


# =============================================================================
# Host-prepared stencil rules (box-polymorphic, JIT-safe at runtime)
# =============================================================================


class PreparedSquareStencilFromBox(SpecRule):
    """
    Regular-grid fixed-K stencil whose *structural* arrays are prepared on the host.

    This rule is the “JIT-friendly” way to use grid-based differential operators:

    - At **prepare time** (Python side, not under JIT), we:
        * read small box descriptors from `extras` (e.g. grid shape, maybe dx later),
        * build large structural arrays (hyper table, slot mask, slot offsets),
        * store them back into `extras` under a cache prefix.

    - At **build/eval time** (can be under JIT), we:
        * only *read* those already-prepared arrays from `extras`,
        * never call NumPy, never hash tracer values, never do Python caching.

    Structural extras policy
    ------------------------
    The keys written under the cache prefix are *structural*: they are consumed by the
    dispatcher/spec builder and must not be forwarded to child local-ops.

    Therefore, these keys are returned by `structural_extras()` and **not** by
    `required_extras()`.

    Cache naming
    ------------
    This class supports two equivalent APIs:

    - `key_prefix="..."`
        Explicit prefix where arrays are stored/read.

    Parameters
    ----------
    offsets
        Integer offsets, shape (K, ndim). Slot order must match the operator.
    bc, order
        Boundary condition and flattening order passed to `hyperfixed_square_stencil`.
    key_grid_shape
        Extras key holding the grid shape (length ndim). Must be *concrete* at prepare time.
    key_prefix
        Prefix used to store structural arrays in extras.
        Keys written/read:
          - f"{key_prefix}/hyper"
          - f"{key_prefix}/slot_mask"
          - optionally f"{key_prefix}/slot_offset"
    include_slot_extras_offset
        If True, store per-slot offset vectors under f"{key_prefix}/slot_offset".
        This is what you want for direction-aware operators later.
    """

    offsets: tuple = eqx.field(static=True)  # tuple-of-tuples of ints
    bc: Literal["pbc", "noflux", "drop"] = eqx.field(static=True)
    order: Literal["C", "F"] = eqx.field(static=True)
    key_grid_shape: str = eqx.field(static=True)
    key_prefix: str = eqx.field(static=True)
    include_slot_extras_offset: bool = eqx.field(static=True)

    # Internal reader: JIT-safe, just reads arrays from extras.
    _reader: "FromExtrasHyperFixed" = eqx.field(static=True)

    def __init__(
        self,
        *,
        offsets: Array,
        bc: Literal["pbc", "noflux", "drop"] = "noflux",
        order: Literal["C", "F"] = "C",
        key_grid_shape: str,
        key_prefix: str,
        include_slot_extras_offset: bool = True,
    ):
        object.__setattr__(self, "key_prefix", str(key_prefix))
        object.__setattr__(self, "key_grid_shape", str(key_grid_shape))

        _arr = np.asarray(offsets, dtype=np.int32)
        object.__setattr__(self, "offsets", tuple(tuple(int(x) for x in row) for row in _arr))
        object.__setattr__(self, "bc", bc)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "include_slot_extras_offset", bool(include_slot_extras_offset))

        # --- structural keys written/read under key_prefix ----------------
        key_h = f"{self.key_prefix}/hyper"
        key_m = f"{self.key_prefix}/slot_mask"
        key_o = f"{self.key_prefix}/slot_offset" if self.include_slot_extras_offset else None

        # Reader rule is purely a “read from extras” spec builder: JIT-safe.
        object.__setattr__(
            self,
            "_reader",
            FromExtrasHyperFixed(
                key_hyper=key_h,
                key_slot_mask=key_m,
                key_slot_offset=key_o,
            ),
        )

    def arity(self):
        # Fixed-K stencil: K is the number of offsets (slots).
        return ("fixed", len(self.offsets))

    def required_extras(self) -> tuple[str, ...]:
        # Small “box descriptors” should remain forwardable and available to children.
        # They are also what `prepare_extras` needs.
        return (self.key_grid_shape,)

    def structural_extras(self) -> tuple[str, ...]:
        # Structural arrays live under the cache prefix; these must not be forwarded.
        return self._reader.structural_extras()

    # ------------------------------------------------------------------
    # Host-side preparation hook
    # ------------------------------------------------------------------
    def prepare_extras(self, extras: Mapping[str, Any]) -> dict[str, Any]:
        """
        Host-side hook: build and insert structural stencil arrays into `extras`.

        Contract
        --------
        - Input `extras` must provide `key_grid_shape` as a *concrete* value.
        - Returns a dict of additions/updates to be merged into extras_global.
        - If keys already exist in extras, does nothing (returns {}).

        Notes
        -----
        This function is intentionally *not JIT-safe* (uses Python/NumPy),
        and must be called outside traced code (e.g. in process.initialize(),
        inference entry points, etc.).
        """
        if extras is None:
            raise KeyError("PreparedSquareStencilFromBox.prepare_extras: extras is required")

        # If the structural arrays are already present, do nothing.
        key_h = f"{self.key_prefix}/hyper"
        key_m = f"{self.key_prefix}/slot_mask"
        key_o = f"{self.key_prefix}/slot_offset"
        have_h = key_h in extras
        have_m = key_m in extras
        have_o = (key_o in extras) if self.include_slot_extras_offset else True
        if have_h and have_m and have_o:
            return {}

        # Read grid shape (must be concrete here).
        gshape = _as_int_tuple(extras[self.key_grid_shape])

        # Build heavy structural arrays (NumPy-heavy helper).
        hyper, slot_mask, slot_offs = hyperfixed_square_stencil(
            grid_shape=gshape,
            offsets=np.array(self.offsets, dtype=np.int32),
            bc=self.bc,
            order=self.order,
            include_slot_extras_offset=self.include_slot_extras_offset,
        )

        # Normalize outputs.
        hyper = jnp.asarray(hyper, dtype=jnp.int32)
        if slot_mask is None:
            slot_mask = jnp.ones_like(hyper, dtype=jnp.bool_)
        else:
            slot_mask = jnp.asarray(slot_mask, dtype=jnp.bool_)

        built: dict[str, Any] = {
            key_h: hyper,
            key_m: slot_mask,
        }
        if self.include_slot_extras_offset:
            # slot_offs has shape (M, K, ndim) with integer offsets.
            built[key_o] = jnp.asarray(slot_offs, dtype=jnp.int32)

        return built

    # ------------------------------------------------------------------
    # JIT-safe spec builder: just read arrays prepared above
    # ------------------------------------------------------------------
    def build(self, x, *, v=None, mask=None, extras: Mapping[str, Any] | None = None) -> HyperFixed:
        """
        Build a `HyperFixed` spec by reading arrays from `extras`.

        This is JIT-safe provided `extras` already contains the prepared keys.
        """
        return self._reader.build(x, v=v, mask=mask, extras=extras)
