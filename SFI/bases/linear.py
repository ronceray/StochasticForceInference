# SFI.bases.linear
# =================
# Lightweight linear utilities for building first-order bases in x and v.
#
# Contract assumed here:
# - User functions are called on a SINGLE SAMPLE (no leading batch axes).
# - For rank=VECTOR leaves, x and v arrive as shape (dim,).
# - For rank=SCALAR leaves, the function may return () or (k,) where k=n_features.
# - Feature axis must be present in the final leaf output; for n_features=1 on
#   a scalar leaf, BasisLeaf will auto-insert it from a scalar return.
#
# pdepth=0 (non-interacting) everywhere in this helper module.

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp

from ..statefunc import Basis, Rank, make_basis
from .monomials import monomials_degree  # degree-specific builder

__all__ = [
    "linear_basis",
    "X",
    "V",
    "x_coordinate",
    "x_coordinates",
    "v_coordinate",
    "v_coordinates",
    "x_components",
    "v_components",
    "unit_axes",
    "frame",
]


def linear_basis(dim: int, *, include_x: bool = True, include_v: bool = False):
    """
    Degree-1 monomial basis in (x, v).

    Parameters
    ----------
    dim : int
        Spatial dimension.
    include_x : bool
        Include linear x terms.
    include_v : bool
        Include linear v terms.

    Returns
    -------
    Basis
        Rank-1 (vector) basis concatenating requested degree-1 monomials.
    """
    return monomials_degree(1, dim=dim, include_x=include_x, include_v=include_v)


def X(dim: int, *, label: Optional[str] = None) -> Basis:
    """
    Identity in x with an explicit feature axis.

    Input : x ∈ R^dim
    Output: Y ∈ R^{dim×1}
    """

    def _eval(x):
        return x[:, None]  # (dim, 1)

    label = "x" if label is None else label
    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.VECTOR,
        n_features=1,
        needs_v=False,
        labels=[label],
    )


def V(dim: int, *, label: Optional[str] = None) -> Basis:
    """
    Identity in v with an explicit feature axis.

    Input : v ∈ R^dim  (provided via keyword v=...)
    Output: Y ∈ R^{dim×1}
    """

    def _eval(x, v):
        return v[:, None]  # (dim, 1)

    label = "v" if label is None else label
    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.VECTOR,
        n_features=1,
        needs_v=True,
        labels=[label],
    )


def x_coordinate(index: int, *, dim: int, label: Optional[str] = None) -> Basis:
    """
    Single x-coordinate as a scalar feature.

    Input : x ∈ R^dim
    Return: scalar (); BasisLeaf will auto-insert feature axis → (1,)
    """

    def _eval(x):
        return x[index]  # ()

    label = f"x{index}" if label is None else label
    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=1,
        needs_v=False,
        labels=[label],
    )


def field_component(index: int, *, n_fields: int, label: Optional[str] = None) -> Basis:
    """Extract a single field component from an SPDE state vector.

    Alias for :func:`x_coordinate` with SPDE-oriented naming.

    Parameters
    ----------
    index : int
        Zero-based index of the field component to extract.
    n_fields : int
        Total number of field components per grid site (= ``dim``).
    label : str, optional
        Human-readable label; defaults to ``"field[{index}]"``.
    """
    if label is None:
        label = f"field[{index}]"
    return x_coordinate(index, dim=n_fields, label=label)


def x_coordinates(indices: Sequence[int], *, dim: int, labels: Optional[Sequence[str]] = None) -> Basis:
    """
    Multiple x-coordinates as scalar features.

    Input : x ∈ R^dim
    Output: y ∈ R^{k} with k=len(indices)
    """
    indices = jnp.array(indices)

    def _eval(x):
        return x[indices]  # (k,)

    if labels is None:
        labels = [f"x{i}" for i in indices]
    elif len(labels) != len(indices):
        raise ValueError("x_coordinates: labels length must match number of indices")

    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=len(indices),
        needs_v=False,
        labels=list(labels),
    )


def v_coordinate(index: int, *, dim: int, label: Optional[str] = None) -> Basis:
    """
    Single v-coordinate as a scalar feature.

    Input : v ∈ R^dim (provided via keyword v=...)
    Return: scalar (); BasisLeaf will auto-insert feature axis → (1,)
    """

    def _eval(x, v):
        return v[index]  # ()

    label = f"v{index}" if label is None else label
    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=1,
        needs_v=True,
        labels=[label],
    )


def v_coordinates(indices: Sequence[int], *, dim: int, labels: Optional[Sequence[str]] = None) -> Basis:
    """
    Multiple v-coordinates as scalar features.

    Input : v ∈ R^dim (provided via keyword v=...)
    Output: y ∈ R^{k} with k=len(indices)
    """
    indices = jnp.array(indices)

    def _eval(x, v):
        return v[indices]  # (k,)

    if labels is None:
        labels = [f"v{i}" for i in indices]
    elif len(labels) != len(indices):
        raise ValueError("v_coordinates: labels length must match number of indices")

    return make_basis(
        func=_eval,
        dim=dim,
        rank=Rank.SCALAR,
        n_features=len(indices),
        needs_v=True,
        labels=list(labels),
    )


# ---------------------------------------------------------------------------
# Component / axis unpackers
# ---------------------------------------------------------------------------
_DEFAULT_X_LABELS = ("x", "y", "z", "w")
_DEFAULT_V_LABELS = ("vx", "vy", "vz", "vw")
_DEFAULT_E_LABELS = ("ex", "ey", "ez", "ew")


def _auto_labels(dim: int, defaults: Sequence[str], prefix: str) -> list[str]:
    if dim <= len(defaults):
        return [defaults[i] for i in range(dim)]
    return [f"{prefix}{i}" for i in range(dim)]


def x_components(dim: int, *, labels: Optional[Sequence[str]] = None) -> tuple[Basis, ...]:
    """Unpack scalar x-coordinate bases, one per axis.

    >>> x, y, z = x_components(3)

    Each returned basis is rank-0 with one feature. Labels default to
    ``("x", "y", "z", "w")`` for ``dim <= 4`` and ``("x0", "x1", ...)`` otherwise.
    """
    if labels is None:
        labels = _auto_labels(dim, _DEFAULT_X_LABELS, "x")
    elif len(labels) != dim:
        raise ValueError(f"x_components: labels length ({len(labels)}) must equal dim ({dim})")
    return tuple(x_coordinate(i, dim=dim, label=labels[i]) for i in range(dim))


def v_components(dim: int, *, labels: Optional[Sequence[str]] = None) -> tuple[Basis, ...]:
    """Unpack scalar v-coordinate bases, one per axis.

    >>> vx, vy, vz = v_components(3)
    """
    if labels is None:
        labels = _auto_labels(dim, _DEFAULT_V_LABELS, "v")
    elif len(labels) != dim:
        raise ValueError(f"v_components: labels length ({len(labels)}) must equal dim ({dim})")
    return tuple(v_coordinate(i, dim=dim, label=labels[i]) for i in range(dim))


def unit_axes(dim: int, *, labels: Optional[Sequence[str]] = None) -> tuple[Basis, ...]:
    """Unpack unit-vector bases (one per spatial axis).

    >>> ex, ey, ez = unit_axes(3)

    Each returned basis is rank-1 with a single feature carrying the unit
    vector along that axis. Labels default to ``("ex", "ey", "ez", "ew")``
    for ``dim <= 4`` and ``("e0", "e1", ...)`` otherwise.
    """
    # local import to avoid a circular import at module load
    from .constants import unit_vector_basis

    if labels is None:
        labels = _auto_labels(dim, _DEFAULT_E_LABELS, "e")
    elif len(labels) != dim:
        raise ValueError(f"unit_axes: labels length ({len(labels)}) must equal dim ({dim})")

    out = []
    for i in range(dim):
        e = unit_vector_basis(dim, axes=[i])
        # Override the leaf's label with the friendly default. Bypass the
        # Equinox freeze to patch labels after construction; fragile if
        # Equinox changes its freeze mechanism — prefer constructing with
        # labels set.
        leaf = e.root
        object.__setattr__(leaf, "labels", (labels[i],))
        out.append(e)
    return tuple(out)


def frame(
    dim: int,
    *,
    velocity: bool = False,
    x_labels: Optional[Sequence[str]] = None,
    v_labels: Optional[Sequence[str]] = None,
    e_labels: Optional[Sequence[str]] = None,
) -> tuple[Basis, ...]:
    """Default compositional frame: constant ``1`` + coordinate scalars + unit axes.

    Overdamped (``velocity=False``)::

        one, *x_components(dim), *unit_axes(dim)

    Underdamped (``velocity=True``)::

        one, *x_components(dim), *v_components(dim), *unit_axes(dim)

    Examples
    --------
    >>> one, x, y, z, ex, ey, ez = frame(3)
    >>> one, x, y, z, vx, vy, vz, ex, ey, ez = frame(3, velocity=True)

    Custom labels (useful for ``dim > 4``):

    >>> bundle = frame(5, x_labels=["q0","q1","q2","q3","q4"])
    """
    from .constants import ones_basis

    pieces: list[Basis] = [ones_basis(dim)]
    pieces.extend(x_components(dim, labels=x_labels))
    if velocity:
        pieces.extend(v_components(dim, labels=v_labels))
    pieces.extend(unit_axes(dim, labels=e_labels))
    return tuple(pieces)
