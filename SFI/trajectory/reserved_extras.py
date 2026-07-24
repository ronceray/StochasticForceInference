"""Framework-owned extras and the single resolver that materialises them.

A force or diffusion expression reads two kinds of data through its ``extras``
mapping: **user** values attached to the trajectory (drive protocols,
per-particle properties, geometry) and **reserved** values supplied by the
framework. The reserved keys are defined once here, in a small registry, and
assembled together with the user values by :func:`resolve_extras` — the single
entry point used by simulation, inference, and diagnostics.

Reserved keys:

``time``
    Absolute time at each resolved frame — lets time-dependent bases (e.g.
    :func:`~SFI.bases.time_fourier`) read the clock.
``duration``
    Total trajectory span.
``dataset_index``
    Dense index of the dataset within its collection, for pooled
    multi-experiment models (:func:`~SFI.bases.per_dataset_scalar`,
    :func:`~SFI.bases.dataset_indicator`).
``particle_index``
    Per-particle integer ids, gathered per edge by interaction dispatchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class ExtrasContext:
    """Everything a reserved-key resolver needs for a set of frames.

    Assembled by whoever drives the evaluation (the trajectory producer, the
    diagnostics residual builder, or the simulator) and passed to
    :func:`resolve_extras`.
    """

    n_particles: int
    dataset_index: int
    frame_times: Any
    duration: Any


@dataclass(frozen=True)
class ReservedKey:
    """A framework-owned extras key and how it is materialised."""

    name: str
    resolve: Callable[[ExtrasContext], Any]


_REGISTRY: Dict[str, ReservedKey] = {}


def register(key: ReservedKey) -> None:
    """Add a reserved key to the registry."""
    _REGISTRY[key.name] = key


register(ReservedKey("time", lambda ctx: ctx.frame_times))
register(ReservedKey("duration", lambda ctx: ctx.duration))
register(ReservedKey("dataset_index", lambda ctx: jnp.asarray(ctx.dataset_index, dtype=jnp.int32)))
register(ReservedKey("particle_index", lambda ctx: jnp.arange(int(ctx.n_particles), dtype=jnp.int32)))


#: The set of reserved key names; user extras may not use these.
RESERVED_NAMES = frozenset(_REGISTRY)


def is_reserved(name: str) -> bool:
    """True when ``name`` is a framework-owned reserved key."""
    return name in _REGISTRY


def resolve_reserved(ctx: ExtrasContext) -> Dict[str, Any]:
    """Materialise every reserved key for ``ctx``."""
    return {name: key.resolve(ctx) for name, key in _REGISTRY.items()}


def resolve_extras(user_extras: Mapping[str, Any], ctx: ExtrasContext) -> Dict[str, Any]:
    """Full per-frame extras: user values plus the resolved reserved keys.

    Reserved names are framework-owned; a user entry colliding with one is
    rejected so the meaning of a reserved key is never ambiguous.
    """
    out = dict(user_extras)
    clash = RESERVED_NAMES.intersection(out)
    if clash:
        raise ValueError(f"extras keys {sorted(clash)} are reserved; rename the user entries.")
    out.update(resolve_reserved(ctx))
    return out


def slice_frame_extras(
    extras_global: Optional[Mapping[str, Any]],
    extras_local: Optional[Mapping[str, Any]],
    *,
    frame_idx: Any,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Materialise user extras at ``frame_idx``.

    * :class:`~SFI.trajectory.dataset.TimeSeriesExtra` → sliced ``value.data[frame_idx]``;
    * :class:`~SFI.trajectory.dataset.FunctionExtra` → its callable, forwarded;
    * plain callable → invoked as ``value(frame_idx, context=context)``;
    * anything else → forwarded unchanged.

    ``extras_local`` overrides ``extras_global`` on key conflicts.
    """
    from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra

    def _materialise(value: Any) -> Any:
        if isinstance(value, FunctionExtra):
            return value.func
        if isinstance(value, TimeSeriesExtra):
            return jnp.asarray(value.data)[frame_idx]
        if callable(value):
            return value(frame_idx, context=context)
        return value

    out: Dict[str, Any] = {}
    for source in (extras_global or {}, extras_local or {}):
        for key, value in source.items():
            out[key] = _materialise(value)
    return out
