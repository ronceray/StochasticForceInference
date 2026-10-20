# SFI/inference/parametric_core/frame_extras.py
"""
Per-frame (time-dependent) extras on the parametric path.

The runner hands each position block its ``extras`` mapping.  Static
values (``extras_global`` / ``extras_local`` arrays, the static reserved
keys) enter as they are; time-dependent values — a
:class:`~SFI.trajectory.dataset.TimeSeriesExtra`, a plain time-generator
callable, or the reserved per-frame ``time`` — are materialised once per
dataset as ``(T, ...)`` arrays (the same runtime contract as
:meth:`TrajectoryDataset.build_extras`) and sliced to the block's frames
as a :class:`FrameSeries` (a pytree, so it enters ``jit`` traced).

The flow consumers vmap over observation intervals and evaluate the
model at each interval with the *left-point* frame value: interval
``k → k+1`` reads frame ``k``, exactly as the simulator holds a recorded
frame's schedule value across that interval's substeps.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = ["FrameSeries", "split_frames", "frame_window", "with_frame",
           "time_varying_keys", "materialize_frame_extras"]


@jax.tree_util.register_pytree_node_class
class FrameSeries:
    """Per-frame values aligned with a position block (leading axis =
    block frame)."""

    def __init__(self, data):
        self.data = data

    def tree_flatten(self):
        return (self.data,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


def split_frames(extras: Optional[Mapping[str, Any]]
                 ) -> Tuple[Optional[Mapping[str, Any]], Optional[Dict[str, Any]]]:
    """``extras → (static, frames)``; ``frames`` is ``None`` when no value
    is per-frame (the historical static path, unchanged)."""
    if not extras:
        return extras, None
    frames = {k: v.data for k, v in extras.items() if isinstance(v, FrameSeries)}
    if not frames:
        return extras, None
    static = {k: v for k, v in extras.items() if not isinstance(v, FrameSeries)}
    return static, frames


def frame_window(frames: Optional[Dict[str, Any]], a: int, b: Optional[int]):
    """Frames ``[a, b)`` of every per-frame value (``None`` passes through)."""
    if frames is None:
        return None
    return {k: v[a:b] for k, v in frames.items()}


def with_frame(static, frame_k):
    """Model-facing extras at one interval: static values plus that
    interval's frame values (``frame_k`` is ``None`` on the static path)."""
    if frame_k is None:
        return static
    return {**(static or {}), **frame_k}


def time_varying_keys(ds, required) -> Tuple[str, ...]:
    """Keys of ``ds``'s extras that vary per frame and that the model reads.

    Mirrors the integrate dispatcher
    (:func:`SFI.integrate.api._has_time_varying_required_extras`): a
    :class:`TimeSeriesExtra`, a plain time-generator callable, and the
    reserved ``time``.  ``required`` is the set of keys the model reads —
    its ``required_extras`` and its dispatchers' ``structural_extras``
    (per-frame neighbour tables); ``None`` (model does not report its
    extras) selects every time-varying key of the dataset.
    """
    from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra

    keys = []
    for src in (ds.extras_global, ds.extras_local):
        for k, v in (src or {}).items():
            if isinstance(v, TimeSeriesExtra) or (
                    callable(v) and not isinstance(v, FunctionExtra)):
                keys.append(k)
    keys.append("time")
    if required is not None:
        keys = [k for k in keys if k in required]
    return tuple(dict.fromkeys(keys))


def materialize_frame_extras(ds, keys, *, dataset_index: int) -> Dict[str, Any]:
    """``{key: (T, ...) array}`` — :meth:`TrajectoryDataset.build_extras`
    evaluated at every frame (vmapped over the frame index), restricted to
    ``keys``."""
    if not keys:
        return {}
    T = int(jnp.shape(ds.X)[0])

    def _at(t):
        ex = ds.build_extras(t, dataset_index=dataset_index)
        return {k: jnp.asarray(ex[k]) for k in keys}

    return jax.vmap(_at)(jnp.arange(T))
