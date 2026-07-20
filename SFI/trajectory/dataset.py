# dataset.py
"""
Trajectory dataset: single-index producer with explicit valid index window.

This module defines :class:`TrajectoryDataset`, an immutable container for one
trajectory and a JAX-traceable *single-t* row producer used by the integration
runtime. No shape heuristics are used. Time bounds are enforced by an explicit
``valid_indices(require)`` window; all gathers assume indices are in-bounds.

Key APIs
--------
- :meth:`valid_indices(require, subsampling=None)`
    Returns the time indices ``t`` for which all requested streams are in-bounds.
- :meth:`make_producer(require, include_mask=True, include_dt=True, ...)`
    Returns ``producer(t)`` that builds a fixed-structure single-row mapping.
- :meth:`build_extras(t, dataset_index=0, context=None)`
    Assembles user + reserved extras at a single time index.

Streams
-------
- States: ``"X"``, ``"X_minus"``, ``"X_plus"``,
          ``"X_minusminus"``, ``"X_plusplus"``.
- Increments: ``"dX_minus" = X[t]-X[t-1]``, ``"dX" = X[t+1]-X[t]``,
              ``"dX_plus" = X[t+2]-X[t+1]``.
- Windows: ``"X_window:<W>"`` returns ``(W, N, d)`` containing
  ``X[t - (W-1)//2], ..., X[t + W//2]`` (W positions, any positive int).
  Odd W → symmetric; even W → one extra to the right.
- Mask: ``"mask"`` (per-particle validity at ``t``).
- Time steps: if ``include_dt=True``, provides ``"dt"`` and, when required,
  ``"dt_minus"``, ``"dt_plus"`` from either scalar/array ``dt`` or absolute
  times ``t``.
- Extras: include by adding ``"extras"`` to ``require``. Values are static
  objects or :class:`TimeSeriesExtra` that are sliced at time ``t``. Callables
  are allowed if JAX-traceable and accept ``(t_idx, context=None)``.

Boundary policy
---------------
- ``valid_indices`` computes the exact interior window from stream offsets.
- All gathers assume in-bounds indices. No clamping. Passing an invalid
  index is a logic error and leads to undefined behavior under XLA.
- Edge effects must be handled by downstream masking via ``"mask_out"``.

Example
-------
>>> ds = TrajectoryDataset(X, dt=0.01, extras_global={"box": jnp.array([Lx, Ly])})
>>> req = {"X", "dX", "mask", "extras"}
>>> t_idx = ds.valid_indices(req)                  # e.g. arange(0, T-1)
>>> producer = ds.make_producer(req, include_dt=True)
>>> # Integrator does: Ys = jax.vmap(lambda tt: program(**producer(tt)))(t_idx)
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from SFI.trajectory.reserved_extras import ExtrasContext, resolve_extras, slice_frame_extras

Array = jax.Array

__all__ = [
    "FunctionExtra",
    "function_extra",
    "TimeSeriesExtra",
    "time_series_extra",
    "TrajectoryDataset",
]

# --------------------------------------------------------------------------- #
# Time-series extras
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TimeSeriesExtra:
    """Wrapper for time-dependent extras with an explicit leading time axis.

    Parameters
    ----------
    data :
        Array with shape ``(T, ...)`` for globals or ``(T, N, ...)`` for
        per-particle extras. The dataset will slice ``data[t]``.
    """

    data: Array


def time_series_extra(x: Any) -> TimeSeriesExtra:
    """Build a :class:`TimeSeriesExtra` from an array-like."""
    return TimeSeriesExtra(jnp.asarray(x))


@dataclass(frozen=True)
class FunctionExtra:
    """Wrapper for a JAX-compatible callable passed through extras.

    Unlike plain callables (which are invoked eagerly by
    :meth:`TrajectoryDataset.build_extras` as time-dependent generators),
    a ``FunctionExtra`` is **passed through unchanged** so the user's basis
    function can call it inside JIT.

    Parameters
    ----------
    func : callable
        A JAX-traceable function, e.g. ``func(x) -> Array``.  It will be
        captured as a compile-time constant inside ``@jax.jit``.

    Examples
    --------
    >>> adhesion = FunctionExtra(lambda x: jnp.exp(-jnp.sum(x**2)))
    >>> coll = TrajectoryCollection.from_arrays(
    ...     X=X, dt=0.01,
    ...     extras_global={"adhesion": adhesion},
    ... )
    """

    func: Callable


def function_extra(func: Callable) -> FunctionExtra:
    """Build a :class:`FunctionExtra` from a callable."""
    if not callable(func):
        raise TypeError(f"function_extra() requires a callable, got {type(func)}")
    return FunctionExtra(func)


def _slice_time_extras(extras: Optional[Mapping[str, Any]], a: int, b: int, T: int) -> Dict[str, Any]:
    """Restrict an extras mapping to frames ``[a, b)``.

    Applies the runtime contract of :meth:`TrajectoryDataset.build_extras`:
    :class:`TimeSeriesExtra` values are sliced on their leading time axis
    (validated against ``T``), :class:`FunctionExtra` and static values
    pass through, and plain callables (time-dependent generators
    ``f(t_idx, context=None)``) are offset-wrapped so the slice keeps
    seeing its original time indices.
    """
    out: Dict[str, Any] = {}
    for key, val in (extras or {}).items():
        if isinstance(val, TimeSeriesExtra):
            data = val.data
            if int(data.shape[0]) != T:
                raise ValueError(
                    f"TimeSeriesExtra {key!r} has leading axis {int(data.shape[0])} != T={T}; "
                    "cannot slice it consistently."
                )
            out[key] = TimeSeriesExtra(data[a:b])
        elif isinstance(val, FunctionExtra):
            out[key] = val
        elif callable(val):
            if a == 0:
                out[key] = val
            else:

                def _shifted(t_idx, context=None, _f=val, _a=a):
                    return _f(t_idx + _a, context=context)

                out[key] = _shifted
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

# Streams and their required time offsets relative to the central index t.
# Values are (min_offset, max_offset).


def _parse_window_key(name: str) -> Optional[int]:
    """Parse ``'X_window:<W>'`` → W (positive int), or None for other keys.

    The window delivers W consecutive positions at offsets
    ``-(W-1)//2, ..., W//2`` relative to the central time index.
    For odd W the window is symmetric; for even W it extends one
    extra step to the right.
    """
    if not name.startswith("X_window:"):
        return None
    w = int(name.split(":", 1)[1])
    if w < 1:
        raise ValueError(f"X_window width must be a positive integer, got {w}")
    return w


STREAM_OFFSETS: Mapping[str, Tuple[int, int]] = {
    "X_m4": (-4, 0),
    "X_m3": (-3, 0),
    "X_minusminus": (-2, 0),
    "X_minus": (-1, 0),
    "X": (0, 0),
    "X_plus": (0, +1),
    "X_plusplus": (0, +2),
    "X_p3": (0, +3),
    "X_p4": (0, +4),
    "dX_minus": (-1, 0),
    "dX": (0, +1),
    "dX_plus": (+1, +2),
    "mask": (0, 0),
    "__dt__": (0, +1),  # pseudo-key: excluded from stream slicing but forces valid t+1 for dt
}


@dataclass(frozen=True)
class TrajectoryDataset:
    """Immutable dataset for a single trajectory.

    Parameters
    ----------
    X :
        State array of shape ``(T, N, d)`` or ``(T, d)``. If ``(T, d)``, N is 1.
    dt :
        Either a scalar step, an array of shape ``(T,)`` (per-step), or ``None``.
        If ``None`` and ``t`` is provided, steps are derived from ``t``.
    t :
        Optional absolute time vector of shape ``(T,)``. If provided, it defines
        dt via finite differences when requested.
    mask :
        Optional boolean mask of shape ``(T, N)`` or ``(T,)`` marking valid
        observations at time t and particle n ("static mask").  If ``None``,
        all ones.  A True entry means the particle's *position* is known and
        can be used for state evaluation (e.g. neighbor forces).
    dynamic_mask :
        Optional boolean mask of shape ``(T, N)`` or ``(T,)`` marking entries
        whose *increments* are reliable and should contribute to parameter
        fitting ("dynamic mask").  Must be a subset of ``mask``
        (``dynamic_mask ⊆ mask``).  If ``None``, defaults to ``mask``.
        Typical use: particles near open boundaries are statically valid
        (their positions are known) but dynamically masked (their
        neighborhoods are incomplete, biasing their increments).
    extras_global :
        Dict of global extras. Values are static objects, :class:`TimeSeriesExtra`,
        or JAX-traceable callables ``f(t_idx, context=None) -> Array`` with a
        leading time axis.
    extras_local :
        Dict of per-particle extras. Same typing as ``extras_global``.
        Time-series entries typically have shape ``(T, N, ...)``.
    metadata :
        Free-form metadata.
    """

    X: Array
    dt: Optional[Array | float] = None
    t: Optional[Array] = None
    mask: Optional[Array] = None
    dynamic_mask: Optional[Array] = None
    extras_global: Dict[str, Any] | None = field(default_factory=dict)
    extras_local: Dict[str, Any] | None = field(default_factory=dict)
    meta: Dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Stable per-dataset identity, carried in ``meta`` so it survives
        # degradation, splitting, and save/load, and used to derive the dense
        # ``dataset_index`` within a collection.
        if self.meta is None:
            object.__setattr__(self, "meta", {})
        self.meta.setdefault("uuid", _uuid.uuid4().hex)

    @property
    def uuid(self) -> str:
        """Stable identity of this dataset."""
        return self.meta["uuid"]

    # ---- basic shapes ----------------------------------------------------- #
    @property
    def T(self) -> int:
        return int(jnp.asarray(self.X).shape[0])

    @property
    def N(self) -> int:
        X = jnp.asarray(self.X)
        return int(X.shape[1]) if X.ndim == 3 else 1

    @property
    def d(self) -> int:
        X = jnp.asarray(self.X)
        return int(X.shape[-1])

    def Teff(self, required: Set[str], *, subsampling: int = 1) -> float:
        """Effective exposure time over valid indices for weighting.

        Defined as

            Teff = sum_t N_active[t] * dt[t],

        where N_active[t] is the number of active (unmasked) particles at time
        index t under the same stream requirements used by the integration
        runtime.

        This reuses the same dt logic as :meth:`_dt_fields_single` and the same
        masking logic as :meth:`_output_mask_single`, so that weighting matches
        exactly what the runtime sees.
        """
        idx = self.valid_indices(required, subsampling=subsampling)
        if idx.size == 0:
            return 0.0

        # Per-step dt for the central index t, using the same rules as the
        # runtime (t vs dt array, scalar dt, etc.).
        force_dt_keys = {"dt"}

        def _dt_single(t_scalar: Array) -> Array:
            dt_fields = self._dt_fields_single(
                required,
                t_scalar,
                force_dt_keys=force_dt_keys,
            )
            if "dt" not in dt_fields:
                raise ValueError(
                    "Teff requires dt to be computable; this should not happen if either 't' or 'dt' is provided."
                )
            return dt_fields["dt"]

        dt_per = jax.vmap(_dt_single)(idx)  # (K,)

        # N_active[t]: same semantics as the "mask_out" produced by
        # make_producer(require, include_mask=True).
        mask_out = jax.vmap(lambda t_scalar: self._output_mask_single(required, t_scalar))(idx)  # (K, N)
        N_active = jnp.sum(mask_out.astype(jnp.float32), axis=-1)  # (K,)

        return float(jnp.sum(N_active * dt_per.astype(jnp.float32)))

    # Convenience builder
    @classmethod
    def from_arrays(
        cls,
        *,
        X: Any,
        dt: Optional[float] = None,
        t: Optional[Any] = None,
        mask: Optional[Any] = None,
        dynamic_mask: Optional[Any] = None,
        extras_global: Optional[Dict[str, Any]] = None,
        extras_local: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "TrajectoryDataset":
        """Construct a dataset from array-likes.

        All inputs are converted to JAX arrays where relevant; extras and meta
        are stored as-is (no deep conversion).

        Returns
        -------
        TrajectoryDataset

        Raises
        ------
        ValueError
            If X contains NaN/Inf, has wrong dimensionality, dt <= 0,
            or the trajectory is too short for any useful computation.
        """
        import warnings

        X = jnp.asarray(X)

        # ---- shape validation ----
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError(
                f"X must have shape (T, d) or (T, N, d), got shape {tuple(X.shape)}. "
                f"For a single scalar time series, reshape to (T, 1)."
            )

        T = X.shape[0]
        if T < 1:
            raise ValueError(f"Trajectory must have at least 1 time step (got T={T}).")
        if T < 4:
            warnings.warn(
                f"Very short trajectory (T={T}). Most inference methods need T >> 1 for meaningful results.",
                stacklevel=2,
            )

        # ---- finiteness check (valid positions only) ----
        # Masked positions may legitimately contain fill values (e.g. 0.0 or
        # even NaN when data is sparse/patchy). Only check finite-ness for the
        # entries that are actually valid (mask=True).
        _check_X: Any = X
        if mask is not None:
            _m = jnp.asarray(mask, dtype=bool)
            if X.ndim == 3 and _m.ndim == 2:  # (T,N,d) + (T,N)
                _check_X = X[_m]  # (K, d)
            elif X.ndim == 2 and _m.ndim == 1:  # (T,d) + (T,)
                _check_X = X[_m]
        if not jnp.all(jnp.isfinite(_check_X)):
            n_nan = int(jnp.sum(jnp.isnan(_check_X)))
            n_inf = int(jnp.sum(jnp.isinf(_check_X)))
            raise ValueError(
                f"X contains non-finite values ({n_nan} NaN, {n_inf} Inf) "
                f"in unmasked positions. "
                f"Clean or mask your data before constructing a TrajectoryDataset."
            )

        # ---- dt validation ----
        if dt is not None:
            dt_arr = jnp.asarray(dt)
            if dt_arr.ndim == 0:
                if float(dt_arr) <= 0:
                    raise ValueError(f"Scalar dt must be positive, got dt={float(dt_arr)}.")
            else:
                if jnp.any(dt_arr <= 0):
                    raise ValueError(
                        f"All dt values must be positive. Found {int(jnp.sum(dt_arr <= 0))} "
                        f"non-positive entries (min={float(jnp.min(dt_arr))})."
                    )

        # ---- t validation ----
        t = None if t is None else jnp.asarray(t)
        if t is not None:
            if t.shape[0] != T:
                raise ValueError(f"Time vector length ({t.shape[0]}) must match X's time dimension ({T}).")
            if dt is not None:
                warnings.warn(
                    "Both 't' and 'dt' were provided. 't' takes precedence; 'dt' will be ignored.",
                    stacklevel=2,
                )

        mask = None if mask is None else jnp.asarray(mask)
        dynamic_mask = None if dynamic_mask is None else jnp.asarray(dynamic_mask)

        # ---- dynamic_mask ⊆ mask validation ----
        if dynamic_mask is not None and mask is not None:
            _s = jnp.asarray(mask, dtype=bool)
            _d = jnp.asarray(dynamic_mask, dtype=bool)
            # Broadcast to compatible shapes for the subset check.
            if _s.ndim == 1 and _d.ndim == 1:
                pass
            elif _s.ndim == 2 and _d.ndim == 2:
                pass
            elif _s.ndim == 1 and _d.ndim == 2:
                _s = _s[:, None]
            elif _s.ndim == 2 and _d.ndim == 1:
                _d = _d[:, None]
            if bool(jnp.any(_d & ~_s)):
                raise ValueError(
                    "dynamic_mask must be a subset of mask: found entries where dynamic_mask is True but mask is False."
                )

        return cls(
            X=X,
            dt=dt,
            t=t,
            mask=mask,
            dynamic_mask=dynamic_mask,
            extras_global=extras_global or {},
            extras_local=extras_local or {},
            meta=meta or {},
        )

    # ---- normalized views ------------------------------------------------- #
    def _X3d(self) -> Array:
        """Return X with shape (T, N, d)."""
        X = jnp.asarray(self.X)
        if X.ndim == 3:
            return X
        if X.ndim == 2:
            return X[:, None, :]
        raise ValueError(f"X must have shape (T,N,d) or (T,d), got {tuple(X.shape)}")

    def _M2d(self) -> Array:
        """Return static mask with shape (T, N) and dtype bool."""
        if self.mask is None:
            return jnp.ones((self.T, self.N), dtype=bool)
        M = jnp.asarray(self.mask).astype(bool)
        if M.ndim == 2:
            return M
        if M.ndim == 1:
            return M[:, None]
        raise ValueError(f"mask must have shape (T,N) or (T,), got {tuple(M.shape)}")

    def _dynamic_M2d(self) -> Array:
        """Return dynamic mask with shape (T, N) and dtype bool.

        Falls back to the static mask if no dynamic_mask was provided.
        """
        if self.dynamic_mask is None:
            return self._M2d()
        M = jnp.asarray(self.dynamic_mask).astype(bool)
        if M.ndim == 2:
            return M
        if M.ndim == 1:
            return M[:, None]
        raise ValueError(f"dynamic_mask must have shape (T,N) or (T,), got {tuple(M.shape)}")

    # ---- offsets and valid window ---------------------------------------- #
    @staticmethod
    def _required_offsets(required: Set[str]) -> Tuple[int, int]:
        """Aggregate min/max offsets implied by required streams."""
        amin, amax = 0, 0
        for k in required:
            if k in STREAM_OFFSETS:
                a, b = STREAM_OFFSETS[k]
                amin = min(amin, a)
                amax = max(amax, b)
            else:
                w = _parse_window_key(k)
                if w is not None:
                    left = (w - 1) // 2
                    right = w - 1 - left  # = w // 2
                    amin = min(amin, -left)
                    amax = max(amax, +right)
        return amin, amax

    def valid_indices(self, required: Set[str], subsampling: Optional[int] = None) -> Array:
        """Return valid time indices given required streams.

        A time index ``t`` is valid iff ``t + amin >= 0`` and ``t + amax <= T-1``,
        where ``(amin, amax)`` aggregates all offsets required by streams in
        ``required``. Extras do not affect the window.

        Parameters
        ----------
        required :
            Set of stream names and possibly ``"extras"``.
        subsampling :
            Optional positive integer. If provided, keep only indices where
            ``t % subsampling == 0`` (grid-aligned to multiples of
            ``subsampling``).  This may exclude the first valid index if it
            is not a multiple of ``subsampling``.

        Returns
        -------
        jax.Array
            1-D array of valid time indices (dtype=int32), possibly empty.
        """
        T = self.T
        amin, amax = self._required_offsets(required)
        start = max(0, -amin)
        stop_incl = T - 1 - max(0, amax)
        if stop_incl < start:
            idx = jnp.array([], dtype=jnp.int32)
        else:
            idx = jnp.arange(start, stop_incl + 1, dtype=jnp.int32)
        if subsampling is not None:
            if subsampling <= 0:
                raise ValueError("subsampling must be a positive integer.")
            if idx.size == 0:
                return idx
            idx = idx[idx % subsampling == 0]
        return idx

    # ---- low-level time gather (assumes in-bounds) ------------------------ #
    @staticmethod
    def _take_t(arr: Array, t: Array) -> Array:
        """Gather arr[t] on axis 0. Assumes 0 <= t < arr.shape[0]."""
        arr = jnp.asarray(arr)
        return lax.dynamic_index_in_dim(arr, t, axis=0, keepdims=False)

    # ---- single-t stream access ------------------------------------------ #
    def _stream_single(self, name: str, t: Array) -> Array:
        X = self._X3d()
        if name == "mask":
            return self._take_t(self._M2d(), t)

        if name in ("X", "X_minus", "X_plus", "X_minusminus", "X_plusplus", "X_m3", "X_m4", "X_p3", "X_p4"):
            offsets = {
                "X": 0,
                "X_minus": -1,
                "X_plus": +1,
                "X_minusminus": -2,
                "X_plusplus": +2,
                "X_m3": -3,
                "X_m4": -4,
                "X_p3": +3,
                "X_p4": +4,
            }
            return self._take_t(X, t + offsets[name])

        if name in ("dX_minus", "dX", "dX_plus"):
            if name == "dX_minus":
                return self._take_t(X, t) - self._take_t(X, t - 1)
            if name == "dX":
                return self._take_t(X, t + 1) - self._take_t(X, t)
            if name == "dX_plus":
                return self._take_t(X, t + 2) - self._take_t(X, t + 1)

        w = _parse_window_key(name)
        if w is not None:
            left = (w - 1) // 2
            offsets = jnp.arange(w) - left  # (W,)
            return X[t + offsets]  # (W, N, d)

        raise KeyError(f"Unknown stream '{name}'")

    # ---- output mask given requirements ---------------------------------- #
    def _output_mask_single(self, required: Set[str], t: Array) -> Array:
        """Per-particle validity for this row, based on requested streams.

        At the central index ``t``, uses the *dynamic mask* (increment
        reliable) so that boundary particles are excluded from fitting even
        when their positions are known.  At all neighbor offsets, uses the
        *static mask* (position known) so that force-evaluation neighbours
        are only required to be visible, not dynamically reliable.
        """
        M = self._M2d()           # static mask: position known
        D = self._dynamic_M2d()   # dynamic mask: increment reliable (⊆ M)
        m = self._take_t(D, t)  # (N,) — central index uses dynamic mask
        for k, (a, b) in STREAM_OFFSETS.items():
            if k in required:
                if a < 0:
                    m = jnp.logical_and(m, self._take_t(M, t + a))
                if b > 0:
                    m = jnp.logical_and(m, self._take_t(M, t + b))
        for k in required:
            w = _parse_window_key(k)
            if w is not None:
                left = (w - 1) // 2
                for off in range(-left, -left + w):
                    if off != 0:
                        m = jnp.logical_and(m, self._take_t(M, t + off))
        return m

    # ---- dt fields for this row ------------------------------------------ #
    def _dt_fields_single(
        self,
        required: Set[str],
        t: Array,
        *,
        force_dt_keys: Optional[Set[str]] = None,
    ) -> Dict[str, Array]:
        """Compute required dt fields for this row (assumes in-bounds).

        Note: requesting ``"X"`` or ``"mask"`` (in addition to increment
        streams) also triggers ``dt`` computation because the integration
        runtime always needs the time step for exposure-time weighting even
        when only positions are requested.
        """
        force_dt_keys = force_dt_keys or set()

        need_dt_minus = any(
            k in required for k in ("dX_minus", "X_minus", "X_minusminus", "dt_minus")
        ) or ("dt_minus" in force_dt_keys)
        need_dt_plus = any(
            k in required for k in ("dX_plus", "X_plus", "X_plusplus", "dt_plus")
        ) or ("dt_plus" in force_dt_keys)
        need_dt = ("dt" in force_dt_keys) or any(k in required for k in ("dX", "X", "mask", "dt"))

        out: Dict[str, Array] = {}

        if self.t is not None:
            tv = jnp.asarray(self.t)
            if need_dt:
                out["dt"] = self._take_t(tv, t + 1) - self._take_t(tv, t)
            if need_dt_minus:
                out["dt_minus"] = self._take_t(tv, t) - self._take_t(tv, t - 1)
            if need_dt_plus:
                out["dt_plus"] = self._take_t(tv, t + 2) - self._take_t(tv, t + 1)
            return out

        if self.dt is None:
            raise ValueError("Both 't' and 'dt' are None; one must be provided.")

        dta = jnp.asarray(self.dt)
        if dta.ndim == 0:
            if need_dt:
                out["dt"] = dta
            if need_dt_minus:
                out["dt_minus"] = dta
            if need_dt_plus:
                out["dt_plus"] = dta
            return out

        if need_dt:
            out["dt"] = self._take_t(dta, t)
        if need_dt_minus:
            out["dt_minus"] = self._take_t(dta, t - 1)
        if need_dt_plus:
            out["dt_plus"] = self._take_t(dta, t + 1)
        return out

    # ---- extras at single t ---------------------------------------------- #
    def build_extras(
        self, t_idx: Array, *, dataset_index: int = 0, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full model-facing extras at ``t_idx``: user values plus reserved keys.

        User extras are sliced at the frame(s) — a
        :class:`~SFI.trajectory.dataset.TimeSeriesExtra` is indexed, a callable
        invoked, anything else forwarded — and the reserved ``time`` /
        ``duration`` / ``dataset_index`` / ``particle_index`` are resolved for
        this dataset. This single mapping is what every consumer (inference,
        simulation, diagnostics) feeds the force/diffusion expression.
        ``extras_local`` overrides ``extras_global`` on key conflicts.
        """
        try:
            t_vec = jnp.asarray(self.materialize_time(as_numpy=False))
        except ValueError:
            t_vec = jnp.arange(self.T, dtype=float)
        ctx = ExtrasContext(
            n_particles=int(self.N),
            dataset_index=int(dataset_index),
            frame_times=t_vec[t_idx],
            duration=(t_vec[-1] - t_vec[0]) if self.T > 1 else jnp.asarray(1.0, dtype=float),
        )
        user = slice_frame_extras(self.extras_global, self.extras_local, frame_idx=t_idx, context=context)
        extras = resolve_extras(user, ctx)
        if context is not None:
            extras["context"] = context
        return extras

    # ---- train/test splitting -------------------------------------------- #
    def split_time(
        self, fraction: float = 0.8, *, gap: int = 0
    ) -> Tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split into ``(train, test)`` datasets along the time axis.

        A side feature for data-abundant scenarios: SFI estimates its own
        accuracy from the training data (``force_predicted_MSE``) and
        validates fits through the diagnostics suite, neither of which
        costs any data.  Hold out a test fraction only when data is
        plentiful, or to confirm a suspected bias floor.

        Parameters
        ----------
        fraction : float
            Fraction of frames assigned to the train half: train is
            ``[0, round(fraction*T))``, test is the remainder (after the
            optional ``gap``).
        gap : int
            Number of frames dropped between the halves.  ``0`` is safe
            for increment-based estimators (the boundary increment
            belongs to neither half by construction); use a few
            correlation times for slowly mixing systems.
        """
        if not (0.0 < fraction < 1.0):
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        T = int(np.asarray(self.X).shape[0])
        k = int(round(fraction * T))
        start_test = k + gap
        if k < 2 or T - start_test < 2:
            raise ValueError(
                f"split_time(fraction={fraction}, gap={gap}) leaves "
                f"train={k} and test={T - start_test} frames (T={T}); "
                "need at least 2 frames on each side."
            )
        prov = {"fraction": fraction, "gap": gap}
        return (
            self._slice_frames(0, k, role="train", **prov),
            self._slice_frames(start_test, T, role="test", **prov),
        )

    def _slice_frames(self, a: int, b: int, *, role: str, **prov) -> "TrajectoryDataset":
        """Return a copy restricted to frames ``[a, b)`` (absolute time kept)."""
        from SFI.statefunc.nodes.interactions.prepare import purge_cache_extras

        T = int(np.asarray(self.X).shape[0])

        X = jnp.asarray(self.X)[a:b]
        mask = None if self.mask is None else jnp.asarray(self.mask)[a:b]
        dynamic_mask = None if self.dynamic_mask is None else jnp.asarray(self.dynamic_mask)[a:b]

        t_arg = None if self.t is None else np.asarray(self.t)[a:b]
        dt_arg = None
        if self.t is None and self.dt is not None:
            dta = np.asarray(self.dt)
            dt_arg = float(dta) if dta.ndim == 0 else dta[a:b]

        extras_g = purge_cache_extras(_slice_time_extras(self.extras_global, a, b, T))
        extras_l = purge_cache_extras(_slice_time_extras(self.extras_local, a, b, T))

        meta = dict(self.meta or {})
        meta["split_time"] = {"role": role, "start": int(a), "stop": int(b), **prov}

        return TrajectoryDataset.from_arrays(
            X=X,
            t=t_arg,
            dt=dt_arg,
            mask=mask,
            dynamic_mask=dynamic_mask,
            extras_global=extras_g,
            extras_local=extras_l,
            meta=meta,
        )

    # ---- main API: single-t row producer --------------------------------- #
    def make_producer(
        self,
        require: Set[str],
        *,
        include_mask: bool = True,
        include_dt: bool = True,
        context: Optional[str] = None,
        force_dt_keys: Optional[Set[str]] = None,
        dataset_index: int = 0,
    ) -> Callable[[Array], Dict[str, Any]]:
        """Return a JAX-traceable function that builds a single-t row.

        Parameters
        ----------
        require :
            Set of stream names and the special key ``"extras"`` if extras are
            needed by downstream expressions.
        include_mask :
            If True, include ``"mask_out"`` computed from ``require``.
        include_dt :
            If True, include ``"dt"`` and neighbors when needed.
        context :
            Optional string to pass through to extras callables.
        force_dt_keys :
            Extra dt fields to force, e.g. ``{"dt_plus"}``.
        dataset_index :
            Position of this dataset within its collection; resolves the
            reserved ``dataset_index`` extra on every row.

        Returns
        -------
        producer : Callable[[Array], Dict[str, Any]]
            A function such that ``producer(t)`` returns a dict whose leaves are
            single-row arrays. Structure is fixed across calls.

        Notes
        -----
        - Use :meth:`valid_indices` to generate in-bounds indices. The producer
          assumes its input is valid and does not clamp.
        """
        # exclude pseudo-keys like "__dt__" from actual stream slicing
        required_streams = {
            n
            for n in require
            if (n in STREAM_OFFSETS and not n.startswith("__")) or n == "mask" or _parse_window_key(n) is not None
        }
        need_extras = "extras" in require
        force_dt_keys = set(force_dt_keys or ())

        def producer(t: Array) -> Dict[str, Any]:
            row: Dict[str, Any] = {}

            # streams
            for name in required_streams:
                row[name] = self._stream_single(name, t)

            # mask_out
            if include_mask:
                row["mask_out"] = self._output_mask_single(require, t)

            # dt fields
            if include_dt:
                dtf = self._dt_fields_single(require, t, force_dt_keys=force_dt_keys)
                row.update(dtf)

            # scalar counts
            row["N_total"] = jnp.array(float(self.N), dtype=jnp.float32)
            row["N_active"] = (
                jnp.sum(row["mask_out"].astype(jnp.float32))
                if "mask_out" in row
                else jnp.array(float(self.N), dtype=jnp.float32)
            )

            # extras
            if need_extras:
                row["extras"] = self.build_extras(t, dataset_index=dataset_index, context=context)

            return row

        return producer

    # ---- batch-t producer (vectorised gather) ----------------------------- #

    def _stream_batch(self, name: str, t_block: Array) -> Array:
        """Gather a stream for a block of time indices.

        Parameters
        ----------
        t_block : shape ``(K,)``
            Vector of valid time indices.

        Returns
        -------
        Array with shape ``(K, N, d)`` for position/increment streams,
        or ``(K, N)`` for mask.
        """
        X = self._X3d()
        if name == "mask":
            M = self._M2d()
            return M[t_block]  # (K, N)

        offsets = {
            "X": 0,
            "X_minus": -1,
            "X_plus": +1,
            "X_minusminus": -2,
            "X_plusplus": +2,
            "X_m3": -3,
            "X_m4": -4,
            "X_p3": +3,
            "X_p4": +4,
        }
        if name in offsets:
            return X[t_block + offsets[name]]  # (K, N, d)

        if name == "dX_minus":
            return X[t_block] - X[t_block - 1]
        if name == "dX":
            return X[t_block + 1] - X[t_block]
        if name == "dX_plus":
            return X[t_block + 2] - X[t_block + 1]

        w = _parse_window_key(name)
        if w is not None:
            left = (w - 1) // 2
            win_offsets = jnp.arange(w) - left  # (W,)
            idx = t_block[:, None] + win_offsets[None, :]  # (K, W)
            return X[idx]  # (K, W, N, d)

        raise KeyError(f"Unknown stream '{name}'")

    def _output_mask_batch(self, required: Set[str], t_block: Array) -> Array:
        """Per-particle validity for a block of rows.

        Same semantics as :meth:`_output_mask_single`: uses ``dynamic_mask``
        at the central indices and ``mask`` at neighbour offsets.

        Returns
        -------
        Array of shape ``(K, N)``, dtype bool.
        """
        M = self._M2d()           # static mask: position known
        D = self._dynamic_M2d()   # dynamic mask: increment reliable
        m = D[t_block]  # (K, N) — central indices use dynamic mask
        for k, (a, b) in STREAM_OFFSETS.items():
            if k in required:
                if a < 0:
                    m = jnp.logical_and(m, M[t_block + a])
                if b > 0:
                    m = jnp.logical_and(m, M[t_block + b])
        for k in required:
            w = _parse_window_key(k)
            if w is not None:
                left = (w - 1) // 2
                for off in range(-left, -left + w):
                    if off != 0:
                        m = jnp.logical_and(m, M[t_block + off])
        return m

    def _dt_fields_batch(
        self,
        required: Set[str],
        t_block: Array,
        *,
        force_dt_keys: Optional[Set[str]] = None,
    ) -> Dict[str, Array]:
        """Compute dt fields for a block of time indices.

        Returns a dict mapping, e.g., ``"dt"`` to an array of shape ``(K,)``.
        """
        force_dt_keys = force_dt_keys or set()

        need_dt_minus = any(
            k in required for k in ("dX_minus", "X_minus", "X_minusminus", "dt_minus")
        ) or ("dt_minus" in force_dt_keys)
        need_dt_plus = any(
            k in required for k in ("dX_plus", "X_plus", "X_plusplus", "dt_plus")
        ) or ("dt_plus" in force_dt_keys)
        need_dt = ("dt" in force_dt_keys) or any(k in required for k in ("dX", "X", "mask", "dt"))

        out: Dict[str, Array] = {}

        if self.t is not None:
            tv = jnp.asarray(self.t)
            if need_dt:
                out["dt"] = tv[t_block + 1] - tv[t_block]
            if need_dt_minus:
                out["dt_minus"] = tv[t_block] - tv[t_block - 1]
            if need_dt_plus:
                out["dt_plus"] = tv[t_block + 2] - tv[t_block + 1]
            return out

        if self.dt is None:
            raise ValueError("Both 't' and 'dt' are None; one must be provided.")

        dta = jnp.asarray(self.dt)
        if dta.ndim == 0:
            # Scalar dt → broadcast to (K,)
            K = t_block.shape[0]
            if need_dt:
                out["dt"] = jnp.broadcast_to(dta, (K,))
            if need_dt_minus:
                out["dt_minus"] = jnp.broadcast_to(dta, (K,))
            if need_dt_plus:
                out["dt_plus"] = jnp.broadcast_to(dta, (K,))
            return out

        # Per-step dt array
        if need_dt:
            out["dt"] = dta[t_block]
        if need_dt_minus:
            out["dt_minus"] = dta[t_block - 1]
        if need_dt_plus:
            out["dt_plus"] = dta[t_block + 1]
        return out

    def make_batch_producer(
        self,
        require: Set[str],
        *,
        include_mask: bool = True,
        include_dt: bool = True,
        context: Optional[str] = None,
        force_dt_keys: Optional[Set[str]] = None,
        dataset_index: int = 0,
    ) -> Callable[[Array], Dict[str, Any]]:
        """Return a function that gathers a batch of rows in one vectorised pass.

        This is the batch counterpart of :meth:`make_producer`. Instead of
        building one row at a time (designed for use inside ``jax.vmap``), this
        function gathers *K* rows at once using array indexing, producing
        arrays with a leading ``K`` axis.

        Parameters
        ----------
        require, include_mask, include_dt, context, force_dt_keys
            Same meaning as in :meth:`make_producer`.

        Returns
        -------
        batch_producer : ``Callable[[Array], Dict[str, Any]]``
            ``batch_producer(t_block)`` with ``t_block`` of shape ``(K,)``
            returns a dict whose arrays have a leading ``K`` axis.

        Notes
        -----
        **Extras limitation**: time-varying extras (``TimeSeriesExtra`` and
        callables) are evaluated at ``t_block[0]`` only, not at each index in
        the block.  This batch producer is designed for use cases where extras
        are global constants across the chunk (e.g. static boundary tensors).
        For per-step time-varying extras, use :meth:`make_producer` with
        ``jax.vmap`` instead.
        """
        required_streams = {
            n
            for n in require
            if (n in STREAM_OFFSETS and not n.startswith("__")) or n == "mask" or _parse_window_key(n) is not None
        }
        need_extras = "extras" in require
        force_dt_keys = set(force_dt_keys or ())

        def batch_producer(t_block: Array) -> Dict[str, Any]:
            row: Dict[str, Any] = {}
            K = t_block.shape[0]

            for name in required_streams:
                row[name] = self._stream_batch(name, t_block)

            if include_mask:
                row["mask_out"] = self._output_mask_batch(require, t_block)

            if include_dt:
                dtf = self._dt_fields_batch(
                    require,
                    t_block,
                    force_dt_keys=force_dt_keys,
                )
                row.update(dtf)

            row["N_total"] = jnp.array(float(self.N), dtype=jnp.float32)
            row["N_active"] = (
                jnp.sum(row["mask_out"].astype(jnp.float32), axis=1)  # (K,)
                if "mask_out" in row
                else jnp.full((K,), float(self.N), dtype=jnp.float32)
            )

            if need_extras:
                # Extras are batch-constant — resolve them at the first valid index.
                row["extras"] = self.build_extras(t_block[0], dataset_index=dataset_index, context=context)

            return row

        return batch_producer

    # ---- convenience: dense views for plotting / inspection --------------- #
    def materialize_time(self, *, as_numpy: bool = True) -> Array | np.ndarray:
        """
        Return a dense absolute time vector ``t`` of shape ``(T,)``.

        Rules
        -----
        - If ``self.t`` is not None, it is returned as-is.
        - Else, if ``self.dt`` is a scalar, use ``t[k] = k * dt``.
        - Else, if ``self.dt`` is an array of shape ``(T,)``, interpret
          ``dt[k]`` as the step between ``X[k]`` and ``X[k+1]`` and build

              t[0]   = 0
              t[k+1] = t[k] + dt[k]   for k = 0..T-2

          The last entry ``dt[T-1]`` (if any) is ignored.
        - If both ``t`` and ``dt`` are None, a ValueError is raised.

        Parameters
        ----------
        as_numpy :
            If True, return a NumPy array; otherwise a JAX array.

        Returns
        -------
        t : array, shape (T,)
        """
        T = self.T

        if self.t is not None:
            t = jnp.asarray(self.t)
        else:
            if self.dt is None:
                raise ValueError("Cannot materialize time: both 't' and 'dt' are None.")
            dta = jnp.asarray(self.dt)
            if dta.ndim == 0:
                t = jnp.arange(T, dtype=float) * dta
            elif dta.ndim == 1:
                if T == 0:
                    t = jnp.zeros((0,), dtype=float)
                else:
                    # t[0] = 0; t[1:] = cumsum(dt[:-1])
                    t0 = jnp.zeros((1,), dtype=float)
                    rest = jnp.cumsum(dta[:-1])
                    t = jnp.concatenate([t0, rest], axis=0)
            else:
                raise ValueError("dt must be scalar or (T,) to build a time axis.")

        return np.asarray(t) if as_numpy else t

    def to_arrays(
        self,
        *,
        as_numpy: bool = True,
        include_mask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Array, Array, Array | None]:
        """
        Materialize dense trajectory arrays for this dataset.

        This is intended for plotting and quick inspection, not for JAX
        integration (use :meth:`make_producer` for that).

        Parameters
        ----------
        as_numpy :
            If True (default), return NumPy arrays.
        include_mask :
            If True (default), return the per-particle validity mask.

        Returns
        -------
        t :
            Absolute time vector of shape ``(T,)``; see :meth:`materialize_time`.
        X :
            State tensor of shape ``(T, N, d)``.
        mask :
            Boolean mask of shape ``(T, N)`` if ``include_mask`` is True,
            otherwise ``None``.
        """
        X3 = self._X3d()
        M2 = self._M2d() if include_mask else None
        t = self.materialize_time(as_numpy=False)

        if as_numpy:
            import numpy as _np

            t = _np.asarray(t)
            X3 = _np.asarray(X3)
            if M2 is not None:
                M2 = _np.asarray(M2)

        return (t, X3, M2)
