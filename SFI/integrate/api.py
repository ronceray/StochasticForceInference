"""
Integration runtime: vmapped over time indices with dataset-owned producers.

Contract
--------
- `collection.peek_row(require=...)` returns a single-row sample mapping for
  memory sizing.
- `collection.iter_slices(require=..., bytes_hint=..., chunk_target_bytes=..., subsampling=..., context=...)`
  yields dictionaries with:

    - "producer": Callable[[t], row]  — JAX-traceable single-t builder,
    - "t_idx": jax.Array[int32]       — indices for this chunk,
    - "weight": float                 — dataset-level weight in [0,1],
    - "dataset_index": int            — for bookkeeping.
- `program` implements:
    - `require() -> set[str]` of streams (plus "extras" if needed),
    - `estimate_bytes_per_sample(sample_row) -> Optional[int]`,
    - `__call__(**streams)` for one time slice; for the parametric route
      it additionally supports a keyword-only argument `params`.

This module provides:

- `integrate(...)`: one-off integration using an `Integrand` `program`
  (backwards compatible front-end).
- `make_parametric_integrator(...)`: build a reusable, jittable integrator
  for a parameterised `Integrand`, with a clear separation between host-side
  planning and JAX-side runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

Row = Mapping[str, Any]
Producer = Callable[[int], Row]
BatchProducer = Callable[[jnp.ndarray], Row]
RowEval = Callable[[Row, Any], jnp.ndarray]


@dataclass(frozen=True)
class ChunkSpec:
    """One time-chunk (possibly padded) for a given dataset."""

    dataset_index: int
    weight: float
    t_block: jnp.ndarray  # shape (K_chunk,)
    valid_block: jnp.ndarray  # shape (K_chunk,), bool


@dataclass(frozen=True)
class IntegrationPlan:
    """
    Host-side integration plan.

    Contains:
      - producers: per-dataset single-t row builders,
      - batch_producers: per-dataset batch-t row builders,
      - chunks: padded time blocks with validity masks and weights,
      - reduction semantics and memory hints.
    """

    producers: Dict[int, Producer]
    batch_producers: Dict[int, BatchProducer]
    chunks: Tuple[ChunkSpec, ...]
    reduce: str
    reduce_over_particles: bool
    weight_by_dt: bool
    bytes_hint: Optional[int]
    K_fixed: Optional[int]
    context: Optional[str]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _build_plan(
    collection,
    program,
    *,
    reduce: str,
    reduce_over_particles: bool,
    weight_by_dt: bool = True,
    subsampling: int,
    chunk_target_bytes: int,
    context: Optional[str],
    bytes_per_sample: Optional[int] = None,
) -> IntegrationPlan:
    """
    Plan an integration once on the host side.

    Computes:
      - required streams from `program.require()`,
      - one real sample to estimate bytes per sample,
      - a fixed chunk size K for stable vmapped kernels,
      - the list of ChunkSpec and per-dataset producers.
    """
    # Required streams; include sentinel for dt window if datasets support it.
    require = set(program.require())
    require.add("__dt__")  # dataset.valid_indices should treat as offsets (0,+1)

    # Size hint from a real row
    try:
        sample_row = collection.peek_row(require=require, context=context)
    except ValueError:
        # ``peek_row`` raises ValueError both when no dataset has a usable time
        # window (the legitimate empty-plan case) and when materialising a row
        # genuinely fails (e.g. a malformed extra). Only the former should be
        # swallowed into an empty plan — masking the latter turns an informative
        # error into a cryptic downstream crash (a scalar Gram fed to swapaxes).
        if any(ds.valid_indices(require).size > 0 for ds in collection.datasets):
            raise
        # No dataset has valid rows for these requirements
        return IntegrationPlan(
            producers={},
            batch_producers={},
            chunks=tuple(),
            reduce=reduce,
            reduce_over_particles=reduce_over_particles,
            weight_by_dt=weight_by_dt,
            bytes_hint=None,
            K_fixed=None,
            context=context,
        )

    if bytes_per_sample is not None:
        bytes_hint = int(bytes_per_sample)
    else:
        estimator = getattr(program, "estimate_bytes_per_sample", None)
        if callable(estimator):
            bytes_hint = estimator(sample_row)
        else:
            bytes_hint = None

    # Derive fixed K per chunk from hint
    if not bytes_hint or bytes_hint <= 0:
        K_fixed: Optional[int] = None
    else:
        # Conservative: at least one row
        K_fixed = max(1, int(chunk_target_bytes // int(bytes_hint)))

    producers: Dict[int, Producer] = {}
    batch_producers: Dict[int, BatchProducer] = {}
    chunks: list[ChunkSpec] = []

    # Collect all payloads first so we can cap K_fixed at the actual data size.
    # Without this cap, tiny datasets get absurdly padded (e.g. 19 valid indices
    # padded to 6.4M when bytes_hint is small and chunk_target_bytes is large).
    payloads = list(
        collection.iter_slices(
            require=require,
            bytes_hint=bytes_hint,
            chunk_target_bytes=chunk_target_bytes,
            subsampling=subsampling,
            context=context,
        )
    )

    if K_fixed is not None and payloads:
        max_payload = max(int(p["t_idx"].shape[0]) for p in payloads)
        K_fixed = min(K_fixed, max_payload)

    for payload in payloads:
        ds_idx: int = int(payload["dataset_index"])
        producer = payload["producer"]
        t_idx = payload["t_idx"]
        weight = float(payload.get("weight", 1.0))

        if ds_idx not in producers:
            producers[ds_idx] = producer
            # Build the matching batch producer from the underlying dataset
            ds = collection.datasets[ds_idx]
            batch_producers[ds_idx] = ds.make_batch_producer(
                require,
                include_mask=True,
                include_dt=True,
                context=context,
                force_dt_keys={"dt"},
                dataset_index=collection.dataset_index(ds_idx),
            )

        if K_fixed is None:
            # No padding, one chunk per payload
            t_block = t_idx
            valid_block = jnp.ones_like(t_idx, dtype=bool)
            chunks.append(
                ChunkSpec(
                    dataset_index=ds_idx,
                    weight=weight,
                    t_block=t_block,
                    valid_block=valid_block,
                )
            )
        else:
            # Split t_idx into blocks of size K_fixed and pad the last one
            K_total = int(t_idx.shape[0])
            for start in range(0, K_total, K_fixed):
                stop = min(start + K_fixed, K_total)
                cur = t_idx[start:stop]
                K = int(cur.shape[0])
                pad = K_fixed - K

                if pad > 0:
                    pad_idx = jnp.pad(cur, (0, pad), mode="edge")
                    valid = jnp.concatenate([jnp.ones((K,), dtype=bool), jnp.zeros((pad,), dtype=bool)])
                else:
                    pad_idx = cur
                    valid = jnp.ones((K_fixed,), dtype=bool)

                chunks.append(
                    ChunkSpec(
                        dataset_index=ds_idx,
                        weight=weight,
                        t_block=pad_idx,
                        valid_block=valid,
                    )
                )

    return IntegrationPlan(
        producers=producers,
        batch_producers=batch_producers,
        chunks=tuple(chunks),
        reduce=reduce,
        reduce_over_particles=reduce_over_particles,
        weight_by_dt=weight_by_dt,
        bytes_hint=bytes_hint,
        K_fixed=K_fixed,
        context=context,
    )


# ---------------------------------------------------------------------------
# Core row kernel and runner (shared)
# ---------------------------------------------------------------------------


def _row_kernel(
    row_eval: RowEval,
    row: Row,
    theta: Any,
    *,
    reduce_over_particles: bool,
    weight_by_dt: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single-row kernel.

    ``row_eval(row, theta) -> y``, same semantics as
    ``program(**row)`` in the non-parametric case.
    Returns ``(weighted_value, dt_eff)`` where ``weighted_value``
    already includes dt when *weight_by_dt* is True.
    """
    y = row_eval(row, theta)

    # Optional particle reduction
    if reduce_over_particles:
        if y.ndim == 0:
            raise ValueError("reduce_over_particles=True but row_eval returned a scalar.")
        m = row.get("mask_out", None)
        if m is not None:
            if y.ndim == 0 or y.shape[0] != m.shape[0]:
                raise ValueError(
                    "mask_out mismatch: row_eval must return an array with "
                    f"leading particle axis of size {m.shape[0]} when mask_out is present."
                )
            mexp = m.reshape((m.shape[0],) + (1,) * (y.ndim - 1))
            y = jnp.where(mexp, y, 0.0)
        y = jnp.sum(y, axis=0)

    dt = row["dt"]
    if weight_by_dt:
        dteff = dt * row["N_active"]
        y_w = y * dt
    else:
        dteff = row["N_active"]
        y_w = y
    return y_w, dteff


def _run_plan_core(
    plan: IntegrationPlan,
    row_eval: RowEval,
    theta: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Core runtime: integrates according to a pre-built IntegrationPlan and a given row_eval.

    Returns ``(acc, Teff_total)`` where:

    - ``acc`` is the sum over chunks of ``y_w``,
    - ``Teff_total`` is the sum over chunks of effective exposure.

    Suitable for jitting when ``row_eval`` and ``theta`` are JAX-traceable.
    """
    if not plan.chunks:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero

    reduce_over_particles = plan.reduce_over_particles
    weight_by_dt = plan.weight_by_dt
    producer_by_idx: Dict[int, Producer] = plan.producers

    # Per-dataset weights are applied in every reduction (sum and mean) so the
    # force Gram, diffusion average, and parametric Gram pool datasets the same
    # way.  Within-dataset weighting (per-dt vs per-point) is the caller's
    # ``weight_by_dt``; the per-dataset multiplier is orthogonal.
    use_weight = True  # per-dataset weights applied in all reductions (unit weights => no-op)

    acc = None
    Teff_total = jnp.asarray(0.0, dtype=jnp.float32)

    for chunk in plan.chunks:
        producer = producer_by_idx[chunk.dataset_index]
        t_block = chunk.t_block  # (K,)
        valid_block = chunk.valid_block  # (K,)
        weight = chunk.weight

        def row_masked(t, is_valid, theta_):
            row = producer(t)
            y_w, dteff = _row_kernel(
                row_eval,
                row,
                theta_,
                reduce_over_particles=reduce_over_particles,
                weight_by_dt=weight_by_dt,
            )
            assert y_w is not None
            maskf = is_valid.astype(y_w.dtype)
            if use_weight:
                return (
                    y_w * maskf * weight,
                    dteff * is_valid.astype(dteff.dtype) * weight,
                )
            else:
                return (
                    y_w * maskf,
                    dteff * is_valid.astype(dteff.dtype),
                )

        Ys, Dteffs = jax.vmap(row_masked, in_axes=(0, 0, None))(t_block, valid_block, theta)
        y_sum = jnp.sum(Ys, axis=0)
        dteff_sum = jnp.sum(Dteffs, axis=0)

        Teff_total = Teff_total + dteff_sum
        acc = y_sum if acc is None else (acc + y_sum)

    return acc, Teff_total


# ---------------------------------------------------------------------------
# Batched runner: batch gather + batch statefunc + vmapped einsum
# ---------------------------------------------------------------------------

BatchRowEval = Callable[[Row, Any], jnp.ndarray]


def _run_plan_batched(
    plan: IntegrationPlan,
    batch_row_eval: BatchRowEval,
    theta: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batched runtime: gathers K rows at once and evaluates in batch.

    Instead of vmapping over individual time indices (which nests vmaps
    inside the state-expression leaves), this path:

    1. Gathers K rows with a single batch producer (one XLA gather),
    2. Passes ``(K, N, d)`` tensors to state expressions, which handle
       the leading batch dimensions in a single fused vmap,
    3. vmaps the einsum contractions over the K axis.

    The result shapes are ``(K, ...)`` where ``...`` is the per-row result
    shape. Particle and time reductions happen outside the batch evaluator.

    Returns ``(acc, Teff_total)`` with the same semantics as
    :func:`_run_plan_core`.
    """
    if not plan.chunks:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero

    reduce_over_particles = plan.reduce_over_particles
    batch_producer_by_idx: Dict[int, BatchProducer] = plan.batch_producers

    use_weight = True  # per-dataset weights applied in all reductions (unit weights => no-op)

    acc = None
    Teff_total = jnp.asarray(0.0, dtype=jnp.float32)

    for chunk in plan.chunks:
        batch_producer = batch_producer_by_idx[chunk.dataset_index]
        t_block = chunk.t_block  # (K,)
        valid_block = chunk.valid_block  # (K,), bool
        weight = chunk.weight

        # 1) Batch gather — one XLA gather per stream
        batch_row = batch_producer(t_block)
        # {X: (K,N,d), dX: (K,N,d), dt: (K,), mask_out: (K,N), ...}

        # 2) Batch evaluate — statefunc sees (K,N,d), einsum vmapped over K
        y = batch_row_eval(batch_row, theta)
        # y shape: (K, N, ..., F) if particle axis is present,
        #          (K, ..., F) if the einsum already contracted particles.

        # 3) Particle reduction over axis 1
        if reduce_over_particles:
            if y.ndim < 2:
                raise ValueError(
                    "reduce_over_particles=True but batch_row_eval returned an array with fewer than 2 dimensions."
                )
            m = batch_row.get("mask_out", None)  # (K, N)
            if m is not None:
                # Expand mask to broadcast: (K, N, 1, ..., 1)
                n_trail = y.ndim - 2  # dims after particle axis
                mexp = m.reshape(m.shape + (1,) * n_trail)
                y = jnp.where(mexp, y, 0.0)
            y = jnp.sum(y, axis=1)  # (K, ..., F)

        # 4) dt weighting
        dt = batch_row["dt"]  # (K,)
        if plan.weight_by_dt:
            dteff = dt * batch_row["N_active"]  # (K,)
            dt_exp = dt.reshape((dt.shape[0],) + (1,) * (y.ndim - 1))
            y_w = y * dt_exp  # (K, ..., F)
        else:
            dteff = batch_row["N_active"]  # (K,)
            y_w = y

        # 5) Valid masking
        valid_f = valid_block.astype(y_w.dtype)
        valid_exp = valid_f.reshape((valid_f.shape[0],) + (1,) * (y_w.ndim - 1))
        y_w = y_w * valid_exp
        dteff = dteff * valid_block.astype(dteff.dtype)

        if use_weight:
            y_w = y_w * weight
            dteff = dteff * weight

        # 6) Sum over K
        y_sum = jnp.sum(y_w, axis=0)
        dteff_sum = jnp.sum(dteff, axis=0)

        Teff_total = Teff_total + dteff_sum
        acc = y_sum if acc is None else (acc + y_sum)

    return acc, Teff_total


# ---------------------------------------------------------------------------
# Public API: integrate (non-parametric)
# ---------------------------------------------------------------------------


def _has_time_varying_required_extras(collection, program) -> bool:
    """True when the program reads an extras key that is time-varying.

    The batched runtime gathers extras once per chunk
    (``make_batch_producer`` collects them at the chunk's first index),
    which is only correct for static extras.  Programs that read
    :class:`TimeSeriesExtra` values or plain time-generator callables must
    run on the per-``t`` core runtime, where ``build_extras(t)`` slices
    them per frame.
    """
    req = getattr(program, "required_extras", None)
    keys: tuple = tuple(req() or ()) if callable(req) else ()
    if not keys:
        return False
    # The reserved ``time`` extra (auto-injected per frame by
    # ``build_extras``) is inherently time-varying, so any program that
    # reads it — e.g. a :func:`~SFI.bases.time_fourier` dictionary — must
    # run on the per-``t`` core runtime even though it is not stored as a
    # TimeSeriesExtra on the dataset.
    if "time" in keys:
        return True
    from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra

    for ds in collection.datasets:
        for src in (ds.extras_global, ds.extras_local):
            for k in keys:
                v = (src or {}).get(k)
                if isinstance(v, TimeSeriesExtra) or (callable(v) and not isinstance(v, FunctionExtra)):
                    return True
    return False


def integrate(
    collection,
    program,
    *,
    reduce: str = "sum",  # {'sum','mean'}
    reduce_over_particles: bool = True,  # sum over leading i if present
    weight_by_dt: bool = True,
    subsampling: int = 1,
    chunk_target_bytes: int = 512 * 1024**2,
    context: Optional[str] = None,
    batch: bool = True,
) -> jnp.ndarray:
    """
    Integrate an instantaneous program over time and datasets.

    Parameters
    ----------
    collection
        TrajectoryCollection exposing producers and time-index chunks.
    program
        Integrand object with `require`, `estimate_bytes_per_sample`, and `__call__`.
    reduce : {'sum','mean'}
        Dataset-and-time reduction. `'mean'` divides by the accumulated
        effective exposure computed from the same `dt` used in the numerator.
    reduce_over_particles : bool
        If the program output has a leading particle axis, apply `mask_out`,
        then sum that axis before the time reduction.
    weight_by_dt : bool
        If True (default), multiply each program output by ``dt`` before
        accumulation.  Set to False for programs whose output should be
        summed without dt weighting (e.g. parametric Gram matrices).
    subsampling : int
        Keep indices with `t % subsampling == 0`.
    chunk_target_bytes : int
        Target working-set size for the vmapped kernel.
    context : str, optional
        Forwarded to dataset extras via producers.

    Returns
    -------
    jax.Array
        Reduced value with particle axis removed if requested. Shapes match the
        program’s output after optional particle reduction.
    """
    if reduce not in {"sum", "mean"}:
        raise ValueError("reduce must be 'sum' or 'mean'")

    # Time-varying extras must be sliced per frame: only the per-t core
    # runtime does that (the batch producer gathers extras once per chunk).
    if batch and _has_time_varying_required_extras(collection, program):
        batch = False

    plan = _build_plan(
        collection,
        program,
        reduce=reduce,
        reduce_over_particles=reduce_over_particles,
        weight_by_dt=weight_by_dt,
        subsampling=subsampling,
        chunk_target_bytes=chunk_target_bytes,
        context=context,
    )

    if batch:
        batch_row_eval = _build_batch_row_eval(program, context=context, parametric=False)
        acc, Teff_total = _run_plan_batched(plan, batch_row_eval, theta=None)
    else:
        # Original vmap-over-t path
        def row_eval(row: Row, _theta: Any) -> jnp.ndarray:
            return program(**row)

        acc, Teff_total = _run_plan_core(plan, row_eval, theta=None)

    if reduce == "sum":
        return acc

    # mean: check Teff_total on host for backwards-compatible error behaviour
    Teff_val = float(Teff_total)
    if Teff_val <= 0.0:
        raise ValueError("Mean reduction requested but total exposure is non-positive.")
    return acc / jnp.asarray(Teff_total, dtype=acc.dtype)


# ---------------------------------------------------------------------------
# Public API: parametric integrator
# ---------------------------------------------------------------------------


def make_parametric_integrator(
    collection,
    program,
    *,
    reduce: str = "sum",
    reduce_over_particles: bool = True,
    weight_by_dt: bool = True,
    subsampling: int = 1,
    chunk_target_bytes: int = 512 * 1024**2,
    context: Optional[str] = None,
    bytes_per_sample: Optional[int] = None,
    batch: bool = True,
) -> Tuple[IntegrationPlan, Callable[[Any], jnp.ndarray]]:
    """
    Build a reusable, jittable integrator for a parametric Integrand.

    Parameters
    ----------
    collection
        TrajectoryCollection exposing producers and time-index chunks.
    program
        Integrand object with `require`, `estimate_bytes_per_sample`, and
        a call signature ``program(**streams, params=theta)`` where `theta`
        is a PyTree of parameters.
    reduce, reduce_over_particles, weight_by_dt, subsampling, chunk_target_bytes, context
        Same meaning as in :func:`integrate`.
    bytes_per_sample : int, optional
        Optional override for the per-sample memory estimate. If None, the
        program's `estimate_bytes_per_sample` is used.
    batch : bool
        If True, use the batched integration path (see :func:`integrate`).

    Returns
    -------
    plan : IntegrationPlan
        Host-side plan describing the chunks and producers.
    run  : callable
        JAX-jitted function ``run(theta) -> value`` that evaluates the
        integration for a given set of parameters.
    """
    if reduce not in {"sum", "mean"}:
        raise ValueError("reduce must be 'sum' or 'mean'")

    # Same correctness rule as `integrate`: time-varying extras require the
    # per-t core runtime.
    if batch and _has_time_varying_required_extras(collection, program):
        batch = False

    plan = _build_plan(
        collection,
        program,
        reduce=reduce,
        reduce_over_particles=reduce_over_particles,
        weight_by_dt=weight_by_dt,
        subsampling=subsampling,
        chunk_target_bytes=chunk_target_bytes,
        context=context,
        bytes_per_sample=bytes_per_sample,
    )

    if not plan.chunks:
        # Empty-plan edge case: always return 0.0
        @jax.jit
        def run_empty(theta):
            del theta
            return jnp.asarray(0.0, dtype=jnp.float32)

        return plan, run_empty

    if batch:
        batch_row_eval = _build_batch_row_eval(program, context=context)

        def run(theta):
            acc, Teff_total = _run_plan_batched(plan, batch_row_eval, theta)
            if reduce == "sum":
                return acc
            Teff_safe = jnp.where(Teff_total > 0, Teff_total, jnp.ones_like(Teff_total))
            return acc / Teff_safe.astype(acc.dtype)
    else:
        # Original vmap-over-t path
        def row_eval(row: Row, theta: Any) -> jnp.ndarray:
            return program(params=theta, **row)

        def run(theta):
            acc, Teff_total = _run_plan_core(plan, row_eval, theta)
            if reduce == "sum":
                return acc
            Teff_safe = jnp.where(Teff_total > 0, Teff_total, jnp.ones_like(Teff_total))
            return acc / Teff_safe.astype(acc.dtype)

    return plan, run


# ---------------------------------------------------------------------------
# Mini-batch parametric integrator
# ---------------------------------------------------------------------------


def _build_batch_row_eval(program, context=None, *, parametric: bool = True):
    """Build a JIT-compiled batch row evaluator.

    Parameters
    ----------
    program
        Integrand or duck-typed program with ``require`` / ``__call__``.
    context : str, optional
        Forwarded to extras as a static constant.
    parametric : bool
        If True (default), forward ``theta`` as ``params=theta`` on every
        call.  Set False for non-parametric programs whose ``__call__``
        does not accept a ``params`` keyword.

    Returns
    -------
    batch_row_eval : callable
        ``batch_row_eval(row, theta) -> jnp.ndarray``.
    """
    _base_static: Dict[str, Any] = {}
    if context is not None:
        _base_static["context"] = context

    def _split_extras(row: Row):
        ext = row.get("extras")
        if ext is None or not isinstance(ext, dict):
            return row, {}, ()
        arr_ext: Dict[str, Any] = {}
        fn_ext: Dict[str, Any] = {}
        for k, v in ext.items():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                arr_ext[k] = v
            elif callable(v):
                fn_ext[k] = v
            elif isinstance(v, (str, type(None), bool, int, float)):
                fn_ext[k] = v
            else:
                arr_ext[k] = v
        if not fn_ext:
            return row, {}, ()
        row = dict(row)
        row["extras"] = arr_ext if arr_ext else {}
        cache_key = tuple(sorted((k, id(v)) for k, v in fn_ext.items()))
        return row, fn_ext, cache_key

    _jit_cache: Dict[tuple, Any] = {}

    def _make_jit_fn(fn_extras: Dict[str, Any]):
        static = {**_base_static, **fn_extras}
        if hasattr(program, "batch_call"):

            @jax.jit
            def jit_fn(row: Row, theta: Any) -> jnp.ndarray:
                if static:
                    ext = dict(row.get("extras", {}))
                    ext.update(static)
                    row = {**row, "extras": ext}
                if parametric:
                    return program.batch_call(params=theta, **row)
                return program.batch_call(**row)
        else:

            @jax.jit
            def jit_fn(row: Row, theta: Any) -> jnp.ndarray:
                if static:
                    ext = dict(row.get("extras", {}))
                    ext.update(static)
                    row = {**row, "extras": ext}
                batched = {k: v for k, v in row.items() if hasattr(v, "ndim") and v.ndim >= 1}
                scalars = {k: v for k, v in row.items() if not hasattr(v, "ndim") or v.ndim < 1}

                def _call(b):
                    if parametric:
                        return program(params=theta, **{**b, **scalars})
                    return program(**{**b, **scalars})

                return jax.vmap(_call)(batched)

        return jit_fn

    _default_jit = _make_jit_fn({})

    def batch_row_eval(row: Row, theta: Any) -> jnp.ndarray:
        stripped, fn_extras, cache_key = _split_extras(row)
        if not cache_key:
            return _default_jit(stripped, theta)
        if cache_key not in _jit_cache:
            _jit_cache[cache_key] = _make_jit_fn(fn_extras)
        return _jit_cache[cache_key](stripped, theta)

    return batch_row_eval


# [new code — parametric update] minibatch infrastructure below
def _build_minibatch_runner(
    plan: IntegrationPlan,
    program,
    *,
    batch_size: int,
    context: Optional[str] = None,
):
    """Build a stochastic mini-batch evaluator from an existing plan.

    Parameters
    ----------
    plan : IntegrationPlan
        A plan already built by ``_build_plan`` / ``make_parametric_integrator``.
    program : Integrand
        The same program used to build *plan*.
    batch_size : int
        Number of time indices to sample per evaluation.
    context : str, optional
        Forwarded to extras.

    Returns
    -------
    run_batch : callable
        ``run_batch(theta, rng_key) -> scalar``.  An unbiased estimator
        of the full-data loss (with ``reduce="sum"`` semantics).
    """
    if not plan.chunks:

        def run_batch_empty(theta, rng_key):
            del theta, rng_key
            return jnp.asarray(0.0, dtype=jnp.float32)

        return run_batch_empty

    reduce_over_particles = plan.reduce_over_particles
    batch_row_eval = _build_batch_row_eval(program, context=context)

    # Pool valid indices per dataset from the plan.
    ds_indices: Dict[int, jnp.ndarray] = {}
    for chunk in plan.chunks:
        ds_idx = chunk.dataset_index
        valid = chunk.t_block[chunk.valid_block]
        if ds_idx in ds_indices:
            ds_indices[ds_idx] = jnp.concatenate([ds_indices[ds_idx], valid])
        else:
            ds_indices[ds_idx] = valid

    # Pre-compute per-dataset batch sizes (proportional allocation).
    total_valid = sum(int(idx.shape[0]) for idx in ds_indices.values())
    ds_batch_info = []  # list of (ds_idx, all_idx, n_batch)
    for ds_idx, all_idx in ds_indices.items():
        n_ds = int(all_idx.shape[0])
        n_batch = max(1, min(n_ds, round(batch_size * n_ds / total_valid)))
        ds_batch_info.append((ds_idx, all_idx, n_ds, n_batch))

    def run_batch(theta, rng_key):
        acc = jnp.asarray(0.0, dtype=jnp.float32)

        for ds_idx, all_idx, n_ds, n_batch in ds_batch_info:
            rng_key, subkey = jax.random.split(rng_key)
            # Sample without replacement
            perm = jax.random.permutation(subkey, n_ds)[:n_batch]
            sampled = all_idx[perm]  # (n_batch,)

            batch_producer = plan.batch_producers[ds_idx]
            batch_row = batch_producer(sampled)

            y = batch_row_eval(batch_row, theta)
            # y shape: (n_batch, N, ...) or (n_batch, ...)

            if reduce_over_particles:
                if y.ndim < 2:
                    raise ValueError("reduce_over_particles=True but batch_row_eval returned < 2 dimensions.")
                m = batch_row.get("mask_out", None)
                if m is not None:
                    n_trail = y.ndim - 2
                    mexp = m.reshape(m.shape + (1,) * n_trail)
                    y = jnp.where(mexp, y, 0.0)
                y = jnp.sum(y, axis=1)

            # dt weighting
            if plan.weight_by_dt:
                dt = batch_row["dt"]
                dt_exp = dt.reshape((dt.shape[0],) + (1,) * (y.ndim - 1))
                y_w = y * dt_exp
            else:
                y_w = y

            # Sum over batch and scale to be unbiased
            y_sum = jnp.sum(y_w, axis=0)
            scale = jnp.asarray(n_ds / n_batch, dtype=y_sum.dtype)
            acc = acc + y_sum * scale

        return acc

    return run_batch


def make_minibatch_parametric_integrator(
    collection,
    program,
    *,
    batch_size: int,
    reduce: str = "sum",
    reduce_over_particles: bool = True,
    weight_by_dt: bool = True,
    subsampling: int = 1,
    chunk_target_bytes: int = 512 * 1024**2,
    context: Optional[str] = None,
    bytes_per_sample: Optional[int] = None,
    batch: bool = True,
) -> Tuple[IntegrationPlan, Callable, Callable]:
    """Build a parametric integrator with both full and mini-batch runners.

    Parameters
    ----------
    collection, program, reduce, reduce_over_particles, weight_by_dt, subsampling, chunk_target_bytes, context, bytes_per_sample, batch
        Same as :func:`make_parametric_integrator`.
    batch_size : int
        Number of time indices to sample per mini-batch evaluation.

    Returns
    -------
    plan : IntegrationPlan
    run_full : callable
        ``run_full(theta) -> scalar`` — full-data evaluator.
    run_batch : callable
        ``run_batch(theta, rng_key) -> scalar`` — stochastic mini-batch
        evaluator.  Unbiased estimator of the full-data value.
    """
    plan, run_full = make_parametric_integrator(
        collection,
        program,
        reduce=reduce,
        reduce_over_particles=reduce_over_particles,
        weight_by_dt=weight_by_dt,
        subsampling=subsampling,
        chunk_target_bytes=chunk_target_bytes,
        context=context,
        bytes_per_sample=bytes_per_sample,
        batch=batch,
    )
    run_batch = _build_minibatch_runner(
        plan,
        program,
        batch_size=batch_size,
        context=context,
    )
    return plan, run_full, run_batch
