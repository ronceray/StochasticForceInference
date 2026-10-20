# SFI/trajectory/degrade.py
"""
SFI.trajectory.degrade
======================

Degrade synthetic trajectories to mimic real data:
- motion blur (temporal window average)
- downsampling
- additive measurement noise
- ROI filtering (mask points outside a region)
- random data loss

Two front doors:
----------------
1) Dataset/Collection API (recommended for internal use)
   - degrade_dataset(ds, ...)
   - degrade_collection(coll, ...)

2) Columns API (back-compat for I/O scripts)
   - degrade_columns(meta, particle_idx, time_idx, state_vectors, ...)

Why two? Column flow is convenient for simple scripts and file round-trips;
dataset/collection flow keeps everything rectangular so we can blur/downsample
time-dependent extras cleanly without flatten/unflatten gymnastics.

Extras semantics
----------------
Time resampling follows the runtime contract of
:meth:`TrajectoryDataset.build_extras` (which slices only explicit
time-dependent extras and forwards everything else unchanged):

- :class:`~SFI.trajectory.dataset.TimeSeriesExtra` (global ``(T, ...)`` or
  local ``(T, N, ...)``): resampled at the kept frames, exactly like ``X``
  and ``t`` — floating data are averaged over the blur window (a plain slice
  when ``motion_blur=0``), integer/bool data are taken at the first frame of
  each window.  The leading axis must equal ``T``.
- plain callables ``f(t_idx, context=None)`` (time-dependent generators):
  re-indexed so that new frame ``k`` reads original frame ``keep_times[k]``.
- :class:`~SFI.trajectory.dataset.FunctionExtra` and other non-array
  objects: passed through unchanged.
- plain **floating** arrays whose leading axis is exactly ``T`` (global
  ``(T, ...)``, local ``(T, N, ...)``) are treated as sampled time series
  (legacy convention) and blurred/downsampled like ``X``, keeping their
  dtype.  Wrap a time series in :func:`~SFI.trajectory.time_series_extra`
  to make the intent explicit.
- every other plain array is **static** and passed through unchanged — in
  particular integer/bool arrays (CSR ``indptr``/``indices`` neighbour
  lists, species labels, index tables) are never resampled, whatever their
  length, and per-particle ``(N, ...)`` constants are kept as is.

Noise/ROI/data-loss are applied on the **mask** (not by deleting rows), so
tensor shapes remain intact. Flattening to columns (if needed) happens last.

Cache-only extras (auto-generated structural tables)
----------------------------------------------------
Keys starting with ``_cache/`` are considered auto-generated structural extras
(e.g. CSR neighbor lists, stencil hyper tables). They are **not** degraded and
are **dropped** from outputs, because any degradation/context change invalidates
such cached structural objects. They can be regenerated on demand by calling
the appropriate host-side preparation routine.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from SFI.statefunc.nodes.interactions.prepare import (
    CACHE_PREFIX,
    is_cache_key,
    purge_cache_extras,
)
from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset

# -------------------------- public: dataset/collection -------------------------- #


def degrade_dataset(
    ds: TrajectoryDataset,
    *,
    downsample: int = 1,
    motion_blur: int = 0,
    data_loss_fraction: float = 0.0,
    noise: Union[None, float, np.ndarray] = None,
    ROI: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]] = None,
    seed: Optional[int] = None,
) -> TrajectoryDataset:
    """
    Degrade a single :class:`TrajectoryDataset`.

    The function operates in tensor space; it returns a new dataset where:

    - ``X`` is motion-blurred over ``motion_blur + 1`` frames and downsampled
      by ``downsample``,
    - the mask is AND-reduced over the blur window, then modified by ROI and
      random data loss,
    - ``t`` (if present) is averaged over the blur window and downsampled,
      otherwise scalar ``dt`` is multiplied by ``downsample``,
    - extras are processed consistently (see module docstring).

    Parameters
    ----------
    ds
        Input dataset to degrade.
    downsample
        Integer downsampling factor along the time axis (must be ``>= 1``).
    motion_blur
        Temporal averaging window size minus one. The actual blur window is
        ``motion_blur + 1`` frames and must satisfy
        ``0 <= motion_blur < downsample``.
    data_loss_fraction
        Fraction of currently valid entries to drop uniformly at random
        after ROI filtering (in ``[0, 1)``).
    noise
        Additive Gaussian noise scale. If a float, isotropic noise with
        standard deviation ``noise`` is applied. If an array, broadcast to
        the state dimension.
    ROI
        Region-of-interest predicate or mask. Can be:

        - float: radial cutoff — keeps positions with ``‖x‖₂ ≤ ROI``,
        - ``(2, d)`` ndarray: axis-aligned box (row 0 = lower bound, row 1 = upper bound),
        - ``Callable[[np.ndarray], bool]``: predicate evaluated on each
          observed position.
    seed
        Optional RNG seed for the noise and data-loss generators.

    Returns
    -------
    TrajectoryDataset
        Degraded dataset with the same number of particles but fewer time
        steps.
    """
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if motion_blur < 0 or motion_blur >= downsample:
        raise ValueError("motion_blur must satisfy 0 <= motion_blur < downsample.")
    if not (0.0 <= data_loss_fraction < 1.0):
        raise ValueError("data_loss_fraction must be in [0, 1).")

    rng = np.random.default_rng(seed)

    # Shapes
    X = np.asarray(ds._X3d())  # (T, N, d)
    M = np.asarray(ds._M2d())  # (T, N)
    T, N, d = X.shape
    has_dynamic = ds.dynamic_mask is not None
    if has_dynamic:
        M_dyn = np.asarray(ds.dynamic_mask)
        if M_dyn.ndim == 1:
            M_dyn = M_dyn[:, None]
    else:
        M_dyn = None

    # Downsampling schedule
    window = motion_blur + 1
    keep_times = np.arange(0, max(T - motion_blur, 0), downsample, dtype=int)
    K = keep_times.size
    if K == 0:
        # Degenerate (e.g. very small T vs blur) → return an empty dataset with copied meta/extras
        return TrajectoryDataset.from_arrays(
            X=np.zeros((0, N, d), dtype=X.dtype),
            t=None if ds.t is None else np.zeros((0,), dtype=float),
            dt=None if ds.t is not None else ds.dt,
            mask=np.zeros((0, N), dtype=bool),
            extras_global=purge_cache_extras(_downsample_extras_global(ds.extras_global, keep_times, window, T)),
            extras_local=purge_cache_extras(_downsample_extras_local(ds.extras_local, keep_times, window, T)),
            meta=_update_meta(
                dict(ds.meta),
                downsample,
                motion_blur,
                data_loss_fraction,
                noise,
                ROI,
                seed,
            ),
        )

    # 1) Blur/Downsample X and mask along time
    X_ds = np.empty((K, N, d), dtype=X.dtype)
    M_ds = np.empty((K, N), dtype=bool)
    M_dyn_ds = np.empty((K, N), dtype=bool) if has_dynamic else None
    for k, t0 in enumerate(keep_times):
        segX = X[t0 : t0 + window]  # (window, N, d)
        segM = M[t0 : t0 + window]  # (window, N)
        with np.errstate(invalid="ignore"):
            X_ds[k] = np.nanmean(np.where(segM[..., None], segX, np.nan), axis=0)
        M_ds[k] = segM.all(axis=0)  # require all valid within window
        if has_dynamic and M_dyn_ds is not None and M_dyn is not None:
            M_dyn_ds[k] = M_dyn[t0 : t0 + window].all(axis=0)

    # 2) Add measurement noise to observed entries
    if noise is not None:
        X_ds = _add_noise_to_masked(X_ds, M_ds, rng, noise)

    # 3) ROI on observed positions at blurred times
    if ROI is not None:
        inside = _roi_predicate(ROI, d)
        # evaluate on every (k,n) that is currently valid
        flat = X_ds.reshape(K * N, d)
        M_flat = M_ds.reshape(K * N)
        sel = np.where(M_flat)[0]
        if sel.size > 0:
            inside_mask = np.array([inside(flat[i]) for i in sel], dtype=bool)
            M_flat[sel] = M_flat[sel] & inside_mask
            if has_dynamic:
                assert M_dyn_ds is not None
                M_dyn_flat = M_dyn_ds.reshape(K * N)
                M_dyn_flat[sel] = M_dyn_flat[sel] & inside_mask
                M_dyn_ds = M_dyn_flat.reshape(K, N)
        M_ds = M_flat.reshape(K, N)

    # 4) Random data loss on valid entries
    if data_loss_fraction > 0.0:
        M_flat = M_ds.reshape(-1)
        valid_idx = np.where(M_flat)[0]
        keep = int(round(valid_idx.size * (1.0 - data_loss_fraction)))
        if keep < valid_idx.size:
            drop = rng.choice(valid_idx, size=valid_idx.size - keep, replace=False)
            M_flat[drop] = False
            if has_dynamic:
                assert M_dyn_ds is not None
                M_dyn_flat = M_dyn_ds.reshape(-1)
                M_dyn_flat[drop] = False
                M_dyn_ds = M_dyn_flat.reshape(K, N)
        M_ds = M_flat.reshape(K, N)

    # 5) Downsample 't' or scale 'dt'
    if ds.t is not None:
        t = np.asarray(ds.t)
        t_out = np.array([np.mean(t[t0 : t0 + window]) for t0 in keep_times], dtype=float)
        dt_arg = None
        t_arg = t_out
    else:
        t_arg = None
        if ds.dt is None:
            dt_arg = None
        else:
            dta = np.asarray(ds.dt)
            if dta.ndim == 0:
                # Scalar dt: new step is downsample × original step.
                dt_arg = float(dta) * downsample
            else:
                # Variable dt array: sum over each downsampled window.
                dt_arg = np.array(
                    [float(np.sum(dta[t0 : t0 + downsample])) for t0 in keep_times],
                    dtype=float,
                )

    # 6) Downsample/blur extras
    extras_g = purge_cache_extras(_downsample_extras_global(ds.extras_global, keep_times, window, T))
    extras_l = purge_cache_extras(_downsample_extras_local(ds.extras_local, keep_times, window, T))

    # 7) Update meta with provenance
    meta2 = _update_meta(dict(ds.meta), downsample, motion_blur, data_loss_fraction, noise, ROI, seed)
    if ds.dt is not None and dt_arg is not None:
        meta2.setdefault("original_dt", np.asarray(ds.dt).tolist())
        meta2["dt"] = dt_arg if np.ndim(dt_arg) == 0 else np.asarray(dt_arg).tolist()

    return TrajectoryDataset.from_arrays(
        X=X_ds,
        t=t_arg,
        dt=dt_arg,
        mask=M_ds,
        dynamic_mask=M_dyn_ds,
        extras_global=extras_g,
        extras_local=extras_l,
        meta=meta2,
    )


def degrade_collection(
    coll: TrajectoryCollection,
    *,
    downsample: int = 1,
    motion_blur: int = 0,
    data_loss_fraction: float = 0.0,
    noise: Union[None, float, np.ndarray] = None,
    ROI: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]] = None,
    seed: Optional[int] = None,
    reweight: Literal["pool", "keep"] = "pool",
) -> TrajectoryCollection:
    """
    Degrade all datasets in a collection and optionally recompute weights.

    Parameters
    ----------
    coll
        Input collection to degrade.
    downsample, motion_blur, data_loss_fraction, noise, ROI, seed
        Same semantics as in :func:`degrade_dataset`.
    reweight
        Policy for updating collection-level weights after degradation:

        - ``"pool"``: recompute weights via ``with_weights("pool")``.
        - ``"keep"``: preserve the relative weights from ``coll.weights``.

    Returns
    -------
    TrajectoryCollection
        New collection whose datasets have been degraded in the same way.

    Notes
    -----
    This function is purely functional: the input collection is not modified.
    """
    ds2: List[TrajectoryDataset] = [
        degrade_dataset(
            ds,
            downsample=downsample,
            motion_blur=motion_blur,
            data_loss_fraction=data_loss_fraction,
            noise=noise,
            ROI=ROI,
            seed=seed,
        )
        for ds in coll.datasets
    ]
    out = TrajectoryCollection(ds2, coll.weights)
    if reweight == "pool":
        return out.with_weights("pool")
    if reweight == "keep":
        return out.with_weights(list(map(float, coll.weights)))
    raise ValueError(f"Unknown reweight policy: {reweight!r}")


# ------------------------------- helpers -------------------------------- #


def _add_noise_to_masked(
    X: np.ndarray,
    M: np.ndarray,
    rng: np.random.Generator,
    noise: Union[float, np.ndarray],
) -> np.ndarray:
    """Add Gaussian noise to X **only where M is True**."""
    Y = X.copy()
    T, N, d = Y.shape
    if np.isscalar(noise):
        # iid σ on each coordinate
        eps = rng.normal(scale=float(noise), size=Y.shape)
        Y[M] = Y[M] + eps[M]
        return Y
    noise_arr = np.asarray(noise, dtype=float)
    if noise_arr.ndim == 1:
        if noise_arr.shape[0] != d:
            raise ValueError("Noise vector length must match state dimension d.")
        eps = rng.normal(size=Y.shape) * noise_arr.reshape((1, 1, d))
        Y[M] = Y[M] + eps[M]
        return Y
    if noise_arr.ndim == 2:
        if noise_arr.shape != (d, d):
            raise ValueError("Noise matrix must be (d, d).")
        eps = rng.normal(size=Y.shape) @ noise_arr
        Y[M] = Y[M] + eps[M]
        return Y
    raise ValueError("noise must be scalar, (d,), or (d,d).")


def _roi_predicate(
    ROI: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]], d: int
) -> Callable:
    """Build an ROI predicate in R^d.

    ``float`` ROI is a radial cutoff: ``‖x‖₂ ≤ ROI``.
    ``(2, d)`` array ROI is an axis-aligned box: ``lo ≤ x ≤ hi`` element-wise.
    """
    if ROI is None:
        return lambda x: True
    if np.isscalar(ROI):
        r = float(ROI)
        return lambda x: bool(np.linalg.norm(x) <= r)
    if isinstance(ROI, np.ndarray) or hasattr(ROI, "__array__"):
        arr = np.asarray(ROI, dtype=float)
        if arr.shape == (2, d):
            lo, hi = arr[0].copy(), arr[1].copy()
            return lambda x: bool(np.all((lo <= x) & (x <= hi)))
        raise ValueError(f"ndarray ROI must have shape (2, {d}), got {arr.shape}.")
    if callable(ROI):
        return ROI  # trusted
    raise ValueError("ROI must be None, scalar float, (2,d) ndarray, or callable.")


def _resample_time_axis(arr: np.ndarray, keep_times: np.ndarray, window: int) -> np.ndarray:
    """Resample a ``(T, ...)`` array at the kept frames, preserving its dtype.

    Floating (and complex) data are NaN-aware averaged over the blur window
    ``[t0, t0 + window)`` — the rule applied to ``X`` and ``t``; with
    ``window == 1`` this is an exact slice.  Integer / bool data cannot be
    averaged and are taken at the first frame of each window.
    """
    K = keep_times.size
    if K == 0:
        return arr[:0]
    if window == 1 or not np.issubdtype(arr.dtype, np.inexact):
        return arr[keep_times]
    buf = np.empty((K,) + arr.shape[1:], dtype=arr.dtype)
    for i, t0 in enumerate(keep_times):
        with np.errstate(invalid="ignore"):
            buf[i] = np.nanmean(arr[t0 : t0 + window], axis=0)
    return buf


def _downsample_extras(
    extras: Dict[str, Any],
    keep_times: np.ndarray,
    window: int,
    T: int,
    *,
    min_ndim: int,
    where: str,
) -> Dict[str, Any]:
    """Shared time-resampling rule for ``extras_global`` / ``extras_local``.

    See the module docstring ("Extras semantics").  ``min_ndim`` is the
    minimum rank of a plain array to count as a sampled time series: 1 for
    globals ``(T, ...)``, 2 for locals ``(T, N, ...)``.
    """
    from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra

    keep_idx = np.asarray(keep_times, dtype=int)
    # downsample=1 without blur keeps every frame: time extras pass unchanged.
    identity = window == 1 and keep_idx.size == T and bool(np.all(keep_idx == np.arange(T)))
    out: Dict[str, Any] = {}
    for k, v in (extras or {}).items():
        if is_cache_key(k, prefix=CACHE_PREFIX):
            continue
        if identity:
            out[k] = v
            continue

        if isinstance(v, TimeSeriesExtra):
            data = np.asarray(v.data)
            if data.ndim == 0 or int(data.shape[0]) != T:
                raise ValueError(
                    f"TimeSeriesExtra {where}[{k!r}] has leading axis "
                    f"{None if data.ndim == 0 else int(data.shape[0])} != T={T}; "
                    "cannot resample it consistently with the trajectory."
                )
            out[k] = TimeSeriesExtra(jnp.asarray(_resample_time_axis(data, keep_idx, window)))
            continue

        if isinstance(v, FunctionExtra):
            out[k] = v
            continue

        if callable(v):
            # Time-dependent generator f(t_idx, context=None): new frame k is
            # original frame keep_times[k] (same contract as split_time's offset).
            if keep_idx.size == 0:
                out[k] = v
            else:
                kt = jnp.asarray(keep_idx)

                def _reindexed(t_idx, context=None, _f=v, _kt=kt):
                    return _f(_kt[t_idx], context=context)

                out[k] = _reindexed
            continue

        if isinstance(v, (np.ndarray, jax.Array, list, tuple)):
            try:
                arr = np.asarray(v)
            except ValueError:  # ragged sequence: not an array, keep as is
                arr = None
            if (
                arr is not None
                and arr.ndim >= min_ndim
                and arr.shape[0] == T
                and np.issubdtype(arr.dtype, np.inexact)
            ):
                out[k] = _resample_time_axis(arr, keep_idx, window)
                continue

        # Static payload (integer/bool tables such as CSR neighbour lists,
        # per-particle constants, scalars, metadata): never resampled.
        out[k] = v
    return out


def _downsample_extras_global(
    eg: Dict[str, Any], keep_times: np.ndarray, window: int, T: int
) -> Dict[str, Any]:
    """
    Resample extras_global entries along time (see "Extras semantics").

    Only explicit :class:`TimeSeriesExtra` values, plain time-indexed
    callables and plain **floating** arrays with leading axis exactly ``T``
    are resampled; integer/bool arrays (e.g. CSR neighbour lists) and all
    other static values are passed through unchanged, dtype included.

    Cache-only extras policy
    ------------------------
    Keys under ``_cache/`` are structural/derivable objects and are **dropped**
    during degradation. They must be regenerated later from the new context.
    """
    return _downsample_extras(eg, keep_times, window, T, min_ndim=1, where="extras_global")


def _downsample_extras_local(
    el: Dict[str, Any], keep_times: np.ndarray, window: int, T: int
) -> Dict[str, Any]:
    """
    Resample extras_local entries along time (see "Extras semantics").

    Same rule as :func:`_downsample_extras_global`, with plain floating
    arrays resampled only when shaped ``(T, N, ...)``; per-particle
    ``(N, ...)`` constants are kept.

    Cache-only extras policy
    ------------------------
    Keys under ``_cache/`` are structural/derivable objects and are **dropped**
    during degradation.
    """
    return _downsample_extras(el, keep_times, window, T, min_ndim=2, where="extras_local")


def _update_meta(
    meta: Dict[str, Any],
    downsample: int,
    motion_blur: int,
    data_loss_fraction: float,
    noise: Union[None, float, np.ndarray],
    ROI: Any,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Record degradation provenance in meta dict (shallow copy upstream)."""
    meta.update(
        {
            "degrade_downsample": downsample,
            "degrade_motion_blur": motion_blur,
            "degrade_data_loss_frac": data_loss_fraction,
            "degrade_noise_spec": (
                None if noise is None else float(noise) if np.isscalar(noise) else np.asarray(noise).tolist()
            ),
            "degrade_ROI_spec": (None if ROI is None else ("callable" if callable(ROI) else np.asarray(ROI).tolist())),
            "degrade_rng_seed": seed,
        }
    )
    return meta


# =============================================================================
# Spatial degradation for grid-based SPDE datasets
# =============================================================================


def degrade_spatial_data(
    coll: TrajectoryCollection,
    *,
    downscale: int | Tuple[int, ...] = 2,
    method: Literal["mean", "subsample"] = "mean",
    blur_radius: int = 0,
    data_loss_fraction: float = 0.0,
    noise: Union[None, float, np.ndarray] = None,
    seed: Optional[int] = None,
    mask_threshold: float = 0.5,
    bc: Literal["noflux", "pbc"] = "noflux",
    prefix: str = "box",
    order: Literal["C", "F"] = "C",
    stencil_radius: int = 0,
) -> TrajectoryCollection:
    """
    Degrade an SPDE-style collection in *space* (blur/coarsen/pixel-loss/noise).

    Assumes the standard SPDE convention:
      - particle axis N is a flattened grid of shape `grid_shape`,
      - state dim d is #fields per site.

    ``dx`` is read from ``extras_global['{prefix}/dx']`` and updated
    automatically; it does not need to be supplied here.

    Also updates 'box/' box parameters and erases structural outputs starting
    with _cache (regenerated on next use).

    Missing pixels (``data_loss_fraction``) only hide their own value, but a
    finite-difference feature at a valid cell reads its neighbours: set
    ``stencil_radius`` to the reach of the basis' stencils (1 for ∇ and ∇²,
    2 for ∇⁴ or composed operators such as ∇(∇·m)) and every cell within that
    lattice (Manhattan) distance of a pixel missing at any frame a row reads
    (``t-1`` … ``t+3``) is excluded from the fit at ``t`` through the dataset's
    ``dynamic_mask``, while its value stays visible to its neighbours'
    stencils.
    """
    rng = np.random.default_rng(seed)

    ds2 = []
    for ds in coll.datasets:
        ds2.append(
            degrade_spatial_dataset(
                ds,
                downscale=downscale,
                method=method,
                blur_radius=blur_radius,
                data_loss_fraction=data_loss_fraction,
                noise=noise,
                rng=rng,
                mask_threshold=mask_threshold,
                bc=bc,
                prefix=prefix,
                order=order,
                stencil_radius=stencil_radius,
            )
        )

    out = TrajectoryCollection(ds2, coll.weights)
    return out.with_weights(list(map(float, coll.weights)))


def degrade_spatial_dataset(
    ds: TrajectoryDataset,
    *,
    downscale: int | Tuple[int, ...] = 1,
    method: Literal["mean", "subsample"] = "mean",
    blur_radius: int = 0,
    data_loss_fraction: float = 0.0,
    noise: Union[None, float, np.ndarray] = None,
    rng: np.random.Generator,
    mask_threshold: float = 0.5,
    bc: Literal["noflux", "pbc"] = "noflux",
    prefix: str = "box",
    order: Literal["C", "F"] = "C",
    stencil_radius: int = 0,
) -> TrajectoryDataset:
    """Spatial degradation of a single SPDE-style dataset.

    Key invariants ensured by this routine
    --------------------------------------
    1) The flattening convention is preserved (``order="C"`` or ``"F"``).
    2) Box metadata (grid_shape, dx) is updated consistently after coarsening.
    3) Any prepared structural stencil payload is dropped so it is rebuilt for the new grid.
    4) Mask handling is conservative: a coarse cell is valid only if enough fine pixels are valid.
    5) With ``stencil_radius > 0``, cells whose stencil reaches a missing pixel
       are excluded from the fit (``dynamic_mask``), not hidden.
    """
    # ---- materialize tensors ----
    X = np.asarray(ds._X3d())  # (T, N, d)
    M = np.asarray(ds._M2d())  # (T, N)
    T, N, d = X.shape

    # Fetch grid shape
    gs = (ds.extras_global or {}).get(f"{prefix}/grid_shape", None)
    if gs is None:
        raise ValueError(
            "degrade_spatial_dataset: grid_shape not provided and could not be "
            f"inferred from extras_global['{prefix}/grid_shape']."
        )
    grid_shape = tuple(int(v) for v in np.asarray(gs).tolist())

    ndim = len(grid_shape)
    if int(np.prod(grid_shape, dtype=np.int64)) != N:
        raise ValueError(f"grid_shape={grid_shape} incompatible with N={N}.")

    # ---- downscale factors ----
    if isinstance(downscale, int):
        fac = (int(downscale),) * ndim
    else:
        fac = tuple(int(v) for v in downscale)
        if len(fac) != ndim:
            raise ValueError(f"downscale must have length {ndim} (got {len(fac)})")
    if any(v < 1 for v in fac):
        raise ValueError("downscale factors must be >= 1")

    # ---- infer dx (needed to update metadata after coarsening) ----
    dx_ex = (ds.extras_global or {}).get(f"{prefix}/dx", None)
    dx_in = _normalize_dx(dx_ex, ndim=ndim)

    # ---- reshape to grid (respect flattening convention!) ----
    # IMPORTANT: if you ever use order="F" anywhere in stencil construction, you must
    # preserve it here, otherwise the coarse field will be indexed differently than
    # the (re)built neighbor tables.
    Xg = X.reshape((T,) + grid_shape + (d,), order=order)
    Mg = M.reshape((T,) + grid_shape, order=order)

    # ---- optional blur (masked box blur; ignores missing pixels) ----
    if blur_radius > 0:
        Xg, Mg = _box_blur_nd(Xg, Mg, radius=int(blur_radius), bc=bc)

    # ---- downscale ----
    Xc, Mc = _downscale_nd(
        Xg,
        Mg,
        factors=fac,
        method=method,
        mask_threshold=mask_threshold,
    )
    out_shape = tuple(int(s) for s in Xc.shape[1:-1])
    N2 = int(np.prod(out_shape, dtype=np.int64))

    # ---- flatten back (same order) ----
    Xc2 = Xc.reshape((T, N2, d), order=order)
    Mc2 = Mc.reshape((T, N2), order=order)

    # ---- optional measurement noise on observed pixels ----
    if noise is not None:
        Xc2 = _add_noise_to_masked(Xc2, Mc2, rng, noise)

    # ---- random pixel loss (on the coarse grid) ----
    if data_loss_fraction > 0.0:
        flat = Mc2.reshape(-1)
        valid_idx = np.where(flat)[0]
        keep = int(round(valid_idx.size * (1.0 - data_loss_fraction)))
        if keep < valid_idx.size:
            drop = rng.choice(valid_idx, size=valid_idx.size - keep, replace=False)
            flat[drop] = False
        Mc2 = flat.reshape((T, N2))

    # ---- fit mask: exclude cells whose stencil touches a missing pixel ----
    dyn = None
    if stencil_radius > 0 and not bool(np.all(Mc2)):
        # A row at t reads frames t-1 … t+3 (Itô: t, t+1; the noise-robust
        # Stratonovich/shift estimators also t+2, and their default
        # "noisy_lag2" correction t-1 and t+3): a neighbour missing in any of
        # them is hidden from the row's stencils (state mask), so the cell
        # must leave the fit.
        Mrow = Mc2.copy()
        Mrow[1:] &= Mc2[:-1]
        for lag in (1, 2, 3):
            Mrow[:-lag] &= Mc2[lag:]
        dyn = _erode_grid_mask(
            Mrow.reshape((T,) + out_shape, order=order), int(stencil_radius), bc=bc
        ).reshape((T, N2), order=order)

    # ---- extras: purge invalid structural payloads, then update box metadata ----
    extras_g = dict(purge_cache_extras(ds.extras_global) or {})

    # Update dx for the coarse grid: dx_out = dx_in * factor
    dx_out = None
    if dx_in is not None:
        dx_out = tuple(float(dx_in[ax]) * float(fac[ax]) for ax in range(ndim))

    # Keep the box extras in sync with the new grid.
    from SFI.bases.spde import square_grid_extras

    extras_g.update(
        square_grid_extras(
            grid_shape=out_shape,
            dx=(1.0 if dx_out is None else dx_out),
            prefix=prefix,
        )
    )

    # ---- extras_local: downscale per-site fields if present ----
    extras_l = None
    if ds.extras_local:
        extras_l = _downscale_extras_local_grid(
            extras_local=dict(ds.extras_local),
            grid_shape=grid_shape,
            factors=fac,
            method=method,
            # If your helper supports it, pass Mg/mask_threshold so it can
            # handle invalid blocks consistently. Otherwise omit.
            mask=Mg,
            mask_threshold=mask_threshold,
            order=order,
        )

    # ---- meta ----
    meta2 = dict(ds.meta)
    meta2.update(
        dict(
            degrade_spatial_downscale=fac,
            degrade_spatial_method=method,
            degrade_spatial_blur_radius=int(blur_radius),
            degrade_spatial_data_loss_frac=float(data_loss_fraction),
            degrade_spatial_noise_spec=(
                None if noise is None else float(noise) if np.isscalar(noise) else np.asarray(noise).tolist()
            ),
            degrade_spatial_bc=bc,
            degrade_spatial_order=order,
        )
    )

    return TrajectoryDataset.from_arrays(
        X=Xc2,
        t=ds.t,
        dt=ds.dt,
        mask=Mc2,
        dynamic_mask=dyn,
        extras_global=extras_g,
        extras_local=extras_l,
        meta=meta2,
    )


def _erode_grid_mask(Mg: np.ndarray, radius: int, *, bc: str = "noflux") -> np.ndarray:
    """Erode a ``(T, *grid)`` boolean mask by a Manhattan-distance ``radius``.

    A cell stays True only if every cell within lattice distance ``radius`` is
    True.  Periodic grids wrap; otherwise out-of-grid neighbours count as
    valid (the boundary condition supplies them, they are not missing).
    """
    Mg = np.asarray(Mg, dtype=bool)
    ndim = Mg.ndim - 1
    out = Mg.copy()
    rng_ = range(-radius, radius + 1)
    for off in np.array(np.meshgrid(*([list(rng_)] * ndim), indexing="ij")).reshape(ndim, -1).T:
        if not 0 < np.abs(off).sum() <= radius:
            continue
        shifted = Mg
        for ax, o in enumerate(off):
            if o == 0:
                continue
            if bc == "pbc":
                shifted = np.roll(shifted, -int(o), axis=ax + 1)
            else:
                pad = [(0, 0)] * Mg.ndim
                pad[ax + 1] = (0, int(o)) if o > 0 else (int(-o), 0)
                sl = [slice(None)] * Mg.ndim
                sl[ax + 1] = slice(int(o), None) if o > 0 else slice(0, int(o))
                shifted = np.pad(shifted[tuple(sl)], pad, constant_values=True)
        out &= shifted
    return out


def _box_filter_1d(
    X: np.ndarray,
    M: np.ndarray,
    *,
    radius: int,
    axis: int,
    bc: Literal["pbc", "noflux", "drop"] = "noflux",
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D **masked** box filter along one spatial axis.

    This is the core primitive used by `_box_blur_nd` (separable N-D blur).

    Parameters
    ----------
    X
        Array of shape ``(..., L, C)`` *along the chosen axis*, where:
          - L is the axis length being blurred,
          - C is the "value/channel" axis (typically state dimension, or a flattened tail).
        In your SPDE pipeline this will typically be ``(T, *grid, d)`` reshaped/moved.
    M
        Boolean mask of shape ``(..., L)`` aligned with `X` (same leading axes, no channel axis).
        Masked entries (False) are ignored in the average.
    radius
        Blur radius r. Window size is ``W = 2r + 1``.
    axis
        Axis index in `X` to blur over (same logical axis in `M`).
        In your grid convention: time is axis 0, grid axes are 1..ndim, channel is last.
    bc
        Boundary condition:
          - "pbc": periodic wrapping
          - "noflux": clamp to edge (replicate boundary samples)
          - "drop": treat out-of-bounds as missing (do not contribute)

    Returns
    -------
    Xf, Mf
        Filtered values and updated mask:
          - `Mf` is True where at least one valid sample contributed to the window.
          - Where `Mf` is False, `Xf` is set to 0 (by convention).
    """
    r = int(radius)
    if r <= 0:
        return X, M

    # We implement the blur by:
    #   1) moving the target axis next to the channel axis,
    #   2) padding according to `bc`,
    #   3) sliding-window sums via cumsum (O(L), not O(L*W)).
    #
    # This is *masked* averaging:
    #   num = sum( X * M )
    #   den = sum( M )
    #   X_out = num / den where den>0
    #   M_out = den>0

    # Move blur axis to be the second-to-last axis (right before channels).
    # After this: X1 has shape (..., L, C), M1 has shape (..., L).
    X1 = np.moveaxis(X, axis, -2)
    M1 = np.moveaxis(M, axis, -1)

    if X1.shape[-2] != M1.shape[-1]:
        raise ValueError("X and M incompatible along blur axis.")

    W = 2 * r + 1

    # --- pad along L according to bc ---
    if bc == "pbc":
        # Wrap: [..., L, C] -> [..., L+2r, C]
        Xp = np.concatenate([X1[..., -r:, :], X1, X1[..., :r, :]], axis=-2)
        Mp = np.concatenate([M1[..., -r:], M1, M1[..., :r]], axis=-1)
    elif bc == "noflux":
        # Clamp: replicate edge values.
        padX = [(0, 0)] * X1.ndim
        padM = [(0, 0)] * M1.ndim
        padX[-2] = (r, r)
        padM[-1] = (r, r)
        Xp = np.pad(X1, pad_width=padX, mode="edge")
        Mp = np.pad(M1, pad_width=padM, mode="edge")
    elif bc == "drop":
        # Out-of-bounds are invalid: pad mask with False, values arbitrary (0).
        padX = [(0, 0)] * X1.ndim
        padM = [(0, 0)] * M1.ndim
        padX[-2] = (r, r)
        padM[-1] = (r, r)
        Xp = np.pad(X1, pad_width=padX, mode="constant", constant_values=0.0)
        Mp = np.pad(M1, pad_width=padM, mode="constant", constant_values=False)
    else:
        raise ValueError(f"Unknown bc={bc!r}")

    # --- masked sliding sums with cumsum ---
    # Numerator: sum(X * M) over the window; denominator: sum(M) over the window.
    # We prepend a zero so "sum over [i, i+W)" becomes csum[i+W] - csum[i].
    Xw = Xp * Mp[..., None]  # broadcast mask onto channels
    num_csum = np.cumsum(Xw, axis=-2)
    den_csum = np.cumsum(Mp.astype(np.int64), axis=-1)

    # Prepend zeros along the summed axis to use the classic sliding window diff.
    num0 = np.concatenate([np.zeros_like(num_csum[..., :1, :]), num_csum], axis=-2)
    den0 = np.concatenate([np.zeros_like(den_csum[..., :1]), den_csum], axis=-1)

    num = num0[..., W:, :] - num0[..., :-W, :]  # (..., L, C)
    den = den0[..., W:] - den0[..., :-W]  # (..., L)

    # Average where den>0; else output 0 and mask False.
    Xf = np.divide(
        num,
        den[..., None],
        out=np.zeros_like(num, dtype=X1.dtype),
        where=(den[..., None] > 0),
    )
    Mf = den > 0

    # Move axes back to original positions.
    X_out = np.moveaxis(Xf, -2, axis)
    M_out = np.moveaxis(Mf, -1, axis)
    return X_out, M_out


def _box_blur_nd(
    Xg: np.ndarray,
    Mg: np.ndarray,
    *,
    radius: int,
    bc: Literal["pbc", "noflux", "drop"] = "noflux",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separable N-D masked box blur over **all grid axes**.

    Parameters
    ----------
    Xg
        Grid data of shape ``(T, *grid_shape, d)``.
    Mg
        Grid mask of shape ``(T, *grid_shape)``.
    radius
        Blur radius (same on every grid axis). Window size per axis: ``2r+1``.
    bc
        Boundary handling; passed to `_box_filter_1d`.

    Notes
    -----
    This is implemented as consecutive 1D masked box filters along each grid axis.
    That yields an N-D box kernel because the box is separable.

    The mask is updated at each 1D pass:
      a voxel survives if it had **at least one** valid contributor in the window
      along that axis (and recursively along all axes after all passes).
    """
    r = int(radius)
    if r <= 0:
        return Xg, Mg

    X = Xg
    M = Mg
    # Convention assumed by your SPDE code: time axis 0, grid axes 1..ndim, channel last.
    ndim = X.ndim - 2
    for ax in range(1, 1 + ndim):
        X, M = _box_filter_1d(X, M, radius=r, axis=ax, bc=bc)
    return X, M


def _downscale_nd(
    Xg: np.ndarray,
    Mg: np.ndarray,
    *,
    factors: tuple[int, ...],
    method: Literal["average", "mean", "subsample", "nearest"] = "average",
    mask_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downscale a masked grid by integer block factors.

    Parameters
    ----------
    Xg
        Shape ``(T, *grid_shape, d)``.
    Mg
        Shape ``(T, *grid_shape)``.
    factors
        Integer factors per grid axis, e.g. (2,2) maps (Nx,Ny)->(Nx/2,Ny/2).
    method
        - "average"/"mean": masked block-average (recommended)
        - "subsample"/"nearest": pick the [0,0,...] corner of each block
    mask_threshold
        Fraction in [0,1]. A coarse cell is marked valid iff the number of valid
        fine pixels in the block is at least:
            ceil(mask_threshold * prod(factors))
        Special case: if mask_threshold <= 0, we only require at least 1 valid pixel.

    Returns
    -------
    Xc, Mc
        Coarse grid values and mask, shapes ``(T, *coarse_shape, d)``, ``(T, *coarse_shape)``.

    Notes
    -----
    - If a grid axis is not divisible by its factor, we **crop** the high end.
      (This avoids inventing boundary conventions at the degradation level.)
    - For "average": values are averaged over **valid** pixels only.
      Blocks with no valid pixels output X=0 and mask=False.
    """
    if Xg.ndim < 3:
        raise ValueError("Xg must have shape (T, *grid, d).")
    if Mg.shape != Xg.shape[:-1]:
        raise ValueError("Mg must have shape (T, *grid) aligned with Xg.")
    ndim = Xg.ndim - 2
    if len(factors) != ndim:
        raise ValueError(f"factors must have length {ndim} (got {len(factors)})")
    fac = tuple(int(f) for f in factors)
    if any(f < 1 for f in fac):
        raise ValueError("All downscale factors must be >= 1.")

    T = Xg.shape[0]
    grid_shape = Xg.shape[1:-1]
    d = Xg.shape[-1]

    coarse_shape = tuple(int(gs // f) for gs, f in zip(grid_shape, fac))
    crop_shape = tuple(cs * f for cs, f in zip(coarse_shape, fac))

    # Crop fine grid so each axis is divisible by its factor.
    slicer = (slice(None),) + tuple(slice(0, Lc) for Lc in crop_shape) + (slice(None),)
    X = Xg[slicer]
    M = Mg[(slice(None),) + tuple(slice(0, Lc) for Lc in crop_shape)]

    # Reshape into interleaved (coarse_axis, block_axis) pairs:
    #   (T, n1, f1, n2, f2, ..., d)
    new_shape_X = [T]
    new_shape_M = [T]
    for cs, f in zip(coarse_shape, fac):
        new_shape_X.extend([cs, f])
        new_shape_M.extend([cs, f])
    new_shape_X.append(d)

    Xr = X.reshape(tuple(new_shape_X))
    Mr = M.reshape(tuple(new_shape_M))

    # Block axes are the "f" axes: indices 2,4,6,... in Mr; in Xr they are 2,4,6,... as well.
    block_axes = tuple(1 + 2 * i + 1 for i in range(ndim))  # (2,4,6,... in 1-based count) -> actual indices
    # Example ndim=2: shape (T, n1, f1, n2, f2, d) -> block_axes=(2,4)

    # Count valid pixels per block.
    den = Mr.sum(axis=block_axes)  # (T, *coarse_shape)

    # Decide coarse mask according to threshold.
    block_size = int(np.prod(fac, dtype=np.int64))
    if mask_threshold <= 0.0:
        required = 1
    else:
        required = int(np.ceil(float(mask_threshold) * block_size))
        required = max(1, min(required, block_size))
    Mc = den >= required

    meth = method.lower()
    if meth in ("average", "mean"):
        # Masked sum over blocks.
        num = (Xr * Mr[..., None]).sum(axis=block_axes)  # (T, *coarse_shape, d)
        Xc = np.divide(
            num,
            den[..., None],
            out=np.zeros_like(num, dtype=Xg.dtype),
            where=(den[..., None] > 0),
        )
        # Ensure blocks deemed invalid are neutral (for downstream ops expecting mask-consistency).
        Xc = np.where(Mc[..., None], Xc, 0.0)
        return Xc, Mc

    if meth in ("subsample", "nearest"):
        # Take the [0,0,...] entry of each block.
        # Build an index tuple selecting 0 on each block axis.
        idx = [slice(None)]
        for _ in range(ndim):
            idx.append(slice(None))  # coarse axis
            idx.append(0)  # block axis
        idx.append(slice(None))  # channels
        Xc = Xr[tuple(idx)]
        # For mask, do the same slicing.
        idxM = [slice(None)]
        for _ in range(ndim):
            idxM.append(slice(None))
            idxM.append(0)
        Mc0 = Mr[tuple(idxM)]
        # Still respect the threshold rule if requested.
        Mc = Mc & Mc0
        Xc = np.where(Mc[..., None], Xc, 0.0)
        return Xc, Mc

    raise ValueError(f"Unknown downscale method: {method!r}")


def _downscale_extras_local_grid(
    extras_local: Optional[Mapping[str, Any]],
    *,
    grid_shape: tuple[int, ...],
    factors: tuple[int, ...],
    method: Literal["average", "mean", "subsample", "nearest"] = "average",
    mask: Optional[np.ndarray] = None,
    mask_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Downscale **flattened per-site** extras alongside a grid downscale.

    This is meant for datasets where "particles" are actually grid sites
    (N = prod(grid_shape)), and `extras_local` may store per-site arrays like:
      - coordinates (x,y) per pixel,
      - edge vectors per pixel,
      - per-pixel calibration factors, etc.

    Heuristics (non-destructive):
    -----------------------------
    - If an entry looks like per-site constants with shape (N, ...),
      it is reshaped to (*grid_shape, ...), downscaled, then re-flattened.
    - If an entry looks like time-dependent per-site extras with shape (T, N, ...),
      it is reshaped to (T, *grid_shape, ...), downscaled, then re-flattened.
    - Anything else is passed through unchanged.

    Parameters
    ----------
    extras_local
        Dict-like mapping of extras.
    grid_shape
        Fine grid shape used to interpret the flattened N axis.
    factors
        Integer downscale factors per grid axis.
    method, mask_threshold
        Passed to `_downscale_nd`.
    mask
        Optional fine-grid mask to use for time-dependent extras, either:
          - shape (T, N) / (T, *grid_shape), or
          - shape (T, *grid_shape).
        If omitted, extras are downscaled as if fully observed.

    Returns
    -------
    dict
        New extras_local dictionary with downscaled per-site entries.
    """
    out: Dict[str, Any] = dict(extras_local or {})
    if not extras_local:
        return out

    grid_shape = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape)
    N = int(np.prod(grid_shape, dtype=np.int64))

    # Normalize mask to (T, *grid) if provided.
    Mg = None
    if mask is not None:
        Mm = np.asarray(mask)
        if Mm.ndim == 2 and Mm.shape[1] == N:
            # (T, N) flattened -> (T, *grid)
            Tm = Mm.shape[0]
            Mg = Mm.reshape((Tm,) + grid_shape)
        elif Mm.ndim == 1 and Mm.shape[0] == N:
            Mg = Mm.reshape((1,) + grid_shape)
        elif Mm.ndim == 1 + ndim:
            Mg = Mm
        else:
            raise ValueError(f"mask has incompatible shape {Mm.shape} for grid_shape={grid_shape} (N={N}).")

    for k, v in list(extras_local.items()):
        arr = np.asarray(v)
        # Case A: per-site constants (N, ...)
        if arr.ndim >= 1 and arr.shape[0] == N:
            tail = arr.shape[1:]
            # Represent as (T=1, *grid, C) by flattening tail into channels.
            C = int(np.prod(tail, dtype=np.int64)) if tail else 1
            Xg = arr.reshape(grid_shape + tail).reshape((1,) + grid_shape + (C,))
            Mg_use = np.ones((1,) + grid_shape, dtype=bool)
            Xc, Mc = _downscale_nd(
                Xg,
                Mg_use,
                factors=factors,
                method=method,
                mask_threshold=mask_threshold,
            )
            # Flatten back: (*coarse, C) -> (N2, tail)
            coarse_shape = Xc.shape[1:-1]
            N2 = int(np.prod(coarse_shape, dtype=np.int64))
            out[k] = Xc.reshape((1, N2, C))[0].reshape((N2,) + tail)
            continue

        # Case B: time-dependent per-site extras (T, N, ...)
        if arr.ndim >= 2 and arr.shape[1] == N:
            Tm = arr.shape[0]
            tail = arr.shape[2:]
            C = int(np.prod(tail, dtype=np.int64)) if tail else 1
            Xg = arr.reshape((Tm,) + grid_shape + tail).reshape((Tm,) + grid_shape + (C,))
            Mg_use = Mg
            if Mg_use is None:
                Mg_use = np.ones((Tm,) + grid_shape, dtype=bool)
            elif Mg_use.shape[0] == 1 and Tm != 1:
                # Broadcast a static mask across time if needed.
                Mg_use = np.broadcast_to(Mg_use, (Tm,) + grid_shape)
            Xc, Mc = _downscale_nd(
                Xg,
                Mg_use,
                factors=factors,
                method=method,
                mask_threshold=mask_threshold,
            )
            coarse_shape = Xc.shape[1:-1]
            N2 = int(np.prod(coarse_shape, dtype=np.int64))
            out[k] = Xc.reshape((Tm, N2, C)).reshape((Tm, N2) + tail)
            continue

        # Otherwise: keep as-is.
        out[k] = v

    return out


def _normalize_dx(dx, *, ndim: int) -> tuple[float, ...] | None:
    """Normalize dx to a length-ndim tuple of floats (or None)."""
    if dx is None:
        return None
    arr = np.asarray(dx)
    if arr.ndim == 0:
        return (float(arr),) * ndim
    vals = tuple(float(v) for v in arr.reshape(-1).tolist())
    if len(vals) == 1 and ndim > 1:
        return (vals[0],) * ndim
    if len(vals) != ndim:
        raise ValueError(f"dx must be scalar or length {ndim} (got {len(vals)})")
    return vals
