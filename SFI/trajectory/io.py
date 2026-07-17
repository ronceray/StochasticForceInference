"""SFI.trajectory.io
====================

CSV I/O utilities and columnar ↔ tensor conversion for trajectory data.

File format
-----------
We support a single CSV with optional YAML header. The numerical columns include:

- ``particle_id`` (optional): if absent, it is a *single-trajectory* file.
- ``time_step``           : integer time index t (0-based after relabel).
- ``x0, x1, ..., x{d-1}`` : state vector components.

Extras are stored either in the header (YAML) or as extra numeric columns:

Prefixes (numeric columns)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``TG_`` : *time-dependent globals* — values depend on ``t`` only.
- ``P_``  : *per-particle constants* — values depend on particle only.
- ``TP_`` : *time-dependent per-particle* — values depend on ``(t, n)``.
- ``G_``  : *global scalars* — constants stored in the header via averaging.

Note: ``TG_``/``TP_`` columns are parsed as time series and wrapped into
:class:`TimeSeriesExtra`. Header `extras_global` entries are treated as
static unless explicitly wrapped when building the dataset.

Header
~~~~~~
The YAML header may include a mapping ``extras_global`` of arbitrary scalars or
arrays. A special key ``"t"`` (vector of length ``T``) is recognized as the time
axis; when present, it replaces scalar ``dt`` in downstream builders.

Round-trip helpers
------------------
- :func:`flatten_X_to_columns` / :func:`assemble_X_from_columns` convert between
  structured tensors ``(T,N,d)`` and flat columns.
- :func:`save_trajectory_csv_with_extras` / :func:`load_trajectory_csv_with_extras`
  handle extras and header metadata.
- :func:`columns_and_extras_to_dataset` builds a :class:`TrajectoryDataset`
  ready for inference.

All functions are NumPy-based; JAX is optional for basic dtype detection only.

"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra  # local import to avoid cycles

__all__ = [
    "save_trajectory",
    "load_trajectory",
    "columns_and_extras_to_dataset",
]
# ---------------- utilities ----------------


def _sanitize_metadata(obj: Any) -> Any:
    """Convert arrays/scalars into plain Python types recursively for YAML/JSON."""
    # Catch any array-like (numpy, JAX, PyTorch, …) by duck-typing.
    # The old check ("ndarray" in type(obj).__name__) missed JAX ArrayImpl.
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            return np.asarray(obj).tolist()
        except Exception:
            pass
    if hasattr(obj, "item") and not isinstance(obj, (dict, list, tuple)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _sanitize_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(v) for v in obj]
    return obj


# ---------------- core converters ----------------


def flatten_X_to_columns(X: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor trajectory to columnar representation.

    Parameters
    ----------
    X : ndarray, shape (T, N, d)
        State tensor.
    mask : ndarray, optional
        Boolean mask with shape ``(T,N)`` or ``(T,)``; invalid rows are filtered out.

    Returns
    -------
    particle_idx : ndarray, shape (L,)
    time_idx : ndarray, shape (L,)
    state_vectors : ndarray, shape (L, d)

    Notes
    -----
    Rows with NaNs in ``state_vectors`` are dropped. If ``mask`` is provided,
    rows where mask is False are also dropped.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be (T,N,d); got {X.shape}")
    T, N, d = X.shape
    time_idx = np.repeat(np.arange(T, dtype=int), N)
    particle_idx = np.tile(np.arange(N, dtype=int), T)
    state_vectors = X.reshape(T * N, d)
    valid = ~np.isnan(state_vectors).any(axis=1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == (T,):
            m = np.broadcast_to(m[:, None], (T, N))
        if m.shape != (T, N):
            raise ValueError(f"mask must be (T,) or (T,N); got {m.shape}")
        valid &= m.reshape(T * N)
    return (
        particle_idx[valid].astype(int, copy=False),
        time_idx[valid].astype(int, copy=False),
        state_vectors[valid],
    )


def assemble_X_from_columns(
    particle_idx: np.ndarray,
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    *,
    fill_value: float = np.nan,
    relabel: bool = True,
    compress_particles: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Columnar to tensor trajectory (and mask).

    Parameters
    ----------
    particle_idx, time_idx : ndarray
        Integer columns.
    state_vectors : ndarray, shape (L, d)
        Flat state vectors.
    fill_value : float
        Value to fill missing entries in ``X``.
    relabel : bool
        If True, compress particle IDs to contiguous ``0..N-1`` and shift
        ``time_idx`` to start at 0.
    compress_particles : bool
        If True, apply greedy interval packing to further reduce the number of
        columns by merging particles whose time supports do not overlap (with a
        2-frame buffer).  Implies relabeling of particle IDs first.
        See :func:`_greedy_compress_particles`.

    Returns
    -------
    X : ndarray, shape (T, N, d)
    mask : ndarray, shape (T, N)
        True where entries are present in the columns.
    id_map : ndarray of shape ``(N,)`` or dict or None
        * ``relabel=True``, ``compress_particles=False``: shape ``(N,)`` array
          of original particle IDs in column order.
        * ``compress_particles=True``: dict with keys ``'column_origins'``
          (list of lists of compact IDs per column), ``'t_first'``, ``'t_last'``
          (ndarrays of shape ``(N_orig,)`` giving the time span of each compact
          particle before compression).
        * Otherwise: ``None``.
    """
    particle_idx = np.asarray(particle_idx, dtype=int)
    time_idx = np.asarray(time_idx, dtype=int)
    state_vectors = np.asarray(state_vectors)
    if state_vectors.ndim != 2:
        raise ValueError("state_vectors must be 2D (L, d)")
    time_idx = time_idx - time_idx.min()
    if (time_idx < 0).any():
        raise ValueError("time_idx normalization failed (negative after shift).")
    id_map: Optional[Any] = None
    if relabel or compress_particles:
        uniq = np.unique(particle_idx)
        remap = {old: new for new, old in enumerate(uniq)}
        particle_idx = np.vectorize(remap.__getitem__, otypes=[int])(particle_idx)
        N = len(uniq)
        if not compress_particles:
            # Only record the map when IDs were not already 0..N-1
            if not np.array_equal(uniq, np.arange(len(uniq))):
                id_map = uniq  # original IDs in column order
    else:
        N = int(particle_idx.max()) + 1 if len(particle_idx) > 0 else 0
    if compress_particles:
        particle_idx, column_origins, t_first, t_last = _greedy_compress_particles(particle_idx, time_idx)
        N = len(column_origins)
        id_map = {
            "column_origins": column_origins,
            "t_first": t_first,
            "t_last": t_last,
        }
    T = int(time_idx.max()) + 1 if len(time_idx) > 0 else 0
    d = int(state_vectors.shape[1])
    X = np.full((T, N, d), fill_value, dtype=state_vectors.dtype)
    mask = np.zeros((T, N), dtype=bool)
    X[time_idx, particle_idx] = state_vectors
    mask[time_idx, particle_idx] = True
    return X, mask, id_map


def _greedy_compress_particles(
    particle_idx: np.ndarray,
    time_idx: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray]:
    """Greedy interval packing: merge particle columns with non-overlapping time windows.

    Two particles assigned to the same column are always separated by at least
    one masked frame (gap ≥ 2 time steps between time supports), preventing
    spurious increments across identity changes.

    Parameters
    ----------
    particle_idx : ndarray, shape (L,)
        Compact particle indices in ``0..N_orig-1``.
    time_idx : ndarray, shape (L,)
        Compact time indices in ``0..T-1``.

    Returns
    -------
    new_particle_idx : ndarray, shape (L,)
        Updated column index for each observation row.
    column_origins : list[list[int]]
        ``column_origins[c]`` lists the compact IDs packed into column ``c``,
        in temporal order.
    t_first : ndarray, shape (N_orig,)
        First time index for each compact particle.
    t_last : ndarray, shape (N_orig,)
        Last time index for each compact particle.
    """
    if len(particle_idx) == 0:
        return (
            particle_idx.copy(),
            [],
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
        )
    N_orig = int(particle_idx.max()) + 1
    t_first = np.full(N_orig, np.iinfo(np.intp).max, dtype=np.intp)
    t_last = np.full(N_orig, -1, dtype=np.intp)
    np.minimum.at(t_first, particle_idx, time_idx)
    np.maximum.at(t_last, particle_idx, time_idx)
    # Sort particles by first appearance (stable to make assignment deterministic)
    order = np.argsort(t_first, kind="stable")
    column_last: List[int] = []  # last time index of each open column
    column_origins: List[List[int]] = []
    assignment = np.empty(N_orig, dtype=np.intp)
    for p in order:
        tf = int(t_first[p])
        tl = int(t_last[p])
        assigned = -1
        for c, clast in enumerate(column_last):
            if clast <= tf - 2:  # ≥ 2-frame gap ensures at least one masked frame
                assigned = c
                break
        if assigned < 0:
            assigned = len(column_last)
            column_last.append(tl)
            column_origins.append([int(p)])
        else:
            column_last[assigned] = tl
            column_origins[assigned].append(int(p))
        assignment[p] = assigned
    return assignment[particle_idx], column_origins, t_first, t_last


def _reindex_extras_local_for_compression(
    extras_local: Dict[str, Any],
    column_origins: List[List[int]],
    t_first: np.ndarray,
    t_last: np.ndarray,
    T: int,
    N_comp: int,
) -> Dict[str, Any]:
    """Reindex per-particle extras from ``N_orig`` to ``N_comp`` columns.

    After :func:`_greedy_compress_particles` has merged particles into fewer
    columns, the extras stored in ``extras_local`` still refer to the original
    compact particle indices.  This function rebuilds them so they match the
    compressed shape ``(T, N_comp, …)``.

    * ``TimeSeriesExtra`` with shape ``(T, N_orig, …)`` → ``(T, N_comp, …)``.
    * ndarray with leading axis ``N_orig`` → promoted to ``(T, N_comp, …)``
      wrapped in a ``TimeSeriesExtra`` so each ``(t, c)`` slot returns the
      value of whichever compact particle is active in column ``c`` at time ``t``.
    * Callables / ``FunctionExtra`` and other shapes are kept unchanged.
    """
    from SFI.trajectory.dataset import TimeSeriesExtra, time_series_extra

    N_orig = len(t_first)
    out: Dict[str, Any] = {}
    for key, val in extras_local.items():
        if callable(val):
            out[key] = val
            continue
        if isinstance(val, TimeSeriesExtra):
            arr = np.asarray(val.data)
            if arr.ndim >= 2 and arr.shape[0] == T and arr.shape[1] == N_orig:
                tail = arr.shape[2:]
                new_arr = np.full((T, N_comp) + tail, np.nan, dtype=float)
                for c, orig_list in enumerate(column_origins):
                    for p in orig_list:
                        tf, tl = int(t_first[p]), int(t_last[p])
                        new_arr[tf : tl + 1, c] = arr[tf : tl + 1, p]
                out[key] = time_series_extra(new_arr)
            else:
                out[key] = val
        else:
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == N_orig:
                # Per-particle constant (N_orig, …) → time-varying (T, N_comp, …)
                tail = arr.shape[1:]
                new_arr = np.full((T, N_comp) + tail, np.nan, dtype=float)
                for c, orig_list in enumerate(column_origins):
                    for p in orig_list:
                        tf, tl = int(t_first[p]), int(t_last[p])
                        new_arr[tf : tl + 1, c] = arr[p]
                out[key] = time_series_extra(new_arr)
            else:
                out[key] = val
    return out


# -------- builder that wires t from extras_global if present --------


def columns_and_extras_to_dataset(
    particle_idx: np.ndarray,
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    *,
    extras_global: Mapping[str, Any] | None = None,
    extras_local: Mapping[str, Any] | None = None,
    dt: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    mask_fill_value: float = np.nan,
    relabel: bool = True,
    compress_particles: bool = False,
    meta: Optional[Dict[str, Any]] = None,
):
    """Build a :class:`TrajectoryDataset` from columns and parsed extras.

    Preference order for the time axis:
    1) explicit ``t`` argument,
    2) ``extras_global['t']`` (from header or ``TG_t``),
    3) fallback to scalar ``dt``.

    Parameters
    ----------
    compress_particles : bool
        If True, apply greedy interval packing so that particles with
        non-overlapping time supports share the same column index.  This can
        dramatically reduce ``N`` for open-boundary systems where particles
        enter and leave the field of view over time.  Per-particle extras are
        automatically reindexed to the compressed column layout.  The mapping
        is stored as ``meta['particle_column_map']``.
        When False (default) and ``relabel=True``, the original particle IDs
        are recorded as ``extras_local['original_particle_id']``.

    Returns
    -------
    TrajectoryDataset
    """
    if isinstance(t, TimeSeriesExtra):
        t = np.asarray(t.data)
    from SFI.trajectory.dataset import TrajectoryDataset

    X, mask, id_map = assemble_X_from_columns(
        particle_idx=particle_idx,
        time_idx=time_idx,
        state_vectors=state_vectors,
        fill_value=mask_fill_value,
        relabel=relabel,
        compress_particles=compress_particles,
    )
    eg = dict(extras_global or {})
    el = dict(extras_local or {})
    meta_out = dict(meta or {})
    if id_map is not None:
        if isinstance(id_map, dict):
            # compress_particles=True: store column map and reindex per-particle extras
            meta_out["particle_column_map"] = id_map["column_origins"]
            el = _reindex_extras_local_for_compression(
                el,
                id_map["column_origins"],
                id_map["t_first"],
                id_map["t_last"],
                T=X.shape[0],
                N_comp=X.shape[1],
            )
        else:
            # relabel=True: record the original→compact ID mapping
            el["original_particle_id"] = id_map
    t_vec = t
    if t_vec is None and "t" in eg:
        tv = eg["t"]
        raw = tv.data if isinstance(tv, TimeSeriesExtra) else tv
        arr = np.asarray(raw)
        if arr.ndim == 1:
            t_vec = arr
    return TrajectoryDataset.from_arrays(
        X=X,
        dt=None if t_vec is not None else dt,
        t=None if t_vec is None else t_vec,
        mask=mask,
        extras_global=eg,
        extras_local=el,
        meta=meta_out,
    )


# ---------------- unified save/load with extras (csv/parquet/h5) ----------------


def _build_tabular_with_extras(
    *,
    particle_idx: Optional[np.ndarray],
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    extras_global: Optional[Dict[str, Any]] = None,
    extras_local: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    prefix_G: str = "G_",
    prefix_TG: str = "TG_",
    prefix_P: str = "P_",
    prefix_TP: str = "TP_",
):
    """Create a pandas DataFrame with all columns (x*, IDs, prefixed extras)
    and a YAML-able metadata dict for header/schema.

    Returns
    -------
    df : pandas.DataFrame
    yaml_meta : dict (already sanitized to plain Python types)
    base_cols : list[str]  # column ordering hint
    """
    import pandas as pd

    time_idx = np.asarray(time_idx, dtype=int)
    particle_idx = None if particle_idx is None else np.asarray(particle_idx, dtype=int)
    state_vectors = np.asarray(state_vectors)
    L, d = state_vectors.shape

    cols: Dict[str, Any] = {}
    if particle_idx is not None:
        cols["particle_id"] = particle_idx
    cols["time_step"] = time_idx
    for j in range(d):
        cols[f"x{j}"] = state_vectors[:, j]

    # Shapes
    T = int(time_idx.max()) + 1 if L else 0
    N = (int(particle_idx.max()) + 1) if (L and particle_idx is not None) else (1 if L else 0)
    part = particle_idx if particle_idx is not None else np.zeros_like(time_idx, dtype=int)

    # Global extras: TimeSeriesExtra → TG_* columns; others → header
    header_globals: Dict[str, Any] = {}
    for key, val in (extras_global or {}).items():
        if isinstance(val, TimeSeriesExtra):
            arr = np.asarray(val.data)
            if arr.shape[0] != T and L:
                raise ValueError(
                    f"extras_global['{key}'] TimeSeriesExtra expects leading axis T={T}, got {arr.shape[0]}"
                )
            vals = arr[time_idx] if L else arr.reshape(0, *arr.shape[1:])
            flat = vals.reshape(L, -1) if L else vals.reshape(0, -1)
            for kcol in range(flat.shape[1] if L else (arr.reshape(-1).shape[0] or 1)):
                name = (
                    f"{prefix_TG}{key}"
                    if (flat.shape[1] if L else arr.reshape(-1).shape[0]) == 1
                    else f"{prefix_TG}{key}_{kcol}"
                )
                cols[name] = flat[:, kcol] if L else np.asarray([], dtype=float)
        else:
            if callable(val) or isinstance(val, FunctionExtra):
                warnings.warn(
                    f"extras_global['{key}'] is a callable and cannot be serialized to disk. "
                    "It will be omitted from the saved file. Re-attach it to the dataset after loading.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            header_globals[key] = _sanitize_metadata(val)

    # Local extras:
    #   - TimeSeriesExtra → TP_* columns
    #   - array-like with first axis N → P_* columns
    #   - otherwise → header
    for key, val in (extras_local or {}).items():
        if isinstance(val, TimeSeriesExtra):
            arr = np.asarray(val.data)  # expected (T, N, …) or (T, 1, …) for single-trajectory
            if arr.ndim < 2:
                raise ValueError(
                    f"extras_local['{key}'] TimeSeriesExtra must have at least 2 dims (T,N,...), got {arr.shape}"
                )
            if L and arr.shape[0] != T:
                raise ValueError(f"extras_local['{key}'] TimeSeriesExtra expects T={T}, got {arr.shape[0]}")
            pid = part
            if (particle_idx is None) and arr.shape[1] == 1:
                pid = np.zeros_like(time_idx)
            if L:
                vals = arr[time_idx, pid]
                flat = vals.reshape(L, -1)
                for kcol in range(flat.shape[1]):
                    name = f"{prefix_TP}{key}" if flat.shape[1] == 1 else f"{prefix_TP}{key}_{kcol}"
                    cols[name] = flat[:, kcol]
            else:
                # empty table, still create no data columns; metadata only
                header_globals[key] = _sanitize_metadata(arr)
        else:
            if callable(val) or isinstance(val, FunctionExtra):
                warnings.warn(
                    f"extras_local['{key}'] is a callable and cannot be serialized to disk. "
                    "It will be omitted from the saved file. Re-attach it to the dataset after loading.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            arr = np.asarray(val)
            if arr.ndim >= 1 and (N == 0 or arr.shape[0] == N):
                # per-particle constants: (N, …)
                vals = arr[part if particle_idx is not None else np.zeros_like(time_idx)]
                flat = vals.reshape(L, -1)
                for kcol in range(flat.shape[1]):
                    name = f"{prefix_P}{key}" if flat.shape[1] == 1 else f"{prefix_P}{key}_{kcol}"
                    cols[name] = flat[:, kcol]
            else:
                header_globals[key] = _sanitize_metadata(arr)

    df = pd.DataFrame(cols)

    yaml_meta = dict(_sanitize_metadata(metadata or {}))
    if header_globals:
        yaml_meta.setdefault("extras_global", {}).update(header_globals)

    base_cols = ([] if particle_idx is None else ["particle_id"]) + ["time_step"] + [f"x{j}" for j in range(d)]
    return df, yaml_meta, base_cols


def _parse_tabular_with_extras(
    df,
    yaml_meta: Dict[str, Any],
    *,
    particle_column: Optional[str],
    time_column: str,
    relabel: bool,
    prefix_G: str = "G_",
    prefix_TG: str = "TG_",
    prefix_P: str = "P_",
    prefix_TP: str = "TP_",
):
    """Inverse of _build_tabular_with_extras for a pandas DataFrame + header/metadata dict."""
    import numpy as np

    from SFI.trajectory.dataset import time_series_extra

    colnames = list(df.columns)
    has_particles = particle_column is not None and (particle_column in df.columns)

    if has_particles:
        particle_indices = df[particle_column].to_numpy(dtype=int)
    else:
        particle_indices = np.zeros((len(df),), dtype=int)

    time_indices = df[time_column].to_numpy(dtype=int)

    # identify state columns: not ids and not extras prefixes
    def is_extra(name: str) -> bool:
        return any(name.startswith(px) for px in (prefix_G, prefix_TG, prefix_P, prefix_TP)) or name in {
            particle_column,
            time_column,
        }

    state_cols = [c for c in colnames if not is_extra(c)]
    state_vectors = df[state_cols].to_numpy(dtype=float)

    # relabel like CSV loader
    if relabel and len(state_vectors) > 0:
        if has_particles:
            _, inv = np.unique(particle_indices, return_inverse=True)
            particle_indices = inv.astype(int, copy=False)
        time_indices = time_indices - int(time_indices.min())

    L = len(df)
    T = int(time_indices.max()) + 1 if L else 0
    N = int(particle_indices.max()) + 1 if L else 0

    # Start from header-provided extras (may contain globals)
    extras_global: Dict[str, Any] = dict(yaml_meta.get("extras_global", {}) or {})
    extras_local: Dict[str, Any] = {}

    # Collect prefixed numeric columns
    def collect(prefix: str) -> Dict[str, np.ndarray]:
        return {name: df[name].to_numpy() for name in df.columns if name.startswith(prefix)}

    TG_cols = collect(prefix_TG)
    P_cols = collect(prefix_P)
    TP_cols = collect(prefix_TP)
    G_cols = collect(prefix_G)

    # TG_: time-dependent globals → TimeSeriesExtra(T, …)
    tg_grouped: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, vals in TG_cols.items():
        base = name[len(prefix_TG) :].split("_")[0]
        tg_grouped.setdefault(base, []).append((name, vals))
    for key, items in tg_grouped.items():
        items_sorted = sorted(items, key=lambda kv: kv[0])
        mat = np.column_stack([v for _, v in items_sorted])  # (L, k)
        tg_matrix = np.full((T, mat.shape[1]), np.nan, dtype=float)
        for t in range(T):
            sel = time_indices == t
            if np.any(sel):
                tg_matrix[t] = np.nanmean(mat[sel], axis=0)
        tg_matrix = tg_matrix.squeeze(-1) if tg_matrix.shape[1] == 1 else tg_matrix
        extras_global[key] = time_series_extra(tg_matrix)

    # TP_: time-dependent per-particle → TimeSeriesExtra(T, N, …)
    tp_grouped: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, vals in TP_cols.items():
        base = name[len(prefix_TP) :].split("_")[0]
        tp_grouped.setdefault(base, []).append((name, vals))
    for key, items in tp_grouped.items():
        items_sorted = sorted(items, key=lambda kv: kv[0])
        mat = np.column_stack([v for _, v in items_sorted])  # (L, k)
        out = np.full((T, N, mat.shape[1]), np.nan, dtype=float)
        for t in range(T):
            sel_t = time_indices == t
            if not np.any(sel_t):
                continue
            for n in range(N):
                sel = sel_t & ((particle_indices == n) if has_particles else (particle_indices == 0))
                if np.any(sel):
                    out[t, n] = np.nanmean(mat[sel], axis=0)
        out = out.squeeze(-1) if out.shape[2] == 1 else out
        extras_local[key] = time_series_extra(out)

    # P_: per-particle constants
    p_grouped: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, vals in P_cols.items():
        base = name[len(prefix_P) :].split("_")[0]
        p_grouped.setdefault(base, []).append((name, vals))
    for key, items in p_grouped.items():
        items_sorted = sorted(items, key=lambda kv: kv[0])
        mat = np.column_stack([v for _, v in items_sorted])
        out = np.full((N, mat.shape[1]), np.nan, dtype=float)
        for n in range(N):
            sel = particle_indices == n
            if np.any(sel):
                out[n] = np.nanmean(mat[sel], axis=0)
        extras_local[key] = out.squeeze(-1) if out.shape[1] == 1 else out

    # G_: global scalars → average
    for name, vals in G_cols.items():
        key = name[len(prefix_G) :]
        extras_global[key] = float(np.nanmean(vals))

    return (
        yaml_meta,
        colnames,
        particle_indices,
        time_indices,
        state_vectors,
        extras_global,
        extras_local,
    )


def _resolve_column_name(spec: Union[int, str], colnames: List[str], *, what: str) -> str:
    """Resolve an ``int`` index or ``str`` name to a column name.

    Raises a ``ValueError`` naming the available columns on a miss.
    """
    if isinstance(spec, str):
        if spec not in colnames:
            raise ValueError(f"{what} column {spec!r} not found; available columns: {colnames}")
        return spec
    i = int(spec)
    if not (-len(colnames) <= i < len(colnames)):
        raise ValueError(f"{what} column index {i} out of range for {len(colnames)} columns: {colnames}")
    return colnames[i]


def _subselect_state_columns(
    df,
    *,
    particle_name: Optional[str],
    time_name: str,
    state_names: List[str],
    prefixes: Tuple[str, ...],
):
    """Keep only id/time, the selected state columns (in order), and extras columns."""
    keep = ([particle_name] if particle_name else []) + [time_name] + list(state_names)
    keep += [c for c in df.columns if any(c.startswith(px) for px in prefixes) and c not in keep]
    return df[keep]


def _apply_named_knobs(
    df,
    *,
    fmt: str,
    particle_column: Optional[Union[int, str]],
    time_column: Union[int, str],
    state_columns: Optional[Sequence[Union[int, str]]],
    prefixes: Tuple[str, ...],
):
    """Resolve column knobs for the name-addressed formats (parquet / h5).

    ``str`` values are honored (and validated); ``int`` values keep the
    historical behavior — they cannot be distinguished from the defaults,
    so the canonical names ``particle_id`` / ``time_step`` are used.
    Returns ``(particle_name_or_None, time_name, df_possibly_subselected)``.
    """
    colnames = list(df.columns)
    if isinstance(particle_column, str):
        p_name: Optional[str] = _resolve_column_name(particle_column, colnames, what="particle")
    elif particle_column is None:
        p_name = None
    else:
        p_name = "particle_id" if "particle_id" in colnames else None
    t_name = (
        _resolve_column_name(time_column, colnames, what="time")
        if isinstance(time_column, str)
        else "time_step"
    )
    if state_columns is not None:
        bad = [c for c in state_columns if not isinstance(c, str)]
        if bad:
            raise ValueError(f"state_columns must be column names (str) for {fmt} files; got {bad!r}")
        state_names = [_resolve_column_name(c, colnames, what="state") for c in state_columns]
        df = _subselect_state_columns(
            df, particle_name=p_name, time_name=t_name, state_names=state_names, prefixes=prefixes
        )
    return p_name, t_name, df


def _infer_format_from_suffix(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return "parquet"
    if lower.endswith(".h5") or lower.endswith(".hdf5"):
        return "h5"
    return "csv"


def save_trajectory(
    filename: str,
    *,
    particle_idx: Optional[np.ndarray],
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    extras_global: Optional[Dict[str, Any]] = None,
    extras_local: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    format: Optional[str] = None,
    # CSV-only knobs:
    float_fmt: str = "%.8f",
    # Parquet-only knobs:
    compression: str = "snappy",
    # Prefixes (shared semantics across formats):
    prefix_G: str = "G_",
    prefix_TG: str = "TG_",
    prefix_P: str = "P_",
    prefix_TP: str = "TP_",
) -> None:
    """Unified saver for {'csv','parquet','h5'} (inferred from filename if format=None)."""
    fmt = (format or _infer_format_from_suffix(filename)).lower()

    # Dispatcher-owned structural arrays (``_cache/`` keys) are derivable and must
    # never be serialized; strip them here so the saver upholds the invariant on
    # its own, mirroring ``simulate`` output and ``degrade`` (no normal path
    # persists them onto a dataset, so this is a defense-in-depth guard).
    from SFI.statefunc.nodes.interactions.prepare import purge_cache_extras

    extras_global = purge_cache_extras(extras_global)
    extras_local = purge_cache_extras(extras_local)

    df, yaml_meta, base_cols = _build_tabular_with_extras(
        particle_idx=particle_idx,
        time_idx=time_idx,
        state_vectors=state_vectors,
        extras_global=extras_global,
        extras_local=extras_local,
        metadata=metadata,
        prefix_G=prefix_G,
        prefix_TG=prefix_TG,
        prefix_P=prefix_P,
        prefix_TP=prefix_TP,
    )

    if fmt == "csv":
        import yaml

        # YAML header as comment lines + CSV body
        yaml_str = yaml.safe_dump(yaml_meta, sort_keys=False)
        yaml_header = "# ---\n" + "\n".join(f"# {line}" for line in yaml_str.strip().splitlines())
        ordered = base_cols + [c for c in df.columns if c not in base_cols]
        with open(filename, "w") as f:
            f.write(yaml_header + "\n")
            df.to_csv(f, index=False, columns=ordered, float_format=float_fmt)
        return

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import yaml
        except Exception as e:  # pragma: no cover
            raise ImportError("Saving Parquet requires pyarrow and yaml.") from e
        table = pa.Table.from_pandas(df, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[b"sfi_yaml_header"] = yaml.safe_dump(yaml_meta, sort_keys=False).encode("utf-8")
        table = table.replace_schema_metadata(md)
        pq.write_table(table, filename, compression=compression)
        return

    if fmt == "h5":
        try:
            import h5py
            import yaml
        except Exception as e:  # pragma: no cover
            raise ImportError("Saving HDF5 requires h5py and yaml.") from e
        with h5py.File(filename, "w") as h5:
            grp = h5.create_group("table")
            for c in df.columns:
                data = np.asarray(df[c].to_numpy())
                grp.create_dataset(c, data=data, compression="gzip", shuffle=True, fletcher32=True)
            # store YAML in root attr
            h5.attrs["sfi_yaml_header"] = yaml.safe_dump(yaml_meta, sort_keys=False)
        return

    raise ValueError("format must be one of {'csv','parquet','h5'}")


def load_trajectory(
    filename: str,
    *,
    format: Optional[str] = None,
    # Column-selection knobs (int index or str name):
    particle_column: Optional[Union[int, str]] = 0,  # None => single-trajectory file
    time_column: Union[int, str] = 1,
    state_columns: Optional[Sequence[Union[int, str]]] = None,
    relabel: bool = True,
    # Prefixes:
    prefix_G: str = "G_",
    prefix_TG: str = "TG_",
    prefix_P: str = "P_",
    prefix_TP: str = "TP_",
):
    """Unified loader for {'csv','parquet','h5'} (inferred from filename if format=None).

    Parameters
    ----------
    particle_column, time_column
        Which columns hold the particle ID and the time index.  Accept a
        column *name* (``str``) for any format, or a positional *index*
        (``int``) for CSV files only.  CSV defaults are positional
        (column 0 = particle, column 1 = time); parquet and HDF5 default
        to the canonical names ``"particle_id"`` and ``"time_step"`` and
        ignore ``int`` values (their defaults cannot be distinguished
        from "unspecified").  ``particle_column=None`` marks a
        single-trajectory file.
    state_columns
        Optional explicit selection of the state-vector columns (names,
        or indices for CSV), in order.  When given, every other
        non-extras column is dropped.  Default: every column that is not
        an ID and does not carry an extras prefix is a state component.

    Returns the standard tuple:
      (metadata, column_headers, particle_indices, time_indices, state_vectors, extras_global, extras_local)
    """
    fmt = (format or _infer_format_from_suffix(filename)).lower()
    extra_prefixes = (prefix_G, prefix_TG, prefix_P, prefix_TP)

    if fmt == "csv":
        # Parse YAML header then read CSV
        import pandas as pd
        import yaml

        metadata: Dict[str, Any] = {}
        yaml_lines: List[str] = []
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("# "):
                    yaml_lines.append(line[2:])
                elif not line.startswith("#"):
                    break
        if yaml_lines:
            try:
                metadata = yaml.safe_load("".join(yaml_lines)) or {}
            except Exception:
                metadata = {}

        df = pd.read_csv(filename, comment="#")
        colnames = list(df.columns)

        # Resolve int/str knobs to column names.
        particle_name = (
            _resolve_column_name(particle_column, colnames, what="particle")
            if particle_column is not None
            else None
        )
        time_name = _resolve_column_name(time_column, colnames, what="time")

        # Optional explicit state-column selection (drops everything else).
        if state_columns is not None:
            state_names = [_resolve_column_name(c, colnames, what="state") for c in state_columns]
            df = _subselect_state_columns(
                df,
                particle_name=particle_name,
                time_name=time_name,
                state_names=state_names,
                prefixes=extra_prefixes,
            )

        # Canonical names for the uniform parser.
        rename: Dict[str, str] = {}
        if particle_name is not None and particle_name != "particle_id":
            if "particle_id" in df.columns:
                raise ValueError(
                    f"particle column {particle_name!r} selected, but a distinct "
                    "'particle_id' column also exists — drop or rename one."
                )
            rename[particle_name] = "particle_id"
        if time_name != "time_step":
            if "time_step" in df.columns:
                raise ValueError(
                    f"time column {time_name!r} selected, but a distinct "
                    "'time_step' column also exists — drop or rename one."
                )
            rename[time_name] = "time_step"
        if rename:
            # pandas-stubs rename overloads reject a plain dict mapper (false positive).
            df = df.rename(columns=rename)  # type: ignore[call-overload]

        # parse
        return _parse_tabular_with_extras(
            df,
            metadata,
            particle_column="particle_id" if particle_name is not None else None,
            time_column="time_step",
            relabel=relabel,
            prefix_G=prefix_G,
            prefix_TG=prefix_TG,
            prefix_P=prefix_P,
            prefix_TP=prefix_TP,
        )

    if fmt == "parquet":
        import pandas as pd
        import pyarrow.parquet as pq
        import yaml

        table = pq.read_table(filename)
        md = dict(table.schema.metadata or {})
        yaml_bytes = md.get(b"sfi_yaml_header", None)
        metadata = yaml.safe_load(yaml_bytes.decode("utf-8")) if yaml_bytes else {}
        df = table.to_pandas(types_mapper=None)
        p_name, t_name, df = _apply_named_knobs(
            df,
            fmt=fmt,
            particle_column=particle_column,
            time_column=time_column,
            state_columns=state_columns,
            prefixes=extra_prefixes,
        )
        return _parse_tabular_with_extras(
            df,
            metadata,
            particle_column=p_name,
            time_column=t_name,
            relabel=relabel,
            prefix_G=prefix_G,
            prefix_TG=prefix_TG,
            prefix_P=prefix_P,
            prefix_TP=prefix_TP,
        )

    if fmt == "h5":
        import h5py
        import pandas as pd
        import yaml

        with h5py.File(filename, "r") as h5:
            metadata = {}
            if "sfi_yaml_header" in h5.attrs:
                try:
                    # h5py stubs type attrs values as array-like; the header is str/bytes.
                    metadata = yaml.safe_load(h5.attrs["sfi_yaml_header"]) or {}  # type: ignore[arg-type]
                except Exception:
                    metadata = {}
            if "table" not in h5:
                raise ValueError("HDF5 file missing 'table' group.")
            grp = h5["table"]
            if not isinstance(grp, h5py.Group):
                raise ValueError("HDF5 'table' entry is not a group.")
            # reconstruct DataFrame from datasets (the group holds only Datasets;
            # h5py stubs widen __getitem__ to include Datatype)
            cols = {name: grp[name][...] for name in grp.keys()}  # type: ignore[index]
            df = pd.DataFrame(cols)
        p_name, t_name, df = _apply_named_knobs(
            df,
            fmt=fmt,
            particle_column=particle_column,
            time_column=time_column,
            state_columns=state_columns,
            prefixes=extra_prefixes,
        )
        return _parse_tabular_with_extras(
            df,
            metadata,
            particle_column=p_name,
            time_column=t_name,
            relabel=relabel,
            prefix_G=prefix_G,
            prefix_TG=prefix_TG,
            prefix_P=prefix_P,
            prefix_TP=prefix_TP,
        )

    raise ValueError("format must be one of {'csv','parquet','h5'}")
