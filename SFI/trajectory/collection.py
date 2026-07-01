# collection.py
"""
Trajectory collection: index-driven streaming over datasets.

This module defines :class:`TrajectoryCollection`, a thin coordinator that:
- stores multiple :class:`TrajectoryDataset` objects,
- computes per-dataset weights,
- yields **(producer, t_idx_chunk)** pairs for vmapped integration.

No chunk heuristics live here. The dataset owns valid windows and single-t row
production. The integrator vmaps over integer indices and reduces.

Typical loop
------------
>>> coll = TrajectoryCollection.from_dataset(ds).with_weights("pool")
>>> for payload in coll.iter_slices(require=req, bytes_hint=bh, chunk_target_bytes=64<<20):
...     producer = payload["producer"]                  # Callable[[t], row]
...     t_idx     = payload["t_idx"]                    # (K_chunk,)
...     w_ds      = payload["weight"]                   # dataset scalar weight
...     # integrator: vmap(lambda t: program(**producer(t)))(t_idx)

Weights
-------
Per-dataset weights are **unnormalised** multipliers applied to every
estimator (force, diffusion, parametric).  Within-dataset weighting is
intrinsic to each estimator: the force is per-dt, the diffusion per-point.

- "pool" (default): multiplier 1 for every dataset — pool all increments on
  equal footing (each dataset then contributes by its effective time for the
  force, by its point count for the diffusion).
- "per_dataset": each dataset contributes equally (multiplier mean(Teff)/Teff_d).
- a sequence of floats: explicit unnormalised multipliers.

Notes
-----
- No cross-dataset vectorization. A small Python loop over datasets is intended.
- `bytes_hint` is the per-row memory estimate supplied by the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
)

import jax.numpy as jnp
import numpy as np

from SFI.trajectory.dataset import TimeSeriesExtra, TrajectoryDataset, time_series_extra

from .io import (
    _parse_tabular_with_extras,
    columns_and_extras_to_dataset,
    flatten_X_to_columns,
    load_trajectory,
    save_trajectory,
)

WeightSpec = Union[str, Sequence[float]]


def _is_single_file(path: Union[str, Path]) -> bool:
    p = Path(path)
    return p.suffix.lower() in {".csv", ".parquet", ".pq", ".h5", ".hdf5"}


@dataclass
class TrajectoryCollection:
    """
    Container for one or more trajectories plus per-dataset weights.

    This is the main user-facing trajectory object. It wraps a list of
    :class:`TrajectoryDataset` instances and exposes an index-based streaming
    interface used by the integration runtime.

    Most users should construct collections via :meth:`from_arrays`,
    :meth:`from_dataset` or :meth:`load` rather than instantiating this
    dataclass directly.

    Parameters
    ----------
    datasets
        List of underlying :class:`TrajectoryDataset` objects. The order is
        preserved in iteration and determines the ordering of the ``weights``
        vector.
    weights
        1D JAX array of shape ``(D,)`` with non-negative entries, where
        ``D = len(datasets)``. The vector is normalized to sum to 1 by
        :meth:`with_weights`.

    Notes
    -----
    The collection itself does not impose any chunking heuristic. It only
    coordinates datasets and their weights; the integrator decides how to
    vmap over the indices returned by :meth:`iter_slices`.
    """

    datasets: List[TrajectoryDataset]
    weights: jnp.ndarray  # shape (D,), normalized to sum to 1

    # ---------- construction ----------
    @classmethod
    def from_dataset(cls, ds: TrajectoryDataset, *, weights: WeightSpec = "pool") -> "TrajectoryCollection":
        """Wrap a single :class:`TrajectoryDataset` in a collection.

        Parameters
        ----------
        ds
            The dataset to wrap.
        weights
            Initial weight specification; default ``"Teff"``.
            See :meth:`with_weights`.

        Returns
        -------
        TrajectoryCollection
            A single-dataset collection with weights computed from ``ds``.
        """
        coll = cls([ds], jnp.array([1.0], dtype=jnp.float32))
        return coll.with_weights(weights)

    def concat(
        self,
        items: Sequence[Union["TrajectoryCollection", TrajectoryDataset]],
        *,
        weights: WeightSpec = "pool",
    ) -> "TrajectoryCollection":
        """
        Concatenate this collection with other collections or datasets.

        Parameters
        ----------
        items
            Sequence of :class:`TrajectoryCollection` or
            :class:`TrajectoryDataset` instances. Collections are flattened
            into their constituent datasets.
        weights
            Weight specification for the concatenated collection. See
            :meth:`with_weights` for accepted values.

        Returns
        -------
        TrajectoryCollection
            New collection containing all datasets from ``self`` followed
            by all datasets from ``items``.
        """
        merged: List[TrajectoryDataset] = []
        merged.extend(self.datasets)
        for it in items:
            if isinstance(it, TrajectoryCollection):
                merged.extend(it.datasets)
            else:
                merged.append(it)
        out = TrajectoryCollection(merged, jnp.ones((len(merged),), dtype=jnp.float32))
        return out.with_weights(weights)

    def __and__(
        self, other: Union["TrajectoryCollection", TrajectoryDataset]
    ) -> "TrajectoryCollection":
        """Merge collections (or a collection and a dataset) with ``&``.

        ``c1 & c2`` appends the datasets of ``other`` to those of ``self``
        with the default ``"pool"`` policy (every increment on equal
        footing).  It chains naturally (``c1 & c2 & c3``); call
        :meth:`concat` or :meth:`with_weights` for the ``"per_dataset"``
        policy or explicit weights.
        """
        return self.concat([other])

    # ---------- weighting ----------
    def with_weights(
        self,
        spec: WeightSpec = "pool",
        *,
        required: Set[str] = frozenset({"X", "dX"}),
        subsampling: int = 1,
    ) -> "TrajectoryCollection":
        """
        Set the per-dataset weights (an **unnormalised** multiplier).

        Parameters
        ----------
        spec
            Inter-dataset weight policy — a per-dataset multiplier applied to
            every estimator (force, diffusion, parametric).  Accepted values:

            - ``"pool"`` (default): multiplier ``1`` for all datasets, i.e.
              pool every increment on equal footing.  Combined with each
              estimator's intrinsic within-dataset weighting (force is per-dt,
              diffusion per-point), this weights each dataset by its effective
              time (force) or point count (diffusion) — the natural
              maximum-likelihood pooling.
            - ``"per_dataset"``: each dataset contributes equally regardless of
              length (multiplier ``mean(Teff)/Teff_d``).  Exact for the force;
              for the diffusion it is exact when ``dt`` is uniform.
            - a sequence of floats: explicit unnormalised multipliers.

        required
            Streams used to compute ``Teff`` in the ``"per_dataset"`` policy.
            See :meth:`TrajectoryDataset.Teff`.
        subsampling
            Optional subsampling factor used when counting valid indices.

        Returns
        -------
        TrajectoryCollection
            The same collection with its :attr:`weights` field updated.

        Notes
        -----
        Weights are exposed to the integrator via the ``"weight"`` entry in the
        payloads yielded by :meth:`iter_slices` and applied in every reduction
        (sum and mean).  They are deliberately **unnormalised**: the absolute
        scale cancels in the mean-reduced estimates, while for the force Gram /
        covariance it sets the information scale (a single dataset carries unit
        weight).
        """
        D = len(self.datasets)
        if isinstance(spec, str):
            if spec == "pool":
                w = jnp.ones((D,), dtype=jnp.float32)
            elif spec == "per_dataset":
                teffs = [float(self.datasets[i].Teff(required, subsampling=subsampling)) for i in range(D)]
                pos = [t for t in teffs if t > 0]
                mean_teff = (sum(pos) / len(pos)) if pos else 1.0
                w = jnp.array([(mean_teff / t) if t > 0 else 0.0 for t in teffs], dtype=jnp.float32)
            else:
                raise ValueError(
                    f"unknown weight policy {spec!r}; use 'pool', 'per_dataset', "
                    "or an explicit per-dataset multiplier array."
                )
        else:
            w = jnp.array(spec, dtype=jnp.float32)
            if w.shape != (D,):
                raise ValueError(f"weights length mismatch: got {w.shape}, expected {(D,)}")
        # Unnormalised relative multipliers: scale cancels in mean-reduced
        # estimates but sets the force Gram / covariance scale.
        self.weights = w
        return self

    # ---------- streaming ----------
    def iter_slices(
        self,
        *,
        require: Set[str],
        bytes_hint: Optional[int],
        chunk_target_bytes: int = 64 * 1024**2,
        subsampling: int = 1,
        context: Optional[str] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """
        Yield chunks as (producer, t_idx) pairs for vmapped integration.

        Parameters
        ----------
        require
            Set of stream names required by the integrator (e.g. ``{"X","dX","mask"}``).
            Passed to :meth:`TrajectoryDataset.valid_indices` and
            :meth:`TrajectoryDataset.make_producer`.
        bytes_hint
            Approximate per-row memory footprint (in bytes) of the values
            produced by the program. If ``None`` or ``<= 0``, no chunking is
            performed and all valid indices are yielded at once.
        chunk_target_bytes
            Target chunk size in bytes. Combined with ``bytes_hint`` to
            determine how many rows to include in each chunk.
        subsampling
            Optional subsampling factor applied to the time indices before
            chunking.
        context
            Optional context string passed through to the dataset producer,
            typically used to switch extra fields.

        Yields
        ------
        dict
            Mapping with keys:

            - ``"producer"``: ``Callable[[jax.Array], dict]``, single-t row builder.
            - ``"t_idx"``: 1D JAX array of integer time indices.
            - ``"dataset_index"``: index of the underlying dataset in
              :attr:`datasets`.
            - ``"weight"``: float dataset weight, taken from :attr:`weights`.
        """
        if subsampling <= 0:
            raise ValueError("subsampling must be a positive integer")

        for ds_idx, ds in enumerate(self.datasets):
            base_idx = ds.valid_indices(require, subsampling=subsampling)
            if base_idx.size == 0:
                continue

            producer = ds.make_producer(
                require,
                include_mask=True,
                include_dt=True,
                context=context,
                force_dt_keys={"dt"},
                dataset_index=self.dataset_index(ds_idx),
            )

            if not bytes_hint or bytes_hint <= 0:
                yield {
                    "producer": producer,
                    "t_idx": base_idx,
                    "dataset_index": ds_idx,
                    "weight": float(self.weights[ds_idx]),
                }
                continue

            total = int(base_idx.size)
            rows_per_chunk = min(total, max(1, int(chunk_target_bytes // int(bytes_hint))))
            for start in range(0, total, rows_per_chunk):
                sel = base_idx[start : start + rows_per_chunk]
                yield {
                    "producer": producer,
                    "t_idx": sel,
                    "dataset_index": ds_idx,
                    "weight": float(self.weights[ds_idx]),
                }

    def peek_row(
        self,
        *,
        require: Set[str] = frozenset({"X", "dX"}),
        context: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """
        Return a single-t sample row from the first dataset with valid indices.

        Parameters
        ----------
        require
            Set of stream names required for the sample (as in
            :meth:`iter_slices`).
        context
            Optional context string forwarded to the producer.

        Returns
        -------
        dict
            Structure matching ``producer(t)`` for the chosen dataset.

        Notes
        -----
        Useful for memory estimation and debugging program outputs.
        """
        for ds_idx, ds in enumerate(self.datasets):
            idx = ds.valid_indices(require)
            if idx.size == 0:
                continue
            t0 = idx[:1][0]
            producer = ds.make_producer(
                require,
                include_mask=True,
                include_dt=True,
                context=context,
                force_dt_keys={"dt"},
                dataset_index=self.dataset_index(ds_idx),
            )
            return producer(t0)
        raise ValueError("peek_row: no dataset has valid indices for the requested streams.")

    def peek_X(self):
        """Convenience helper: peek at the "X" stream. Shape-aligned with the first valid row of "X" from peek_row."""
        row = self.peek_row(require={"X"})
        return row["X"]

    def peek_dX(self):
        """Convenience helper: peek at the "dX" stream. Shape-aligned with the first valid row of "dX" from peek_row."""
        row = self.peek_row(require={"dX"})
        return row["dX"]

    def peek_mask(self):
        """Convenience helper: peek at the "mask" stream.

        Shape-aligned with the first valid row of "mask" from peek_row.
        """
        row = self.peek_row(require={"mask"})
        return row["mask"]

    def peek_dt(self):
        """Convenience helper: peek at the "dt" stream. Shape-aligned with the first valid row of "dt" from peek_row."""
        row = self.peek_row(require={"dt"})
        return row["dt"]

    # ---------- aggregate Teff over datasets ---------- #
    def Teff(self, required: Set[str], *, subsampling: int = 1) -> float:
        """Total effective exposure time across all datasets.

        This is simply the sum of per-dataset Teff values:

            sum_d datasets[d].Teff(required, subsampling=subsampling).
        """
        return float(sum(ds.Teff(required, subsampling=subsampling) for ds in self.datasets))

    # ---------- persistence API ----------
    def save(
        self,
        path: Union[str, Path],
        *,
        format: Optional[str] = None,
        **format_kw: Any,
    ) -> Path:
        """
        Save the collection.

        Rules
        -----
        - Single file path (.csv/.parquet/.h5): collection must have exactly one dataset.
        - Directory path: write one file per dataset + manifest.yaml.
        - Masked samples are dropped (no masked rows written).
        - No relabeling at save-time; relabeling is handled at load-time.
        - ``dynamic_mask`` is not persisted; after a save/load round-trip it
          will be ``None`` (equivalent to the static mask).
        """
        dst = Path(path)

        def _write_one(ds: TrajectoryDataset, filename: Path, fmt: Optional[str]) -> None:
            # flatten and drop masked rows
            X = np.asarray(ds._X3d())
            M = np.asarray(ds._M2d(), dtype=bool)
            pid, tidx, vecs = flatten_X_to_columns(X, mask=M)

            # extras: persist t or dt if present
            eg = dict(ds.extras_global or {})
            el = dict(ds.extras_local or {})
            if ds.t is not None and "t" not in eg:
                eg["t"] = time_series_extra(np.asarray(np.array(ds.t)))
            elif ds.dt is not None and "t" not in eg and "dt" not in eg:
                dta = np.asarray(ds.dt)
                eg["dt"] = time_series_extra(dta) if dta.ndim == 1 else float(dta)

            meta = dict(ds.meta or {})

            save_trajectory(
                str(filename),
                particle_idx=pid,
                time_idx=tidx,
                state_vectors=vecs,
                extras_global=eg,
                extras_local=el,
                metadata=meta,
                format=fmt,
                **format_kw,
            )

        # Case A: single file
        if _is_single_file(dst):
            if len(self.datasets) != 1:
                raise ValueError("Saving to a single file requires exactly one dataset.")
            _write_one(self.datasets[0], dst, format)
            return dst

        # Case B: directory of per-dataset files
        dst.mkdir(parents=True, exist_ok=True)
        fmt = (format or "parquet").lower()
        if fmt not in {"csv", "parquet", "h5"}:
            raise ValueError("format must be 'csv', 'parquet', or 'h5'.")
        ext = {"csv": ".csv", "parquet": ".parquet", "h5": ".h5"}[fmt]

        entries = []
        for i, ds in enumerate(self.datasets):
            rel = Path(f"ds_{i:03d}{ext}")
            _write_one(ds, dst / rel, fmt)
            entries.append({"name": getattr(ds, "name", f"dataset_{i}"), "file": rel.as_posix()})

        import yaml

        manifest = {"version": 1, "n_datasets": len(entries), "datasets": entries}
        (dst / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
        return dst

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        relabel: bool = True,
        compress_particles: bool = False,
        particle_column: Union[int, str, None] = "auto",
        time_column: Union[int, str] = "auto",
        state_columns: Optional[Sequence[Union[int, str]]] = None,
    ) -> "TrajectoryCollection":
        """
        Load a collection from a single file or a directory.

        Parameters
        ----------
        relabel
            If True, compress particle IDs to 0..N-1 and shift time to start at 0.
        compress_particles
            If True, further reduce the column count by merging particles whose
            time supports do not overlap (greedy interval packing with a 2-frame
            buffer).  Useful for open-boundary systems where particles enter and
            leave the field of view, causing the naive N to grow as the total
            number of unique particle IDs rather than the concurrent count.
            Per-particle extras are reindexed automatically; the mapping is
            stored in ``dataset.meta['particle_column_map']``.
        particle_column, time_column
            Which columns hold the particle ID and the time index, as a
            column *name* (any format) or a positional *index* (CSV only).
            ``"auto"`` (default) keeps the loader defaults: CSV positional
            (column 0 = particle, column 1 = time), parquet/HDF5 the
            canonical names ``"particle_id"`` / ``"time_step"``.  Pass
            ``particle_column=None`` for single-trajectory files.
        state_columns
            Optional explicit selection of the state-vector columns
            (names, or indices for CSV), in order; every other non-extras
            column is dropped.  Default: all non-ID, non-extras columns.

        Notes
        -----
        The default weight policy differs by path: a single-file load uses
        ``"Teff"`` (via :meth:`from_dataset`); a directory load uses
        ``"equal"``.  Call :meth:`with_weights` after loading if a consistent
        policy is needed.
        """
        src = Path(path)

        column_kw: Dict[str, Any] = {}
        if particle_column != "auto":
            column_kw["particle_column"] = particle_column
        if time_column != "auto":
            column_kw["time_column"] = time_column
        if state_columns is not None:
            column_kw["state_columns"] = state_columns

        def _read_one(filename: Path) -> TrajectoryDataset:
            yaml_meta, _cols, pid, tidx, vecs, eg, el = load_trajectory(
                str(filename), relabel=relabel, **column_kw
            )
            # Prefer t from extras; else use dt if present in extras or YAML meta
            dt_pass = None
            if "t" not in eg and "dt" in eg:
                v = eg["dt"]
                dt_pass = v.data if isinstance(v, TimeSeriesExtra) else float(v)
            elif "t" not in eg and yaml_meta and "dt" in yaml_meta:
                dt_pass = float(yaml_meta["dt"])
            return columns_and_extras_to_dataset(
                pid,
                tidx,
                vecs,
                extras_global=eg,
                extras_local=el,
                dt=dt_pass,
                relabel=relabel,
                compress_particles=compress_particles,
                meta=yaml_meta,
            )

        # Single file → one dataset
        if _is_single_file(src):
            ds = _read_one(src)
            return cls.from_dataset(ds)

        # Directory → many datasets
        if not src.is_dir():
            raise FileNotFoundError(f"No such file or directory: {src}")

        files: Iterable[Path]
        manifest = src / "manifest.yaml"
        if manifest.exists():
            import yaml

            man = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            files = [src / Path(e["file"]) for e in man.get("datasets", [])]
        else:
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".csv", ".parquet", ".pq", ".h5", ".hdf5"})

        datasets = [_read_one(fp) for fp in files]
        return cls(datasets=datasets, weights=jnp.ones((len(datasets),), dtype=jnp.float32)).with_weights("pool")

    # Constructors:
    @classmethod
    def from_arrays(
        cls,
        *,
        X: Any,
        dt: Optional[float] = None,
        t: Optional[Any] = None,
        mask: Optional[Any] = None,
        extras_global: Optional[Dict[str, Any]] = None,
        extras_local: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        weights: WeightSpec = "pool",
    ) -> "TrajectoryCollection":
        """
        Build a single-dataset collection from array-likes.

        This is the recommended entry point when you already have tensors
        in memory.

        Parameters
        ----------
        X
            State array of shape ``(T, N, d)`` or ``(T, d)``. If ``(T, d)``,
            a single particle is assumed.
        dt
            Either a scalar step, an array of shape ``(T,)`` (per-step),
            or ``None``. If ``None`` and ``t`` is provided, effective steps
            are derived from ``t`` on demand.
        t
            Optional absolute time vector of shape ``(T,)``. If provided, it
            defines time steps when needed.
        mask
            Optional boolean mask of shape ``(T, N)`` or ``(T,)`` marking valid
            observations. If ``None``, all entries are considered valid.
        extras_global
            Mapping of global extras. Values can be static objects,
            :class:`TimeSeriesExtra`, or JAX-traceable callables
            ``f(t_idx, context=None) -> Array`` with a leading time axis.
        extras_local
            Mapping of per-particle extras, with the same typing as
            ``extras_global``. Time-series entries typically have shape
            ``(T, N, ...)``.
        meta
            Free-form metadata dictionary attached to the underlying dataset.
        weights
            Initial weight specification for the resulting collection.
            See :meth:`with_weights`.

        Returns
        -------
        TrajectoryCollection
            A collection with one dataset built from the provided arrays.
        """
        ds = TrajectoryDataset.from_arrays(
            X=X,
            dt=dt,
            t=t,
            mask=mask,
            extras_global=extras_global,
            extras_local=extras_local,
            meta=meta,
        )
        return cls.from_dataset(ds, weights=weights)

    @classmethod
    def from_columns(
        cls,
        particle_idx: np.ndarray,
        time_idx: np.ndarray,
        state_vectors: np.ndarray,
        *,
        extras_global: Mapping[str, Any] | None = None,
        extras_local: Mapping[str, Any] | None = None,
        dt: Optional[float] = None,
        t: Optional[np.ndarray] = None,
        relabel: bool = True,
        compress_particles: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        weights: WeightSpec = "pool",
    ) -> "TrajectoryCollection":
        """
        Build a single-dataset collection from flat (particle, time) columns.

        This constructor is convenient when reading trajectories from a
        tabular format or a custom pipeline.

        Parameters
        ----------
        particle_idx
            Integer array of shape ``(L,)`` with particle IDs for each row.
        time_idx
            Integer array of shape ``(L,)`` with time indices ``t`` for each row.
        state_vectors
            Array of shape ``(L, d)`` with state vectors.
        extras_global
            Parsed global extras (e.g. from YAML header), as described in
            :mod:`SFI.trajectory.io`.
        extras_local
            Parsed local extras, including time-series extras, as described
            in :mod:`SFI.trajectory.io`.
        dt
            Optional scalar step; used only if no absolute time axis is
            provided via ``t`` or ``extras_global['t']``.
        t
            Optional time vector of shape ``(T,)`` overriding any time axis
            inferred from extras.
        relabel
            If True, compress particle IDs to ``0..N-1`` and shift time to
            start at 0.
        compress_particles
            If True, apply greedy interval packing to reduce the column count
            by merging particles whose time supports do not overlap (with a
            2-frame buffer).  Per-particle extras are reindexed automatically.
            The mapping is stored in ``dataset.meta['particle_column_map']``.
        meta
            Metadata dictionary to attach to the dataset.
        weights
            Initial weight specification for the resulting collection.

        Returns
        -------
        TrajectoryCollection
            A collection with one dataset assembled from the columns.
        """
        ds = columns_and_extras_to_dataset(
            particle_idx,
            time_idx,
            state_vectors,
            extras_global=extras_global,
            extras_local=extras_local,
            dt=dt,
            t=t,
            relabel=relabel,
            compress_particles=compress_particles,
            meta=meta,
        )
        return cls.from_dataset(ds, weights=weights)

    #: Column-name candidates tried (case-insensitively) by `from_dataframe`.
    _PARTICLE_COLUMN_CANDIDATES = ("particle_id", "particle", "track_id", "track", "traj_id")
    _TIME_COLUMN_CANDIDATES = ("time_step", "frame", "time", "t")

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        particle: Optional[str] = None,
        time: Optional[str] = None,
        coords: Optional[Sequence[str]] = None,
        dt: Optional[float] = None,
        t: Optional[Any] = None,
        extras_global: Mapping[str, Any] | None = None,
        extras_local: Mapping[str, Any] | None = None,
        relabel: bool = True,
        compress_particles: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        weights: WeightSpec = "pool",
    ) -> "TrajectoryCollection":
        """
        Build a single-dataset collection from a pandas DataFrame.

        The natural entry point for raw tracking tables (trackpy,
        TrackMate, custom pipelines): columns are addressed by *name*, in
        any order, and junk columns are dropped.

        Parameters
        ----------
        df
            A pandas DataFrame with one row per ``(particle, time)``
            observation.
        particle
            Name of the particle/track-ID column.  Default: case-insensitive
            auto-detection among ``particle_id, particle, track_id, track,
            traj_id``; if none is present the table is treated as a single
            trajectory, and if several are present a ``ValueError`` asks
            for an explicit choice.
        time
            Name of the time column.  Default: auto-detection among
            ``time_step, frame, time, t`` (same ambiguity rule).  Integer
            columns are used as frame indices; float columns are
            factorized into frame indices and, unless ``t`` or ``dt`` is
            given, their sorted unique values become the absolute time
            axis.
        coords
            State-vector column names, in order.  Default: every remaining
            column without an extras prefix (``G_``, ``TG_``, ``P_``,
            ``TP_``), in dataframe order.  Columns not selected are
            silently dropped.
        dt, t
            Time-axis specification, as in :meth:`from_columns`.
        extras_global, extras_local
            Extra fields merged **over** any extras parsed from prefixed
            columns (user values win).
        relabel, compress_particles, meta, weights
            As in :meth:`from_columns`.

        Examples
        --------
        >>> coll = TrajectoryCollection.from_dataframe(
        ...     tracks, particle="track_id", time="frame",
        ...     coords=("x", "y"), dt=0.05,
        ... )
        """
        try:
            import pandas as pd  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("TrajectoryCollection.from_dataframe requires pandas.") from e

        colnames = list(df.columns)
        prefixes = ("G_", "TG_", "P_", "TP_")

        def _pick(explicit: Optional[str], candidates: Sequence[str], what: str, required: bool) -> Optional[str]:
            if explicit is not None:
                if explicit not in colnames:
                    raise ValueError(f"{what} column {explicit!r} not found; available columns: {colnames}")
                return explicit
            lower = {c.lower(): c for c in colnames}
            hits = [lower[c] for c in candidates if c in lower]
            if len(hits) > 1:
                raise ValueError(
                    f"ambiguous {what} column — found {hits}; pass {what}= explicitly"
                )
            if hits:
                return hits[0]
            if required:
                raise ValueError(
                    f"no {what} column found (tried {tuple(candidates)}); pass {what}= explicitly"
                )
            return None

        particle_name = _pick(particle, cls._PARTICLE_COLUMN_CANDIDATES, "particle", required=False)
        time_name = _pick(time, cls._TIME_COLUMN_CANDIDATES, "time", required=True)

        if coords is None:
            skip = {particle_name, time_name}
            coord_names = [
                c for c in colnames if c not in skip and not any(c.startswith(p) for p in prefixes)
            ]
        else:
            coord_names = list(coords)
            missing = [c for c in coord_names if c not in colnames]
            if missing:
                raise ValueError(f"coords columns not found: {missing}; available columns: {colnames}")
        if not coord_names:
            raise ValueError("no state (coordinate) columns selected")

        keep = ([particle_name] if particle_name else []) + [time_name] + coord_names
        keep += [c for c in colnames if any(c.startswith(p) for p in prefixes) and c not in keep]
        sub = df[keep].copy()

        # Float time column → factorize to frame indices (+ time axis).
        t_resolved = t
        tvals = np.asarray(sub[time_name].to_numpy())
        if not np.issubdtype(tvals.dtype, np.integer):
            uniq, inv = np.unique(tvals, return_inverse=True)
            if t is None and dt is None:
                t_resolved = uniq
            sub[time_name] = inv.astype(int)

        _meta, _cols, pid, tidx, vecs, eg, el = _parse_tabular_with_extras(
            sub,
            {},
            particle_column=particle_name,
            time_column=time_name,
            relabel=relabel,
        )
        eg = {**eg, **(dict(extras_global) if extras_global else {})}
        el = {**el, **(dict(extras_local) if extras_local else {})}

        return cls.from_columns(
            pid,
            tidx,
            vecs,
            extras_global=eg,
            extras_local=el,
            dt=dt,
            t=t_resolved,
            relabel=relabel,
            compress_particles=compress_particles,
            meta=meta,
            weights=weights,
        )

    def split_time(
        self,
        fraction: float = 0.8,
        *,
        gap: int = 0,
        reweight: Literal["pool", "keep"] = "pool",
    ) -> tuple["TrajectoryCollection", "TrajectoryCollection"]:
        """Split every dataset along time into ``(train, test)`` collections.

        A side feature for data-abundant scenarios: SFI estimates its own
        accuracy from the training data (``force_predicted_MSE``) and
        validates fits through the diagnostics suite, neither of which
        costs any data.  Hold out a test fraction only when data is
        plentiful, or to confirm a suspected bias floor with
        :meth:`~SFI.inference.base.BaseLangevinInference.holdout_score`.

        Parameters
        ----------
        fraction : float
            Fraction of frames per dataset assigned to the train half.
        gap : int
            Frames dropped between the halves (decorrelation; ``0`` is
            safe for increment-based estimators).
        reweight : {"Teff", "keep"}
            ``"Teff"`` (default) recomputes per-dataset weights on each
            half; ``"keep"`` carries over the current relative weights.

        Examples
        --------
        >>> train, test = coll.split_time(0.8)
        >>> inf = OverdampedLangevinInference(train)
        >>> # ... fit ...
        >>> inf.holdout_score(test)
        """
        pairs = [ds.split_time(fraction, gap=gap) for ds in self.datasets]
        train_ds = [p[0] for p in pairs]
        test_ds = [p[1] for p in pairs]
        n = len(self.datasets)
        spec: WeightSpec = "pool" if reweight == "pool" else np.asarray(self.weights)
        train = type(self)(datasets=train_ds, weights=jnp.ones((n,), dtype=jnp.float32)).with_weights(spec)
        test = type(self)(datasets=test_ds, weights=jnp.ones((n,), dtype=jnp.float32)).with_weights(spec)
        return train, test

    def dataset_index(self, position: int) -> int:
        """Dense index of dataset ``position``, keyed on its stable identity.

        Datasets are numbered by first appearance of their ``uuid``, so the
        index a force sees (e.g. via :func:`~SFI.bases.per_dataset_scalar` or
        :func:`~SFI.bases.dataset_indicator`) is tied to the dataset itself, not
        its slot — stable under concatenation and reordering.
        """
        order: Dict[str, int] = {}
        for ds in self.datasets:
            order.setdefault(ds.uuid, len(order))
        return order[self.datasets[position].uuid]

    def degrade(
        self,
        *,
        downsample: int = 1,
        motion_blur: int = 0,
        data_loss_fraction: float = 0.0,
        noise: Union[None, float, np.ndarray] = None,
        ROI: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]] = None,
        seed: Optional[int] = None,
        reweight: Literal["pool", "keep"] = "pool",
    ) -> "TrajectoryCollection":
        """
        Return a new degraded collection; the original is not modified.

        This is the preferred user-facing API for degrading synthetic
        trajectories to mimic experimental noise, blur, and data loss.

        Parameters
        ----------
        downsample, motion_blur, data_loss_fraction, noise, ROI, seed, reweight
            See :func:`SFI.trajectory.degrade.degrade_collection` for a full
            description of each parameter.

        Returns
        -------
        TrajectoryCollection
            New degraded collection. The original collection is not modified.
        """
        from SFI.trajectory.degrade import degrade_collection

        return degrade_collection(
            self,
            downsample=downsample,
            motion_blur=motion_blur,
            data_loss_fraction=data_loss_fraction,
            noise=noise,
            ROI=ROI,
            seed=seed,
            reweight=reweight,
        )

    def to_arrays(
        self,
        *,
        dataset: int = 0,
        as_numpy: bool = True,
        include_mask: bool = True,
    ):
        """
        Convenience helper: materialize one dataset as dense arrays.

        Parameters
        ----------
        dataset :
            Index of the dataset inside the collection (default 0).
        as_numpy :
            If True, return NumPy arrays.
        include_mask :
            If True, also return the per-particle mask.

        Returns
        -------
        t, X, mask :
            See :meth:`TrajectoryDataset.to_arrays`.
        """
        if not (0 <= dataset < len(self.datasets)):
            raise IndexError(f"dataset index {dataset} out of range for D={len(self.datasets)}")
        return self.datasets[dataset].to_arrays(
            as_numpy=as_numpy,
            include_mask=include_mask,
        )

    def merge(
        self,
        items: Sequence[Union["TrajectoryCollection", TrajectoryDataset]],
        *,
        weights: WeightSpec = "pool",
    ) -> "TrajectoryCollection":
        """Combine this collection with others into one collection.

        Convenience alias for :meth:`concat` — useful for assembling an
        ensemble from several single-trajectory collections
        (``base.merge([c1, c2, ...])``). See :meth:`concat` for the
        ``weights`` policy.
        """
        return self.concat(items, weights=weights)

    def to_array(self, *, axis: Literal["time"] = "time", as_numpy: bool = True):
        """Materialize the whole collection as one dense ``(T, N, d)`` array.

        Concatenates every dataset along the time axis into a single array
        of positions.  Use this for the legitimate non-plotting reach-ins
        (disk caching, ensemble bootstrap initial conditions, neighbour
        lists); for plotting, prefer the toolkit functions in
        :mod:`SFI.utils.plotting`, and for ``(t, X, mask)`` of a single
        dataset use :meth:`to_arrays`.

        Parameters
        ----------
        axis :
            Only ``"time"`` is supported (axis-0 concatenation).
        as_numpy :
            If True (default), return a NumPy array; else a JAX array.

        Returns
        -------
        ndarray, shape ``(sum_T, N, d)``
        """
        if axis != "time":
            raise ValueError(f"to_array only supports axis='time', got {axis!r}.")
        if not self.datasets:
            raise ValueError("Empty TrajectoryCollection.")
        out = jnp.concatenate([ds._X3d() for ds in self.datasets], axis=0)
        return np.asarray(out) if as_numpy else out

    def velocity_array(
        self,
        *,
        dataset: int = 0,
        scheme: Literal["central", "forward", "backward"] = "central",
        as_numpy: bool = True,
    ):
        """Finite-difference velocity ``v(t)`` for one dataset.

        Reconstructs velocities from stored positions with
        :func:`SFI.utils.maths.fd_velocity`, matching the secant-velocity
        convention of the underdamped engine.  Handy for building
        ``(x, v)`` phase portraits or held-out evaluation grids from
        position-only recordings.

        Parameters
        ----------
        dataset :
            Dataset index inside the collection (default 0).
        scheme :
            Finite-difference stencil; see :func:`SFI.utils.maths.fd_velocity`.
        as_numpy :
            If True (default), return a NumPy array; else a JAX array.

        Returns
        -------
        v : ndarray, shape ``(T, N, d)``
        """
        from SFI.utils.maths import fd_velocity

        t, X, _ = self.to_arrays(dataset=dataset, as_numpy=True, include_mask=True)
        dt = np.diff(np.asarray(t, dtype=float))
        if dt.size == 0:
            raise ValueError("velocity_array needs at least 2 frames.")
        v = fd_velocity(X, dt, scheme=scheme)
        return np.asarray(v) if as_numpy else v

    # ------------------------------------------------------------------
    # Attribute forwarding for the most common case: a single dataset
    # ------------------------------------------------------------------
    def _single_dataset(self):
        """Return the unique dataset if the collection has exactly one, else None."""
        if len(self.datasets) == 1:
            return self.datasets[0]
        return None

    def __getattr__(self, name):
        """
        Forward attribute access to the sole dataset when exactly one is present.
        Does not intercept existing attributes on the collection itself.
        """
        # Guard: during pickle/unpickle, __dict__ may not yet contain
        # 'datasets', so _single_dataset() would recurse back here.
        if name == "datasets":
            raise AttributeError(name)
        ds = self._single_dataset()
        if ds is not None and hasattr(ds, name):
            return getattr(ds, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __getitem__(self, key):
        """
        Optional: dict-style forwarding for convenience.
        Example: coll["extras_global"].
        """
        ds = self._single_dataset()
        if ds is not None and hasattr(ds, key):
            return getattr(ds, key)
        raise KeyError(key)

    def __dir__(self):
        """
        Extend tab completion: if a single dataset is present,
        expose its attributes as if they were part of the collection.
        """
        base = super().__dir__()
        ds = self._single_dataset()
        if ds is not None:
            return sorted(set(base) | set(dir(ds)))
        return base
