"""Chunked simulation with periodic neighbor-list rebuilds.

Provides :func:`simulate_chunked`, a thin wrapper around
:meth:`OverdampedProcess.simulate` that breaks a long simulation into
shorter chunks and rebuilds the CSR neighbor list (via
:func:`~SFI.utils.neighbors.build_neighbor_csr`) between chunks.

This avoids the O(N²) cost of ``AutoPairs`` for large particle systems
while keeping the extras interface exactly as-is.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from SFI.trajectory.collection import TrajectoryCollection
from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr


def simulate_chunked(
    proc,
    dt: float,
    Nsteps: int,
    key,
    *,
    cutoff: float,
    box: np.ndarray,
    skin: float = 0.0,
    rebuild_every: int = 50,
    save_every: Optional[int] = None,
    spatial_dims: slice = slice(None, 2),
    indptr_key: str = "indptr",
    indices_key: str = "indices",
    nnz_safety: float = 1.25,
    oversampling: int = 1,
    prerun: int = 0,
    compute_observables: bool = False,
    jit_compile: bool = True,
    verbose: bool = False,
) -> TrajectoryCollection:
    """Run a chunked overdamped simulation with periodic neighbor rebuilds.

    Parameters
    ----------
    proc : OverdampedProcess
        An initialized process whose force uses
        ``dispatch_pairs_from_extras(indptr_key, indices_key)``.
    dt : float
        Time step.
    Nsteps : int
        Total number of steps.
    key : jax PRNG key
        Random key for the simulation.
    cutoff : float
        Cutoff radius for neighbor list construction.
    box : array-like, shape ``(d,)``
        Periodic box lengths.
    skin : float
        Verlet skin width.  The neighbor list is built with radius
        ``cutoff + skin`` so that particles drifting into range between
        rebuilds are already included.  After each chunk the maximum
        particle displacement is checked; a warning is printed if it
        exceeds ``skin / 2`` (the Verlet safety threshold).
    rebuild_every : int
        Number of simulation steps between neighbor-list rebuilds.
    save_every : int, optional
        Number of simulation steps per output dataset.  If *None*,
        defaults to ``rebuild_every``.  Must be a multiple of
        ``rebuild_every``.
    spatial_dims : slice
        Slice into the state vector that selects spatial coordinates
        (default: first two components ``[:2]``).
    indptr_key, indices_key : str
        Extras keys for the CSR neighbor list.
    nnz_safety : float
        Fraction by which ``max_nnz`` is enlarged beyond the initial
        neighbor count to absorb fluctuations (default 1.25 = 25%).
    oversampling, prerun, compute_observables, jit_compile
        Forwarded to ``proc.simulate()``.
    verbose : bool
        Print progress info.

    Returns
    -------
    TrajectoryCollection
        Concatenated trajectory from all chunks.
    """
    # Time-dependent extras are not supported here: simulate() is re-invoked
    # per rebuild chunk, which would mis-slice frame-aligned schedules.
    from SFI.trajectory.dataset import TimeSeriesExtra

    for src in (proc.extras_global, proc.extras_local):
        for k, v in (src or {}).items():
            if isinstance(v, TimeSeriesExtra) or (callable(v) and not hasattr(v, "func")):
                raise NotImplementedError(
                    f"simulate_chunked does not support time-dependent extras (got {k!r}); "
                    "use proc.simulate() directly."
                )

    box = np.asarray(box, dtype=np.float64)
    cutoff_list = cutoff + skin  # Verlet list radius

    if save_every is None:
        save_every = rebuild_every
    if save_every % rebuild_every != 0:
        raise ValueError(f"save_every ({save_every}) must be a multiple of rebuild_every ({rebuild_every})")
    rebuilds_per_save = save_every // rebuild_every

    # --- initial neighbor list ---
    positions = np.asarray(proc._x)  # (P, d) or (d,)
    # Assumes layout (P, d) for ndim==2 or (d,) for ndim==1; (d, P) would silently mislabel.
    pos_spatial = positions[:, spatial_dims] if positions.ndim == 2 else positions[spatial_dims]
    indptr, indices = build_neighbor_csr(pos_spatial, cutoff_list, box)
    pos_at_rebuild = pos_spatial.copy()  # track displacements

    # Fixed max_nnz with safety margin
    max_nnz = max(int(len(indices) * nnz_safety), len(indices) + 1)
    indptr_pad, indices_pad = pad_neighbor_csr(indptr, indices, max_nnz)

    # Merge CSR into existing extras (don't clobber other keys)
    base_extras = dict(proc.extras_global or {})
    base_extras[indptr_key] = jnp.array(indptr_pad)
    base_extras[indices_key] = jnp.array(indices_pad)
    proc.set_extras(extras_global=base_extras)

    # --- chunk the simulation ---
    n_rebuilds = max(1, (Nsteps + rebuild_every - 1) // rebuild_every)
    remaining = Nsteps
    collections = []  # final output datasets
    sub_collections = []  # accumulate rebuild-chunks within a save-chunk
    step_done = 0
    max_disp_in_save = 0.0  # track across rebuilds within a save-chunk

    for rebuild_i in range(n_rebuilds):
        chunk_steps = min(rebuild_every, remaining)
        if chunk_steps <= 0:
            break

        key, sub_key = jax.random.split(key)

        coll = proc.simulate(
            dt,
            Nsteps=chunk_steps,
            key=sub_key,
            oversampling=oversampling,
            prerun=prerun if rebuild_i == 0 else 0,
            compute_observables=compute_observables,
            jit_compile=jit_compile,
        )
        sub_collections.append(coll)
        remaining -= chunk_steps
        step_done += chunk_steps

        # Emit a save-chunk when we've accumulated enough rebuilds
        if (rebuild_i + 1) % rebuilds_per_save == 0 or remaining <= 0:
            # Merge sub-collection X arrays into one contiguous dataset.
            # Each sub-collection has one dataset with X shape (T_sub, N, d)
            # where T_sub = rebuild_every (no duplicate frames).
            X_parts = [np.asarray(sc.datasets[0].X) for sc in sub_collections]
            X_merged = np.concatenate(X_parts, axis=0)
            if verbose:
                print(f"    merged {len(sub_collections)} sub-chunks → X shape {X_merged.shape}")
            merged = TrajectoryCollection.from_arrays(
                X=X_merged,
                dt=dt,
            )
            collections.append(merged)
            sub_collections = []

            if verbose:
                print(f"  chunk {len(collections)}: {step_done}/{Nsteps} steps")

        # Rebuild neighbors from updated positions
        if remaining > 0:
            positions = np.asarray(proc._x)  # (P, d) or (d,)
            pos_spatial = positions[:, spatial_dims] if positions.ndim == 2 else positions[spatial_dims]

            # --- Verlet displacement check ---
            disp = pos_spatial - pos_at_rebuild
            if box is not None:
                disp = disp - box * np.round(disp / box)
            max_disp = float(np.sqrt((disp * disp).sum(axis=-1)).max())
            max_disp_in_save = max(max_disp_in_save, max_disp)
            if skin > 0 and max_disp > skin / 2:
                import warnings

                warnings.warn(
                    f"Rebuild {rebuild_i}: max displacement {max_disp:.3f} "
                    f"> skin/2 = {skin / 2:.3f}.  Neighbor list may have "
                    f"missed interactions.  Increase skin or decrease "
                    f"rebuild_every.",
                    stacklevel=2,
                )
            if verbose and (rebuild_i + 1) % rebuilds_per_save == 0:
                print(f"    max displacement = {max_disp_in_save:.3f}  (skin/2 = {skin / 2:.3f})")
                max_disp_in_save = 0.0

            indptr, indices = build_neighbor_csr(pos_spatial, cutoff_list, box)
            pos_at_rebuild = pos_spatial.copy()

            # Grow max_nnz if needed
            if len(indices) > max_nnz:
                max_nnz = int(len(indices) * nnz_safety)
                if verbose:
                    print(f"    max_nnz grew to {max_nnz}")

            indptr_pad, indices_pad = pad_neighbor_csr(indptr, indices, max_nnz)

            # Update extras, preserving all non-CSR keys
            updated_extras = dict(proc.extras_global or {})
            updated_extras[indptr_key] = jnp.array(indptr_pad)
            updated_extras[indices_key] = jnp.array(indices_pad)
            proc.set_extras(extras_global=updated_extras)

            # Free old JIT caches to prevent GPU memory buildup
            jax.clear_caches()

    # --- concatenate chunks ---
    if len(collections) == 1:
        return collections[0]
    return collections[0].concat(collections[1:], weights="pool")
