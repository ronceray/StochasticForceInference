"""Cell-list neighbor builder for truncated-range pair interactions.

Provides :func:`build_neighbor_csr` which constructs a sparse CSR
neighbor list from particle positions and a cutoff radius, using
``scipy.spatial.cKDTree``.  The returned ``(indptr, indices)`` arrays
plug directly into ``dispatch_pairs_from_extras``.

All routines run on the host (pure NumPy) and are meant to be called
*between* JIT-compiled simulation chunks, not inside ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree


def build_neighbor_csr(
    positions: np.ndarray,
    cutoff: float,
    box: Optional[np.ndarray] = None,
    *,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a CSR neighbor list using ``scipy.spatial.cKDTree``.

    Parameters
    ----------
    positions : ndarray, shape ``(N, d)``
        Particle positions (spatial coordinates only).
    cutoff : float
        Cutoff radius.  Pairs with ``r_ij > cutoff`` are excluded.
    box : ndarray, shape ``(d,)``, optional
        Periodic box lengths.  If *None*, open (non-periodic) boundaries.
    exclude_self : bool
        If *True* (default), self-pairs ``(i, i)`` are never included.

    Returns
    -------
    indptr : ndarray, shape ``(N + 1,)``, dtype int32
        CSR row pointers.
    indices : ndarray, shape ``(nnz,)``, dtype int32
        CSR column indices (neighbour particle indices).
    """
    positions = np.asarray(positions, dtype=np.float64)
    N, d = positions.shape

    if N == 0:
        return (
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    # --- wrap positions into the primary box ---
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        positions = positions % box

    # --- build KD-tree and query pairs ---
    boxsize = box if box is not None else None
    tree = cKDTree(positions, boxsize=boxsize)
    csr = tree.sparse_distance_matrix(tree, cutoff, output_type="coo_matrix")
    csr = csr.tocsr()

    if exclude_self:
        csr.setdiag(0)
        csr.eliminate_zeros()

    indptr = csr.indptr.astype(np.int32)
    indices = csr.indices.astype(np.int32)

    return indptr, indices


def make_neighbor_extras(
    positions: np.ndarray,
    cutoff: float,
    box: Optional[np.ndarray] = None,
    *,
    indptr_key: str = "indptr",
    indices_key: str = "indices",
    exclude_self: bool = True,
) -> dict:
    """Build a CSR neighbor list and return it as an extras dict.

    Convenience wrapper around :func:`build_neighbor_csr`.  The returned
    dict is ready to be merged into ``extras_global`` for a process that
    uses ``dispatch_pairs_from_extras(indptr_key, indices_key)``.

    Parameters
    ----------
    positions, cutoff, box, exclude_self
        Forwarded to :func:`build_neighbor_csr`.
    indptr_key, indices_key
        Keys under which CSR arrays are stored.

    Returns
    -------
    dict
        ``{indptr_key: indptr, indices_key: indices}``
    """
    indptr, indices = build_neighbor_csr(
        positions,
        cutoff,
        box,
        exclude_self=exclude_self,
    )
    return {
        indptr_key: jnp.array(indptr),
        indices_key: jnp.array(indices),
    }


def pad_neighbor_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    max_nnz: int,
    *,
    fill_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a CSR neighbor list to a fixed ``max_nnz``.

    JAX JIT recompiles when array shapes change.  Padding the indices
    array to a fixed length avoids recompilation across simulation
    chunks with fluctuating neighbor counts.

    Excess entries are filled with ``fill_index`` (default 0).  Because
    ``indptr`` keeps the true lengths, the dispatcher will only iterate
    over the real neighbours — the padded entries are never evaluated.

    .. note::
       This only pads ``indices``.  ``indptr`` is left unchanged (always
       ``N + 1`` long).  If the actual nnz exceeds ``max_nnz``, a
       ``ValueError`` is raised.

    Parameters
    ----------
    indptr, indices
        As returned by :func:`build_neighbor_csr`.
    max_nnz : int
        Target length for ``indices``.
    fill_index : int
        Index used to fill padded entries.

    Returns
    -------
    indptr, indices_padded
        Same ``indptr``, padded ``indices`` of length ``max_nnz``.
    """
    nnz = len(indices)
    if nnz > max_nnz:
        raise ValueError(
            f"Actual nnz ({nnz}) exceeds max_nnz ({max_nnz}). Increase max_nnz or enlarge the cutoff safety margin."
        )
    if nnz == max_nnz:
        return indptr, indices
    padded = np.full(max_nnz, fill_index, dtype=indices.dtype)
    padded[:nnz] = indices
    return indptr, padded
