# TODO: review this file
"""Tests for SFI.utils.neighbors."""

import numpy as np
import pytest

from SFI.utils.neighbors import build_neighbor_csr, make_neighbor_extras, pad_neighbor_csr


# ---------- helpers ----------

def _brute_force_neighbors(positions, cutoff, box=None, exclude_self=True):
    """O(N²) reference neighbor list."""
    N = len(positions)
    neighbors = [[] for _ in range(N)]
    cutoff2 = cutoff * cutoff
    for i in range(N):
        for j in range(N):
            if exclude_self and i == j:
                continue
            dx = positions[j] - positions[i]
            if box is not None:
                dx = dx - box * np.round(dx / box)
            if (dx * dx).sum() <= cutoff2:
                neighbors[i].append(j)
    return neighbors


def _csr_to_lists(indptr, indices):
    """Convert CSR to list-of-lists for easy comparison."""
    N = len(indptr) - 1
    return [sorted(indices[indptr[i]:indptr[i + 1]].tolist()) for i in range(N)]


# ---------- build_neighbor_csr ----------

class TestBuildNeighborCSR:
    def test_basic_square(self):
        """Four particles on a square, cutoff includes nearest neighbors."""
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        cutoff = 1.01  # includes edges but not diagonals
        indptr, indices = build_neighbor_csr(positions, cutoff)
        nbrs = _csr_to_lists(indptr, indices)
        assert nbrs[0] == [1, 3]
        assert nbrs[1] == [0, 2]
        assert nbrs[2] == [1, 3]
        assert nbrs[3] == [0, 2]

    def test_vs_brute_force(self):
        """Cell-list matches brute-force on random positions."""
        rng = np.random.default_rng(42)
        N = 100
        positions = rng.uniform(0, 10, (N, 2))
        cutoff = 2.5
        box = np.array([10.0, 10.0])

        indptr, indices = build_neighbor_csr(positions, cutoff, box)
        cell_nbrs = _csr_to_lists(indptr, indices)
        ref_nbrs = _brute_force_neighbors(positions, cutoff, box)
        ref_nbrs = [sorted(nb) for nb in ref_nbrs]

        for i in range(N):
            assert cell_nbrs[i] == ref_nbrs[i], f"Mismatch at particle {i}"

    def test_vs_brute_force_3d(self):
        """Works in 3D with PBC."""
        rng = np.random.default_rng(123)
        N = 50
        positions = rng.uniform(0, 8, (N, 3))
        cutoff = 2.0
        box = np.array([8.0, 8.0, 8.0])

        indptr, indices = build_neighbor_csr(positions, cutoff, box)
        cell_nbrs = _csr_to_lists(indptr, indices)
        ref_nbrs = [sorted(nb) for nb in _brute_force_neighbors(positions, cutoff, box)]

        for i in range(N):
            assert cell_nbrs[i] == ref_nbrs[i], f"Mismatch at particle {i}"

    def test_pbc_wrapping(self):
        """Two particles near opposite edges should be neighbors with PBC."""
        positions = np.array([
            [0.1, 5.0],
            [9.9, 5.0],
        ])
        box = np.array([10.0, 10.0])
        cutoff = 0.5
        indptr, indices = build_neighbor_csr(positions, cutoff, box)
        nbrs = _csr_to_lists(indptr, indices)
        assert nbrs[0] == [1]
        assert nbrs[1] == [0]

    def test_no_pbc(self):
        """Without PBC, particles near opposite edges are NOT neighbors."""
        positions = np.array([
            [0.1, 5.0],
            [9.9, 5.0],
        ])
        cutoff = 0.5
        indptr, indices = build_neighbor_csr(positions, cutoff)
        nbrs = _csr_to_lists(indptr, indices)
        assert nbrs[0] == []
        assert nbrs[1] == []

    def test_empty(self):
        """N=0 returns valid empty CSR."""
        positions = np.empty((0, 2))
        indptr, indices = build_neighbor_csr(positions, 1.0)
        assert indptr.shape == (1,)
        assert indices.shape == (0,)

    def test_single_particle(self):
        """Single particle with exclude_self has no neighbors."""
        positions = np.array([[5.0, 5.0]])
        indptr, indices = build_neighbor_csr(positions, 1.0, np.array([10.0, 10.0]))
        assert _csr_to_lists(indptr, indices) == [[]]

    def test_include_self(self):
        """With exclude_self=False, self-pairs appear."""
        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        box = np.array([10.0, 10.0])
        cutoff = 1.0
        indptr, indices = build_neighbor_csr(positions, cutoff, box, exclude_self=False)
        nbrs = _csr_to_lists(indptr, indices)
        assert 0 in nbrs[0]  # self-pair
        assert 1 in nbrs[1]  # self-pair


# ---------- pad_neighbor_csr ----------

class TestPadNeighborCSR:
    def test_padding(self):
        indptr = np.array([0, 2, 5], dtype=np.int32)
        indices = np.array([1, 2, 0, 3, 4], dtype=np.int32)
        ip, ip_idx = pad_neighbor_csr(indptr, indices, 10)
        assert np.array_equal(ip, indptr)
        assert len(ip_idx) == 10
        assert np.array_equal(ip_idx[:5], indices)
        assert np.all(ip_idx[5:] == 0)

    def test_exact_size(self):
        indptr = np.array([0, 2], dtype=np.int32)
        indices = np.array([3, 4], dtype=np.int32)
        ip, ip_idx = pad_neighbor_csr(indptr, indices, 2)
        assert np.array_equal(ip_idx, indices)

    def test_overflow_raises(self):
        indptr = np.array([0, 3], dtype=np.int32)
        indices = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError, match="exceeds max_nnz"):
            pad_neighbor_csr(indptr, indices, 2)


# ---------- make_neighbor_extras ----------

class TestMakeNeighborExtras:
    def test_keys_and_types(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        extras = make_neighbor_extras(positions, cutoff=1.5, box=np.array([10.0, 10.0]))
        assert "indptr" in extras
        assert "indices" in extras
        # Should be JAX arrays
        import jax.numpy as jnp
        assert hasattr(extras["indptr"], "shape")
        assert extras["indptr"].dtype == jnp.int32

    def test_custom_keys(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        extras = make_neighbor_extras(
            positions, cutoff=2.0,
            indptr_key="my_ip", indices_key="my_idx",
        )
        assert "my_ip" in extras
        assert "my_idx" in extras
