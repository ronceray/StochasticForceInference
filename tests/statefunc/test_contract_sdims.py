# TODO: review this file
# tests/statefunc/test_contract_sdims.py
"""Tests for the sdims extension to the outer contract system."""

import jax.numpy as jnp
import numpy as np
import pytest

from SFI.statefunc import Rank, StateExpr, make_basis
from SFI.statefunc.nodes.contract import _ContractMixin
from SFI.statefunc.nodes.ops.concat import ConcatNode
from SFI.statefunc.nodes.ops.einsum import EinsumNode
from SFI.statefunc.nodes.ops.reshape_rank import ReshapeRankNode


# ---------------------------------------------------------------------------
#  Mock node for unit-testing merge_contract / inherit_contract
# ---------------------------------------------------------------------------
def _mock(rank=0, dim=None, n_features=1, sdims=None, pdepth=0, particles_input=False):
    """Return a minimal Basis node with the given contract fields.

    Sets sdims on the root node of the Basis so merge_contract sees it.
    """
    r = int(rank)
    # Build a function that produces the right output shape
    if sdims is not None:
        rank_shape = tuple(sdims)
    elif dim is not None:
        rank_shape = (dim,) * r
    else:
        rank_shape = ()

    def f(x, **_):
        return jnp.ones(rank_shape + (n_features,))

    b = make_basis(f, dim=dim, rank=r, n_features=n_features)
    # Set sdims on the root node (BaseNode)
    object.__setattr__(b.root, "sdims", sdims)
    return b


# ═══════════════════════════════════════════════════════════════════════════════
#  1. inherit_contract propagates sdims
# ═══════════════════════════════════════════════════════════════════════════════
class TestInheritContract:
    def test_inherits_sdims(self):
        m = _mock(rank=2, dim=3, sdims=(3, 3))
        d = _ContractMixin.inherit_contract(m.root)
        assert d["sdims"] == (3, 3)

    def test_inherits_none_sdims(self):
        m = _mock(rank=1, dim=3)
        d = _ContractMixin.inherit_contract(m.root)
        assert d["sdims"] is None

    def test_override_sdims(self):
        m = _mock(rank=1, dim=3)
        d = _ContractMixin.inherit_contract(m.root, sdims=(5,))
        assert d["sdims"] == (5,)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. merge_contract — concat mode
# ═══════════════════════════════════════════════════════════════════════════════
class TestMergeContractConcat:
    def test_concat_none_sdims(self):
        """When all children have sdims=None, output sdims is None."""
        a = _mock(rank=1, dim=3, n_features=2)
        b = _mock(rank=1, dim=3, n_features=3)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="concat")
        assert d["sdims"] is None
        assert d["n_features"] == 5

    def test_concat_matching_sdims(self):
        a = _mock(rank=2, dim=3, sdims=(3, 5), n_features=1)
        b = _mock(rank=2, dim=3, sdims=(3, 5), n_features=2)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="concat")
        assert d["sdims"] == (3, 5)
        assert d["n_features"] == 3

    def test_concat_mismatched_sdims_error(self):
        a = _mock(rank=2, dim=3, sdims=(3, 5), n_features=1)
        b = _mock(rank=2, dim=3, sdims=(3, 4), n_features=1)
        with pytest.raises(ValueError, match="disagree on sdims"):
            _ContractMixin.merge_contract([a.root, b.root], mode="concat")

    def test_concat_mixed_none_sdims_error(self):
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, n_features=1)
        with pytest.raises(ValueError, match="sdims=None"):
            _ContractMixin.merge_contract([a.root, b.root], mode="concat")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. merge_contract — map mode
# ═══════════════════════════════════════════════════════════════════════════════
class TestMergeContractMap:
    def test_map_none_sdims(self):
        a = _mock(rank=1, dim=3, n_features=2)
        b = _mock(rank=1, dim=3, n_features=2)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="map")
        assert d["sdims"] is None

    def test_map_matching_sdims(self):
        a = _mock(rank=2, dim=3, sdims=(3, 5), n_features=2)
        b = _mock(rank=2, dim=3, sdims=(3, 5), n_features=2)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="map")
        assert d["sdims"] == (3, 5)

    def test_map_mismatched_sdims_error(self):
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, sdims=(5,), n_features=1)
        with pytest.raises(ValueError, match="disagree on sdims"):
            _ContractMixin.merge_contract([a.root, b.root], mode="map")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. merge_contract — einsum mode
# ═══════════════════════════════════════════════════════════════════════════════
class TestMergeContractEinsum:
    def test_einsum_none_sdims(self):
        """When no child has sdims, output sdims is None (backward compat)."""
        a = _mock(rank=1, dim=3, n_features=1)
        b = _mock(rank=1, dim=3, n_features=1)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,n->mn")
        assert d["sdims"] is None
        assert d["rank"] == Rank.MATRIX

    def test_einsum_with_sdims_outer_product(self):
        """i(3) × j(5) → ij gives sdims=(3,5)."""
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, sdims=(5,), n_features=1)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,n->mn")
        assert d["sdims"] == (3, 5)
        assert d["rank"] == Rank.MATRIX

    def test_einsum_with_sdims_contraction(self):
        """m(3) · m(3) → scalar, sdims=()."""
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,m->")
        assert d["sdims"] == ()
        assert d["rank"] == Rank.SCALAR

    def test_einsum_with_sdims_mixed_contraction(self):
        """mn(3,5) × n(5) → m(3) gives sdims=(3,)."""
        a = _mock(rank=2, dim=3, sdims=(3, 5), n_features=1)
        b = _mock(rank=1, dim=3, sdims=(5,), n_features=1)
        d = _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="mn,n->m")
        assert d["sdims"] == (3,)

    def test_einsum_inconsistent_letter_size_error(self):
        """Letter 'm' has size 3 in one child, 5 in another → error."""
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, sdims=(5,), n_features=1)
        with pytest.raises(ValueError, match="inconsistent sizes"):
            _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,m->")

    def test_einsum_mixed_none_and_sdims(self):
        """One child has sdims, other has None but known dim → uniform dim used."""
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=3, n_features=1)  # sdims=None, dim=3
        d = _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,n->mn")
        assert d["sdims"] == (3, 3)

    def test_einsum_mixed_none_no_dim_error(self):
        """One child has sdims, other has None and dim=None → error."""
        a = _mock(rank=1, dim=3, sdims=(3,), n_features=1)
        b = _mock(rank=1, dim=None, n_features=1)  # sdims=None, dim=None
        with pytest.raises(ValueError, match="dim=None"):
            _ContractMixin.merge_contract([a.root, b.root], mode="einsum", spec="m,n->mn")

    def test_einsum_scalar_times_vector_with_sdims(self):
        """,n→n: scalar(no axes) × vector(5) → vector sdims=(5,)."""
        s = _mock(rank=0, dim=3, sdims=(), n_features=1)
        v = _mock(rank=1, dim=3, sdims=(5,), n_features=1)
        d = _ContractMixin.merge_contract([s.root, v.root], mode="einsum", spec=",n->n")
        assert d["sdims"] == (5,)
        assert d["rank"] == Rank.VECTOR


# ═══════════════════════════════════════════════════════════════════════════════
#  5. _assert_outputs with sdims
# ═══════════════════════════════════════════════════════════════════════════════
class TestAssertOutputsSdims:
    def test_non_uniform_rank_accepted(self):
        """Node with sdims=(3, 5) accepts output with rank axes (3, 5)."""
        m = _mock(rank=2, dim=3, sdims=(3, 5), n_features=2)
        x = jnp.zeros((4, 3))
        y = jnp.zeros((4, 3, 5, 2))
        m.root._assert_outputs(x, y)  # should not raise

    def test_non_uniform_rank_wrong_shape_rejected(self):
        """Node with sdims=(3, 5) rejects output with rank axes (3, 3)."""
        m = _mock(rank=2, dim=3, sdims=(3, 5), n_features=2)
        x = jnp.zeros((4, 3))
        y = jnp.zeros((4, 3, 3, 2))
        with pytest.raises(ValueError, match="rank axes shape"):
            m.root._assert_outputs(x, y)

    def test_uniform_sdims_matches_dim(self):
        """Node with sdims=(3, 3) and dim=3 accepts the right shape."""
        m = _mock(rank=2, dim=3, sdims=(3, 3), n_features=1)
        x = jnp.zeros((2, 3))
        y = jnp.zeros((2, 3, 3, 1))
        m.root._assert_outputs(x, y)  # should not raise

    def test_scalar_sdims_empty(self):
        """Scalar node with sdims=() works fine."""
        m = _mock(rank=0, dim=3, sdims=(), n_features=4)
        x = jnp.zeros((2, 3))
        y = jnp.zeros((2, 4))
        m.root._assert_outputs(x, y)  # should not raise

    def test_none_sdims_still_uses_dim(self):
        """Classic path: sdims=None falls back to (dim,)*rank validation."""
        m = _mock(rank=1, dim=3, n_features=2)
        x = jnp.zeros((4, 3))
        y_ok = jnp.zeros((4, 3, 2))
        m.root._assert_outputs(x, y_ok)
        y_bad = jnp.zeros((4, 5, 2))
        with pytest.raises(ValueError, match="rank axes shape"):
            m.root._assert_outputs(x, y_bad)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. ReshapeRankNode with sdims
# ═══════════════════════════════════════════════════════════════════════════════
class TestReshapeRankSdims:
    def test_fold_with_sdims(self):
        """Fold rank-2 sdims=(3, 5) to rank-0 → n_features = 1 * 3 * 5 = 15."""
        # Need a node that outputs (..., 3, 5, 1)
        def f(x, **_):
            return jnp.ones((3, 5, 1))

        child = make_basis(f, dim=2, rank=2, n_features=1)
        # Set sdims on the root node (leaf)
        object.__setattr__(child.root, "sdims", (3, 5))

        node = ReshapeRankNode(child.root, target_rank=0)
        assert node.rank == Rank.SCALAR
        assert node.n_features == 15
        # Scalar fold: sdims is None (no rank axes left, but we track it)
        # Actually: when we fold to scalar, out_sdims = axis_sizes[:0] = () if sdims was not None
        # Wait, let's check: `axis_sizes[:tgt]` where tgt=0 → empty tuple ()
        # But _merge_static sets out_sdims = axis_sizes[:tgt] which is ()
        # But () for a scalar... the design says sdims=() for scalar with sdims awareness
        assert node.sdims == ()

    def test_fold_partial_with_sdims(self):
        """Fold rank-2 sdims=(3, 5) to rank-1 → keeps first sdim, features*5."""
        def f(x, **_):
            return jnp.ones((3, 5, 2))

        child = make_basis(f, dim=2, rank=2, n_features=2)
        object.__setattr__(child.root, "sdims", (3, 5))

        node = ReshapeRankNode(child.root, target_rank=1)
        assert node.rank == Rank.VECTOR
        assert node.n_features == 10  # 2 * 5
        assert node.sdims == (3,)

    def test_unfold_with_sdims_rejected(self):
        """Unfolding with non-uniform sdims is not supported."""
        child = _mock(rank=1, dim=3, sdims=(5,), n_features=6)
        with pytest.raises(ValueError, match="non-uniform sdims"):
            ReshapeRankNode(child.root, target_rank=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Default sdims is None for existing code
# ═══════════════════════════════════════════════════════════════════════════════
class TestBackwardCompat:
    def test_make_basis_has_none_sdims(self):
        b = make_basis(lambda x, **_: x, dim=3, rank=1, n_features=1)
        assert b.sdims is None

    def test_concat_preserves_none(self):
        a = make_basis(lambda x, **_: jnp.ones((1,)), dim=3, rank=0, n_features=1)
        b = make_basis(lambda x, **_: jnp.ones((1,)), dim=3, rank=0, n_features=1)
        c = a & b
        assert c.sdims is None

    def test_einsum_preserves_none(self):
        v = make_basis(
            lambda x, **_: jnp.repeat(x[..., None], 1, axis=-1),
            dim=3,
            rank=1,
            n_features=1,
        )
        e = StateExpr.einsum("i,j->ij", v, v)
        assert e.sdims is None
