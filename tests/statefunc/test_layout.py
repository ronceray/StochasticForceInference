# TODO: review this file
"""Tests for SFI.statefunc.layout  (Phase 2 — Layout protocol & Sectors)."""

from __future__ import annotations

import pytest

from SFI.statefunc.layout import (
    IdentityLayout,
    ScalarSector,
    Sector,
    StateLayout,
    SymTensorSector,
    TensorSector,
    VectorSector,
    _BaseLayout,
)
from SFI.statefunc.structexpr import StructuredExpr


# ── ScalarSector ─────────────────────────────────────────────────────


class TestScalarSector:
    def test_basic(self):
        s = ScalarSector([0])
        assert s.sdims == () and s.n_data == 1 and s.indices == (0,)

    def test_list_normalised_to_tuple(self):
        s = ScalarSector([3])
        assert isinstance(s.indices, tuple)

    def test_too_many_indices_raises(self):
        with pytest.raises(ValueError, match="exactly 1"):
            ScalarSector([0, 1])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="exactly 1"):
            ScalarSector([])


# ── VectorSector ─────────────────────────────────────────────────────


class TestVectorSector:
    def test_basic(self):
        v = VectorSector([0, 1], sdim=2)
        assert v.sdims == (2,) and v.n_data == 2

    def test_spatial_flag(self):
        v = VectorSector([0, 1], sdim=2, spatial=True)
        assert v.spatial is True

    def test_spatial_default_false(self):
        v = VectorSector([0, 1], sdim=2)
        assert v.spatial is False

    def test_index_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="requires 2"):
            VectorSector([0, 1, 2], sdim=2)

    def test_sdim_zero_raises(self):
        with pytest.raises(ValueError, match="sdim must be"):
            VectorSector([], sdim=0)


# ── SymTensorSector ──────────────────────────────────────────────────


class TestSymTensorSector:
    def test_2d(self):
        s = SymTensorSector([0, 1, 2], sdim=2)
        assert s.sdims == (2, 2) and s.n_data == 3

    def test_3d(self):
        s = SymTensorSector(range(6), sdim=3)
        assert s.sdims == (3, 3) and s.n_data == 6

    def test_voigt_pairs_2d(self):
        s = SymTensorSector([0, 1, 2], sdim=2)
        assert s.voigt_pairs == [(0, 0), (0, 1), (1, 1)]

    def test_voigt_pairs_3d(self):
        s = SymTensorSector(range(6), sdim=3)
        expected = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        assert s.voigt_pairs == expected

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="requires 3 indices"):
            SymTensorSector([0, 1], sdim=2)


# ── TensorSector ─────────────────────────────────────────────────────


class TestTensorSector:
    def test_basic(self):
        t = TensorSector(range(6), sdims=(2, 3))
        assert t.sdims == (2, 3) and t.n_data == 6

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="requires 6"):
            TensorSector(range(5), sdims=(2, 3))


# ── Index validation (_BaseLayout) ──────────────────────────────────


class TestIndexValidation:
    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="Index 1"):
            _BaseLayout(
                dim=3,
                a=VectorSector([0, 1], sdim=2),
                b=ScalarSector([1]),
            )

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _BaseLayout(dim=2, a=VectorSector([0, 5], sdim=2))

    def test_partial_coverage_ok(self):
        """Unused data indices are allowed."""
        lay = _BaseLayout(dim=4, phi=ScalarSector([0]))
        assert lay.dim == 4


# ── _BaseLayout ──────────────────────────────────────────────────────


class TestBaseLayout:
    def test_field_access(self):
        lay = _BaseLayout(
            dim=4,
            velocity=VectorSector([0, 1], sdim=2),
            director=VectorSector([2, 3], sdim=2),
        )
        v = lay.velocity
        assert isinstance(v, StructuredExpr)
        assert v.sdims == (2,) and v.n_features == 1

    def test_unpack(self):
        lay = _BaseLayout(dim=2, pos=VectorSector([0, 1], sdim=2))
        fields = lay.unpack()
        assert set(fields.keys()) == {"pos"}
        assert fields["pos"].sdims == (2,)

    def test_same_layout_algebra(self):
        lay = _BaseLayout(
            dim=4,
            a=VectorSector([0, 1], sdim=2),
            b=VectorSector([2, 3], sdim=2),
        )
        r = lay.a + lay.b
        assert r.sdims == (2,) and r.n_features == 1

    def test_cross_layout_raises(self):
        lay1 = _BaseLayout(dim=2, x=VectorSector([0, 1], sdim=2))
        lay2 = _BaseLayout(dim=2, y=VectorSector([0, 1], sdim=2))
        with pytest.raises(TypeError, match="different Layouts"):
            lay1.x + lay2.y

    def test_nonexistent_field_raises(self):
        lay = _BaseLayout(dim=2, pos=VectorSector([0, 1], sdim=2))
        with pytest.raises(AttributeError, match="no field"):
            lay.nonexistent

    def test_repr(self):
        lay = _BaseLayout(dim=2, phi=ScalarSector([0]))
        r = repr(lay)
        assert "dim=2" in r and "phi" in r

    def test_labels_from_sector_name(self):
        lay = _BaseLayout(dim=2, pos=VectorSector([0, 1], sdim=2))
        assert lay.pos.labels == ("pos",)


# ── IdentityLayout ──────────────────────────────────────────────────


class TestIdentityLayout:
    def test_basic(self):
        lay = IdentityLayout(dim=3)
        assert lay.dim == 3
        x = lay.state
        assert x.sdims == (3,) and x.n_features == 1

    def test_embed_not_implemented(self):
        lay = IdentityLayout(dim=2)
        with pytest.raises(NotImplementedError):
            lay.embed(rank=1, state=lay.state)


# ── StateLayout protocol ────────────────────────────────────────────


class TestProtocol:
    def test_identity_satisfies_protocol(self):
        lay = IdentityLayout(dim=3)
        assert isinstance(lay, StateLayout)

    def test_base_satisfies_protocol_structurally(self):
        """_BaseLayout doesn't implement embed(), but duck-type check works."""
        lay = _BaseLayout(dim=2, x=VectorSector([0, 1], sdim=2))
        assert hasattr(lay, "dim")
        assert hasattr(lay, "unpack")


# ── Design-note-style layouts ───────────────────────────────────────


class TestDesignNoteLayouts:
    def test_active_nematic_layout(self):
        lay = _BaseLayout(
            dim=4,
            velocity=VectorSector([0, 1], sdim=2, spatial=True),
            director=VectorSector([2, 3], sdim=2),
        )
        v = lay.velocity
        n = lay.director
        assert v.sdims == (2,) and n.sdims == (2,)
        # algebra works between same-layout fields
        diff = v - n
        assert diff.sdims == (2,)

    def test_abp_layout(self):
        lay = _BaseLayout(
            dim=3,
            position=VectorSector([0, 1], sdim=2),
            angle=ScalarSector([2]),
        )
        r = lay.position
        theta = lay.angle
        assert r.sdims == (2,) and theta.sdims == ()

    def test_cahn_hilliard_layout(self):
        lay = _BaseLayout(dim=1, phi=ScalarSector([0]))
        phi = lay.phi
        assert phi.sdims == () and phi.n_features == 1
        # phi**3 - phi works
        expr = phi ** 3 - phi
        assert expr.sdims == () and expr.n_features == 1
