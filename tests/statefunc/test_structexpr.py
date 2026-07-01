# TODO: review this file
"""Tests for SFI.statefunc.structexpr  (Phase 1 — inner-world algebra)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from SFI.statefunc.structexpr import (
    StructuredExpr,
    _BinaryOp,
    _ConcatOp,
    _ConstNode,
    _DenseOp,
    _EinsumOp,
    _FREE_LAYOUT,
    _ReshapeOp,
    _SectorLeaf,
    _SliceOp,
    _StackOp,
    _UnaryOp,
)
from SFI.statefunc.params import ParamSpec, ParamSuite


# ── helpers ──────────────────────────────────────────────────────────


def _scalar(nf: int = 1, lid: int = 0) -> StructuredExpr:
    return StructuredExpr(
        sdims=(), n_features=nf, param_suite=None,
        labels=(), _layout_id=lid,
        _node=_ConstNode(1.0, ()),
    )


def _vec(sdim: int = 2, nf: int = 1, lid: int = 0) -> StructuredExpr:
    return StructuredExpr(
        sdims=(sdim,), n_features=nf, param_suite=None,
        labels=(), _layout_id=lid,
        _node=_SectorLeaf("v", tuple(range(sdim)), (sdim,)),
    )


def _mat(s1: int = 2, s2: int = 3, nf: int = 1, lid: int = 0) -> StructuredExpr:
    return StructuredExpr(
        sdims=(s1, s2), n_features=nf, param_suite=None,
        labels=(), _layout_id=lid,
        _node=_ConstNode(0, (s1, s2)),
    )


# ── Addition / subtraction ──────────────────────────────────────────


class TestAddSub:
    def test_scalar_add(self):
        r = _scalar() + _scalar()
        assert r.sdims == () and r.n_features == 1

    def test_vector_add(self):
        r = _vec() + _vec()
        assert r.sdims == (2,) and r.n_features == 1

    def test_python_scalar_radd(self):
        r = 3 + _scalar()
        assert r.sdims == () and r.n_features == 1

    def test_sub(self):
        r = _scalar() - _scalar()
        assert r.sdims == () and isinstance(r._node, _BinaryOp)
        assert r._node.op == "-"

    def test_rsub(self):
        r = 3 - _scalar()
        assert r.sdims == () and r.n_features == 1

    def test_sdims_mismatch_raises(self):
        with pytest.raises(ValueError, match="sdims mismatch"):
            _scalar() + _vec()

    def test_nfeatures_mismatch_raises(self):
        with pytest.raises(ValueError, match="n_features mismatch"):
            _scalar(nf=1) + _scalar(nf=2)


# ── Multiplication ──────────────────────────────────────────────────


class TestMul:
    def test_scalar_times_vec(self):
        r = _scalar() * _vec()
        assert r.sdims == (2,) and r.n_features == 1

    def test_python_float_times_vec(self):
        r = 0.5 * _vec()
        assert r.sdims == (2,) and r.n_features == 1

    def test_same_sdims_kronecker(self):
        r = _vec(nf=2) * _vec(nf=3)
        assert r.sdims == (2,) and r.n_features == 6

    def test_scalar_times_scalar(self):
        r = _scalar(nf=2) * _scalar(nf=3)
        assert r.sdims == () and r.n_features == 6

    def test_different_srank_raises(self):
        with pytest.raises(TypeError, match="incompatible sdims"):
            _vec() * _mat()

    def test_different_sdims_same_srank_raises(self):
        with pytest.raises(TypeError, match="incompatible sdims"):
            _vec(2) * _vec(3)


# ── Division ────────────────────────────────────────────────────────


class TestDiv:
    def test_div_by_scalar(self):
        r = _vec() / 2.0
        assert r.sdims == (2,) and r.n_features == 1

    def test_div_by_same_nf(self):
        r = _vec(nf=3) / _vec(nf=3)
        assert r.n_features == 3

    def test_div_by_nf_1(self):
        r = _vec(nf=5) / _vec(nf=1)
        assert r.n_features == 5

    def test_div_bad_nf_raises(self):
        with pytest.raises(ValueError, match="n_features"):
            _vec(nf=5) / _vec(nf=3)

    def test_rdiv(self):
        r = 1 / _scalar()
        assert r.sdims == () and r.n_features == 1


# ── Power ───────────────────────────────────────────────────────────


class TestPow:
    def test_pow_int(self):
        r = _scalar() ** 3
        assert r.sdims == () and r.n_features == 1

    def test_pow_preserves_sdims(self):
        r = _vec() ** 2
        assert r.sdims == (2,) and r.n_features == 1

    def test_pow_non_scalar_raises(self):
        with pytest.raises(TypeError, match="scalar"):
            _scalar() ** _vec()


# ── Negation / positive ─────────────────────────────────────────────


class TestUnary:
    def test_neg(self):
        r = -_vec()
        assert r.sdims == (2,) and isinstance(r._node, _UnaryOp)
        assert r._node.op == "neg"

    def test_pos(self):
        v = _vec()
        assert v is (+v)


# ── Concatenation (&) ──────────────────────────────────────────────


class TestConcat:
    def test_basic(self):
        r = _vec(nf=1) & _vec(nf=2)
        assert r.sdims == (2,) and r.n_features == 3

    def test_labels_concat(self):
        a = StructuredExpr(sdims=(), n_features=1, param_suite=None,
                           labels=("a",), _layout_id=0, _node=_ConstNode(0, ()))
        b = StructuredExpr(sdims=(), n_features=1, param_suite=None,
                           labels=("b",), _layout_id=0, _node=_ConstNode(0, ()))
        r = a & b
        assert r.labels == ("a", "b")

    def test_sdims_mismatch_raises(self):
        with pytest.raises(ValueError, match="sdims mismatch"):
            _scalar() & _vec()


# ── Feature selection ──────────────────────────────────────────────


class TestGetitem:
    def test_int_index(self):
        r = _vec(nf=5)[2]
        assert r.n_features == 1

    def test_slice_index(self):
        r = _vec(nf=5)[1:4]
        assert r.n_features == 3

    def test_list_index(self):
        r = _vec(nf=5)[[0, 3]]
        assert r.n_features == 2

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _vec(nf=2)[5]


# ── Transpose ──────────────────────────────────────────────────────


class TestTranspose:
    def test_basic(self):
        m = _mat(2, 3)
        assert m.T.sdims == (3, 2)

    def test_rank3(self):
        t = StructuredExpr(
            sdims=(2, 3, 4), n_features=1, param_suite=None,
            labels=(), _layout_id=0, _node=_ConstNode(0, (2, 3, 4)),
        )
        assert t.T.sdims == (2, 4, 3)

    def test_srank_1_raises(self):
        with pytest.raises(TypeError, match="srank >= 2"):
            _ = _vec().T


# ── Matmul ────────────────────────────────────────────────────────


class TestMatmul:
    def test_matrix_matrix(self):
        r = _mat(2, 3) @ _mat(3, 4)
        assert r.sdims == (2, 4) and r.n_features == 1

    def test_vector_vector(self):
        r = _vec(3) @ _vec(3)
        assert r.sdims == () and r.n_features == 1

    def test_matrix_vector(self):
        A = _mat(2, 3)
        v = _vec(3)
        r = A @ v
        assert r.sdims == (2,) and r.n_features == 1

    def test_contraction_mismatch_raises(self):
        with pytest.raises(ValueError, match="contraction"):
            _mat(2, 3) @ _mat(2, 4)

    def test_scalar_raises(self):
        with pytest.raises(TypeError, match="srank >= 1"):
            _scalar() @ _vec()


# ── Dot ───────────────────────────────────────────────────────────


class TestDot:
    def test_vector_dot(self):
        r = _vec(3).dot(_vec(3))
        assert r.sdims == () and r.n_features == 1

    def test_matrix_vector_dot(self):
        # (2,3).dot((3,)) — last axes match → output (2,)
        m = _mat(2, 3)
        v = _vec(3)
        r = m.dot(v)
        assert r.sdims == (2,) and r.n_features == 1

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="contraction"):
            _vec(2).dot(_vec(3))


# ── Einsum ────────────────────────────────────────────────────────


class TestEinsum:
    def test_outer_product(self):
        r = StructuredExpr.einsum("i,j->ij", _vec(2), _vec(3))
        assert r.sdims == (2, 3) and r.n_features == 1

    def test_contraction(self):
        r = StructuredExpr.einsum("ij,jk->ik", _mat(2, 3), _mat(3, 4))
        assert r.sdims == (2, 4) and r.n_features == 1

    def test_trace(self):
        r = StructuredExpr.einsum("ii->", _mat(3, 3))
        assert r.sdims == () and r.n_features == 1

    def test_letter_conflict_raises(self):
        with pytest.raises(ValueError, match="conflicting sizes"):
            StructuredExpr.einsum("i,i->", _vec(2), _vec(3))

    def test_wrong_operand_count_raises(self):
        with pytest.raises(ValueError, match="operands"):
            StructuredExpr.einsum("i,j->ij", _vec(2))

    def test_wrong_srank_raises(self):
        with pytest.raises(ValueError, match="axes"):
            StructuredExpr.einsum("ij->i", _vec(2))

    def test_features_multiply(self):
        r = StructuredExpr.einsum("i,j->ij", _vec(2, nf=3), _vec(2, nf=2))
        assert r.n_features == 6


# ── Stack ─────────────────────────────────────────────────────────


class TestStack:
    def test_scalars_to_vector(self):
        r = StructuredExpr.stack([_scalar(), _scalar()])
        assert r.sdims == (2,) and r.n_features == 1

    def test_vectors_to_matrix(self):
        r = StructuredExpr.stack([_vec(3), _vec(3), _vec(3)])
        assert r.sdims == (3, 3) and r.n_features == 1

    def test_sdim_default(self):
        r = StructuredExpr.stack([_scalar()] * 5)
        assert r.sdims == (5,)

    def test_sdim_mismatch_raises(self):
        with pytest.raises(ValueError, match="sdim"):
            StructuredExpr.stack([_scalar(), _scalar()], sdim=3)

    def test_mixed_sdims_raises(self):
        with pytest.raises(ValueError, match="same sdims"):
            StructuredExpr.stack([_scalar(), _vec()])

    def test_mixed_nf_raises(self):
        with pytest.raises(ValueError, match="same n_features"):
            StructuredExpr.stack([_scalar(nf=1), _scalar(nf=2)])


# ── Eye ───────────────────────────────────────────────────────────


class TestEye:
    def test_basic(self):
        I = StructuredExpr.eye(3)
        assert I.sdims == (3, 3) and I.n_features == 1
        assert I.labels == ("I",)

    def test_free_layout(self):
        I = StructuredExpr.eye(2)
        assert I._layout_id == _FREE_LAYOUT

    def test_algebra_with_eye(self):
        Q = StructuredExpr.einsum("i,j->ij", _vec(2), _vec(2))
        r = Q - 0.5 * StructuredExpr.eye(2)
        assert r.sdims == (2, 2) and r.n_features == 1


# ── Math methods ──────────────────────────────────────────────────


class TestMath:
    @pytest.mark.parametrize("method", [
        "sin", "cos", "exp", "log", "tanh", "abs", "sqrt",
    ])
    def test_preserves_shape(self, method):
        v = _vec(3, nf=2)
        r = getattr(v, method)()
        assert r.sdims == (3,) and r.n_features == 2

    def test_elementwisemap_custom(self):
        v = _vec()
        r = v.elementwisemap(lambda z: jnp.tanh(z ** 2))
        assert r.sdims == (2,) and isinstance(r._node, _UnaryOp)
        assert r._node.op == "ew"
        assert r._node.fn is not None


# ── Reshape ───────────────────────────────────────────────────────


class TestReshape:
    def test_rank_to_features(self):
        t = _mat(2, 3, nf=5)
        r = t.rank_to_features()
        assert r.sdims == () and r.n_features == 30

    def test_features_to_rank(self):
        s = _scalar(nf=12)
        r = s.features_to_rank((3, 4))
        assert r.sdims == (3, 4) and r.n_features == 1

    def test_features_to_rank_partial(self):
        s = _scalar(nf=12)
        r = s.features_to_rank((3,))
        assert r.sdims == (3,) and r.n_features == 4

    def test_round_trip(self):
        t = _mat(2, 3, nf=5)
        r = t.rank_to_features().features_to_rank((2, 3))
        assert r.sdims == (2, 3) and r.n_features == 5

    def test_features_to_rank_non_scalar_raises(self):
        with pytest.raises(TypeError, match="srank == 0"):
            _vec().features_to_rank((3,))

    def test_features_to_rank_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            _scalar(nf=7).features_to_rank((3,))


# ── Dense ─────────────────────────────────────────────────────────


class TestDense:
    def test_basic(self):
        r = _scalar(nf=4).dense(8, name="W1")
        assert r.sdims == () and r.n_features == 8
        assert r.param_suite is not None
        assert r.param_suite["W1"].shape == (8, 4)

    def test_preserves_sdims(self):
        r = _vec(3, nf=2).dense(5, name="proj")
        assert r.sdims == (3,) and r.n_features == 5


# ── Layout compatibility ─────────────────────────────────────────


class TestLayoutCompat:
    def test_same_layout_ok(self):
        _scalar(lid=1) + _scalar(lid=1)  # no error

    def test_different_layout_raises(self):
        with pytest.raises(TypeError, match="different Layouts"):
            _scalar(lid=1) + _scalar(lid=2)

    def test_free_layout_compatible(self):
        r = _scalar(lid=_FREE_LAYOUT) + _scalar(lid=5)
        assert r._layout_id == 5

    def test_eye_compatible_with_any(self):
        I = StructuredExpr.eye(2)
        v = _mat(2, 2, lid=7)
        r = v + I
        assert r._layout_id == 7


# ── ParamSuite merging ───────────────────────────────────────────


class TestParamMerge:
    def test_none_propagation(self):
        r = _scalar() + _scalar()
        assert r.param_suite is None

    def test_merge_through_add(self):
        ps = ParamSuite([ParamSpec("w", (3,))])
        a = StructuredExpr(sdims=(), n_features=1, param_suite=ps,
                           labels=(), _layout_id=0, _node=_ConstNode(0, ()))
        b = _scalar()
        r = a + b
        assert r.param_suite is not None
        assert r.param_suite["w"].shape == (3,)

    def test_merge_shared_name(self):
        ps = ParamSuite([ParamSpec("w", (3,))])
        a = StructuredExpr(sdims=(), n_features=1, param_suite=ps,
                           labels=(), _layout_id=0, _node=_ConstNode(0, ()))
        b = StructuredExpr(sdims=(), n_features=1, param_suite=ps,
                           labels=(), _layout_id=0, _node=_ConstNode(0, ()))
        r = a + b
        assert len(r.param_suite) == 1  # shared, not duplicated


# ── Composite expressions from design note examples ──────────────


class TestDesignNoteExamples:
    def test_cahn_hilliard(self):
        """phi**3 - phi."""
        phi = _scalar()
        expr = phi ** 3 - phi
        assert expr.sdims == () and expr.n_features == 1

    def test_symmetric_strain_rate(self):
        """0.5 * (grad_v + grad_v.T)."""
        grad_v = _mat(2, 2)
        S = 0.5 * (grad_v + grad_v.T)
        assert S.sdims == (2, 2) and S.n_features == 1

    def test_nematic_order_tensor(self):
        """Q = n⊗n - 0.5 * I."""
        n = _vec(2)
        Q = StructuredExpr.einsum("i,j->ij", n, n) - 0.5 * StructuredExpr.eye(2)
        assert Q.sdims == (2, 2) and Q.n_features == 1

    def test_self_propulsion_vector(self):
        """eθ = [cos(θ), sin(θ)]."""
        theta = _scalar()
        e_theta = StructuredExpr.stack([theta.cos(), theta.sin()])
        assert e_theta.sdims == (2,) and e_theta.n_features == 1

    def test_advection_einsum(self):
        """(v·∇)n  =  einsum('i,ij->j', v, grad_n)."""
        v = _vec(2)
        grad_n = _mat(2, 2)  # mock grad(n) as (ndim, ndim)
        adv = StructuredExpr.einsum("i,ij->j", v, grad_n)
        assert adv.sdims == (2,) and adv.n_features == 1

    def test_active_nematic_forces(self):
        """viscous & active_force → n_features=2."""
        viscous = _vec(2, nf=1)
        active = _vec(2, nf=1)
        combined = viscous & active
        assert combined.sdims == (2,) and combined.n_features == 2


# ── repr ──────────────────────────────────────────────────────────


def test_repr():
    r = repr(_scalar())
    assert "sdims=()" in r and "n_features=1" in r

def test_repr_with_params():
    r = repr(_scalar(nf=4).dense(8))
    assert "has_params=True" in r
