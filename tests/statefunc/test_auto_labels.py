# TODO: review this file
"""Tests for StructuredExpr auto-labelling (inner-world algebra)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from SFI.statefunc.structexpr import (
    StructuredExpr,
    _ConstNode,
    _FREE_LAYOUT,
    _SectorLeaf,
    _int_superscript,
    _complete_labels,
    _paren_for_pow,
    _paren_for_mul,
)
from SFI.statefunc.layout._base import _BaseLayout
from SFI.statefunc.layout._sectors import ScalarSector, VectorSector


# ── helpers ──────────────────────────────────────────────────────────


def _field(name: str, lid: int = 0) -> StructuredExpr:
    """Create a labelled scalar field ``(sdims=(), n_features=1)``."""
    return StructuredExpr(
        sdims=(), n_features=1, param_suite=None,
        labels=(name,), _layout_id=lid,
        _node=_SectorLeaf(name, (0,), ()),
    )


def _unlabelled(lid: int = 0) -> StructuredExpr:
    """Create an unlabelled scalar ``(sdims=(), n_features=1)``."""
    return StructuredExpr(
        sdims=(), n_features=1, param_suite=None,
        labels=(), _layout_id=lid,
        _node=_ConstNode(42.0, ()),
    )


# ── Pure helper tests ───────────────────────────────────────────────


class TestHelpers:
    def test_int_superscript(self):
        assert _int_superscript(0) == "⁰"
        assert _int_superscript(1) == "¹"
        assert _int_superscript(2) == "²"
        assert _int_superscript(3) == "³"
        assert _int_superscript(12) == "¹²"
        assert _int_superscript(-1) == "⁻¹"

    def test_complete_labels(self):
        assert _complete_labels(("U",), 1) is True
        assert _complete_labels(("U", "V"), 2) is True
        assert _complete_labels((), 1) is False
        assert _complete_labels(("U",), 2) is False
        assert _complete_labels(("",), 1) is False

    def test_paren_for_pow(self):
        assert _paren_for_pow("U") == "U"        # single char: no parens
        assert _paren_for_pow("UV") == "(UV)"    # multi char: parens
        assert _paren_for_pow("∇U") == "(∇U)"

    def test_paren_for_mul(self):
        assert _paren_for_mul("U") == "U"
        assert _paren_for_mul("UV") == "UV"
        assert _paren_for_mul("U+V") == "(U+V)"
        assert _paren_for_mul("U-V") == "(U-V)"
        assert _paren_for_mul("U/V") == "(U/V)"


# ── Multiplication ──────────────────────────────────────────────────


class TestMulLabels:
    def test_basic_product(self):
        U, V = _field("U"), _field("V")
        assert (U * V).labels == ("UV",)

    def test_product_reverses_fine(self):
        U, V = _field("U"), _field("V")
        assert (V * U).labels == ("VU",)

    def test_mul_by_const_1_preserves(self):
        U = _field("U")
        assert (U * 1).labels == ("U",)
        assert (1 * U).labels == ("U",)

    def test_mul_by_unlabelled_drops(self):
        U = _field("U")
        nolabel = _unlabelled()
        assert (U * nolabel).labels == ()

    def test_triple_product(self):
        U, V, W = _field("U"), _field("V"), _field("W")
        assert (U * V * W).labels == ("UVW",)

    def test_mul_parens_for_compound(self):
        """Label containing + gets parenthesised in product."""
        U, V = _field("U"), _field("V")
        s = U + V  # label "U+V"
        W = _field("W")
        result = s * W
        assert result.labels == ("(U+V)W",)


# ── Power ────────────────────────────────────────────────────────────


class TestPowLabels:
    def test_basic_square(self):
        U = _field("U")
        assert (U ** 2).labels == ("U²",)

    def test_basic_cube(self):
        U = _field("U")
        assert (U ** 3).labels == ("U³",)

    def test_pow_zero(self):
        U = _field("U")
        assert (U ** 0).labels == ("1",)

    def test_pow_one(self):
        U = _field("U")
        assert (U ** 1).labels == ("U",)

    def test_pow_negative(self):
        U = _field("U")
        assert (U ** -1).labels == ("U⁻¹",)

    def test_compound_label_gets_parens(self):
        """Multi-char labels wrapped in parens before superscript."""
        U, V = _field("U"), _field("V")
        uv = U * V  # label "UV"
        assert (uv ** 2).labels == ("(UV)²",)


# ── Subtraction ─────────────────────────────────────────────────────


class TestSubLabels:
    def test_basic_sub(self):
        U, V = _field("U"), _field("V")
        assert (U - V).labels == ("U-V",)

    def test_sub_unlabelled_drops(self):
        U = _field("U")
        nolabel = _unlabelled()
        assert (U - nolabel).labels == ()


# ── Division ────────────────────────────────────────────────────────


class TestDivLabels:
    def test_basic_div(self):
        U, V = _field("U"), _field("V")
        assert (U / V).labels == ("U/V",)

    def test_div_unlabelled_drops(self):
        U = _field("U")
        nolabel = _unlabelled()
        assert (U / nolabel).labels == ()


# ── Addition ────────────────────────────────────────────────────────


class TestAddLabels:
    def test_both_labelled_produces_sum(self):
        U, V = _field("U"), _field("V")
        assert (U + V).labels == ("U+V",)

    def test_one_unlabelled_fallback(self):
        U = _field("U")
        nolabel = _unlabelled()
        # labelled + unlabelled → keep labelled
        assert (U + nolabel).labels == ("U",)
        # unlabelled + labelled → keep labelled
        assert (nolabel + U).labels == ("U",)

    def test_scalar_add_preserves(self):
        """U + 1 (coerced scalar) → keeps 'U' (first-non-empty wins)."""
        U = _field("U")
        result = U + 1
        assert result.labels == ("U",)


# ── Negation ────────────────────────────────────────────────────────


class TestNegLabels:
    def test_neg_preserves(self):
        U = _field("U")
        assert (-U).labels == ("U",)


# ── Einsum / dot ────────────────────────────────────────────────────


class TestEinsumDotLabels:
    def _vector(self, name: str) -> StructuredExpr:
        """Labelled vector (sdims=(2,), n_features=1)."""
        return StructuredExpr(
            sdims=(2,), n_features=1, param_suite=None,
            labels=(name,), _layout_id=0,
            _node=_SectorLeaf(name, (0, 1), (2,)),
        )

    def test_dot_product(self):
        a = self._vector("∇U")
        b = self._vector("∇V")
        result = a.dot(b)
        assert result.labels == ("∇U·∇V",)

    def test_dot_self(self):
        a = self._vector("∇U")
        result = a.dot(a)
        assert result.labels == ("∇U·∇U",)

    def test_einsum_unlabelled_drops(self):
        a = self._vector("∇U")
        b = StructuredExpr(
            sdims=(2,), n_features=1, param_suite=None,
            labels=(), _layout_id=0,
            _node=_ConstNode(0, (2,)),
        )
        result = a.dot(b)
        assert result.labels == ()


# ── Elementwise map ─────────────────────────────────────────────────


class TestEWLabels:
    def test_sin(self):
        U = _field("U")
        assert U.sin().labels == ("sin(U)",)

    def test_cos(self):
        U = _field("U")
        assert U.cos().labels == ("cos(U)",)

    def test_exp(self):
        U = _field("U")
        assert U.exp().labels == ("exp(U)",)

    def test_log(self):
        U = _field("U")
        assert U.log().labels == ("log(U)",)

    def test_tanh(self):
        U = _field("U")
        assert U.tanh().labels == ("tanh(U)",)

    def test_abs(self):
        U = _field("U")
        assert U.abs().labels == ("abs(U)",)

    def test_sqrt(self):
        U = _field("U")
        assert U.sqrt().labels == ("sqrt(U)",)

    def test_custom_name(self):
        U = _field("U")
        result = U.elementwisemap(jnp.sin, name="mysin")
        assert result.labels == ("mysin(U)",)

    def test_unlabelled_drops(self):
        nolabel = _unlabelled()
        assert nolabel.sin().labels == ()


# ── with_label override ─────────────────────────────────────────────


class TestWithLabelOverride:
    def test_override_auto_label(self):
        """Manual with_label always takes precedence."""
        U, V = _field("U"), _field("V")
        result = (U * V).with_label("custom")
        assert result.labels == ("custom",)


# ── Concatenation (&) preserves labels ──────────────────────────────


class TestConcatLabels:
    def test_concat_auto_labels(self):
        U, V = _field("U"), _field("V")
        usq = U ** 2    # "U²"
        uv = U * V      # "UV"
        result = usq & uv
        assert result.labels == ("U²", "UV")


# ── layout.const ────────────────────────────────────────────────────


class TestLayoutConst:
    def _make_layout(self):
        """Minimal layout with a scalar field."""
        return _BaseLayout(dim=1, x=ScalarSector(indices=(0,)))

    def test_const_int(self):
        layout = self._make_layout()
        one = layout.const(1)
        assert one.labels == ("1",)
        assert one.sdims == ()
        assert one.n_features == 1

    def test_const_zero(self):
        layout = self._make_layout()
        z = layout.const(0)
        assert z.labels == ("0",)

    def test_const_float(self):
        layout = self._make_layout()
        c = layout.const(2.5)
        assert c.labels == ("2.5",)

    def test_const_custom_label(self):
        layout = self._make_layout()
        c = layout.const(1, label="ONE")
        assert c.labels == ("ONE",)

    def test_const_compatible_with_field(self):
        layout = self._make_layout()
        one = layout.const(1)
        x = layout.x
        result = one & x
        assert result.labels == ("1", "x")
        assert result.n_features == 2


# ── Compound expressions (Gray Scott pattern) ───────────────────────


class TestGrayScottPattern:
    """Test the specific label patterns from the Gray Scott example."""

    def test_full_gray_scott_labels(self):
        U, V = _field("U"), _field("V")
        # Powers
        assert (U ** 2).labels == ("U²",)
        assert (U ** 3).labels == ("U³",)
        # Products
        assert (U * V).labels == ("UV",)
        # Power × field
        assert (U ** 2 * V).labels == ("U²V",)
        assert (U * V ** 2).labels == ("UV²",)  # V² superscript binds to V

    def test_pow_then_mul_label_chain(self):
        """U**2 * V should give 'U²V'."""
        U, V = _field("U"), _field("V")
        u_sq = U ** 2       # "U²"
        result = u_sq * V   # "U²" * "V" → paren_for_mul("U²") + "V"
        # "U²" has no +/-/·/  → no parens → "U²V"
        assert result.labels == ("U²V",)

    def test_mul_then_pow(self):
        """(U*V)**2 → '(UV)²'."""
        U, V = _field("U"), _field("V")
        uv = U * V  # "UV"
        result = uv ** 2  # paren_for_pow("UV") = "(UV)" → "(UV)²"
        assert result.labels == ("(UV)²",)
