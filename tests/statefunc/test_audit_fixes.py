"""Tests for statefunc audit fixes (final-review branch)."""

import jax.numpy as jnp
import pytest

from SFI.statefunc import Basis, Rank, make_interactor
from SFI.statefunc.basis import Basis
from SFI.statefunc.interactor import _tree_is_parametric
from SFI.statefunc.nodes.interactions import AutoPairs
from SFI.statefunc.params import ParamSpec, ParamSuite
from SFI.statefunc.psf import PSF


# ---------------------------------------------------------------------------
# _tree_is_parametric — defensive consistency fix
# ---------------------------------------------------------------------------

class TestTreeIsParametric:
    """_tree_is_parametric must detect params even when nodes use child/inner."""

    def _make_parametric_interactor(self, dim=2):
        def f(Xk, *, params):
            return (params["w"] * (Xk[1] - Xk[0]))[..., None]

        return make_interactor(
            f,
            dim=dim,
            rank=Rank.VECTOR,
            K=2,
            params={"w": ()},
        )

    def test_direct_parametric_interactor_detected(self):
        inter = self._make_parametric_interactor()
        assert _tree_is_parametric(inter.root)

    def test_parametric_interactor_after_slice_detected(self):
        inter = self._make_parametric_interactor()
        sliced = inter[0]
        assert _tree_is_parametric(sliced.root)

    def test_dispatch_auto_returns_psf_after_slice(self):
        """dispatch(return_as='auto') on a sliced parametric Interactor returns PSF."""
        inter = self._make_parametric_interactor()
        sliced = inter[0]
        result = sliced.dispatch_pairs(symmetric=True)
        assert isinstance(result, PSF)

    def test_dispatch_auto_returns_basis_for_non_parametric(self):
        def f(Xk):
            return (Xk[1] - Xk[0])[..., None]

        inter = make_interactor(f, dim=2, rank=Rank.VECTOR, K=2)
        result = inter.dispatch_pairs(symmetric=True)
        assert isinstance(result, Basis)


# ---------------------------------------------------------------------------
# params.coerce — regression tests for if/else and guard simplification
# ---------------------------------------------------------------------------

class TestParamCoerce:
    """Coerce behaves correctly after cleanup."""

    def _suite(self, name, shape):
        return ParamSuite([ParamSpec(name, shape)])

    def test_scalar_coercion_python_float(self):
        suite = self._suite("a", ())
        out = suite.coerce({"a": 3.14})
        assert out["a"].shape == ()

    def test_scalar_coercion_jax_array(self):
        suite = self._suite("a", ())
        out = suite.coerce({"a": jnp.array(1.0)})
        assert out["a"].shape == ()

    def test_len1_coercion_scalar_expansion(self):
        suite = self._suite("b", (1,))
        out = suite.coerce({"b": 2.0})
        assert out["b"].shape == (1,)

    def test_len1_exact_shape_accepted(self):
        suite = self._suite("b", (1,))
        out = suite.coerce({"b": jnp.array([5.0])})
        assert out["b"].shape == (1,)

    def test_len1_wrong_shape_raises(self):
        suite = self._suite("b", (1,))
        with pytest.raises(TypeError, match="expected"):
            suite.coerce({"b": jnp.zeros((3,))})

    def test_wrong_dtype_raises_when_cast_false(self):
        suite = self._suite("c", (2,))
        with pytest.raises(TypeError, match="dtype"):
            suite.coerce({"c": jnp.zeros((2,), dtype=jnp.int32)}, cast_dtype=False)


# ---------------------------------------------------------------------------
# stateexpr einsum letters — no duplicate chars (latent bug for rank > 26)
# ---------------------------------------------------------------------------

class TestEinsumLetters:
    """The einsum letters constant must be duplicate-free."""

    _BAD_LETTERS = "ijklmnopqrstuvwxyzabcdefghpqrsuvt"  # the pre-fix string

    def test_pre_fix_string_has_duplicates(self):
        """Document that the old string had duplicates (regression guard)."""
        s = self._BAD_LETTERS
        dups = [c for c in s if s.count(c) > 1]
        assert dups, "Expected pre-fix string to have duplicates — update this test if string changed"

    def test_module_constant_has_no_duplicates(self):
        """After the fix, _EINSUM_LETTERS must be duplicate-free."""
        import SFI.statefunc.stateexpr as se
        letters = getattr(se, "_EINSUM_LETTERS", None)
        assert letters is not None, "_EINSUM_LETTERS constant missing from stateexpr"
        assert len(letters) == len(set(letters)), (
            f"Duplicate chars: {[c for c in letters if letters.count(c) > 1]}"
        )

    def test_rank2_mul_correctness(self):
        """rank-2 * rank-2 must be element-wise, not a trace."""
        from SFI.statefunc import make_basis
        A = make_basis(lambda x: jnp.outer(x, x), dim=2, rank=2, n_features=1)
        x = jnp.array([1.0, 2.0])
        out_A = A(x[None])
        C = A * A
        out_C = C(x[None])
        assert jnp.allclose(out_C, out_A * out_A)
