# SFI/statefunc/nodes/ops/linear.py
"""Linear algebra operator nodes: Dense and Coeff."""

import equinox as eqx
import jax.numpy as jnp

from ...params import ParamSpec, ParamSuite
from ..base import BaseNode, BaseOpNode
from ..contract import _ContractMixin


# ──────────────────────────────────────────────────────────────────────────────
#  DenseNode  –  linear map on the feature axis of one child
# ──────────────────────────────────────────────────────────────────────────────
class DenseNode(BaseOpNode):
    """
    Affine map along the feature axis of a single child.

    Parameters
    ----------
    n_out       : int                 - output feature dimension
    weight      : str = "W"           - parameter, shape ``(n_in, n_out)``
    bias        : str | None = "b"    - parameter, shape ``(n_out,)`` or None
    """

    weight_name: str = eqx.field(static=True)
    bias_name: str | None = eqx.field(static=True)
    _n_out: int = eqx.field(static=True)

    # ------------------------------------------------------------------
    def __init__(self, child: BaseNode, *, n_out: int, weight: str = "W", bias: str | None = "b"):
        self.weight_name = weight
        self.bias_name = bias
        self._n_out = int(n_out)
        super().__init__(child)

        # own parameter template
        W_spec = ParamSpec(weight, (child.n_features, n_out))
        specs = [W_spec]
        if bias is not None:
            specs.append(ParamSpec(bias, (n_out,)))
        own_suite = ParamSuite.from_specs(*specs)

        merged = child.param_suite.merge(own_suite) if child.param_suite else own_suite
        object.__setattr__(self, "param_suite", merged)

    def with_children(self, new_children):
        return DenseNode(
            new_children[0],
            n_out=self._n_out,
            weight=self.weight_name,
            bias=self.bias_name,
        )

    # ------------------------------------------------------------------
    def _merge_static(self, children):
        ch = children[0]
        return _ContractMixin.inherit_contract(ch, n_features=self._n_out)

    # ------------------------------------------------------------------
    def _op(self, outs, *, params):
        y_in = outs[0]
        W = params[self.weight_name]
        b = params[self.bias_name] if self.bias_name else None
        y = jnp.tensordot(y_in, W, axes=[-1, 0])  # "...i,ij->...j"
        return y + b if b is not None else y

    # ------------------------------------------------------------------
    #  Feature-wise flatten
    # ------------------------------------------------------------------
    def flatten(self):
        """
        Return (funcs, labels, descs) where

        * ``funcs[j](...)``  calls the *DenseNode* itself and picks the
          j-th feature slice → keeps maintenance trivial.
        * Label is ``"dense{j}"`` as there's no practical way to propagate
          previous labels through a Dense node.
        * Descriptors are inherited from the **child** – the linear map
          does not alter the mathematical identity, only its coefficients.
        """

        child_funcs, child_labels, child_descs = self.children[0].flatten()
        n_out = self.n_features

        # --- per-feature callables -----------------------------------
        def _slice_feature(j):
            # j is captured as default arg to avoid late binding
            return lambda x, *, v=None, mask=None, extras=None, params=None, _j=j: (
                self(x, v=v, mask=mask, extras=extras, params=params)[..., _j]
            )

        funcs = tuple(_slice_feature(j) for j in range(n_out))

        # --- labels ---------------------------------------------------
        labels = tuple(f"dense{j}" for j in range(n_out))

        # --- descriptors (inherit) ------------------------------------
        descs = tuple({"dense_of": tuple(child_descs), "index": j} for j in range(n_out))

        return funcs, labels, descs


# ──────────────────────────────────────────────────────────────────────────────
#  CoeffNode – “Dense-with-θ-vector” (n_out = 1, bias = None)
# ──────────────────────────────────────────────────────────────────────────────
class CoeffNode(DenseNode):
    """
    Special-case of :class:`DenseNode` that stores its weights as a **vector**
    θ ∈ ℝⁿ rather than a (n_in × 1) column matrix:

        F(x; θ) = θ · f(x)           # dot on feature axis, returns scalar

    The output therefore has a **singleton feature axis** (n_features = 1).

    Parameters
    ----------
    child      : BaseNode
        Basis/dictionary to be linearly combined.
    coeff_key  : str, default ``"coeff"``
        Parameter-dict name under which θ is stored.
    """

    _basis_labels: tuple[str, ...] = eqx.field(static=True, default=())

    # ------------------------------------------------------------------
    def __init__(self, child: BaseNode, *, coeff_key: str = "coeff"):
        # call DenseNode constructor with n_out = 1, bias = None
        super().__init__(child, n_out=1, weight=coeff_key, bias=None)

        # Override the weight spec to a **vector** (n_in,) instead of matrix,
        # with a default of ones so bare `Basis → .to_psf()` simulates out of
        # the box (equivalent to "sum all features").
        theta = ParamSpec(coeff_key, (child.n_features,), default=jnp.ones((child.n_features,)))
        own = ParamSuite.from_specs(theta)
        merged = child.param_suite.merge(own) if child.param_suite else own
        object.__setattr__(self, "param_suite", merged)

        # Preserve the original basis labels for downstream reporting
        _, child_labels, _ = child.flatten()
        object.__setattr__(self, "_basis_labels", tuple(child_labels))

    def with_children(self, new_children):
        return CoeffNode(new_children[0], coeff_key=self.weight_name)

    # ------------------------------------------------------------------
    # DenseNode._op expects W shaped (n_in, 1); we override to use a vector.
    # ------------------------------------------------------------------
    def _op(self, outs, *, params):
        y_in = outs[0]  # (..., n_feat)
        theta = params[self.weight_name]  # (n_feat,)
        y = jnp.einsum("...b,b->...", y_in, theta)[..., None]  # (..., 1)
        return y
