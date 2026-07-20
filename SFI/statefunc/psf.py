"""PSF façade: parametric family of state functions."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .nodes import BaseNode, DerivativeNode
from .params import ParamSuite
from .stateexpr import StateExpr


# ============================================================================
#  PSF  – parametric state-function family
# ============================================================================
class PSF(StateExpr):
    """Parametric State-Function family `F(x; θ)`.

    By default `drop_features=True`: when `n_features==1`, outputs **do not**
    carry a trailing feature axis. `.d_theta()` forces `drop_features=False`.

    Holds a `ParamSuite` template describing names, shapes, and dtypes of θ.
    `__call__` evaluates `F` given a parameter dict matching the template.
    Supports `.d_theta()` in addition to `.d_x()`/`.d_v()`.
    """

    template: ParamSuite = eqx.field(static=True)
    drop_features: bool = eqx.field(static=True, default=True)
    _validated_stamp: tuple | None = eqx.field(static=True, default=None, repr=False)

    def __init__(self, root: BaseNode, *, drop_features: bool = True):
        if root.param_suite is None:
            raise ValueError("PSF root must carry parameters")
        super().__init__(root)
        object.__setattr__(self, "template", root.param_suite)
        object.__setattr__(self, "drop_features", bool(drop_features))

    def _coerce_and_check(self, params, extras):
        # Normalize/validate via the suite
        pnorm = self.template.coerce(params)
        stamp = tuple(sorted((k, pnorm[k].shape, pnorm[k].dtype) for k in self.template._lookup))
        if stamp != self._validated_stamp:
            object.__setattr__(self, "_validated_stamp", stamp)
        # Only presence for extras here; shape/broadcast is handled in node contracts
        self._validate_extras_presence(extras)
        return pnorm

    def __call__(self, x, *, v=None, mask=None, extras=None, params=None):
        if params is None:
            if self.template.size == 0:
                params = {}  # parameter-free PSF: auto-supply
            else:
                defaults = self.template.defaults()
                if defaults is None:
                    raise ValueError("PSF.__call__: params are required (template has parameters without defaults).")
                params = defaults
        params = self._coerce_and_check(params, extras)
        y = self._caller(x, v, mask, extras, params)
        if self.drop_features and self.n_features == 1:
            return jnp.squeeze(y, axis=-1)
        return y

    def bind(self, params: dict[str, jax.Array] | None = None):
        """Freeze parameter dict into an SF with normalized arrays.

        If ``params is None``, fall back to spec defaults (``ParamSuite.defaults()``).
        Raises if the template has parameters without defaults.
        """
        from .sf import SF

        if params is None:
            defaults = self.template.defaults()
            if defaults is None:
                raise ValueError("PSF.bind(): params are required (template has parameters without defaults).")
            params = defaults
        return SF(self, params, drop_features=self.drop_features)

    def d_theta(self, *, mode: str = "auto"):
        """Build an expression for the Jacobian w.r.t. parameters θ.

        Shape effect
        ------------
        The final axis becomes `features × n_params_total`. Batch/pdepth/rank
        prefixes are preserved exactly.

        Notes
        -----
        The parameter PyTree is handled leafwise; each grad leaf is flattened over
        its param part, then all leaves are concatenated along the final axis.
        """
        if self.root.param_suite is None:  # unreachable: enforced by __init__
            raise AttributeError("Expression has no parameters to differentiate")
        node = DerivativeNode(self.root, var="theta", mode=mode)
        return PSF(node, drop_features=False)

    # ------------- SciPy helpers ---------------------------------
    @property
    def labels(self):
        """Basis labels from the underlying CoeffNode (if present)."""
        from .nodes.ops.linear import CoeffNode

        if isinstance(self.root, CoeffNode):
            return self.root._basis_labels
        _, labs, _ = self.root.flatten()
        return labs

    def flatten_params(self, params: dict[str, jax.Array]):
        """Vectorize a parameter dict according to the template order."""
        return self.template.vectorize(params)

    def unflatten_params(self, vec: jax.Array):
        """Materialize a parameter dict from a flat vector (inverse of `flatten_params`)."""
        return self.template.materialize(vec)
