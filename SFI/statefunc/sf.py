"""SF façade: state function with fixed parameters."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .nodes import BaseNode
from .psf import PSF
from .stateexpr import StateExpr, _specialize_node


# ============================================================================
#  SF  – bound (θ-fixed) state-function
# ============================================================================
class SF(StateExpr):
    """State-Function with θ fixed (a thin wrapper over the PSF’s root).

    Behaves like a `Basis` for evaluation purposes (no `.d_theta()`), but you
    can still build `.d_x()` / `.d_v()` expressions.

    Feature axis handling mirrors the parent `PSF`:
    if `drop_features=True` and `n_features==1`, the final axis is removed.
    """

    params: dict[str, jax.Array] = eqx.field(static=False, repr=False)
    _psf: PSF = eqx.field(static=True, repr=False)
    drop_features: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        psf: PSF,
        params: dict[str, jax.Array],
        *,
        drop_features: bool | None = None,
    ):
        super().__init__(psf.root)
        params = psf.template.coerce(params)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "_psf", psf)
        object.__setattr__(
            self,
            "drop_features",
            psf.drop_features if drop_features is None else bool(drop_features),
        )

    def __call__(self, x, *, v=None, mask=None, extras=None):
        """Evaluate the bound function on a **batched** input."""
        self._validate_extras_presence(extras)

        y = self._caller(x, v, mask, extras, self.params)

        if self.drop_features and self.n_features == 1:
            return jnp.squeeze(y, axis=-1)
        return y

    @property
    def labels(self):
        """Basis labels propagated from the parent PSF."""
        return self._psf.labels

    def _with_node(self, new_root: BaseNode):
        new_psf = PSF(new_root, drop_features=self.drop_features)
        return SF(new_psf, self.params, drop_features=self.drop_features)

    def specialize(self, *, dataset: int) -> "SF":
        """Specialize a *bound* function at condition ``dataset``.

        Rewrites the graph (folding ``dataset_index``-reading leaves) and
        projects the bound parameter values onto the shrunken template: a
        per-condition spec whose shape loses a leading axis is sliced at
        ``dataset``; shared specs are kept verbatim.
        """
        k = int(dataset)
        new_root = _specialize_node(self.root, k)
        new_psf = PSF(new_root, drop_features=self.drop_features)
        new_params = _project_params(self.params, self._psf.template, new_psf.template, k)
        return SF(new_psf, new_params, drop_features=self.drop_features)


def _project_params(old_params, old_template, new_template, k: int) -> dict:
    """Map a bound parameter dict onto a specialized template.

    For each spec in ``new_template``: keep the old value when the shape is
    unchanged; slice the leading axis at ``k`` when specialization dropped it
    (the per-condition case, e.g. ``(K,) -> ()``).
    """
    out: dict = {}
    for spec in new_template:
        name = spec.name
        if name not in old_params:
            continue  # genuinely new param (none today); leave to template defaults
        old_val = old_params[name]
        old_shape = tuple(old_template[name].shape)
        new_shape = tuple(spec.shape)
        if old_shape == new_shape:
            out[name] = old_val
        elif len(old_shape) == len(new_shape) + 1 and old_shape[1:] == new_shape:
            out[name] = old_val[k]  # per-condition slice
        else:
            raise ValueError(
                f"Cannot project param {name!r} from shape {old_shape} to "
                f"{new_shape} during specialize(dataset={k})."
            )
    return out
