# SFI/statefunc/nodes/ops/mapn.py
"""MapN: broadcast and apply N-ary function across children."""

from typing import Callable

import equinox as eqx

from ...params import ParamSuite
from ..base import BaseNode, BaseOpNode


# ──────────────────────────────────────────────────────────────────────────────
#  MapNNode – elementwise map of N children (with optional parameters)
# ──────────────────────────────────────────────────────────────────────────────
class MapNNode(BaseOpNode):
    """
    Apply an arbitrary **element-wise** callable to *N* child bases.

    Parameters
    ----------
    mapper : Callable
        A function that will be called as::

            mapper(*arrays[, theta]) -> array

        The *arrays* (one per child) have **identical shapes**, including feature axis.
        If a ``mapper_param_suite`` is supplied, a θ-dict with **only** those names
        is appended as the *last positional argument*; otherwise no parameter payload
        is passed to the mapper.

    mapper_param_suite : ParamSuite | None
        Parameter template **owned by the mapper itself** (not the children).

    Contract rules
    --------------
    - Children must match in (rank, n_features). ``dim`` is unified if concrete.
    - Particle depth follows broadcast rules: you may mix (pdepth=0, particles_input=False)
      with (pdepth=1, True). The output uses the latter, and non-P children treat P as part
      of the batch prefix.
    - The mapper **must** return a tensor with the same last-axis size (``n_features``),
      otherwise ``_assert_outputs`` will trigger.

    Notes
    -----
    This node's full ``param_suite`` is the **shared-by-name** merge of
    ``(children's suites) ∪ (mapper_param_suite)``.

    Reusing the same name across children/mapper **ties** that parameter provided
    the shape/dtype match exactly; incompatible duplicates raise an error.
    """

    CONTRACT_MODE = "map"

    mapper: Callable = eqx.field(static=True)
    mapper_param_suite: ParamSuite | None = eqx.field(static=True)
    label_fn: Callable[[str], str] | None = eqx.field(static=True, default=None, repr=False)

    # ------------------------------------------------------------------
    def __init__(
        self,
        mapper: Callable,
        *children: BaseNode,
        mapper_param_suite: ParamSuite | None = None,
        label_fn: Callable[[str], str] | None = None,
    ):
        object.__setattr__(self, "mapper", mapper)
        object.__setattr__(self, "mapper_param_suite", mapper_param_suite)
        object.__setattr__(self, "label_fn", label_fn)

        # Build base composite first: this sets `param_suite` from children
        super().__init__(*children)

        # Now incorporate the mapper's own params using shared-by-name merge
        # (allows parameter tying across children and mapper when specs match).
        merged = ParamSuite.merge_many(self.param_suite, self.mapper_param_suite)
        object.__setattr__(self, "param_suite", merged)

    def with_children(self, new_children):
        return MapNNode(
            self.mapper,
            *new_children,
            mapper_param_suite=self.mapper_param_suite,
            label_fn=self.label_fn,
        )

    # ------------------------------------------------------------------
    def _op(self, outs, *, params):
        """
        Combine child outputs element-wise via `mapper`.

        The `params` dict passed here is the full suite; we slice out only the
        mapper's subset (if any) and append it as the final positional argument.
        """
        if self.mapper_param_suite:
            theta = {n: params[n] for n in self.mapper_param_suite._lookup}
            return self.mapper(*outs, theta)
        return self.mapper(*outs)

    # ------------------------------------------------------------------
    #  Feature-aware flatten  (N children, element-wise map)
    # ------------------------------------------------------------------
    def flatten(self):
        """
        Build (funcs, labels, descs) where *n_features* matches children exactly.

        - funcs[i](x, ...) := mapper( child0_i(x), child1_i(x), …, θ? )
        - labels are produced by `label_fn` if given, otherwise
          `"{mapper_name}(" + ",".join(child_labels) + ")"`.
        - descs  = tuple of children’s desc[i]  (one per child)  for traceability.
        """
        # ---- gather children ----
        child_flat = [ch.flatten() for ch in self.children]
        funcs_lists, labels_lists, descs_lists = zip(*child_flat)
        n_feat = len(labels_lists[0])  # already validated in _merge_static

        # --- per-feature callables -----------------------------------
        def _slice_feature(j):
            # j is captured as default arg to avoid late binding
            return lambda x, *, v=None, mask=None, extras=None, params=None, _j=j: (
                self(x, v=v, mask=mask, extras=extras, params=params)[..., _j]
            )

        funcs = tuple(_slice_feature(j) for j in range(n_feat))

        # ---- labels ----
        if self.label_fn is not None:
            labels = tuple(self.label_fn(*(lablist[i] for lablist in labels_lists)) for i in range(n_feat))
        else:
            mname = self.mapper.__name__ if getattr(self.mapper, "__name__", "") else "map"
            labels = tuple(f"{mname}(" + ",".join(lab_tuple) + ")" for lab_tuple in zip(*labels_lists))

        # ---- descriptors ----
        descs = [
            tuple(dl[i] for dl in descs_lists)  # one tuple per child
            for i in range(n_feat)
        ]

        return funcs, labels, descs
