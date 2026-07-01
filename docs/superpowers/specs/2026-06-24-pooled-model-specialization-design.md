# Pooled → single-condition model specialization

**Date:** 2026-06-24
**Status:** approved, implementing

## Problem

`dataset_index` (a framework-reserved extra) does two different jobs under one
name:

1. **Pool position** — "which experiment in this collection," auto-derived
   `0…K-1` at inference time (per-dataset weighting, identity). Genuinely a
   *dataset* index.
2. **Per-condition selector** — "which experiment's parameters the model uses."
   In a pooled fit these coincide, but in a *simulation* there is no pool, so the
   name is meaningless and the value has to be smuggled in via
   `_reserved_overrides` / the `dataset=` argument.

This conflation makes simulations counter-intuitive ("a dataset index in a
simulation") and inflexible (the selector is reserved, so it cannot be set
through the normal extras API — setting it triggers the reserved-key clash that
surfaced as a cryptic crash before the error-handling fixes on `main`).

## Decision

Remove the coordinate from the simulation side entirely. When reproducing one
experiment, **partially evaluate the pooled model at that condition**, producing
a standalone model that does not read `dataset_index` at all. `dataset_index`
keeps only its honest job (pool position at inference). Pre-1.0: clean break
allowed (no on-disk format change).

## Design

### `specialize(dataset=k)` — first-class, reusable

A new operation on `StateExpr` (inherited by `Basis` / `PSF` / `SF`,
`SFI/statefunc/stateexpr.py`). It walks the node graph (reusing the
`_walk_nodes` traversal pattern, stateexpr.py:1128) and returns a NEW graph in
which every `dataset_index`-reading leaf is folded at index `k`:

- **`per_dataset_scalar(name, K)`** (`SFI/bases/constants.py:329`): its `(K,)`
  `ParamSpec` collapses to the `k`-th entry. On a **bound** model (`SF`, or
  PSF+θ) it folds to a constant; on an **unbound** `PSF` it becomes a scalar
  param `name` (the condition-`k` slice).
- **`dataset_indicator(K)`** (`SFI/bases/constants.py:380`): constant one-hot for
  `k` (selects which fitted coefficient block survives).
- Every other leaf passes through unchanged. Composite nodes (`MapNNode`,
  `ConcatNode`, `EinsumNode`, `DenseNode`, …) are rebuilt from specialized
  children, re-merging param suites via `ParamSuite.merge_many`. `dataset_index`
  drops out of `required_extras` automatically.

Each `dataset_index`-reading primitive declares **its own** specialization via a
small `specialize_at(k)` hook on the leaf, so future per-condition primitives are
handled without touching the walker.

Param-model transform: a leaf's `(K,)` spec becomes either a scalar `ParamSpec`
(unbound) or is dropped in favour of a folded constant (bound, value sliced
`θ[name] → θ[name][k]`). The PSF-level template shrinks accordingly.

Public use, independent of bootstrap:
`inferred.force_inferred.specialize(dataset=k)`.

### Bootstrap integration + clean trajectory export

`simulate_bootstrapped_trajectory(key, ..., dataset=k)`
(`SFI/inference/underdamped.py:311`, twin at `overdamped.py:348`) becomes:

```python
force_k = self.force_inferred.specialize(dataset=k)      # no dataset_index
diff_k  = self.diffusion_inferred.specialize(dataset=k)  # no-op if D not pooled
proc    = UnderdampedProcess(force_k, diff_k)            # standalone model
```

- The `dataset_index` branch of `_reserved_overrides` / `_reserved_context`
  (`SFI/langevin/base.py:404-408`) becomes dead for this path; `particle_index`
  stays (legitimate in simulation).
- Exported `coll_boot` is a plain single trajectory: one dataset, unit weight,
  `extras_global` with **no `dataset_index`** — so the reserved-clash route is
  structurally impossible.
- Re-inference uses a plain single-condition basis. Producing the matching
  specialized basis is the caller's choice (`B.specialize(dataset=k)` is
  available); the simulation deliverable is the model + clean trajectory.

### Clean-break migration

- Keep `per_dataset_scalar` / `dataset_indicator` as the canonical per-condition
  primitives (correct *at inference*); add the `specialize_at` hook to each.
- Drop the `dataset=` → `_reserved_overrides["dataset_index"]` simulation
  mechanism. Forcing a `dataset_index` during simulation is no longer possible
  by design — that is the break.
- Update the multi-experiment demo
  (`examples/gallery/advanced/multi_experiment_demo.py`) and docs
  (`bases/user_guide.rst`, `trajectory/user_guide.rst`, glossary) to the
  "fit pooled → `specialize(dataset=k)` → simulate" story.
- No on-disk format change.

## Testing

- **`specialize` units** (`tests/statefunc/`): specialized `per_dataset_scalar`
  / `dataset_indicator` have no `dataset_index` in `required_extras` and evaluate
  equal to the pooled model fed `dataset_index=k`, for every `k`; a composite
  `(dataset_indicator(3) * X(2)) & named_scalar("c")` round-trips; param template
  shrinks correctly; bound-`SF` value-fold matches.
- **Bootstrap integration** (`tests/inference/`): `dataset=k` returns a process
  whose force has no `dataset_index` dependence and a `coll_boot` whose
  `extras_global` lacks `dataset_index`; re-inference runs without the
  reserved-clash error (regression for this thread).
- **Equivalence**: a pooled simulation at `k` (old `_reserved_overrides`
  behaviour, captured before removal) and the new specialized simulation produce
  identical trajectories for a fixed key.

## Out of scope

`particle_index` (legitimately needed in simulation). Renaming `dataset_index`.
Any change to how pooled *inference* gathers per-dataset rows.
