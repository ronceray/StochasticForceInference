# Spec note — Re-channel structural (CSR) extras out of the data path

**Status:** design note for a fresh session · **Date:** 2026-06-18
**Decision pending:** do this **before the first release** (the failure mode is a
silent correctness bug — see "Why this may be pre-release"), or defer to
post-release. This note is written to be read cold.

## Background (where this sits)

This is "Phase C" of an extras-system upgrade. **Phase B is already done and
committed** (commits `757beee84` "Resolve reserved extras through a single
registry" and `9ee167697` "Supply reserved extras to bootstrapped
simulations"); its design note is
`docs/superpowers/specs/2026-06-17-extras-resolver-design.md`.

Phase B established the taxonomy (`stateexpr.py` lines ~59–65 name three kinds
**global / particle / structural**) and gave the **reserved** keys (`time`,
`duration`, `dataset_index`, `particle_index`) a single registry + resolver
(`SFI/trajectory/reserved_extras.py`), assembled once in
`TrajectoryDataset.build_extras`. It also added a stable
`TrajectoryDataset.uuid`.

Phase C is the **structural** kind — CSR neighbour tables, hyper-edge tables,
stencil masks/geometry used by interaction dispatchers — which Phase B did *not*
touch.

## The problem

Structural arrays are **stored on the dataset** in `extras_global`, namespaced
under a `_cache/` prefix, with a hand-managed lifecycle:

- **Build**: `prepare_collection_for_expr(data, expr)` runs every spec's
  `prepare_extras()` and writes `_cache/…` arrays into `extras_global`
  (`SFI/statefunc/nodes/interactions/prepare.py`).
- **Use**: the interaction dispatcher reads them to build the edge schedule.
  Kernels never see them ("never forwarded").
- **Purge**: `purge_collection_cache(data)` strips the `_cache/` keys afterward.

This build→use→purge dance is **scattered across ~14 sites** (count via
`grep -rn "purge_collection_cache\|prepare_collection_for_expr" SFI/`):
`SFI/inference/overdamped.py` (~262/265, 344/347, 533/621),
`SFI/inference/underdamped.py` (~225/259/590), `SFI/inference/base.py`
(~842/844/889/918/969), plus purges in `SFI/trajectory/degrade.py` and the
dataset slicer (`SFI/trajectory/dataset.py` ~749–764).

Three problems, in order of seriousness:

1. **Latent correctness bug (the important one).** A `_cache/` CSR is built for a
   specific context (box, particle set, sampling). Forget a `purge` — or carry a
   dataset through a transform that doesn't purge — and a **stale neighbour list
   survives and is silently used**, producing wrong forces with no error. This is
   the same scatter anti-pattern Phase B removed for reserved extras, but with a
   silent-wrong-answer failure mode instead of a `KeyError`.
2. **Namespace pollution.** Computation scaffolding lives in the same dict as
   physical user data (box geometry, conditions, drive protocols).
3. **Fragile `_cache/` prefix.** A naming convention is the only thing stopping
   these arrays from being serialised, sliced per frame, or treated as data;
   every transform must remember to purge.

## Current architecture (the contracts)

- `SFI/statefunc/nodes/interactions/specs.py`
  - `CACHE_PREFIX = "_cache/"`.
  - `SpecRule`: `arity()`, `required_extras()` (forwarded presence-only),
    `structural_extras()` (dispatcher-owned keys, never forwarded), `build()`.
  - `AutoPairs` — builds an all-pairs CSR on the fly (no user input).
  - `FromExtrasPairsCSR(indptr_key, indices_key)` — reads user-supplied CSR
    arrays from extras under chosen keys (see
    `Interactor.dispatch_pairs_from_extras`, `SFI/statefunc/interactor.py` ~89).
  - There is also an internal per-spec `_cache` (eqx static field).
- `SFI/statefunc/nodes/interactions/stencils.py` — `PreparedSquareStencilFromBox`
  etc.: `structural_extras()` + `prepare_extras(extras)` host hook.
- `SFI/statefunc/nodes/interactions/prepare.py` — `prepare_structural_extras_for_expr`,
  `prepare_collection_for_expr`, `purge_cache_extras` / `purge_dataset_cache` /
  `purge_collection_cache`, `is_cache_key`.
- `SFI/statefunc/nodes/interactions/dispatcher.py` — consumes the structural
  tables to gather per-edge inputs (also gathers `particle_extras`).
- Producer interaction (Phase B): `TrajectoryDataset.build_extras` →
  `slice_frame_extras` passes `_cache/` arrays through as **static** extras, so
  they currently reach the dispatcher via the normal extras mapping.

## Goal

Make structural arrays **dispatcher-owned and never stored on the dataset**.
This is stronger (and cleaner) than the originally-sketched "separate channel" —
a parallel dict would still need building and invalidating. Instead:

- The dispatcher builds CSR/stencil tables **on demand** from context
  (positions, box, boundary conditions) and **memoises them dispatcher-side,
  keyed by a context hash** (geometry/params + the dataset `uuid` from Phase B).
- Then: no `prepare_collection_for_expr` injection, no `purge_collection_cache`
  scatter, no `_cache/` prefix, no serialise/slice/transform fragility.
  Invalidation is automatic — context changes ⇒ cache miss ⇒ rebuild.

## The one real tension — preserve memoisation

The on-dataset `_cache/` exists for **performance**: neighbour/CSR construction is
expensive and must not run per chunk/frame. So this is **not** "stop caching" —
it is "cache in the dispatcher keyed by context." The design must keep a single
build per (expr, dataset-context), matching today's effective memoisation.
`FromExtrasPairsCSR` (user *provides* the CSR) stays a first-class path — there,
the user's arrays are the source of truth and need no building.

## Scope / files to touch

- `SFI/statefunc/nodes/interactions/dispatcher.py` + `specs.py` — own and
  memoise the structural tables (context-keyed cache; build on first use).
- `SFI/inference/{overdamped,underdamped,base}.py` — delete the ~14
  `prepare_collection_for_expr` / `purge_collection_cache` calls.
- `SFI/langevin/base.py` — the simulation structural prep (`_prepare_model_extras`
  / `prepare_structural_extras_for_expr`) routes through the dispatcher cache.
- `SFI/trajectory/{degrade,dataset}.py` — remove the `_cache/` purges (nothing to
  purge once arrays aren't stored on the dataset).
- `SFI/statefunc/nodes/interactions/prepare.py` — likely retire
  `prepare_collection_for_expr` and the `purge_*` helpers, or reduce to a thin
  context-hash utility.
- `FromExtrasPairsCSR` — keep; it is the explicit user-provided path.

## Migration / risk

- **High blast radius**: the interaction dispatcher is central to multi-particle,
  ABP, and SPDE inference/simulation. This is the riskiest subsystem in the repo
  to refactor.
- Backward-compat is not a constraint (pre-release), so the `_cache/` convention
  can be removed outright rather than shimmed.
- Land behind the full test suite (`pytest tests/`, ~42 min) plus the gallery
  demos that exercise interactions: `abp_align_demo`, `abp_nonreciprocal_demo`,
  `abp_to_spde_demo`, `active_nematic_demo`, `gray_scott_demo`, `home_range_demo`,
  and `advanced/flocking_3d_demo`, `advanced/multi_experiment_demo`.

## Testing

- Unit: dispatcher builds the same CSR as today for `AutoPairs` /
  `FromExtrasPairsCSR` / stencil specs (golden vs current output on a fixed
  fixture, pinned *before* the refactor).
- **Cache-correctness (the headline):** a dataset reused across two contexts
  (e.g. before/after `degrade` or a box change) must rebuild — assert the
  dispatcher does **not** reuse a stale table (this is the bug class being
  closed).
- Memoisation/perf guard: a repeated evaluation at fixed context builds the
  table once (count builds via an instrumented spec).
- Regression: full suite + the interaction gallery demos above.

## Why this may be pre-release

The motivating defect is **silent wrong forces** from a stale structural cache,
not a crash. Shipping a public release with a maintenance pattern that can
silently corrupt interacting-system fits is the kind of thing worth closing
before v1 — the cost is one focused (if invasive) refactor of a well-bounded
subsystem with strong test coverage. Weigh that against the release-window risk
of touching the dispatcher. If pre-release: schedule it as its own branch with
the full suite + interaction gallery as the gate.

## Open questions for the fresh session

1. **Cache key.** What exactly hashes to a context key — dataset `uuid` +
   box/bc/offsets + N? Confirm every input that changes the CSR is in the key.
2. **Cache location/lifetime.** On the dispatcher instance, or a process-level
   keyed store? How does it interact with JIT (host-side build between traced
   chunks, per the existing "host-only" contract in `prepare.py`)?
3. **`FromExtrasPairsCSR`.** Keep user-provided CSR in `extras` (it is genuine
   input, not scaffolding), or also move it behind a typed dispatcher input?
4. **Variable-N / masked / multi-dataset** pooled fits — the cache must key per
   dataset; verify against `tests/inference/test_per_dataset.py` interactor case.
