# Audit brief — whole-codebase review of extras usage & treatment

**Purpose.** Independently verify that the **entire extras system** is internally
consistent and that nothing is out of place. A recent change moved dispatcher-owned
*structural* arrays (CSR neighbour tables, grid stencils) off datasets/processes and
into transient, build-at-point-of-use overlays; this audit is the cross-check that
the change is complete and that the surrounding extras machinery (reserved keys,
time-varying extras, forwarding, transforms, serialization) is still coherent.

Be **skeptical**: the goal is to find inconsistencies, not to confirm a prior change.
Read cold; do not assume the recent change is correct — verify it.

## What "extras" are (the mental model to confirm)

Extras are auxiliary inputs attached to trajectory data and threaded to force/
diffusion models. There are three kinds (declared in
`SFI/statefunc/stateexpr.py`, search "global / particle / structural"):

- **global** — small user values broadcast to all sites (box geometry, conditions,
  drive protocols, `box/grid_shape`, `box/dx`). Forwarded to kernels.
- **particle** — per-site user data. Forwarded.
- **structural** — dispatcher-owned CSR/hyper/stencil tables. **Never forwarded** to
  kernels; consumed by the interaction dispatcher. Historically namespaced `_cache/`.

Plus **reserved** keys resolved by the framework, not the user: `time`, `duration`,
`dataset_index`, `particle_index` (registry/resolver in
`SFI/trajectory/reserved_extras.py`).

Value wrappers: `TimeSeriesExtra` (leading time axis; sliced per frame), `FunctionExtra`
(on-demand), and plain callables `f(t)`.

## Key files / contracts (start here)

- `SFI/trajectory/reserved_extras.py` — `ExtrasContext`, `resolve_extras` /
  `resolve_reserved`, `slice_frame_extras`, `RESERVED_NAMES`.
- `SFI/trajectory/dataset.py` — `extras_global` / `extras_local`, `build_extras(t_idx)`,
  the per-`t` `producer` / `make_batch_producer` (calls `build_extras`), `uuid`, the
  frame slicer (`_slice_frames` / `_slice_time_extras`).
- `SFI/statefunc/nodes/interactions/prepare.py` — `CACHE_PREFIX`, `is_cache_key`,
  `purge_cache_extras`, `prepare_structural_extras_for_expr`,
  `build_structural_overlay`, `prepare_collection_for_expr`.
- `SFI/statefunc/nodes/interactions/specs.py` — `SpecRule.required_extras()` vs
  `structural_extras()`, `AutoPairs`, `FromExtrasPairsCSR`, `CachedRule`.
- `SFI/statefunc/nodes/interactions/{dispatcher,stencils}.py` — structural filtering;
  `PreparedSquareStencilFromBox`, `HyperFixedSquareStencilFromBox`.
- `SFI/inference/base.py` — `_structural_scope`; `compare_to_exact` extras handling.
- `SFI/langevin/base.py` — `_model_extras`, `_prepare_model_extras`,
  `_prepared_structural`, `_invalidate_prepared_extras`, `set_extras`, schedule
  classification (`_classify` / `_build_schedules`).
- `SFI/trajectory/degrade.py` — extras handling under degradation.
- `SFI/bases/spde.py`, `SFI/statefunc/layout/_grid.py` — who builds stencils.

## Audit checklist

### 1. Persistence / namespace hygiene
- [ ] No code path **persists** `_cache/` keys onto a `TrajectoryDataset` that a user
      can hold, serialize, or transform. `grep -rn "_cache/" SFI/` and confirm every
      write is into a *transient* dict (overlay / private store), not onto a returned
      dataset or `self.extras_global`.
- [ ] `grep -rn "extras_global\[" SFI/ ; grep -rn "\.update(" SFI/ | grep -i extra` —
      confirm no surprise structural writes to user-facing extras.
- [ ] Simulation: after `proc.simulate(...)`, `proc.extras_global` holds **no**
      `_cache/`; the output collection's datasets hold no `_cache/`.
- [ ] Inference: after any `infer_*` call, `inf.data.datasets[i].extras_global` holds
      no `_cache/`, and `inf.data` is the **same object** the caller passed.

### 2. Kinds & forwarding
- [ ] Dispatcher filters `structural_extras()` keys out before forwarding to child
      kernels; `required_extras()` (globals) and particle extras *are* forwarded.
      Check `dispatcher.py` and confirm `FromExtrasPairsCSR`/stencil keys are not
      leaking into kernels.
- [ ] `AutoPairs` builds on the fly (no extras), `FromExtrasPairsCSR` reads
      user-provided CSR (genuine input, never rebuilt/memoised), stencils build from
      `box/grid_shape` descriptors. Confirm these three remain distinct.

### 3. Reserved extras
- [ ] `time`, `duration`, `dataset_index`, `particle_index` resolve through the single
      registry in both inference (`build_extras`) and simulation (`resolve_reserved`
      in the schedule path) so a force sees the **identical** mapping either way.
- [ ] Pooled / multi-dataset fits: `dataset_index` is keyed on dataset `uuid` (stable
      under reorder/concat); `particle_index` matches `N` per dataset. Cross-check
      `tests/inference/test_per_dataset.py` and `tests/trajectory` reserved tests.

### 4. Time-varying extras (subtle — verify carefully)
- [ ] `TimeSeriesExtra` is sliced per frame in inference (`slice_frame_extras`) and
      becomes a per-frame **schedule** in simulation; shapes are `(T, ...)`/`(Nsteps, ...)`.
- [ ] Known runtime caveat: time-varying extras are effectively **batch-constant
      unless the per-`t` core runtime fires** (the batch producer gathers extras once
      per chunk). Confirm programs that read time-dependent extras force the per-`t`
      path (see `integrate(..., batch=...)` and `_has_time_varying_required_extras`).
- [ ] `simulate_chunked` rejects time-dependent extras (it re-invokes `simulate` per
      rebuild chunk) — confirm the guard is intact.

### 5. Structural build / JIT boundary
- [ ] Structural builds are **host-side** (NumPy, concrete `grid_shape`); the JIT eval
      only *reads* prepared arrays. Confirm no NumPy/`int(tracer)` runs under trace.
- [ ] `build_structural_overlay` drops pre-existing `_cache/` before building (a stale
      table cannot be reused); `prepare_collection_for_expr` is non-mutating and
      builds the **union** over multiple exprs (held-out compare path).
- [ ] Memoisation/perf: the stencil host-build runs **once per inference** and **once
      per simulate** (not per frame). Confirm the build-once paths (the `_structural_scope`
      single prepare; the process `_prepared_structural` store + invalidation).

### 6. Transforms (degrade / slice / concat / save-load)
- [ ] `degrade_spatial_data` and the frame slicer slice `TimeSeriesExtra` along time,
      preserve globals, and strip `_cache/` (these strips are now invariant guards —
      confirm they are no-ops in normal flow but still present).
- [ ] `concat` / pooled collections preserve per-dataset extras and `uuid`.
- [ ] Save/load round-trips extras (`SFI/trajectory/io.py`); confirm `_cache/` is never
      serialized.

### 7. Simulation specifics
- [ ] `_invalidate_prepared_extras` (called from `set_extras`/`set_params`) clears
      `_prepared_structural` **and** resets the prepared flag.
- [ ] `_model_extras` and the scan static-extras both include `_prepared_structural`;
      the early-return guards account for it (a process with only structural arrays
      must still surface them).
- [ ] Bootstrapped simulation from an inferred force (`simulate_bootstrapped_trajectory`)
      threads reserved `dataset_index` correctly.

## How to verify (commands)

```bash
# Inventory every extras touchpoint
grep -rn "_cache/\|extras_global\|extras_local\|build_extras\|prepare_collection_for_expr\|build_structural_overlay\|prepare_structural_extras_for_expr\|purge_cache_extras\|structural_extras\|required_extras\|TimeSeriesExtra\|FunctionExtra\|resolve_extras\|resolve_reserved\|_prepared_structural\|_structural_scope" SFI/

# Confirm no leftover collection/dataset-level purge helpers or stray prepare/purge scatter
grep -rn "purge_collection_cache\|purge_dataset_cache" SFI/ tests/

# Targeted tests (fast-ish, exercise the extras paths)
.venv/bin/python -m pytest tests/statefunc/test_structural_overlay.py \
  tests/inference/test_per_dataset.py tests/trajectory -q -p no:cacheprovider

# Reserved + time-dependent extras
.venv/bin/python -m pytest tests/ -q -k "extra or reserved or time_dependent or fourier" -p no:cacheprovider

# Full suite is the definitive gate (~45 min)
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

## Red flags

- A `_cache/` key found on a returned dataset, a saved file, or `self.extras_global`
  after a call completes.
- A `prepare_*` without a matching teardown, or a `self.data = prepare(...)` that is
  not inside `_structural_scope` (re-introduces the forgotten-purge bug).
- `int(...)` / NumPy on an extras value inside a `@jit`/`vmap`/`lax.scan` region.
- A time-varying extra read on the **batched** integration path without forcing the
  per-`t` runtime (silently uses the frame-0 value).
- Reserved keys resolved differently in simulation vs inference (a force that behaves
  differently when simulated vs fit on the same data).
- `FromExtrasPairsCSR` arrays being memoised/rebuilt instead of passed through (breaks
  dynamic neighbour lists / future T1 topology).

## Deliverable

A short report: per checklist section, PASS / FINDING with file:line evidence; a list
of any inconsistencies with suggested fixes; and confirmation that the full suite +
the interaction gallery (`gray_scott`, `abp_to_spde`, `abp_nonreciprocal`) pass.
