# Design spec — Unified extras resolver (Phase B)

**Status:** draft for review · **Date:** 2026-06-17
**Scope:** Phase B (single resolver, reserved-as-declared, simulation/inference
unified). Phase C (re-channel structural/CSR extras out of the user-extras
namespace) is explicitly **out of scope** here and sequenced as a follow-up.

## Context

Extras are SFI's mechanism for passing non-state data to a force/diffusion
expression (drive protocols, per-particle properties, geometry, neighbour
lists, time). The *declaration* of what extras a model needs already lives on
the consumer (`StateExpr`): leaves declare `extras_keys=(...)` and
`particle_extras=(...)`; interaction `SpecRule`s declare `required_extras()`
(forwarded) and `structural_extras()` (CSR, dispatcher-owned, never forwarded).
`stateexpr.py` already names three kinds *by declaration*: **global /
particle / structural**.

The **provision** side, however, is scattered, and that scatter has produced
real bugs (two fixed this session in `SFI/diagnostics/residuals.py`):

- **Reserved keys are provided ad hoc, per call site.** `dataset_index` /
  `particle_index` come from `TrajectoryCollection.injected_extras()`; the
  per-frame `time` key is injected inline by the integrate engine
  (`integrate/api.py`). Diagnostics had to *replicate* this by hand
  (`_force_extras`) and silently drifted — it never injected `dataset_index`
  and (still) never injects `time`.
- **Simulation has a *separate* time pipeline.** `LangevinBase._split_extras`
  classifies user extras into static vs per-frame *schedules*
  (`TimeSeriesExtra` / `f(t)` materialised at `k·dt`) and merges
  `{**static, **sched}` each integrator step. The reserved `time` key is an
  *inference* construct (`collect_extras(t)`), absent from this path — so the
  same `extras["time"]`-reading force is not provided consistently across
  simulate vs infer.
- **No registry.** There is no single list of reserved keys or of how each is
  resolved; the knowledge is duplicated across `collection.py`, `api.py`,
  `factory.py`, and `residuals.py`.

## Goal

One resolver, driven by the consumer's declared contract, used **identically**
by simulation, inference, and diagnostics. Concretely:

1. A single source of truth for reserved extras (which keys, how each resolves).
2. `time` provided the same way in simulation and inference — **no bespoke
   time-handling in simulation**.
3. `dataset_index` resolved from a **stable dataset identity** (robust to
   `concat`/`split`/reorder/held-out), not list position.
4. The diagnostics hand-rolled injection (`_force_extras`) and the inline
   engine/sim injections are all replaced by calls to the resolver.

Non-goals (Phase C): moving CSR/structural extras out of the `extras` dict;
changing the `_cache/` convention; changing interaction dispatch.

## Conceptual model

Extras are a **consumer contract** with a **provider taxonomy**. The consumer
declares *what* it needs; the resolver decides *who* supplies each, by category:

| Category (how declared) | Examples | Provider | Lifetime |
| --- | --- | --- | --- |
| user-global (`extras_keys`) | drive `a`, condition `c`, box | dataset `extras_global` / sim `set_extras` | persists with data |
| user-particle (`particle_extras`) | radius, species, home-range | dataset `extras_local` | persists with data |
| user per-frame (schedule) | `f(t)` / `TimeSeriesExtra` | dataset (sliced at frame) | persists with data |
| **reserved** (`extras_keys`, but registered) | `time`, `dataset_index`, `particle_index` | **resolver/registry** | derived, transient |
| structural (`structural_extras`) | CSR `indptr`/`indices` | rule/dispatcher (`_cache/`) | computed, dropped on transform |

The only *new* idea in B is making the **reserved** row first-class: declared
like any key, but resolved by a registry rather than by hand at each site.
Structural stays exactly as today (Phase C).

## Components

### 1. Reserved-extras registry — `SFI/statefunc/reserved_extras.py` (new)

The single authority. Each reserved key registers a name, a *kind* (shape
contract), and a resolver:

```python
@dataclass(frozen=True)
class ReservedKey:
    name: str
    kind: Literal["per_dataset_scalar", "per_particle", "per_frame"]
    resolve: Callable[["ExtrasContext"], Array]

RESERVED: dict[str, ReservedKey] = {}          # registry
def register_reserved(key: ReservedKey) -> None: ...
def is_reserved(name: str) -> bool: ...
RESERVED_NAMES: frozenset[str]                  # for collision checks
```

Initial registrations:

- `particle_index` (`per_particle`) → `jnp.arange(ctx.n_particles, int32)`
- `dataset_index` (`per_dataset_scalar`) → `ctx.dataset_index` (see §4)
- `time` (`per_frame`) → `ctx.frame_times` (absolute time per resolved frame)

`ExtrasContext` carries everything a resolver can need, assembled once by the
caller:

```python
@dataclass(frozen=True)
class ExtrasContext:
    n_particles: int
    dataset_index: int           # dense index from the collection registry (§4)
    frame_times: Array | None    # (K,) absolute time at the resolved frames
    # room to grow (e.g. dataset_id) without touching call sites
```

### 2. The resolver — `resolve_extras(...)` in the same module

```python
def resolve_extras(required, particle_required, *, data_extras, context) -> dict:
    """Merge user data-extras with resolved reserved extras for a kernel call.

    `required` / `particle_required` come from StateExpr.required_extras /
    .particle_extras. For each required key: if registered reserved -> call its
    resolver(context); else -> take from data_extras (KeyError if absent, as
    today). Returns the kernel-facing extras dict.
    """
```

This is the one place reserved keys are materialised. Inference, diagnostics,
and simulation all call it.

### 3. Per-frame slicing — shared utility

Both inference (`collect_extras(t)` slicing `TimeSeriesExtra` per frame) and
simulation (`_split_extras` materialising schedules) implement "value of a
time-varying extra at frame(s) `t`". Extract one utility,
`slice_frame_extras(data_extras, frame_idx) -> dict`, used by both. The
reserved `time` key flows through the *same* per-frame machinery (it is just a
registered `per_frame` reserved key), which is what removes simulation's
separate time path.

### 4. Dataset identity + collection registry

- `TrajectoryDataset` gains an optional `name: str | None` (stable identity).
  Default `None`. Survives `save`/`load`, `concat`, `split` (it is metadata on
  the dataset).
- `TrajectoryCollection` builds an **ordered identity registry**: each dataset's
  identity (its `name`, or a positional fallback when unset) maps to a dense
  `[0, n)` index. `concat` **appends** unseen identities (existing indices
  preserved); `dataset_index` is read from this registry, not from `enumerate`.
- `injected_extras` / the resolver compute `dataset_index` via this registry.
- Optional (kept for debugging / robust held-out mapping): also expose
  `dataset_id` (the identity string) as a reserved key in a later increment;
  not required for B.

**Backward-compat:** with no `name`s set, the registry degenerates to positional
indices — byte-for-byte today's behaviour.

## Data flow (unified)

```
consumer (StateExpr) ──declares──> required / particle_required
                                          │
caller builds ExtrasContext (n_particles, dataset_index from registry, frame_times)
                                          │
data_extras = slice_frame_extras(dataset user extras, frame_idx)   # time-varying handled here
                                          │
extras = resolve_extras(required, particle_required, data_extras=…, context=…)
                                          │
                          ┌───────────────┼───────────────┐
                       simulate         infer           diagnose      (identical call)
```

## Call-site changes

- **`SFI/diagnostics/residuals.py`** — delete `_force_extras`; both builders call
  `resolve_extras(...)` with an `ExtrasContext` built from the enumerated
  dataset + `frame_times` from the producer at `t_idx` (the anchor frame in
  both the OD increment and UD symmetric stencil — uniform). This *also*
  delivers `time` to diagnostics, fixing the remaining gap, and supersedes the
  two interim fixes from this session.
- **`SFI/integrate/api.py`** — replace inline `time` injection and the per-`t`
  `collect_extras` reserved-key handling with `resolve_extras`. Keep the
  batch-vs-per-`t` routing: a program whose `required` contains a `per_frame`
  key (reserved `time` or a user schedule) runs on the per-`t` core
  (`_has_time_varying_required_extras` stays, but reads the registry).
- **`SFI/langevin/base.py` + `overdamped.py`/`underdamped.py`** — `_split_extras`
  keeps classifying *user* static vs schedule, but the per-frame application and
  the reserved `time` now go through `slice_frame_extras` + `resolve_extras`, so
  there is no simulation-specific time logic. Oversampling semantics
  (schedule held constant across substeps) preserved.
- **`SFI/trajectory/collection.py`** — `injected_extras` delegates to the
  registry; add the identity registry and `dataset_index`-from-identity.
- **`SFI/trajectory/dataset.py`** — add `name`; producer per-frame extras go via
  `slice_frame_extras`.

## Migration / backward-compat

- Consumers unchanged: reserved keys still declared via `extras_keys`; the
  registry classifies them. No basis edits.
- No identities set ⇒ positional `dataset_index` (today's behaviour).
- Reserved collision policy preserved: a user key colliding with a reserved name
  raises (today's `injected_extras` behaviour), now centralised in the registry.
- Saved files unaffected (reserved extras remain transient; `name` is optional
  new metadata, additively serialised).
- The interim `residuals.py` fixes (commits `6726e74e9`, `fe2a02648`) are
  subsumed and their hand-rolled logic removed; their regression tests stay.

## Testing

- **Registry/resolver units:** each reserved key resolves to the expected
  shape/value; `resolve_extras` partitions reserved vs user; missing user key
  raises; reserved collision raises.
- **Golden equivalence:** for `time` / `dataset_index` / `particle_index`,
  resolver output matches the *current* engine/collection injection on a fixed
  fixture (pin old behaviour before refactor).
- **Sim↔infer invariant (the headline):** one force reading `extras["time"]`
  yields identical per-frame values when simulated and when re-evaluated in
  inference/diagnostics on the produced trajectory.
- **Dataset identity:** `concat`/`split`/reorder preserve the
  `identity→dataset_index` map when `name`s are set; held-out
  `assess(inf, data=test)` aligns per-dataset params by identity.
- **Regression:** the two diagnostics extras tests added this session; full
  `tests/diagnostics`, `tests/langevin`, `tests/integrate`, and
  `tests/inference/test_per_dataset.py`.
- **Perf guard:** a static-only program still takes the batch fast-path (no
  regression to per-`t`).

## Risks

- **Per-`t` routing / perf:** time-varying extras must keep forcing the per-`t`
  path; static-only programs must keep the batch fast-path. Driven off the
  registry now — covered by the perf guard test.
- **Simulation integrator semantics:** schedule materialisation at `k·dt` and
  "held constant across oversampling substeps" must be preserved when moving to
  the shared slicing utility.
- **Identity touches save/load + concat/split:** needs golden tests; default-off
  (positional) keeps blast radius small until opted into.

## Sequencing within B (each step independently testable + committable)

1. **Registry + resolver + `ExtrasContext` + `slice_frame_extras`** — pure
   addition, no behaviour change. Add golden tests vs current outputs.
2. **Diagnostics → resolver** — simplest consumer; removes `_force_extras`,
   delivers `time` to diagnostics. (Validates the resolver end-to-end.)
3. **Inference (`api.py`) → resolver** — keep batch/per-`t` routing.
4. **Simulation → resolver + shared slicing** — the sim/inference unification;
   assert the sim↔infer `time` invariant.
5. **Dataset identity + collection registry** — `dataset_index` from identity;
   default-off positional fallback.

## Open decision for review

- **Identity type:** spec assumes a user-settable `name` (string) with
  positional fallback — meaningful for multi-experiment fits, fully
  backward-compatible. Confirm this vs. a UUID/content-hash default.
