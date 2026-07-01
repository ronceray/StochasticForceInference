# Dynamics-order classifier — targeted gallery scenarios

**Date:** 2026-06-27
**Status:** approved (brainstorm)

## Motivation

`SFI.classify_dynamics` (the overdamped-vs-underdamped classifier, added
2026-06-17) shipped with a single gallery demo
(`examples/gallery/dynamics_order_demo.py`) showing one clean overdamped
double-well and one clean underdamped oscillator, both under heavy
localization noise. The demo's *Takeaways* claim two further behaviors —
the honest `inconclusive` verdict at coarse sampling, and operation beyond
1D — that it never actually demonstrates. The classifier is also lightly
tested in those regimes. This work exercises it across two more scenarios
so we get a clear view of its utility (and limits), and pins the new
regimes with unit tests.

## Scope (decided in brainstorm)

- **Goal:** more worked scenarios (illustrative breadth), not a regime-map sweep.
- **Scenarios:** (1) coarse-sampled → `inconclusive`; (2) a 2D system.
- **Placement:** extend the existing `dynamics_order_demo.py` (one coherent page).

## Design

### Section A — Coarse sampling → honest `inconclusive`

Underdamped damped oscillator (`F = -k x - γ v`, `τ_v = 1/γ`) sampled coarsely
so `γ·Δt ≳ 1` and momentum is only partially resolved. The verdict logic
returns `inconclusive` when the inertial model is still statistically
preferred (`ΔAICc > 2`, `V_z > 3`) yet the noise-immune lag-2 persistence
`ρ₂` at the finest stride lands in `[0.05, 0.15]`. The section classifies,
`print_summary()` (showing the `gamma*dt_min ~ …` tail), plots the verdict,
and calls out `meta["gamma_dt_min"]`.

Parameters (`γ`, `Δt`, `Nsteps`, strides, seed, noise) are tuned empirically
so the verdict is deterministically `inconclusive`, verified by running the demo.

### Section B — 2D overdamped with a non-conservative rotational force

2D overdamped trajectory in an isotropic trap with a weak curl component,
`F(x) = -k·x + ω(ẑ × x)` — a detailed-balance-breaking rotational force,
on-theme for the lab. Under the same heavy localization noise the verdict
should be **OD**: a rotational force is first-order and cannot fake inertia
(`ρ₂ → 0`). Exercises isotropic component pooling beyond 1D and makes the
point that non-equilibrium driving ≠ inertia. Shows the 2D trajectory and
the verdict panel.

### Housekeeping
- Update *Takeaways* to reference the now-demonstrated `inconclusive` and 2D cases.
- Update `sphinx_gallery_thumbnail_number` to remain valid (page grows to ~6 figures).
- Extend the header tag lines (`2D`, `non-equilibrium`).

## Testing

Add focused cases to `tests/diagnostics/test_dynamics_order.py`:
- a coarse-sampled UD collection → `verdict == "inconclusive"`;
- a 2D rotational overdamped collection → `verdict == "OD"`.

These lock the demo's claims to behavior, addressing the "not extensively
tested" concern. Tests use the same tuned parameters as the demo (kept modest
for CPU runtime; `cross_check=False` where the OD fit is not the point).

## Build & verify (gallery gotchas)

1. Run the demo standalone (`JAX_PLATFORMS=cpu`) to confirm verdicts + figures.
2. Filtered single-page regen:
   `cd docs && SFI_DOCS_SKIP_STALE=1 SFI_DOCS_RUN_GALLERY=1 SFI_DOCS_GALLERY_FILTER='dynamics_order_demo\.py$' make clean html`.
3. Save the target's regenerated artifacts, `git checkout HEAD -- docs/source/gallery`
   and `docs/source/sg_execution_times.rst`, then restore the target's files.
4. Delete the orphan `docs/source/gallery/advanced/index.rst` if regenerated.
5. Regenerate + sanity-check the standalone notebook.
