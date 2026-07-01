# Changelog

## Unreleased

### Added

- **Dynamics-order classifier** (`SFI.classify_dynamics`): decides whether
  trajectory data is overdamped or underdamped — or *inconclusive* — directly
  from position increments, with no prior choice of inference engine. Robust to
  high localization noise (lag ≥ 2 displacement covariances are
  measurement-noise-immune by construction) and to coarse sampling (reports
  *inconclusive* when `γ·Δt ≳ 1`, where momentum is unresolved). Combines a
  noise-corrected lag-2 / apparent-kinetic-energy `Δt`-scaling test, a
  parametric fit of the diffusion + inertia + localization covariance model
  (recovering the momentum relaxation time `τ_v` with an AICc OD-vs-UD
  comparison), and an overdamped-fit residual-autocorrelation cross-check.
  Returns a `DynamicsOrderReport`; `SFI.diagnostics.plot_dynamics_order`
  visualises the scaling and verdict.

## 2.0.0 (2026-06)

First public release of the rewritten, JAX-based SFI.

### Highlights

- **Two inference paths per engine** (overdamped / underdamped):
  - `infer_force_linear` — the closed-form basis projection (the
    linear estimators: Itô/Strato/symmetric moments, trapeze Gram,
    PASTIS sparsity).
  - `infer_force` — the parametric estimator: a single RK4 flow step
    per observation interval, windowed-precision likelihood, native
    `(D, Σ_η)` profiling, and the skip-trick errors-in-variables
    instrument for consistency under measurement noise.  Takes a
    `Basis` (direct Gauss–Newton) or any differentiable `PSF`
    (frozen-precision L-BFGS), including interacting multi-particle
    models.
- **State-dependent diffusion**: `infer_diffusion` (rank-2 basis or
  direct PSF, velocity-dependent in the underdamped case) on the same
  windowed conditional likelihood.
- **Compositional model building** (`SFI.statefunc`, `SFI.bases`):
  symbolic bases, pair interactions, and neural-network force fields
  from the same expression algebra.
- **Diagnostics** (`SFI.diagnostics.assess`): residual whitening tests,
  autocorrelation/normality batteries, predicted-vs-realised error
  consistency, and a four-point cross-covariance misspecification test.
- **Benchmarks**: systematic NMSE grids (time-step × noise ×
  trajectory length) for Lorenz, double-well, Van der Pol, harmonic,
  and aligning ABPs, plus the v2.0 parametric decision benchmark.
