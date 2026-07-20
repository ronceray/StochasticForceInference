# Changelog

## 2.0.2 (2026-07)

### Fixed

- **Overdamped Stratonovich force fit — gradient-correction contraction
  order.** The Itô-to-Stratonovich spurious-drift term contracts the
  instantaneous diffusion and the inverse-normalization matrix with the
  basis Jacobian as `(D_inst · A⁻¹) : ∇b`, with `D_inst` on the
  derivative index and `A⁻¹` on the output (force) index. The two
  matrices were contracted in the opposite order, computing
  `(A⁻¹ · D_inst) : ∇b`. The results coincide whenever `A⁻¹` and
  `D_inst` commute — isotropic diffusion, the `rectangle`-Gram presets,
  and any simultaneously-diagonalizable pair — so the discrepancy was
  invisible in those cases and appeared only for anisotropic,
  non-aligned instantaneous diffusion combined with a non-trivial
  decorrelating Gram (the `robust`/`shift` presets). Affects Stratonovich
  force coefficients only; Itô and Itô-shift fits, diffusion inference,
  and all commuting-matrix configurations are unchanged.

## 2.0.1 (2026-07)

### Added

- **Entropy production estimation**
  (`OverdampedLangevinInference.compute_entropy_production`): the
  Frishman & Ronceray (2020) dissipation estimator, reconnected to the
  v2 moment engine.  Projects the phase-space velocity onto the force
  basis (Stratonovich v-moments, reused from a `Strato` fit or computed
  on demand after any preset) and contracts with the inverse average
  diffusion; reports the rate, its estimated error, the `2·N_b/τ_N`
  fluctuation bias and the bias-subtracted `Sdot_debiased` /
  `DeltaS_debiased` (the fluctuation bias is exactly the AIC complexity
  correction on the trajectory information — both engines expose the
  debiased fields), and fills the `DeltaS` / `error_DeltaS` /
  `DeltaS_debiased` fields of `print_report()` / `report_dict()`.
  Respects the PASTIS support (`support="current"`, default) or the
  full basis (`support="full"`).  Independent localization noise does
  *not* bias the estimator at leading order — the Stratonovich moments
  are odd under time reversal while i.i.d. noise is even, so the naive
  `O(Λ/Δt)` term cancels (verified analytically and numerically;
  motion-blur-correlated noise does not cancel) — warnings and docs
  state the corrected picture.
  Companion docs: a *Stochastic thermodynamics with SFI* concepts page
  (equilibrium with multiplicative noise `F_Itô = −D∇U + ∇·D`, the
  `D̂⁻¹(F̂ − ∇·D̂)` gradientness test, joint `(U, D)` workflows,
  gradient-basis potential fits) and the `entropy_production_demo`
  gallery page (driven optical trap, detection-limit sweep, parametric
  energy recovery).

- **Underdamped entropy production from positions only**
  (`UnderdampedLangevinInference.compute_entropy_production`): the fitted
  acceleration field is split by time-reversal parity (`x` even, `v` odd;
  `time_reversal_split()` exposes the reversible/irreversible parts) and
  the Stratonovich log path-probability ratio
  `ΔS = Σ (dv̂ − F̂⁺dt)·D̄⁻¹·F̂⁻` is evaluated with the ULI-reconstructed
  kinematics — no velocity measurements needed.  Provisional error bars:
  analytic odd-sector scale `√(2Q⁻ + (2N⁻)²)` (with `N⁻ = rank G⁻⁻`
  counting odd degrees of freedom) combined with an empirical
  block-variance estimate; the same-sample plug-in bias is `O(1/τN)` and
  removable by cross-fitting via `coefficients=`.  Validated against the
  exact parity-aware phase-space OU rate (Lyapunov) and a `γΔt`
  calibration sweep.  The underdamped simulator now implements
  `compute_observables=True` (information + entropy accumulated with the
  true internal velocities, trapezoid parity split).

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

- **Non-uniform sampling in the parametric estimators**: datasets
  carrying an absolute time vector `t` or a per-step `dt` array (both
  long-standing data-model features) now fit directly — force and
  diffusion, overdamped and underdamped — with per-interval flow steps
  and process covariances (the underdamped case requires the default
  Lyapunov-exact covariance blocks).  Previously the parametric solvers
  applied the first interval to every step.

- **`specialize()` derivative support**: `StateExpr.specialize(dataset=k)`
  now folds through `DerivativeNode` / `SliceFeaturesNode` /
  `ReshapeRankNode`, so pooled models built with `.d_x()` (gradient
  bases, potential-parametrized forces) collapse to single-condition
  form like any other expression.

### Performance

- **Parametric estimators (`infer_force` / `infer_diffusion`, OD + UD) are
  much faster and lighter, especially for wide or expensive bases.** The
  regressor `ψ = ∂r/∂θ`, the flow Jacobian, and the errors-in-variables
  instruments now come from the exact per-stage θ-recursion instead of
  `jax.jacfwd` with `n_params` forward tangents through the RK4 flow — one
  force / `∂F/∂x` / `∂F/∂θ` evaluation per stage, with a zero-AD fast path
  for linear bases. This also removes the second-order-AD compile wall for
  bases containing derivatives of fitted fields.  Typical fits run 2–3×
  faster end-to-end.
- `infer_diffusion` and the `(D, Λ)` profile of `infer_force` optimise
  the likelihood over cached fixed-θ tensors: zero force/basis
  evaluations per profile iteration.

### Changed

- **Parametric accuracy upgrades** (default-on):
  - the process covariance integrates the Lyapunov equation
    `Q̇ = JQ + QJᵀ + 2D` on the same RK4 stages as the flow — removes an
    `O((kΔt)²)` covariance error, worth 10–60× in diffusion NMSE at
    coarse sampling (`k·Δt ≳ 0.1`), with force estimates unchanged;
  - a convexity correction cancels the `½∇²Φ:Λ` residual-mean bias that
    measurement noise induces through drift curvature (noise/EIV path,
    non-interacting models);
  - the errors-in-variables instrument stays active for nonlinear-in-θ
    PSF models, which now solve by damped Gauss–Newton (they previously
    fell back to the plain quasi-likelihood); a sensitivity-collapse
    guard detects degenerate nonlinear models and falls back to the
    robust path instead of diverging;
  - substantially more robust at high measurement noise, most visibly
    for underdamped data (up to 25× lower NMSE on noise-dominated
    benchmark cells).
- New guidance for coarse-sampling regimes: on (near-)clean data at
  coarse effective sampling pass `eiv=False` (the `(D, Λ)` profile
  cannot separate a Δt-independent Λ from the process increment
  structure there and can fit a spurious measurement noise) and
  `n_substeps=2` (the single-step flow model degrades when the dynamics
  is fast on the observation interval).
- The `(D, Λ)` profile's default iteration budget is raised to 200
  (profile iterations run on cached tensors and are cheap; the
  historical 20-iteration budget could stall the noise profile far from
  its optimum on badly-scaled landscapes — interacting underdamped
  flocks were the observed case — with the stalled `Λ̂` then biasing the
  force fit through the weights).
- **Müller-Brown gallery demo reworked as a potential fit**
  (`nn_force_demo`): the MLP and the monomial library now parametrize
  the scalar energy landscape `U(x)` (rank-0 expressions) and obtain
  the force by automatic differentiation, `F = −U.d_x()` — every fitted
  model is conservative by construction; the polynomial route stays
  linear (gradient basis `b_i = −∇u_i`).  Adds an inferred-landscape
  comparison figure; potential surfaces use the reversed-viridis
  colormap.

### Deprecated

- The `extra_radius` and `n_cond` kwargs of the parametric estimators
  no longer have any effect (the estimator has no window truncation to
  tune); they are accepted for compatibility and fire a
  `DeprecationWarning` when set.

### Fixed

- The parametric path now honors `max_memory_gb` (it was silently ignored —
  the engine default always applied), and its chunk planning accounts for
  the real working set (ψ/J/r buffers, whitening transients, basis
  memory hints) instead of the packed output only, so wide or expensive
  bases no longer exhaust memory.
- The parametric solvers always run in float64 internally (scoped
  `jax.enable_x64`, float inputs upcast), whatever the session dtype: the
  block Cholesky factors and the Gram accumulation over ~1e5–1e6
  residuals are unreliable in float32. Results are returned in float64;
  the session configuration is untouched.
- The moment diffusion estimators handle non-uniform sampling correctly.
  Overdamped: both the Vestergaard (`noisy`) and the weak-noise stencils
  divide by the two-interval mean `(dt + dt⁻)/2` — exactly unbiased at
  any spacing, where the previous centre-`dt` division was biased by
  `E[(dt+dt⁻)/2dt] − 1` (e.g. +33 % on an alternating 1:3 grid).
  Underdamped (ULI `noisy`): the stencil's ballistic cancellation only
  holds at equal intervals — under non-uniform sampling the leaked
  `⟨v²⟩·(δ⁻² + δ⁺² − 3δ⁻δ⁺)/dt̄³` term could dwarf (or turn negative)
  the estimate; it is now subtracted using the η-clean `dX⁺⊗dX⁻`
  velocity-square estimate (which shares no localization noise, so the
  Λ cancellation is untouched), with the `dt̄³` normalisation on the
  local three-interval mean.  All corrections vanish bitwise at uniform
  sampling.
- The parametric parameter covariance uses a stable pseudo-inverse — finite
  error bars for rank-deficient (collinear-basis) Grams.
- Gallery: the `lorenz_demo` notebook download was missing from the
  repository (an unanchored `.gitignore` pattern swallowed it); the
  per-page tag pills and the tag-filter bar render correctly on the
  online docs.

## 2.0.0 (2026-06)

First public release of the rewritten, JAX-based SFI.

### Highlights

- **Two inference paths per engine** (overdamped / underdamped):
  - `infer_force_linear` — the closed-form basis projection (the
    linear estimators: Itô/Strato/symmetric moments, trapeze Gram,
    PASTIS sparsity).
  - `infer_force` — the parametric estimator: a single RK4 flow step
    per observation interval, block-banded Gaussian likelihood, native
    `(D, Σ_η)` profiling, and the skip-trick errors-in-variables
    instrument for consistency under measurement noise.  Takes a
    `Basis` (direct Gauss–Newton) or any differentiable `PSF`
    (frozen-precision L-BFGS), including interacting multi-particle
    models.
- **State-dependent diffusion**: `infer_diffusion` (rank-2 basis or
  direct PSF, velocity-dependent in the underdamped case) on the same
  conditional likelihood.
- **Compositional model building** (`SFI.statefunc`, `SFI.bases`):
  symbolic bases, pair interactions, and neural-network force fields
  from the same expression algebra.
- **Diagnostics** (`SFI.diagnostics.assess`): residual whitening tests,
  autocorrelation/normality batteries, predicted-vs-realised error
  consistency, and a four-point cross-covariance misspecification test.
- **Benchmarks**: systematic NMSE grids (time-step × noise ×
  trajectory length) for Lorenz, double-well, Van der Pol, harmonic,
  and aligning ABPs, plus the v2.0 parametric decision benchmark.
