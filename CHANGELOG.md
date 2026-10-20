# Changelog

## 2.0.4 (2026-10)

### Changed

- **Overdamped instantaneous-diffusion stencils are attached to their first
  point.**  `"noisy"` and `"WeakNoise"` use the increments `dX_t, dX_{t+1}`
  at `X_t` (conditioned on `X_t` both lie in the future, so the estimate is
  unbiased state-resolved; the shared-point attachment is unbiased only in
  the mean).  With this attachment the lag-2 correction stencil
  (`"noisy_lag2"`) takes its forward three-term form,
  `[-a (x) c - c (x) a + b (x) m] / (2 dt)` added to Vestergaard
  (`a, b, c = dX_t, dX_{t+1}, dX_{t+2}`, `m = dX_{t-1}`), which cancels the
  O(dt) bias for nonlinear dynamics; the single shifted lag-2 product does
  so only for linear dynamics.  The benchmark figures quoted for
  `"noisy_lag2"` below were measured with the backward single-product form.
- **Entropy production and model selection normalize by their own rows.**
  `integrate(..., return_measure=True)` returns, with a sum, the particle-time
  (dataset weights included) of the rows that sum covers; estimators normalize
  by that measure instead of recounting rows.  The overdamped entropy rate
  divided the fit's sums by the particle-time of every `(t, t+1)` row, so with
  missing frames it came out low by the row-count ratio (`preset="robust"`,
  5 % frame loss: 1.44 instead of 1.97, i.e. 0.95³); it now uses the force
  Gram's measure, which also makes it exact for weighted multi-dataset
  collections.  The underdamped parity moments, `tau_N` and the block error
  bar share one row set; the block error bar is built from the masked per-row
  contributions (it used the raw positions and was silently dropped when
  masked frames held NaN) and now covers multi-dataset collections.  The
  BIC/EBIC sample size of `sparsify_force` / `sparsify_diffusion` is the
  linear fit's row measure.
- **Underdamped `compute_entropy_production`: finite-Δt counterterm and
  coefficient-bias diagnostic.**  The moment objects are now evaluated on
  the symmetric kinematics with their derived O(Δt) counterterm (cusp,
  positional-defect, gyroscopic and Stratonovich-cusp terms; returned as
  `DeltaS_dt`), valid for any coefficients.  The coefficients remain the
  force fit's.  Their own O(Δt) error — the trapezoid Gram on secant
  velocities, which on the driven inertial trap inflates the rate to 1.24σ,
  1.52σ, 2.22σ at γΔt = 0.1, 0.2, 0.4 (exact infinite-data values) — is
  reported to first order as `Sdot_dt_bias` (residual of a kinematically
  consistent normal system at the fit's coefficients; an order-of-magnitude
  flag, not subtracted, and `None` for interacting bases or fits with the
  `noisy` correction).
  Interacting bases keep the uncorrected objects, with a warning.

### Performance

- **`simulate_chunked` no longer recompiles at every neighbour rebuild.**
  `OverdampedProcess.simulate` / `UnderdampedProcess.simulate` closed the
  extras over a freshly jitted step, so each rebuilt CSR list was a new
  compile-time constant and a new compile (≈1–2 s per rebuild on the ABP
  demos).  The whole run is now one jitted function, cached on the process,
  that takes array-valued extras as traced arguments: a rebuild at fixed
  `max_nnz` reuses it (≈1 ms per call on a small system instead of ≈0.7 s),
  and only a `max_nnz` growth recompiles.  Trajectories are bitwise
  unchanged; the per-rebuild `jax.clear_caches()` is gone.
- **Parametric estimator: `max_memory_gb` bounds the working set; large
  models no longer exhaust memory.**  The parametric core sized its chunks
  from a static per-point estimate that missed the per-stage θ-recursion
  intermediates of the Gauss–Newton Gram by an order of magnitude, and a
  1000-residual floor overrode small budgets: the Müller–Brown
  neural-network fit (336 weights, 15k frames) built its Gram in one chunk
  of 6–10 GB at the default 1 GB.  Each per-chunk program (Gram, NLL,
  fixed-θ cache, differentiated L-BFGS loss) is now sized from XLA's buffer
  assignment of the compiled program, like the integrate engine's planner
  (no floor), and a multi-chunk loss rematerialises each chunk in its
  backward pass.  `∂F/∂θ` uses reverse-mode AD when the model has more
  parameters than the frame force has entries (forward otherwise), at the
  same template-rounded θ and in float64, so it matches forward mode to
  float64 round-off.  An `inner="auto"` Gauss–Newton runaway on a nonlinear
  model is caught at the step that collapses the Gram, before the (D, Λ)
  reprofile and further Gram builds; the L-BFGS fallback is unchanged.  On
  that fit with the library defaults: 1.8 GB peak RSS (previously > 6 GB)
  and 67 s (the previous `auto` route took 117 s even at
  `max_memory_gb=0.1`).  Fitted θ, D and Λ are bit-identical at equal chunk
  sizes; a different budget changes only the floating-point summation order
  (which an unconverged non-convex fit can amplify).  The gallery demo no
  longer needs `max_memory_gb=0.1`.

### Fixed

- **Float64 parametric fits no longer evaluate models at float32-rounded
  parameters.**  `ParamSpec` defaults to float32 and the parameter suite cast
  every value to the spec dtype, so under x64 (the parametric solvers' scope)
  each model evaluation, L-BFGS gradient and fitted `SF` coefficient used θ
  rounded to float32 (gradients ~3e-8 off, finite differences useless).  A
  floating spec now keeps a wider floating value (`promote_types`); float32
  sessions, integer parameters and Python scalars are unchanged.  Parametric
  results move at the 1e-8–1e-5 relative level (goldens regenerated).

- **Successive fits on one inference object no longer leak state.**  A linear
  diffusion (or force) fit after a parametric one inherited the parametric
  method flag, Fisher metric and error outputs, so `compute_diffusion_error`
  and `sparsify_diffusion` took the parametric branch (wrong covariance, or a
  silent parametric re-fit).  A refused re-fit no longer overwrites the
  existing fit.
- **`param_cov` after sparsification has the full parameter size.**  It held
  the support-sized covariance, so `predict_ci` was wrong (3.7× too wide on a
  one-term support) or raised; sparsify now recomputes the error outputs when
  they had been computed.
- **Compiled-run cache served stale extras.**  A dict or list extra edited in
  place reused the previous compiled simulation; NumPy-scalar extras
  recompiled on every call and typed PRNG-key extras crashed.  Extras are
  flattened (arrays traced, scalars keyed by value) and the cache is bounded.
- **Dead pixels with the noise-robust preset (lattice fields).**  The all-pairs
  Stratonovich correction zeroed the increments of observed neighbours left
  out of the fit; it now zeroes only neighbours with unknown state.
  `degrade_spatial_data(stencil_radius=…)` excludes frames t−1…t+3.
- **`degrade(downsample=…)` corrupted static extras.**  Integer tables such as
  CSR neighbour lists were cut to the kept frames and cast to float64
  (crashing the next fit), while `TimeSeriesExtra` values were not resampled.
- **Parametric integrators and per-frame neighbour lists.**  A program built
  under `shared_rows` and traced later read a different pair-image stream;
  per-frame (`TimeSeriesExtra`) neighbour lists did not reach the parametric
  core.  `"pbc_ref"` is now a reserved extras name.
- **Smaller fixes.**  `predict_ci`/`predict_var`/`predict_cov` take `v=` for
  underdamped fields; a parametric fit returning θ ≡ 0 warns; the
  `[mse_consistency]` flag states the χ² excess instead of reading as a
  measured error ratio; basis indexing with NumPy/JAX index arrays (and
  NumPy integer scalars, boolean masks) works without warnings;
  `model_summary` styles only on a TTY; `SparsityResult.all_ic` reuses the
  recorded selection; plotting papercuts (`comparison_scatter` ticks,
  Pareto legend, `plot_dynamics_order` layout, `plot_field_error` label).

- **Stratonovich moments of periodic pair bases are unbiased.**  The pair
  primitives of `SFI.bases.pairs` re-picked each pair's minimum image at
  every frame, so a direction-carrying feature `φ(r) r̂_ij` with
  `φ(L/2) ≠ 0` flipped sign whenever a pair crossed `|Δx| = L/2` during a
  step.  The trapezoid `½[b(X_t) + b(X_{t+1})]·ΔX` integrated those jumps
  into an O(1) bias that did not shrink with Δt: on the aligning-ABP gallery
  setup (box 30, exp-poly kernels up to length 4) the Stratonovich force NMSE
  was 2.0e-2 against 3.6e-4 for Itô on clean data, and 1.9e-2 for the
  `"robust"` preset on the noisy tracks.  A program that evaluates periodic
  pair features at several frames (Stratonovich trapezoid, shifted Gram, ULI,
  entropy production) now puts every pair of a row on one image (the
  reserved per-particle extra `"pbc_ref"`), fixed at a frame that precedes
  every increment of the row: a frame inside an increment would switch the
  image with its localisation noise, which the increment carries divided by
  Δt (anchoring at `X` raised Itô-shift/shift from 9e-4 to 6e-2 on the noisy
  tracks), and a later frame depends on the increment itself.  A frame the
  estimator already reads is reused, so the overdamped `"auto"`, `"robust"`,
  `"clean"` and `"KM"` fits use exactly the rows of a free-space fit; only
  `"legacy-sfi-v1.0"` (frame t-1) and the underdamped `"robust"` / `"clean"`
  presets (frame t-2) need one more observed frame, and only for periodic
  bases.  `compute_entropy_production` after a fit whose rows read no such
  frame recomputes its Stratonovich moments and Gram on the widened rows.
  Stratonovich now gives 4.1e-4 (clean) and 1.6e-3 (`"robust"`, noisy,
  against 1.0e-3 for Itô-shift/shift); the aligning-ABP demo is back on the
  default preset.  `pbc_displacement` takes the reference as
  `ref=(ref_j, ref_i)` for hand-written interactors.  Hard kernel cutoffs and
  neighbour lists that truncate nonzero kernels are discontinuous in the same
  way and cannot be repaired by an image choice; the bases user guide
  documents the continuity requirement (kernels vanishing at their support,
  per-frame lists with a skin) with measured biases.
- **Per-frame neighbour lists are read per frame in the batched integrator.**
  Time-varying CSR / hyperedge tables are dispatcher-owned (structural)
  extras, absent from `required_extras`, so programs reading only them ran
  on the batched runtime, which gathers extras at a chunk's first frame:
  every row of the chunk used that frame's topology (force NMSE 1e-1 on a
  per-frame-list ABP fit, 7.5e-5 after).  Structural extras now count in the
  per-frame routing test.
- **Parametric `infer_force` / `infer_diffusion` accept time-dependent
  extras.**  The parametric runner forwarded the dataset extras verbatim, so a
  `TimeSeriesExtra` (e.g. a switched-trap protocol read by `extra_scalar`)
  reached the jitted flow as a non-array and raised a `TypeError`, and the
  reserved per-frame `time` (`time_fourier` bases) was dropped.  The
  time-dependent values the model reads (`TimeSeriesExtra`, time-generator
  callables, `time`) are now materialised per frame, sliced per position
  block, and read at each observation interval's first frame — the
  simulator's convention — in the residual, instrument, convexity,
  diffusion and L-BFGS paths of both engines.  Fits with static extras run
  the unchanged code path.  The default `infer_diffusion` start point now
  evaluates the diffusion model with the dataset's extras (it silently fell
  back to θ_D = 0, a singular start, for any model reading extras), and the
  parametric `infer_force` records the basis labels so `coeff_block` can
  locate a `Basis` block.  The time-dependent forcing gallery demo gains a
  localisation-noise section.

- **All-pairs stencils now see the fit mask.**  The stencil time operators of
  `SFI.inference.graph_diffusion` received only their declared streams, so
  the zeroing of excluded particles' increments (`mask_out`) was a no-op.
  Force corrections were unaffected (the basis masks its neighbours, which
  removes their tangents), but the graph-noise intensities
  (`estimate_graph_noise`), which sum increments over the particles of a
  frame, counted the increments of dynamically masked particles.

- **`dynamic_mask` removed particles from their neighbours' interactions.**
  `TrajectoryDataset` documents `dynamic_mask` as "increment unreliable,
  position still valid for neighbour forces" (e.g. particles at an open
  boundary).  The linear estimators instead handed state expressions the fit
  mask `mask_out`, which hyperedge dispatchers fold into slot validity: a
  dynamically masked particle vanished from every neighbour's hyperedges and
  per-slot kernels (e.g. a Voronoi cell's polygon) were evaluated without it.
  Row producers now also emit `mask_state` (static mask at every frame the
  row reads, i.e. positions known), which is what expressions receive;
  `mask_out` still decides which rows enter the sums.  The parametric core and
  the residual diagnostics already evaluated forces without the fit mask.
- **Per-particle time-series extras broke derivatives of single-particle
  bases.**  Derivative nodes (`d_x`, `d_v`) closed over `particle_extras`
  unchanged when vmapping over samples, so a basis reading a per-particle
  extra received the whole `(N,)` / `(K, N)` array: underdamped fits
  (`d_v` in the noise correction) failed with a stacking error, under either
  `cross_particle`.  They are now sliced per sample, as in the leaf's own
  batched evaluation.
- **Pair interactions with masked particles returned NaN fits.**  The pairs
  dispatcher ignored the particle mask when selecting edges, so a particle
  with an unknown position (stored as NaN, e.g. before it enters the field of
  view in `TrajectoryCollection.from_columns` data) still interacted with
  everyone, and a linear fit with any pair basis became NaN.  Pairs with a
  masked member are now excluded (with `jnp.where`, so NaN positions cannot
  leak), consistently with hyperedge slots, in the forward pass and in the
  same-particle Jacobian.  The degree normalisation of `reducer="mean"` now
  counts the contributing neighbours.
- **Bases dispatched on a user-supplied CSR could not be fitted.**  With
  `dispatch_pairs_from_extras` (or any `FromExtras*` rule) the CSR keys are
  dispatcher-owned and never reach `required_extras`, which alone decided
  whether `integrate` streams extras; unless the basis also read a regular
  extra, inference failed with `KeyError: 'FromExtrasPairsCSR: extras is
  required'`.  `StateExpr.structural_extras` now reports the dispatcher-owned
  keys of the whole node tree, and they request extras streaming too.
- **`UnderdampedProcess` crashed with a `NoiseModel` diffusion.**  The
  overdamped simulator accepts any `NoiseModel` (`D=...`), the underdamped
  one stored it but its half-kicks still read the constant-diffusion factor
  (`Constant diffusion not initialized.`).  Both half-kicks now draw the
  velocity noise from the model's `sample`, with the same `sqrt(2 D)`
  scaling.

- **Interaction dispatch on hyperedges and padded neighbour lists.**
  - `HyperCSR` schedules failed under JIT (boolean-mask indexing), required a
    fixed-K interactor (`Kmax` interactors raised), silently truncated groups
    longer than K, and could not be batched. The padding is now JIT-safe,
    uses `K` or `Kmax`, raises on oversized groups, and handles shared or
    per-sample schedules.
  - With `owners="all"`/`"custom"`, padded hyperedge slots (and slots masked
    by `slot_mask` or `exclude_self`) received a copy of the output; with
    `owners="global"`, padding rows entered the sum/mean. Only valid slots
    contribute now, and a row whose focal slot is masked contributes nothing.
  - `PairsCSR` entries past `indptr[-1]` (the padding added by
    `pad_neighbor_csr`) leaked into particle `fill_index` under
    `owners="all"`/`"custom"` and into `owners="global"` reductions; they are
    now ignored everywhere, including the same-particle Jacobian. A global
    `"max"` no longer counts masked entries as zeros.
  - The chunked hyperedge path misaligned per-hyperedge/per-slot spec extras
    in the last chunk when `M` was not a multiple of `chunk_size`.
  - The efficient same-particle Jacobian of hyperedge dispatchers (used by
    the parametric core) ignored the derivative through repeated
    participants (row padding, no-flux stencil slots) and fell back to the
    generic path for `HyperCSR`; it now matches full autodiff and mirrors
    the forward masking.

- **Chunk planner ignored the particle axis of `pdepth=0` bases.**
  `Integrand.estimate_bytes_per_sample` gave `pdepth=0` expressions no
  particle axis, although the runtime evaluates them on the full `(N, d)`
  block of each row and the Gram einsum materializes an `(N, F, F)`
  intermediate per row before the particle sum. The per-row byte hint was
  therefore `N` times too small, chunking was effectively disabled for
  stacked `(T, N, d)` data with a `pdepth=0` basis, and peak memory scaled
  as `T * N * F**2` regardless of `max_memory_gb` (5e5 frames x 100
  particles x 56 features exhausted a 32 GB machine; the same data as 100
  single-particle datasets peaked at 1.5 GB). The estimate now counts
  `max(1, pdepth)` particle axes, scales node-level transients of
  `pdepth=0` graphs by `N`, and takes the item size from the `X` stream
  instead of assuming 4 bytes. Single-particle and `pdepth=1` workloads are
  unaffected. Regression tests pin the hint against the materialized Gram
  block and the chunk count against the budget.
- **Chunk planner under-estimated the memory of derivative and heavy
  kernels.**  The static per-row hints count node outputs and gathers only;
  they cannot see a kernel's internal arrays nor the tangent batches of
  forward-mode Jacobians, and the same-particle derivatives of interaction
  bases (ULI with a hyperedge basis) were under-estimated ~100x, so a
  400-cell fit overran `max_memory_gb` several-fold.  `integrate` now
  compiles the kernel the runtime executes at the statically chosen block
  size, reads XLA's buffer assignment (`memory_analysis()`), and shrinks the
  blocks when they exceed the budget; the probe compilation is the one the
  runtime reuses, so it costs no extra compile.  `measure_memory=False`
  restores the static sizing; `make_parametric_integrator(theta_probe=...)`
  opts in for the parametric path.  Backends without a memory analysis keep
  the static estimate.

- **Noise-robust force fits were biased when particles share noise.**  The
  Stratonovich correction (`preset="robust"`, `"legacy-sfi-v1.0"`,
  `M_mode="Strato"`) and the ULI velocity correction used each particle's own
  Jacobian block, which is exact only when the noise is independent between
  particles.  With shared noise the bias is O(1): on an SPDE lattice with
  conserved noise every coefficient of a Laplacian / biharmonic basis came out
  wrong, including the sign of the diffusion term; a ring with bond noise
  quadrupled the stiffness and flipped the coupling; neighbouring vertex-model
  cell centroids (polarity noise shared through common vertices) got the
  collective velocity relaxation with the wrong sign.  For interacting bases
  both corrections now use the instantaneous diffusion estimator on every pair
  of particles, `sum_j D_t,ij : A^-1 d b_i / d y_j`, evaluated as forward-mode
  derivatives of the basis along the increments: no diffusion model, no
  Jacobian block, and consistent at O(dt) without shared noise too (coefficients
  of interacting bases can shift at O(dt) from the previous release).
  `infer_force_linear(..., cross_particle="same-particle")` restores the old
  correction; it warns on `SFI.bases.spde` stencil bases.  Single-particle
  bases are unchanged.  `metadata["force_noise_correction"]` records the
  choice.

### Added

- **The robust overdamped preset corrects with a lag-2 stencil
  (`correction_stencil="noisy_lag2"`, new default).**  The Stratonovich
  correction now uses the Vestergaard stencil minus half the lag-2 product,
  `-1/2 dX_{t-2} (x) dX_t / dt`.  That term shares no frame with the other
  increments, so localisation noise does not bias it, and it cancels the O(dt)
  bias of Stratonovich moments with the shift Gram for linear dynamics (exact
  analysis).  Benchmarks: force error lower by up to 7.6x (Lorenz) and 34x
  (double well) at intermediate and coarse time steps, 1.3-8x with
  state-dependent noise; `"noisy"` is better (up to 3.5x) only when the
  localisation noise dominates `D dt` at fine time steps.  Coefficients from
  `preset="robust"` (and `"auto"` on noisy data) shift at O(dt) from the
  previous release; `correction_stencil="noisy"` restores the old estimator.
  The legacy preset (rectangle Gram) keeps `"noisy"`.  One more frame is
  dropped at the start of each trajectory.

- **Lattice conserved noise as bond sources (experimental).**
  `LatticeConservedNoise(sigma, grid_shape=..., dx=..., bc="pbc"|"noflux",
  n_fields=..., fields=..., mobility=...)` (in `SFI.langevin.graph_noise`,
  importable from `SFI.langevin.noise`) gives one independent kick per
  nearest-neighbour bond, `+a` to one site and `-a` to the other: covariance
  `D = sigma^2/dV * (-lattice Laplacian)`, the operator of the `Laplacian`
  stencil, with `sum(phi)` conserved for periodic and no-flux boundaries.
  Same `sigma` convention as the spectral `ConservedNoise` (unchanged; they
  agree at large scales, the bond form has the lattice dispersion).  Supports
  per-axis spacings, noise on a subset of the fields, and a field-dependent
  mobility evaluated at the bond mean.  The noise carries its bond tables
  (`.extras`) and adds them when sampling; `estimate_graph_noise(coll, noise)`
  accepts the noise model and returns `sigma^2`.  The SPDE user guide gains a
  section on force estimators for lattice fields (robust preset with the
  all-pairs correction under localisation noise; never the per-site one with
  conserved noise).

- **Directional derivatives and random probes in `integrate`.**
  `ExprOperand(..., tx=..., tv=...)` evaluates the forward-mode derivative of
  a state expression along tangent time operands (one jvp): a Jacobian
  contracted with a vector without forming the Jacobian (e.g. the
  all-particle `sum_j d b_i/d x_j . u_j` of an interaction basis).
  `integrate(..., seed=...)` streams one PRNG key per row
  (`fold_in(seed, dataset, frame)`, independent of chunking) to time
  operands that request it; `SFI.integrate.timeops.normal_probe` draws
  Gaussian probes from it (Hutchinson traces and similar estimators).

- **Per-slot interaction outputs and spec-carried extras.**
  `make_interactor(..., per_slot=True)` lets a hyperedge kernel return one
  value per member, `(K, *rank, F)`, scattered slot by slot (the force a
  many-body energy, e.g. a vertex-model cell, exerts on each member).
  `make_interactor(..., spec_extras=...)` declares arrays carried by the
  dispatch spec; `FromExtrasHyperFixed` gains `edge_extras_keys` /
  `slot_extras_keys`, and the new `FromExtrasHyperCSR` rule reads ragged
  groups (plus slot ids and per-group/per-slot arrays) from `extras`, so a
  topology can change between calls without recompiling.

- **Finite sampling-interval guard.** `SFI.diagnostics.finite_dt_scale(inf)`
  measures the per-step drift-gradient scale `dt*|grad F|` of a fitted
  force (with respect to the velocity for underdamped fits), the size of
  the linear estimators' O(dt) biases, and returns a `fine` / `coarse` /
  `unreliable` verdict; `assess(inf)` carries it in a new `finite_dt`
  section of the `DiagnosticsReport` and flags the coarse regimes.
- **Gram-conditioning warning.** Both linear force estimators record the
  condition number of the force Gram (`metadata["force_gram_cond"]`,
  printed by `print_report`) and warn above a dtype-aware threshold (1e6
  in float32, 1e8 in float64): on a collinear or badly scaled dictionary
  the pseudo-inverse solve succeeds but the coefficients and error
  estimates are meaningless.
- **PSD check before simulating a fitted diffusion.**
  `SFI.diagnostics.psd_fraction(D_sf, coll)` measures the fraction of data
  points where a diffusion field is not positive semi-definite;
  `simulate_bootstrapped_trajectory` runs it on the fitted diffusion
  (`metadata["diffusion_psd_fraction"]`) and warns, since the simulator
  clips negative eigenvalues to zero and the generated noise then differs
  from the fitted field without any error.
- **Noise-consistency warning.** Under detected localization noise the
  drift and diffusion fits must both be noise-corrected (`preset="robust"`
  with the `"noisy"` diffusion stencil, the `"auto"` choice) or both raw:
  correcting only one side leaves an offset `Lambda/dt` in the other,
  which the force `D^-1 Phi` turns into a spurious term along `grad ln P`.
  Both engines now warn about a mixed pair before simulating the fitted
  model (`simulate_bootstrapped_trajectory`) and in `print_report`.
- **Replica batches in the simulators.** `OverdampedProcess.initialize` and
  `UnderdampedProcess.initialize` accept `x0` of shape `(B, d)` for a
  `pdepth=0` force: the `B` rows are integrated as independent replicas of
  the same process (the step, noise and information/entropy observables
  were already batch-agnostic), yielding a `(T, B, d)` dataset with
  `num_particles=B` in the metadata. A `(1, d)` input is still squeezed to
  `(d,)` so single-replica runs keep their `(T, d)` output.

- **Gallery.** Every demo was audited and reworked: model-selection steps
  show their Pareto front and error bars, most demos add a measurement-noise
  or missing-data robustness check, and particle and field demos show the
  data next to bootstrapped simulations of the fitted model as movies
  (`SFI.utils.plotting.animate_particles_comparison`).  New demo: pair
  interactions of self-propelled Janus colloids inferred from published
  experimental tracks (Hem et al. 2025).

## 2.0.3 (2026-07)

### Fixed

- **Structured-field feature selection (`StructuredExpr.__getitem__`).**
  Indexing a structured field selects *features*, but for a compiled inner
  value (which carries no feature axis) the pointwise slice fell through to a
  spatial axis instead: it raised for scalar sectors and silently returned the
  wrong component for vector/tensor sectors. Feature selection is now resolved
  structurally against the `&`-concatenated terms, compiling through the
  feature-concat path; opaque/parametric nodes keep the prior behaviour.
  Regression tests pin `field[i]` to the `i`-th feature column of the full
  field (scalar and vector), with structural dimensions preserved.

### Documentation

- Corrected the force-moment and covariance-block math in the
  `underdamped._force_moments` and `covariance.build_ud_blocks_exact`
  docstrings (the code was already correct; the prose described an earlier,
  buggy contraction pattern), and added hardening comments at the overdamped
  and underdamped gradient-correction einsums noting that the
  `(D_inst · A⁻¹) : ∇b` pairing relies on `D_inst` being symmetric. No
  estimator numerics change.

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
