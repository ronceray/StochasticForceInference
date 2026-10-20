.. _whatsnew:

What's new in SFI v2
====================

SFI 2.0 is a ground-up, JAX-based rewrite of the Stochastic Force
Inference research code that accompanied
`Frishman & Ronceray (2020) <https://doi.org/10.1103/PhysRevX.10.021009>`_
and the follow-up papers (underdamped inference, PASTIS model
selection).  This page is the one place in these docs that talks about
lineage and version history; everywhere else, the two estimator
families are simply called the **linear estimators** and the
**parametric estimators**.


Lineage: where the two estimator families come from
---------------------------------------------------

- The **linear estimators** — :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`,
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`, :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant` — are the
  direct descendants of the published SFI methodology.  With their
  default (``"auto"``) settings they apply that line's best refinements
  (moment-convention auto-selection, trapeze Gram construction,
  Vestergaard noise-aware diffusion, PASTIS sparsity), rewritten on
  JAX and typically orders of magnitude faster than the original
  research code.  They remain a *best-effort* estimate: exact in the
  fine-sampling, low-noise limit, biased outside it.

- The **parametric estimators** — :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` — are **new in 2.0**: a likelihood-based
  estimator built from one or more RK4 flow steps per observation
  interval (``n_substeps``), with native profiling of the diffusion and
  measurement-noise levels
  :math:`(\mathbf{D}, \Lambda)` and an errors-in-variables
  instrument for consistency under measurement noise.  They are the
  robust, flexible path for real experimental data — at an iterative
  compute cost.  See :ref:`choosing-an-estimator` for the regime
  table.


Highlights of 2.0
-----------------

- **Two first-class inference paths per engine** (overdamped and
  underdamped), sharing one API: closed-form linear projection and the
  parametric likelihood fit, both with PASTIS sparsification wired in.
- **State-dependent diffusion** — :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` /
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear` fit :math:`\mathbf{D}(\mathbf{x})` (and
  :math:`\mathbf{D}(\mathbf{x},\mathbf{v})` in the underdamped case).
- **Compositional model building** (:mod:`SFI.statefunc`,
  :mod:`SFI.bases`): symbolic bases, pair interactions, and
  neural-network force fields from the same expression algebra.
- **Diagnostics suite** (:func:`SFI.diagnostics.assess`): residual
  whitening tests, autocorrelation/normality batteries, and
  predicted-vs-realised error consistency.  Every flagged issue
  carries a one-line action hint pointing at the likely cure.
- **Overdamped or underdamped?** — the experimental
  :func:`SFI.classify_dynamics` reads raw positions and returns an
  ``"OD"`` / ``"UD"`` / ``"inconclusive"`` verdict, robust to
  localization noise and coarse sampling, for when the right engine
  isn't obvious a priori.
- **Entropy production, reconnected — and extended to underdamped** —
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
  brings the 2020 PRX dissipation estimator (basis-projected phase-space
  velocity, with fluctuation bias and error bar) to the v2 engine: it works
  after any linear force fit and after PASTIS sparsification.  Its new
  underdamped counterpart
  (:meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`)
  measures dissipation **from positions only**, via the time-reversal
  even/odd split of the fitted acceleration field
  (:meth:`~SFI.inference.UnderdampedLangevinInference.time_reversal_split`)
  and the Stratonovich log path-probability ratio, with provisional
  error bars (analytic + block-variance) and a cross-fitting hook; the
  underdamped simulator gains the matching ground-truth
  information/entropy observables.  Both engines report the
  AIC-debiased rate (``Sdot_debiased``) alongside the raw plug-in
  value, and independent localization noise is benign at leading order
  (time-reversal parity cancels the naive :math:`O(\Lambda/\Delta t)`
  term).
- **Diffusion stencils attached to their first point** — the
  ``"noisy"`` (Vestergaard) and ``"WeakNoise"`` instantaneous estimators
  now use the increments *following* the evaluation point
  (:math:`\Delta x_t, \Delta x_{t+1}` at :math:`x_t`) instead of the
  shared-point pair :math:`\Delta x_{t-1}, \Delta x_t`.  Their
  unconditional mean is unchanged, so
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant`
  is unaffected, but the estimates become unbiased *state-resolved*:
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`
  and the Stratonovich gradient correction no longer carry the
  :math:`O(\Delta t)` conditioning bias of the backward increment.
  Each trajectory now loses its last frame instead of its first.
- **Experimental SPDE toolbox** for spatial field data (linear
  estimators only) — see :doc:`/spde/index`.
- **Direct ingestion of tracking tables** —
  :meth:`TrajectoryCollection.from_dataframe
  <SFI.trajectory.TrajectoryCollection.from_dataframe>` (named,
  auto-detected columns) and named-column selection in
  :meth:`~SFI.trajectory.TrajectoryCollection.load`.
- **Held-out validation** (side feature for data-abundant scenarios) —
  ``coll.split_time(0.8)`` + ``inf.holdout_score(test)``; held-out
  diagnostics via ``assess(inf, data=test)``.
- **Multi-experiment inference** — pool many datasets into a single
  fit and let the model share or split parameters across experiments
  (one drift per condition, or a force field :math:`F(x, T)` learned
  jointly across temperatures).  Experiments are combined through the
  trajectory layer's extras, so no special data plumbing is needed.


2.0.4 (2026-10)
---------------

Correctness fixes — results change for the affected fits:

- **Missing frames no longer bias the linear estimators.**  A linear fit
  combines several time sums (Gram matrix, moments, the Stratonovich /
  ULI correction); with masked or missing frames each sum used to keep
  the rows its *own* stencil could read, and the ratio of sums was biased
  by the ratio of row counts (an overdamped ``preset="robust"`` stiffness
  read :math:`-0.85` for a true :math:`-1` at 5% frame loss; diffusion
  estimates were 10–20% low).  The force and diffusion fits now sum every
  term over one common row set — the frames all their stencils read —
  via the new :func:`SFI.integrate.shared_rows` /
  :func:`SFI.integrate.required_frames`.
- **Entropy production rates with missing frames** — the rate of
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
  was normalized by a particle-time that counted rows the fit had dropped
  (``preset="robust"`` at 5% frame loss: 1.44 instead of 1.97).  Rates now
  use the measure :func:`SFI.integrate.integrate` returns with the fit's own
  sums (``return_measure=True``), dataset weights included; the underdamped
  block error bar honours the mask.
- **Diffusion error bars** —
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_error`
  is now the least-squares sandwich estimator, with the correlation lag
  set by the frame span of the local diffusion estimator.  The previous
  formula under-reported the standard errors by an order of magnitude;
  the new ones match the scatter across independent recordings.
- **Diffusion model selection and parametric diffusion errors** —
  :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_diffusion`
  brings the PASTIS / Pareto search of ``sparsify_force`` to the
  diffusion field.  For the linear fit it scores each support by the
  efficient minimum-distance likelihood in the sandwich metric
  :math:`G^\top S^{-1} G`, so a spurious term gains :math:`\chi^2_1/2`
  and the selection does not depend on units; for the parametric fit it
  uses the NLL Hessian and re-fits the selected support exactly.
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` now
  returns the inverse Fisher information as ``param_cov``, and
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_error`
  covers parametric and sparsified fits.
- **Coefficient blocks after sparsification** —
  :meth:`~SFI.inference.OverdampedLangevinInference.coeff_block` returns
  the covariance embedded in the full basis (pruned terms are exact
  zeros); it used to slice a support-sized covariance with full-basis
  indices.
- **PASTIS selection settings are recorded** — the
  :class:`~SFI.inference.sparse.SparsityResult` returned by
  ``sparsify_force`` stores the criterion, :math:`p_0` and :math:`\tau`
  it was selected with (``selection``); ``select_by_ic`` and
  :func:`~SFI.utils.plotting.plot_pareto_front` reuse them by default,
  so plots mark the model actually selected (they used :math:`p_0 =
  10^{-3}` against the fit's :math:`0.05`).
- **Parametric errors-in-variables selection** — after a parametric fit with
  the EIV instrument, ``sparsify_force`` scores supports in the
  inverse-sandwich metric :math:`G_{\rm raw}^\top H^{-1} G_{\rm raw}` and keeps
  the efficient restricted estimator: selected supports and coefficients can
  change, and the standard errors no longer over-state the scatter
  (``eiv=False`` is unchanged).
- **Underdamped entropy production at finite Δt** —
  :meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`
  adds the derived :math:`O(\Delta t)` counterterm (``DeltaS_dt``) and reports
  the coefficients' own bias as ``Sdot_dt_bias``: underdamped
  :math:`\dot S` values change.
- **χ² consistency test calibrated at finite Δt** — the diagnostics subtract
  the finite-Δt expectation of the residual norm (drift relaxation, stencil
  bias) from the fitted force, so ``excess_z`` moves (correct fits read
  :math:`-4\sigma` before); with MSD diffusion on the fitting data the test
  now reports no information (NaN, with a note).
- **BIC/EBIC sample size** — for linear fits, ``sparsify_force`` /
  ``sparsify_diffusion`` take :math:`\tau` from the row measure of the fit's
  own Gram (dataset weights included), not a recount of the rows.
- **Time-dependent extras in the parametric estimators** —
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` /
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` read a
  ``TimeSeriesExtra``, a time-generator callable or the reserved ``time`` per
  frame (a ``TypeError`` before), and so do per-frame neighbour lists
  (dropped before: ``KeyError``).
- **Successive fits on one object** — a linear diffusion fit after a
  parametric one no longer inherits its method flag, Hessian or error
  analysis (``compute_diffusion_error`` used the parametric Hessian and
  ``sparsify_diffusion`` silently re-fitted the parametric model); likewise
  for a parametric diffusion fit after a linear selection, and a linear force
  fit after a parametric ``PSF`` fit.  A refused re-fit leaves the fit intact.
- **Covariance after sparsification** — ``force_inferred.param_cov`` is in
  the full parameter order (pruned terms: zeros), as for the diffusion: with
  one selected term ``predict_ci`` silently broadcast the support-sized
  matrix (bands 3.7× too wide in one case), with more it raised.
  ``sparsify_force`` / ``sparsify_diffusion`` after an error analysis re-run
  it on the selected support, so covariances, standard errors and
  ``param_cov`` no longer keep their pre-selection values.
- **Parametric integrators built inside** :func:`SFI.integrate.shared_rows`
  evaluate their program in that row scope wherever they run (periodic pair
  programs read another image reference, or raised ``KeyError``).

Smaller additions:

- :meth:`~SFI.inference.OverdampedLangevinInference.predict_time_profile`
  evaluates a time-dependent basis block row by row (it failed on
  ``time_fourier`` blocks) and can return 1-σ errors (``return_std``).
- ``coeff_block(basis, field="diffusion")`` locates a ``Basis`` block of a
  linear or parametric diffusion fit (it always raised).
- ``SparsityResult.all_ic`` defaults to the recorded :math:`p_0` and
  :math:`\tau`, like ``select_by_ic`` (it used :math:`p_0 = 10^{-3}` and
  skipped BIC/EBIC).
- ``"pbc_ref"`` is a reserved extras name: a user extra of that name is
  rejected instead of being read as the periodic-image reference.
- Diagnostics: the realised NMSE comes with a one-sided 95% upper bound,
  and the ``[mse_consistency]`` hint distinguishes residuals larger than
  the noise model (bias) from smaller ones (over-estimated
  :math:`\bar D`, non-white noise).
- :func:`~SFI.trajectory.degrade.degrade_spatial_data` clears cached
  stencil extras itself and accepts ``stencil_radius`` to exclude cells
  whose stencil touches a missing pixel.
- Particle plotters wrap colour values for cyclic colormaps, so an
  unwrapped heading angle with ``vmin=-π, vmax=π`` colours by direction
  (it was clipped to the end colour).
- The gallery was audited and rewritten: every model-selection step
  shows its Pareto front and error bars, and most demos add a
  measurement-noise or missing-data robustness check.  A new demo infers
  the pair interactions of self-propelled Janus colloids from published
  experimental tracks.


Version history
---------------

SFI has gone through three generations.  Each one is a ground-up rewrite
of the one before — the through-line is the *method*, not the code.

**v1.0 (2020) — the original research code.** The NumPy/SciPy
implementation accompanying `Frishman & Ronceray (2020)
<https://doi.org/10.1103/PhysRevX.10.021009>`_: a flat set of scripts
(no installable package) for **overdamped** systems.  It projects
force, velocity, and diffusion fields onto a chosen basis, estimates
reconstruction error, and computes **entropy production** and
probability currents, with a Brownian-dynamics simulator for
bootstrapped validation and 2D/3D plotting helpers.  Underdamped
Langevin Inference (**ULI**) lived in a *separate* companion package
(`Brückner et al. (2020)
<https://doi.org/10.1103/PhysRevLett.125.058103>`_).  Documentation
was a theory-manual PDF.

**v1.5 (July 2025) — the JAX package.**
A ground-up **JAX** rewrite, packaged and pip-installable, that unified
the two methodology papers into one library: **overdamped (OLI) and
underdamped (ULI) inference side by side**, sharing a closed-form linear
estimator (Itô/Stratonovich moment conventions, error reports,
exact-model comparison, bootstrapped resimulation).  Its headline
addition was **PASTIS sparse model selection**
(`Gerardos & Ronceray (2025) <https://doi.org/10.1103/ltdt-hvh7>`_) — a
principled criterion, with a beam search, for deciding which basis terms
the data actually support.  Shipped with runnable OLI/ULI example
scripts; still no rendered documentation.

**v2.0 (this release) — the parametric, documented rewrite.**
A second ground-up rewrite: as with v1.5 before it, none of the previous
generation's code or API carries over.  Alongside faster, refined linear
estimators, 2.0 introduces the **parametric estimators** for robust
inference under measurement noise and coarse sampling, **compositional
model building**, a **diagnostics suite**, an
experimental **SPDE** toolbox, direct ingestion of tracking tables, and
— for the first time — **full rendered documentation** (this site).  See
*Highlights of 2.0* above for the detailed feature list.
