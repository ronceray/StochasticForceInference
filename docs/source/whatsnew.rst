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
