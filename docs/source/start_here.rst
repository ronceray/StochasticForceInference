.. _start_here:

Start here: is SFI right for my data?
======================================

SFI is built for learning **interpretable stochastic dynamics** from
time-ordered observations of a continuous state. It is most useful when
you care about the drift or force field, the diffusion or noise level,
sparse term selection, or bootstrapped simulation of the inferred model.

.. tip::

   **New to SFI?** The :doc:`/gallery/ou_demo` tutorial walks through a
   complete workflow end to end — simulate, infer the force and
   diffusion, select a minimal model, validate, and save — on a simple
   synthetic example. It is the fastest way to see the whole pipeline in
   action before routing your own data below.

When SFI is a good fit
----------------------

SFI is usually a good fit if:

- each observation is a continuous-valued state: positions, angles,
  velocities, concentrations, field values, abundances of large populations, or another trusted state
  variable;
- stochastic fluctuations are part of the dynamics, not just a nuisance
  layered on top of a deterministic fit;
- adjacent frames are close enough in time that they remain correlated;
- you want an explicit dynamical law rather than a black-box forecaster.


When SFI is usually the wrong tool
----------------------------------

SFI is usually not the right tool if:

- the data is categorical, count-based, event-based, or text-like rather
  than continuous coordinates;
- the task is forecasting only, with no interest in an interpretable
  equation of motion;
- the series is dominated by long memory, abrupt regime switches,
  interventions, or hidden controls that cannot be represented in the
  state;
- you only have a few disconnected snapshots rather than a trajectory or a
  field movie.


Pick the right starting route
-----------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Your data or question
     - Start here
     - Then read
   * - One tracked object or a few observed coordinates
     - :doc:`/gallery/experimental_workflow_demo`
     - :doc:`/trajectory/user_guide`, :doc:`/inference/user_guide`
   * - Noisy or coarsely-sampled recordings (localization error,
       low frame rate)
     - :doc:`/inference/noise_and_sampling`
     - :doc:`/gallery/experimental_workflow_demo`
   * - Position-only data with inertia or oscillations
     - :doc:`/inference/underdamped`
     - :doc:`/gallery/van_der_pol_demo`
   * - Many interacting particles or agents
     - :doc:`/particles/user_guide`
     - :doc:`/gallery/abp_align_demo`, :doc:`/gallery/advanced/multi_experiment_demo`
   * - Spatial fields on a regular grid (experimental)
     - :doc:`/spde/user_guide`
     - :doc:`/gallery/gray_scott_demo`, :doc:`/gallery/abp_to_spde_demo`
   * - Large nonlinear parametric force models
     - :doc:`/gallery/advanced/nn_force_demo`
     - :doc:`/inference/user_guide`


Choose the data container first
-------------------------------

If your data is already in memory as an array, start with
:meth:`~SFI.trajectory.TrajectoryCollection.from_arrays`. For a first fit, this is the
simplest and most predictable entry point.

If you have a tracked-particle table where particles appear and disappear
over time, use :meth:`~SFI.trajectory.TrajectoryCollection.from_dataframe` (pandas, columns
addressed by name) or the lower-level :meth:`~SFI.trajectory.TrajectoryCollection.from_columns`.

If your trajectories are already on disk as CSV, Parquet, or HDF5, use
``TrajectoryCollection.load`` — the file format is specified in
:doc:`/trajectory/data_formats`.


Pick an estimator family
------------------------

SFI ships two first-class estimator families — route by data regime,
not by habit:

- **Linear estimators** — ``compute_diffusion_constant()``,
  ``infer_force_linear()``, ``infer_diffusion_linear()`` — a
  closed-form projection: no initial guess, seconds even on large
  datasets, exact in the fine-sampling, low-noise limit.
- **Parametric estimators** — ``infer_force()``, ``infer_diffusion()``
  — an iterative likelihood fit that models measurement noise and
  finite sampling explicitly, and accepts any differentiable model
  (including nonlinear ones).  More compute, more robustness.

If your recordings carry measurement (localization) noise, or the
frame interval is coarse compared to the dynamics, start directly with
the parametric estimators — :doc:`/inference/noise_and_sampling` is
the guide.  Otherwise the linear first pass below is the fastest
start; the full trade-off table is in :ref:`choosing-an-estimator`.

Default first pass (clean, well-sampled data)
---------------------------------------------

1. load or build a :class:`~SFI.trajectory.TrajectoryCollection`;
2. choose a small linear basis;
3. call ``compute_diffusion_constant()``, ``infer_force_linear()``, and
   ``compute_force_error()``;
4. if the basis is large or if you seek a minimal model, run ``sparsify_force()``.

On noisy or coarsely-sampled data the equivalent parametric pass is a
single call — ``inf.infer_force(B)`` — which profiles the diffusion
and measurement-noise levels automatically.  When in doubt, run both:
agreement is itself a diagnostic, and disagreement measures the bias
the linear estimator cannot absorb.


Use diagnostics on real data
----------------------------

For experimental data, the main validation tool is the diagnostics suite:

.. code-block:: python

   from SFI.diagnostics import assess

   inf.compute_force_error()
   report = assess(inf, level="standard")
   report.print_summary()


This is the fastest way to separate three common failure modes:

- missing dynamics or a too-small basis: residual autocorrelation flags;
- diffusion or noise mismatch: whitened residual standard deviation far
  from 1;
- a biased model: the realised NMSE stays well above the predicted
  (sampling-noise) value (the MSE-consistency flag).  A redundant basis
  is handled separately by sparse selection (:meth:`sparsify_force`).

On experimental data, the noise- and bias-type flags usually trace
back to localization noise or coarse sampling — the cure is then the
parametric estimators, not a bigger basis; see
:doc:`/inference/noise_and_sampling`.


What to do next
---------------

- If diagnostics look clean and the coefficients are interpretable, keep
  the linear workflow and consider sparse selection.
- If diagnostics flag noise or sampling effects, switch to the
  parametric estimators: :doc:`/inference/noise_and_sampling`.
- If inertia matters but you only observe positions, switch to
  :class:`~SFI.inference.UnderdampedLangevinInference` — see :doc:`/inference/underdamped`.
- If your system contains interacting agents, move to
  :doc:`/particles/user_guide`.
- If your state is a field on a grid, move to :doc:`/spde/user_guide`
  (experimental).
- If you need a model that is nonlinear in its parameters, use the
  parametric estimators with a PSF — see :ref:`choosing-an-estimator`.