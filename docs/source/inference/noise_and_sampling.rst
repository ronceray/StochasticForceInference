.. _noise-and-sampling:

Measurement noise and coarse sampling
=====================================

Real trajectories are rarely clean: localization error blurs every
position, and the camera frame rate fixes a sampling interval
:math:`\Delta t` that may be coarse compared to the dynamics.  Both
imperfections bias the linear estimators in ways that **more data will
not fix** — they call for the parametric estimators instead.  This page
shows how to recognise the symptoms and what to run.


Recognising the symptoms
------------------------

You likely have a measurement-noise or sampling problem when:

- **Diagnostics flag it.**  After a linear fit,
  :func:`SFI.diagnostics.assess` reports ``[mse_consistency]``
  (realised error well above the predicted, sampling-noise value),
  residual ``[autocorr]`` flags, or a whitened residual standard
  deviation far from 1.  On experimental data these flags usually mean
  noise or coarse sampling, not a wrong basis.
- **The diffusion estimators disagree.**
  ``compute_diffusion_constant(method="noisy")`` (noise-aware)
  and ``method="WeakNoise"`` (clean-data) give clearly different
  values, or ``inf.Lambda`` — the estimated measurement-noise
  covariance — is comparable to :math:`2 D \Delta t`.
- **The error plateaus.**  Adding more data keeps shrinking the
  *predicted* error (``inf.force_predicted_MSE``) while the realised
  error against held-out data stalls: you have hit a bias floor.


Why the linear estimators acquire a bias
----------------------------------------

Two distinct mechanisms, both growing from the finite-difference
construction:

**Errors-in-variables (measurement noise).**  The linear estimators
regress finite-difference velocities on basis functions evaluated at
the *measured* positions.  Localization noise :math:`\eta` of
covariance :math:`\Lambda` enters both sides: it inflates the
velocity estimate (variance :math:`\sim 2\Lambda/\Delta t^2`) and
perturbs the regressors.  On nonlinear systems the resulting
errors-in-variables bias is proportional to the noise level and does
not average away with longer trajectories.

**Euler secant (coarse sampling).**  The linear estimators approximate
the drift over one interval by the straight-line secant
:math:`(\mathbf{x}_{t+\Delta t} - \mathbf{x}_t)/\Delta t`.  When
:math:`\Delta t` is no longer small compared to the dynamical
timescales, the secant mis-tracks the curved true flow and the
estimate acquires an :math:`O(\Delta t)` bias.

See :ref:`parametric-concept` for the quantitative treatment of both
effects.


The parametric workflow
-----------------------

The parametric estimators address both mechanisms natively: a single
RK4 flow step per observation interval replaces the Euler secant, the
measurement noise :math:`\Lambda` is part of the observation
model, and the *skip-trick* errors-in-variables instrument keeps the
estimating equation consistent under noise.

.. code-block:: python

   from SFI import OverdampedLangevinInference
   from SFI.bases import monomials_up_to

   inf = OverdampedLangevinInference(coll)

   B = monomials_up_to(order=3, dim=2, rank='vector')
   inf.infer_force(B)            # profiles (D, Λ) automatically
   inf.infer_diffusion()         # optional: defaults to a symmetric-matrix basis
   inf.compute_force_error()
   inf.print_report()

Notes:

- ``F`` can be a :class:`~SFI.statefunc.Basis` (fast Gauss–Newton path, PASTIS
  sparsification wired) or any differentiable :class:`~SFI.statefunc.PSF` — see
  :ref:`choosing-an-estimator`.
- The noise and diffusion levels :math:`(\mathbf{D}, \Lambda)` are
  **profiled automatically**: closed-form moment estimators initialise
  them, and one conditional-NLL refinement updates them at the fitted
  parameters.  Nothing to tune.
- If you know the noise from calibration (e.g. the localization
  precision of your microscope), pass it explicitly — and pass the
  diffusion too if known, which skips profiling entirely:

  .. code-block:: python

     inf.infer_force(B, D=D_known, Lambda=Sigma_known)   # fast path

- The errors-in-variables instrument is on by default (``eiv="auto"``);
  you should not need to touch it.

**Runtime expectations.**  The parametric fit is iterative: expect
minutes where the linear estimators take seconds on large problems,
though on moderate data the gap is small (an underdamped solve at
:math:`T \approx 10^4`, :math:`n = 8` runs in ~20 s on a laptop CPU
core, vs. ~10 s for the linear estimator — see
:ref:`parametric-algorithm` for scaling).

**Cross-checking.**  Running both estimator families on the same basis
is itself a diagnostic: if they agree, noise and sampling effects are
under control and you can keep the cheaper linear workflow; if they
disagree, trust the parametric fit — the discrepancy measures the
linear bias.


Worked examples and validation
------------------------------

- :doc:`/gallery/experimental_workflow_demo` — an end-to-end
  experimental pipeline where the diagnostics flag localization noise
  and the parametric estimator removes the bias.


.. seealso::

   - :ref:`choosing-an-estimator` — the regime table.
   - :ref:`parametric-concept` — the observation model and estimator
     theory.
   - :ref:`parametric-algorithm` — algorithm details and the full
     parameter reference.
   - :doc:`/inference/underdamped` — noise is doubly harmful for
     inertial systems; the underdamped page covers the specifics.
