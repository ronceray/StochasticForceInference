.. _diagnostics_user_guide:

Diagnostics
===========

Once an inference object is fitted, the next question is *should we
trust it?* The :mod:`SFI.diagnostics` submodule answers that by
recomputing standardised residuals from the fitted force and the
inferred constant diffusion, then testing whether they look like an
independent :math:`\mathcal N(0, 1)` sample — the signature of a
correctly specified model.

Quick start
-----------

.. code-block:: python

   import SFI
   from SFI.diagnostics import assess

   inf = SFI.OverdampedLangevinInference(collection)
   inf.compute_diffusion_constant()
   inf.infer_force_linear(basis)
   inf.compute_force_error()

   report = assess(inf, level="standard")   # or inf.diagnose()
   report.print_summary()

The same call works for :class:`~SFI.UnderdampedLangevinInference`,
multi-particle and multi-dataset collections, and for the linear,
parametric, and nonlinear inference paths.  Data access is routed
through ``TrajectoryDataset.make_batch_producer`` (the same streaming
layer used internally by :mod:`SFI.integrate`), so masked frames and a
variable per-step ``dt`` are handled uniformly.

What is computed
----------------

For an **overdamped** fit, the residual is the Euler–Maruyama
innovation

.. math::

   r_t = \Delta x_t - F(x_t)\,\Delta t,
   \qquad \mathrm{Cov}(r_t) = 2\,\bar D\,\Delta t,

whitened by :math:`(2\bar D\,\Delta t)^{-1/2}` so that, under a correct
model, the standardised residual :math:`z_t` is :math:`\mathcal N(0, I)`.

For an **underdamped** fit (positions only), the residual is the
symmetric finite-difference acceleration minus the fitted force — the
same quantity the underdamped estimator itself fits:

.. math::

   r_t = \hat a_t - F(\hat x_t, \hat v_t),
   \qquad
   \hat a_t = \frac{x_{t+1} - 2x_t + x_{t-1}}{\Delta t^2},

with :math:`\hat x_t = \tfrac13(x_{t-1} + x_t + x_{t+1})` and
:math:`\hat v_t = (x_{t+1} - x_{t-1}) / (2\Delta t)`.  The noise of this
second difference is the second difference of integrated white noise,
with variance :math:`\tfrac23\,(2\bar D)/\Delta t`; this sets the
whitening.  Because adjacent accelerations share two of three
positions, the residual is a moving-average process in time, so the
builder keeps every second valid index — leaving the pooled
innovations serially independent.

The standard suite (``level="standard"``, the default) runs:

* **Residual moments** — pooled mean, standard deviation, skewness, and
  excess kurtosis, with a per-component breakdown.
* **Autocorrelation** — Ljung–Box on :math:`z_t` and on :math:`z_t^2`
  (the latter detects volatility clustering / a mis-estimated
  diffusion).
* **Normality** — Kolmogorov–Smirnov against :math:`\mathcal N(0, 1)`,
  plus the raw Q–Q data.
* **MSE consistency** — compares the realised mean-square residual to
  the predicted (sampling-noise) NMSE via a sampling-noise-aware
  chi-square z-score; :math:`|z| > 5` flags bias or a mis-specified
  diffusion.

The ``"minimal"`` level computes the residual moments only.

Output
------

:func:`~SFI.diagnostics.assess` returns a
:class:`~SFI.diagnostics.DiagnosticsReport` carrying a ``residuals``
section and a ``meta`` dict (backend, regime, sample sizes).  The
report exposes:

* :meth:`~SFI.diagnostics.DiagnosticsReport.print_summary` — a concise
  human-readable table with ``✓/✗`` marks at a chosen significance
  level, ending in a ``-- Flags --`` block;
* :meth:`~SFI.diagnostics.DiagnosticsReport.flag_issues` — the list of
  warnings (one string per failing check) that ``print_summary`` uses;
* :meth:`~SFI.diagnostics.DiagnosticsReport.to_dict` /
  :meth:`~SFI.diagnostics.DiagnosticsReport.to_json` — JSON-clean
  serialisation.

Plotting helpers
----------------

.. code-block:: python

   from SFI.diagnostics import (
       plot_qq, plot_residual_histogram, plot_residual_acf, plot_summary,
   )

   fig = plot_summary(report)        # 1×3 figure, all panels at once

* :func:`~SFI.diagnostics.plot_qq` — normal Q–Q with the :math:`y = x`
  reference;
* :func:`~SFI.diagnostics.plot_residual_histogram` — histogram with the
  :math:`\mathcal N(0, 1)` density overlaid;
* :func:`~SFI.diagnostics.plot_residual_acf` — autocorrelation of
  :math:`z` and :math:`z^2` with the 95 % Bartlett band;
* :func:`~SFI.diagnostics.plot_summary` — the three panels side by side.

All panels accept either a fitted inferer or a precomputed
:class:`~SFI.diagnostics.DiagnosticsReport`.

Interpreting flags
------------------

* ``[autocorr/ljung_box]`` — the model is missing a time-correlated
  feature (for example a constant-only fit on a mean-reverting process).
* ``[autocorr/ljung_box_squared]`` — the diffusion is mis-estimated or
  state-dependent.
* ``[normality/ks]`` — the residuals are non-Gaussian: rare events not
  captured by the basis, or a non-Gaussian noise structure.
* ``[moments/std]`` — a whitened standard deviation
  :math:`\not\approx 1` usually means :math:`\bar D` is wrong (try a
  different :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant` ``method``).
* ``[moments/mean]`` — a non-zero residual mean points to a systematic
  drift bias.
* ``[mse_consistency]`` — realised NMSE far above predicted: the model
  is biased; widen the basis or run :meth:`sparsify_force` and refit.

Each printed flag carries the corresponding action hint inline
(``"[mse_consistency] … — … consider the parametric estimator
(infer_force)"``); pass ``hints=False`` to
:meth:`~SFI.diagnostics.DiagnosticsReport.flag_issues` or
:meth:`~SFI.diagnostics.DiagnosticsReport.print_summary` for bare
statistics (machine parsing).

How visible a flag is depends on the sampling interval.  At very fine
:math:`\Delta t` the diffusion estimate can absorb a weak
force-misspecification, leaving the marginal tests looking clean;
coarser sampling makes the leftover structure show up in the
autocorrelation and MSE-consistency checks.

When a flag points beyond the linear estimators
-----------------------------------------------

On experimental data, ``[mse_consistency]``, ``[moments/std]``, and
``[autocorr/ljung_box_squared]`` very often trace back to measurement
noise or coarse sampling rather than a wrong basis.  The cure is then
not a bigger basis but the noise-aware **parametric estimators**
(:meth:`~SFI.inference.OverdampedLangevinInference.infer_force` / :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`), which model the
measurement-noise covariance :math:`\Lambda` explicitly and
replace the Euler secant with an RK4 flow step:

* ``[mse_consistency]`` with localization noise suspected — refit with
  the parametric estimator on a fresh inference object and compare;
  :doc:`/inference/noise_and_sampling` walks through it.
* ``[moments/std]`` far from 1 after trying both
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant` methods — the noise level is biasing
  the diffusion estimate; the parametric estimators profile
  :math:`(\mathbf{D}, \Lambda)` jointly.
* ``[autocorr/ljung_box]`` that persists after widening the basis —
  suspect coarse sampling; the parametric RK4 flow step extends the
  usable :math:`\Delta t` range.

If the parametric and linear fits *agree*, the flags point back at the
model itself: widen or restructure the basis
(:doc:`/bases/user_guide`) and re-run :meth:`sparsify_force`.

When data is plentiful, a suspected bias floor can also be confirmed
with an explicit train/test split — ``coll.split_time(0.8)`` +
``inf.holdout_score(test)``, and ``assess(inf, data=test)`` for
held-out diagnostics.  This is a side feature: the predicted error and
this suite cost no data, which is SFI's preferred, data-efficient
route.

.. _dynamics-order-classifier:

Classifying overdamped vs. underdamped dynamics
-----------------------------------------------

.. note::

   This classifier is **experimental** — the discriminator and verdict
   thresholds may still change, and the ``"inconclusive"`` band is
   deliberately conservative.

Before choosing an inference engine you can ask the data which regime it is
in.  :func:`~SFI.diagnostics.classify_dynamics` (experimental) reads raw
positions and returns an ``"OD"`` / ``"UD"`` / ``"inconclusive"`` verdict,
robust to high localization noise and coarse sampling:

.. code-block:: python

    from SFI import classify_dynamics
    from SFI.diagnostics import plot_dynamics_order

    report = classify_dynamics(collection)
    report.print_summary()        # verdict + tau_v, sigma, D, AICc, scaling slope
    plot_dynamics_order(report)   # rho2(dt) and apparent-KE scaling panels

The discriminator is the lag-resolved displacement covariance
:math:`C_k=\langle\Delta x_t\cdot\Delta x_{t+k}\rangle`.  White localization
noise touches only :math:`C_0` and :math:`C_1`, so lag-2 statistics are
measurement-noise-immune; scanning :math:`\Delta t` separates the overdamped
force confound (which vanishes as :math:`\Delta t\to0`) from genuine momentum
persistence (which saturates).  The verdict combines a model-free scaling
test, a parametric fit of the diffusion + inertia + localization covariance
model (recovering the momentum relaxation time :math:`\tau_v`), and an
overdamped-fit residual-autocorrelation cross-check.  At coarse sampling
(:math:`\gamma\,\Delta t\gtrsim1`) momentum is unresolved and the verdict is
``"inconclusive"`` by design.  The method assumes *white* localization noise;
strong memory (a generalized-Langevin / viscoelastic bath) can also produce
velocity persistence and is out of scope.  A worked example ships in the
gallery (*Overdamped or underdamped? Classifying dynamics from data*).

References
----------

Ljung & Box (1978); Diebold, Gunther & Tay (1998).
