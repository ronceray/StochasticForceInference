.. _inference-user-guide:

Running inference
=================

The :mod:`SFI.inference` subpackage is the core of SFI: it estimates the
force (drift) and diffusion fields of a stochastic system from trajectory
data.

.. tip::

   **Which engine?**

   - **Overdamped** (:class:`~SFI.inference.OverdampedLangevinInference`):
     positions evolve as
     :math:`\mathrm{d}\mathbf{x} = \mathbf{F}\,\mathrm{d}t + \sqrt{2\mathbf{D}}\,\mathrm{d}W`.
     Use when inertia is negligible — colloids, molecular motors, cells
     migrating on a substrate.
   - **Underdamped** (:class:`~SFI.inference.UnderdampedLangevinInference`):
     positions and velocities evolve jointly.  Use when inertia matters —
     vibrated grains, swimming organisms, underdamped Brownian motion.

   Unsure which fits your data?  The experimental
   :ref:`overdamped/underdamped classifier <dynamics-order-classifier>`
   decides from raw positions.

Both engines share one workflow and API (inherited from
``BaseLangevinInference``); they differ only in the physics of the moment
estimators.


Workflow at a glance
--------------------

A fit is a short sequence of method calls on the engine.  Take the linear
path or the parametric path (see :ref:`choosing-an-estimator`):

.. code-block:: python

   inf = SFI.OverdampedLangevinInference(coll)   # or Underdamped...

   # Linear path — fast closed-form projection
   inf.compute_diffusion_constant()               # 1. constant diffusion D
   inf.infer_force_linear(basis)                  # 2. force F(x)
   inf.infer_diffusion_linear(basis_D)            # 3. optional: state-dependent D(x)
   inf.sparsify_force(criterion="PASTIS")         # 4. optional: sparse selection
   inf.compute_force_error()                      # 5. predicted error
   inf.diagnose()                                 # 6. consistency diagnostics
   inf.print_report()                             # 7. inspect

.. code-block:: python

   inf = SFI.OverdampedLangevinInference(coll)

   # Parametric path — iterative likelihood fit,
   # robust to measurement noise and coarse sampling
   inf.infer_force(F)                             # 1. force; profiles (D, Λ)
   inf.infer_diffusion(basis_D)                   # 2. optional: diffusion field D(x)
   inf.compute_force_error()                      # 3. predicted error
   inf.diagnose()                                 # 4. consistency diagnostics
   inf.print_report()                             # 5. inspect

Each call populates attributes on the engine (``diffusion_average``,
``force_inferred``, ``force_predicted_MSE``, …).  The fit is itself a usable
model — evaluate it or simulate from it:

.. code-block:: python

   F_hat = inf.force_inferred                     # fitted force (an SF): F_hat(x)
   coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(key)


.. _choosing-an-estimator:

Choosing an estimator
---------------------

SFI offers two families of force/diffusion estimators that share the same
API.  Choose by your data, not by habit.

**Linear estimators** —
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`,
:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`,
:meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant`
— solve a closed-form projection onto a basis: no initial guess, no
iterations, seconds even on very large datasets.  They are exact in the
fine-sampling, low-noise limit and stay the fastest first look outside it, at
the cost of a bias that more data will not remove.

**Parametric estimators** —
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`
— fit a likelihood instead.  Each observation interval is advanced with an
accurate **RK4** step (a fourth-order Runge–Kutta integrator), and the
diffusion and measurement-noise levels :math:`(\mathbf{D}, \Lambda)` are
estimated as part of the fit rather than assumed.  This buys robustness to
measurement noise and coarse sampling, and it fits force models a basis
cannot express — families nonlinear in their parameters, such as
neural-network or gated drifts.  The price is compute: an iterative fit,
typically minutes where the linear path takes seconds.

Route by data regime:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Data regime
     - Use
     - Why
   * - Clean data, fine sampling
     - linear
     - exact in this limit; fastest
   * - Quick exploration / first pass
     - linear
     - no initial guess, runs in seconds
   * - Huge dataset
     - linear
     - closed form, streams through data
   * - Measurement (localization) noise
     - parametric
     - estimates the noise level :math:`\Lambda` and corrects the bias
       it would otherwise cause
   * - Coarse sampling (large Δt)
     - parametric
     - the RK4 step follows the motion across each interval, where a
       finite-difference velocity breaks down
   * - Model nonlinear in its parameters (e.g. neural-network drift)
     - parametric
     - only the parametric estimators can fit models of this kind
   * - Multi-particle / interacting
     - either
     - linear for speed; parametric when noise or coarse Δt is also
       present
   * - SPDE / grid fields
     - linear
     - the (experimental) SPDE toolbox is validated on the linear
       estimators only

.. note::

   The split is not "linear = clean data only".  The linear estimators'
   ``"auto"`` defaults already absorb a good deal of measurement noise and
   finite-:math:`\Delta t` bias, so the fast path stays useful on real data
   without extra knobs.  Reach for the parametric estimators when linear and
   parametric disagree, not reflexively;
   :doc:`/inference/noise_and_sampling` covers the symptoms and the workflow
   in depth.

When in doubt, run the linear estimator first to fix scales and candidate
terms, then confirm with the parametric estimator — agreement is itself a
diagnostic, and disagreement usually flags noise or sampling effects the
linear path cannot absorb.


Diffusion estimation
--------------------

The linear estimators need a constant diffusion tensor up front:
:meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant`
estimates it — together with the measurement-noise covariance
:math:`\Lambda` — before force inference.  **The parametric estimators do not
need this call**; they profile :math:`(\mathbf{D}, \Lambda)` natively as part
of the fit.

The method is chosen automatically (``method="auto"``), but can be forced:

.. list-table::
   :header-rows: 1

   * - Method
     - When to use
   * - ``"noisy"``
     - Measurement noise present (:math:`\mathrm{tr}(\Lambda) > 0`); the
       noise-robust estimator (overdamped: Vestergaard–Blainey–Flyvbjerg)
   * - ``"WeakNoise"``
     - Clean data (no localization error)
   * - ``"MSD"``
     - Simplest mean-square-displacement estimator (biased by noise)

The estimator names are the same in both engines.  After this step,
``inf.diffusion_average`` holds the :math:`d \times d` diffusion matrix and
``inf.Lambda`` the estimated measurement-noise covariance :math:`\Lambda` (the
localization-error matrix).  The ``"noisy"`` estimator fits :math:`\Lambda`
jointly with the diffusion; if your measurement noise is known from
calibration instead, pass it directly to the parametric estimators via
``Lambda=``.

State-dependent diffusion :math:`D(x)` can be inferred via
:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`,
which takes a rank-2 :class:`~SFI.statefunc.Basis`, or via the parametric
:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`.


Force inference
---------------

Linear estimators
^^^^^^^^^^^^^^^^^

Express the force as a linear combination of basis functions
:math:`F(x) = \sum_a c_a \phi_a(x)` and solve for the coefficient vector
:math:`c` by generalized least squares
(:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`):

.. code-block:: python

   from SFI.bases import monomials_up_to

   B = monomials_up_to(order=3, dim=2, rank='vector')
   inf.infer_force_linear(B, preset="auto")

**Conventions.** A single ``preset`` keyword bundles the moment/Gram
conventions; both engines pick it automatically, so you rarely set this by
hand.  The names mean the same thing in both engines — ``"auto"``
(noise-aware), ``"robust"`` (handles measurement noise), ``"clean"`` (sharper
on verified clean data) — plus engine-specific ``legacy-*`` presets that
reproduce the published SFI v1.0 conventions.

*Overdamped* — the ``preset`` sets the moment convention ``M_mode`` and the
Gram construction ``G_mode``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``preset``
     - Resolves to (M_mode, G_mode)
   * - ``"auto"``
     - ``"robust"`` if measurement noise is detected, else ``"clean"`` (default)
   * - ``"robust"``
     - Stratonovich moments + ``"shift"`` Gram (noise-robust)
   * - ``"clean"``
     - Itô moments + ``"trapeze"`` Gram (sharper on clean / fine-sampled data)
   * - ``"KM"``
     - Kramers–Moyal: Itô moments + ``"rectangle"`` Gram
   * - ``"legacy-sfi-v1.0"``
     - Stratonovich + ``"rectangle"`` (the published SFI v1.0 convention)

*Underdamped* — the velocity is reconstructed from positions, so the preset
additionally picks the instantaneous diffusion estimator
``diffusion_method``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``preset``
     - Resolves to (M_mode, G_mode, diffusion_method)
   * - ``"auto"``
     - ``"robust"`` if measurement noise is detected, else ``"clean"`` (default)
   * - ``"robust"``
     - ``symmetric`` + ``trapeze`` + ``noisy``
   * - ``"clean"``
     - ``symmetric`` + ``trapeze`` + ``WeakNoise``
   * - ``"legacy-clean-v1.0"``
     - ``early`` + ``rectangle`` + ``MSD`` (SFI v1.0)
   * - ``"legacy-noisy-v1.0"``
     - ``symmetric`` + ``rectangle`` + ``noisy`` (SFI v1.0)

Power users can override the individual axes: ``M_mode``, ``G_mode`` (and,
underdamped, ``diffusion_method``) take precedence over the preset when set
explicitly.

Parametric estimators
^^^^^^^^^^^^^^^^^^^^^

For models that are not a simple linear combination of basis functions, or
when measurement noise and coarse sampling dominate.  Pass the model to
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`; each
observation interval is advanced with an accurate RK4 step (refined into
``n_substeps`` sub-steps for very coarse data), and
:math:`(\mathbf{D}, \Lambda)` are estimated as part of the fit:

.. code-block:: python

   # Overdamped
   inf = SFI.OverdampedLangevinInference(coll)
   inf.infer_force(F)                            # Basis or PSF
   inf.infer_diffusion(basis_D)                  # state-dependent D(x)

   # Underdamped
   inf_ud = SFI.UnderdampedLangevinInference(coll_ud)
   inf_ud.infer_force(F_ud)                      # F(x, v), needs_v=True
   inf_ud.infer_diffusion(basis_D_ud)            # D(x, v)

``F`` can be either kind of model:

- a **Basis** — the force is a linear combination of fixed functions (the
  same object the linear estimators take).  This takes the fast
  :term:`Gauss–Newton` route and needs **no starting guess**: the fit begins
  at zero.  PASTIS sparsification is wired in automatically.
- a **PSF** (*parametric state function*) — any differentiable force model
  with adjustable parameters, including ones where the parameters enter
  nonlinearly (a saturating force, a gated term, a neural-network drift).
  This takes the :term:`L-BFGS` route and **needs an initial guess** for the
  parameters, since a general nonlinear fit will not converge from an
  arbitrary start.  For large models raise the inner budget with
  ``inner_maxiter``:

.. code-block:: python

   from SFI.statefunc import make_psf

   # Define a parametric force (single-sample function)
   def my_force(x, *, params, mask=None, extras=None):
       k, a = params["k"], params["a"]
       return -k * x + a * x**3

   F = make_psf(my_force, dim=2, rank=1, n_features=1,
                params={"k": (), "a": ()})

   inf.infer_force(F, {"k": 1.0, "a": 0.0})

Beyond fitting non-linear models, the parametric estimators add: the
measurement-noise level :math:`\Lambda` is **modelled** and its bias
corrected (the linear estimators can only partly absorb it); the RK4 step
gives **coarse-sampling robustness**, following the deterministic motion
across each interval instead of a finite-difference velocity; and
**diffusion and noise are estimated automatically** during the force fit.

A note on large :math:`\Delta t`: on a **linear basis** the Gauss–Newton
route is the most robust option of all; the L-BFGS route used for general
PSFs is *less* tolerant of coarse sampling.  When both ``D=`` and ``Lambda=``
are given they are held fixed and the estimation step is skipped — the fast
path when the noise levels are already known.  See
:doc:`/inference/noise_and_sampling` for when these effects matter and how to
diagnose them.

.. seealso::

   :ref:`parametric-concept` — mathematical foundations.

   :ref:`parametric-algorithm` — detailed algorithm description and full
   parameter reference.


Interacting particle systems
----------------------------

Both estimator families handle interacting particles — forces that depend on
the relative positions (and, underdamped, velocities) of neighbours.  You
build the interaction model **compositionally** from the pair primitives in
:mod:`SFI.bases.pairs` and dispatch it over neighbours; the data layout and
full recipes are in :doc:`/particles/user_guide`.  In brief:

- **Linear** — stack ready-made pair bases
  (:func:`~SFI.bases.pairs.radial_pair_basis`,
  :func:`~SFI.bases.pairs.angular_pair_basis`, …) alongside any
  single-particle basis with ``&``, then pass the result to
  ``infer_force_linear`` — nothing else changes.
- **Parametric** — compose the same primitives with *learnable* kernels:
  :func:`~SFI.bases.pairs.parametric_radial_kernel` fits a scalar radial
  kernel :math:`k(r;\theta)`, multiplied by a direction
  (:func:`~SFI.bases.pairs.pair_direction`) or an angular coupling, and you
  hand the resulting model to ``infer_force``.  Worked examples:
  :doc:`/gallery/abp_nonreciprocal_demo`.

For a fully custom K-body rule, drop to
:func:`~SFI.statefunc.make_interactor` directly
(:doc:`/particles/user_guide`); the pre-built primitives cover the common
cases.

The underdamped interacting solver scales **linearly in the number of
particles**: it keeps the within-particle coupling and drops the small
cross-particle terms, which contribute only at higher order in
:math:`\Delta t`.  Worked example:
:doc:`/gallery/advanced/flocking_3d_demo` — 3-D underdamped flocking of 20
particles, with the two solvers checked head-to-head.


Sparse model selection
----------------------

When the basis is larger than the true model, many coefficients should be
zero.  :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force`
explores the space of sparse supports and selects the optimal model via an
information criterion:

.. code-block:: python

   result = inf.sparsify_force(criterion="PASTIS", method="beam")

Four search methods are available: **beam search** (default, PASTIS
original), **greedy stepwise**, **STLSQ** (SINDy-style), and **LASSO**.  The
last two are deterministic-regression sparsifiers kept for benchmarking
against those toolchains, not the recommended path for stochastic data — see
:ref:`sparsity-user-guide`.  Available criteria: ``"PASTIS"`` (recommended),
``"AIC"``, ``"BIC"``.

The returned :class:`~SFI.inference.sparse.SparsityResult` holds the full
Pareto front and can be re-queried for any criterion without re-running the
search.

.. seealso::

   :ref:`sparsity-user-guide` — full user guide with all parameters.

   :ref:`sparsity-theory` — mathematical foundations and software
   architecture.


Error estimation and data requirements
---------------------------------------

After force inference,
:meth:`~SFI.inference.OverdampedLangevinInference.compute_force_error`
computes the parameter covariance and the predicted error:

.. code-block:: python

   inf.compute_force_error()

It sets:

- ``inf.force_coefficients_covariance`` — :math:`\Sigma_c`
- ``inf.force_coefficients_stderr`` — :math:`\sqrt{\mathrm{diag}(\Sigma_c)}`
- ``inf.force_predicted_MSE`` — the expected normalised mean-square error
  (NMSE) of the inferred force

That predicted NMSE is your main guide to whether you have enough data — SFI
estimates its own error, so you rarely need to guess a threshold:

- As a rule of thumb, values below ~0.1 indicate a quantitatively reliable
  fit; values above ~0.5 an underpowered one.
- In the variance-limited regime the error falls inversely with the amount of
  data: doubling the total observation time roughly halves the predicted
  NMSE.  A larger basis costs proportionally more data for the same accuracy —
  sparsification
  (:meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force`) recovers
  most of that cost when the true model is sparse.
- When the predicted error keeps shrinking with more data but the *realised*
  error (against held-out data or ground truth) plateaus, you have hit a bias
  floor — typically measurement noise or coarse sampling.  Switch to the
  parametric estimators (:doc:`/inference/noise_and_sampling`).

To confirm a suspected bias floor with an explicit train/test split, use
:meth:`~SFI.inference.OverdampedLangevinInference.holdout_score`:

.. code-block:: python

   train, test = coll.split_time(0.8)
   inf = SFI.OverdampedLangevinInference(train)
   # ... fit on train ...
   inf.compute_force_error()
   inf.holdout_score(test)
   # {"holdout_NMSE": ..., "predicted_NMSE": ..., "ratio": ..., ...}
   # ratio ≈ 1: sampling-limited.  ratio ≫ 1: bias floor.

.. note::

   Held-out validation is a **side feature for data-abundant scenarios**.
   SFI works hard to be data-efficient — the predicted error and the
   diagnostics suite cost no data at all — and holding out a significant
   fraction for testing runs against that spirit.  Use it when data is
   plentiful, or as the decisive check on a suspected bias floor.  The score
   is a bias detector, not a precision instrument: its resolution is set by
   the χ² fluctuations of the residuals (the calibrated quantity is
   ``excess_z``).


Consistency diagnostics
-----------------------

Once the fit and its predicted error are in hand, the canonical sanity check
is :func:`SFI.diagnostics.assess` — equivalently,
:meth:`~SFI.inference.OverdampedLangevinInference.diagnose`:

.. code-block:: python

   from SFI.diagnostics import assess

   report = assess(inf, level="standard")   # or inf.diagnose()
   report.print_summary()
   for msg in report.flag_issues(alpha=0.01):
       print("warn:", msg)

The report contains residual moments, Ljung--Box / Box--Pierce
autocorrelation tests on :math:`z_t` and :math:`z_t^2`, Kolmogorov--Smirnov /
Anderson--Darling / Jarque--Bera normality tests, the probability integral
transform, the conditioning of the Gram matrix, coefficient z-scores, and a
comparison of predicted vs realised :math:`\chi^2`.  See the
:ref:`Diagnostics user guide <diagnostics_user_guide>` for the full
inventory, plotting helpers, and interpretation guide.


Validation against ground truth
-------------------------------

When the true model is available (simulation benchmarks), compare against it
with :meth:`~SFI.inference.OverdampedLangevinInference.compare_to_exact`:

.. code-block:: python

   inf.compare_to_exact(
       model_exact=proc,       # LangevinProcess with .force_sf, .diffusion_sf
       data_exact=coll,        # clean trajectory for evaluation
       maxpoints=2000,
   )
   print(f"NMSE_force = {inf.NMSE_force:.4f}")

Alternatively, pass explicit callables:

.. code-block:: python

   inf.compare_to_exact(
       force_exact=my_force_sf,
       diffusion_exact=my_D_matrix,
   )


Bootstrapped simulation
-----------------------

Generate a trajectory from the inferred model as a qualitative validation
with
:meth:`~SFI.inference.OverdampedLangevinInference.simulate_bootstrapped_trajectory`:

.. code-block:: python

   coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(
       key, oversampling=10,
   )

This builds an :class:`~SFI.langevin.OverdampedProcess` (or
:class:`~SFI.langevin.UnderdampedProcess`) from the inferred force and
diffusion, simulates it, and returns the trajectory and process object.

For a **pooled multi-experiment fit** (one with :term:`per-dataset
parameters <per-dataset parameter>`), pass ``dataset=k`` to reproduce a
specific experiment.  The inferred model is collapsed to that condition — its
per-dataset parameters are folded at ``k`` and the ``dataset_index``
dependence is removed — so the returned process and trajectory are a
standalone single experiment:

.. code-block:: python

   coll_k, proc_k = inf.simulate_bootstrapped_trajectory(key, dataset=k)

``proc_k`` is experiment ``k``'s own model and ``coll_k`` a plain single
trajectory, so re-inferring on it uses an ordinary (non-pooled) basis.  You
can collapse the fitted model the same way without simulating, via
``inf.force_inferred.specialize(dataset=k)``.


Reporting and serialization
---------------------------

:meth:`~SFI.inference.OverdampedLangevinInference.print_report` writes a
human-readable summary to stdout;
:meth:`~SFI.inference.OverdampedLangevinInference.report_dict` returns the
same results as a machine-readable dict.  Persist a fit with
:func:`~SFI.inference.save_results` / :func:`~SFI.inference.load_results`
(lightweight summary) or :func:`~SFI.inference.save_model` /
:func:`~SFI.inference.load_model` (the full callable model, a JAX pytree via
equinox):

.. code-block:: python

   inf.print_report()          # human-readable summary to stdout
   d = inf.report_dict()       # machine-readable dict of all results

   # Save / load lightweight summary
   from SFI.inference import save_results, load_results
   save_results(inf, "results.npz")
   summary = load_results("results.npz")

   # Save / load the full callable model (JAX pytree via equinox)
   from SFI.inference import save_model, load_model
   save_model(inf.force_inferred, "force.eqx")


Troubleshooting
---------------

**NMSE is large (> 0.5).**
The basis may be too small, or the data too short.  Try increasing the
polynomial order or adding more kernel length scales, and check that the
trajectory covers the relevant state space.

**All coefficients are pruned by PASTIS.**
The signal is too weak for the library size.  Reduce the basis, increase the
trajectory length, or check that ``dt`` is appropriate.

**Gram matrix is singular.**
The basis contains linearly dependent features (e.g. redundant monomials).
Remove duplicates or use a smaller basis.

**Diffusion estimate is negative or noisy.**
The trajectory may be too short, or ``dt`` too large.  Try a different
diffusion estimator (``method="WeakNoise"`` or ``"noisy"``).

**Diagnostics flag residual autocorrelation or MSE inconsistency.**
On experimental data this usually signals measurement noise or coarse
sampling rather than a wrong basis — see
:doc:`/inference/noise_and_sampling`.
