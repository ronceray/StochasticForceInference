.. _glossary:

Glossary
========

Short definitions of the jargon that appears across the SFI docs.
Cross-link from any reference page with ``:term:\`PASTIS\``` (etc.).

.. glossary::
   :sorted:

   PASTIS
      Parsimonious Stochastic Inference — the canonical information
      criterion used by :meth:`~SFI.OverdampedLangevinInference.sparsify_force`.
      Penalises support cardinality with a Bayes-factor-like prior set
      by ``p``.  Gerardos & Ronceray, *Phys. Rev. Lett.* **135**,
      167401 (2025).

   AIC
      Akaike Information Criterion.  Penalises support cardinality by
      ``2k``; the classical "thin" prior.

   BIC
      Bayesian Information Criterion.  Penalises support cardinality by
      ``k log N``; stricter than AIC at large sample sizes.

   LASSO
      L\ :sub:`1`-penalised least-squares for sparse model selection.
      Implemented as :class:`~SFI.inference.LassoStrategy`.

   STLSQ
      Sequential Thresholded Least Squares — the SINDy-style strategy.
      Implemented as :class:`~SFI.inference.STLSQStrategy`.

   L-BFGS
      Limited-memory BFGS, a quasi-Newton optimiser; the inner solver
      of the parametric estimator for nonlinear-in-θ :class:`~SFI.statefunc.PSF` families
      (``inner="lbfgs"``) and for the :math:`(D, \Lambda)` profile.

   RK4
      Classical fourth-order Runge–Kutta scheme; used by the parametric
      estimator to integrate the deterministic drift flow over each
      observation interval.

   Heun
      Stochastic Heun predictor–corrector integrator (weak order 2);
      the **default** scheme of
      :class:`~SFI.langevin.OverdampedProcess` (``method="heun"``).
      ``method="euler"`` selects the classical Euler–Maruyama
      integrator (weak order 1).

   Gauss–Newton
      Linearisation-then-least-squares method for parametric inference,
      the fast path for linear-in-θ bases (``inner="gn"``).  Replaces
      the Hessian of the loss with :math:`J^\top J` of the
      test-function Jacobian.

   Gram matrix
      :math:`G_{\alpha\beta} = \langle \phi_\alpha, \phi_\beta \rangle`,
      the normal-equation matrix assembled by
      :mod:`SFI.integrate` from time-averaged basis evaluations.

   Itô convention
      SDE interpretation where the stochastic increment
      :math:`\sqrt{2D(x_t)}\,dW_t` is evaluated at the *left* endpoint
      of the time step.  See :doc:`/physics_reference`.

   Stratonovich convention
      Mid-point evaluation of the stochastic increment.  Required for
      state-dependent ``D``.  See :doc:`/physics_reference`.

   secant velocity
      Centred finite-difference velocity
      :math:`v_t = (x_{t+1} - x_{t-1})/(2\Delta t)` used by the
      underdamped diagnostics and the ULI residual.  See
      :doc:`/diagnostics`.

   local-precision NLL
      Negative log-likelihood weighted by the inverse of the *locally*
      estimated noise covariance, used in the parametric path to
      handle heteroscedastic measurement noise.  See
      :doc:`/inference/parametric_concept`.

   neighbour list
      CSR-encoded list of neighbour indices for each particle, used by
      pair-interaction bases.  Built host-side via
      :func:`SFI.utils.neighbors.build_neighbor_csr` between JIT
      chunks; see AGENTS.md §4.8.

   JAX persistent cache
      On-disk cache of compiled JAX traces, opt-in via
      ``SFI_JAX_CACHE_DIR=~/.cache/sfi/jax_cache``.  Saves seconds to
      minutes per session on repeated runs.

   NMSE
      Normalised Mean Square Error — the canonical force/diffusion
      accuracy metric: mean squared error of the inferred field divided
      by the mean square of the true field.  Available as
      ``inf.NMSE_force`` after
      :meth:`~SFI.OverdampedLangevinInference.compare_to_exact`;
      ``inf.force_predicted_MSE`` is the a-priori estimate that needs no
      ground truth.

   SPDE
      Stochastic Partial Differential Equation — field dynamics on a
      regular grid, where the drift is a spatial-operator functional of
      the field.  SFI infers them via composable stencil operators
      (experimental toolbox); see :doc:`/spde/user_guide`.

   ABP
      Active Brownian Particle — a self-propelled particle carrying a
      position and a heading angle, the canonical active-matter model.
      See the ABP gallery demos.

   PBC
      Periodic Boundary Conditions — wrap-around boundaries on a box or
      grid.  Minimum-image inter-particle displacements are computed by
      :func:`SFI.bases.pairs.pbc_displacement`.

   Extras
      User-defined fields attached to a
      :class:`~SFI.trajectory.TrajectoryCollection` and passed to state
      functions at evaluation time — ``extras_global`` (per experiment)
      and ``extras_local`` (per particle).  Used for box sizes, species
      labels, neighbour lists, trap centres, and other contextual data.

   Interactor
      A local K-body interaction rule
      (:class:`~SFI.statefunc.Interactor`) that is dispatched over a
      neighbour graph to build a global multi-particle Basis/PSF/SF.
      See :doc:`/particles/user_guide`.

   particles
      The ``N`` axis of a trajectory's ``(T, N, d)`` state array — the
      independent or interacting bodies tracked over time (cells,
      colloids, agents, …).  State functions declare how they consume
      this axis through ``pdepth``: ``pdepth=0`` evaluates one particle
      at a time (the same law applied independently to each), while
      ``pdepth=1`` sees all particles together for interactions.  The
      particle count may vary over time; the :term:`mask` records
      entries and exits.  See :doc:`/particles/user_guide`.

   linear estimators
      The closed-form estimator family:
      :meth:`~SFI.OverdampedLangevinInference.infer_force_linear`,
      :meth:`~SFI.OverdampedLangevinInference.infer_diffusion_linear`,
      :meth:`~SFI.OverdampedLangevinInference.compute_diffusion_constant`.
      A projection onto a basis — no initial guess, no iterations —
      exact in the fine-sampling, low-noise limit, biased outside it.
      See :ref:`choosing-an-estimator`.

   parametric estimators
      The likelihood-based estimator family:
      :meth:`~SFI.OverdampedLangevinInference.infer_force`,
      :meth:`~SFI.OverdampedLangevinInference.infer_diffusion`.  One or
      more RK4 flow steps per observation interval, windowed-precision
      NLL,
      native :math:`(\mathbf{D}, \Lambda)` profiling; robust to
      measurement noise and coarse sampling, accepts nonlinear-in-θ
      models.  See :ref:`choosing-an-estimator`.

   errors-in-variables
      Regression bias arising when the regressors themselves carry
      noise.  In SFI, localization noise enters both the
      finite-difference velocities and the basis evaluations at
      measured positions, biasing the linear estimators on nonlinear
      systems; the parametric estimators correct it via the
      :term:`skip-trick` instrument.

   skip-trick
      The errors-in-variables :term:`instrument` of the parametric
      Gauss–Newton path: test functions are evaluated at temporally
      separated (skipped) observations, decorrelating the instrument
      from the measurement noise of the residual and restoring
      consistency.  On by default (``eiv="auto"``).

   instrument
      In errors-in-variables regression, a quantity correlated with
      the true regressor but uncorrelated with its measurement noise,
      used to build an unbiased estimating equation.

   windowed precision
      The banded inverse covariance of the parametric residuals over a
      short time window, providing the weights of the parametric NLL.
      Captures the correlations that measurement noise induces between
      consecutive residuals.

   conditional NLL
      The negative log-likelihood seen as a function of
      :math:`(\mathbf{D}, \Lambda)` with the model parameters
      :math:`\theta` held at their fitted values — minimised once to
      refine the profiled noise levels.

   profiling
      Internal estimation of nuisance parameters — in SFI, the
      diffusion level :math:`\mathbf{D}` and measurement-noise
      covariance :math:`\Lambda` during a parametric fit — so the
      user does not have to supply them.  Skipped entirely when both
      are passed explicitly.

   measurement-noise covariance
      The covariance :math:`\Lambda` of the localization error on each
      recorded position.  Estimated jointly with the diffusion by the
      :term:`Vestergaard` method (linear estimators, exposed as
      ``inf.Lambda``) or profiled natively (parametric estimators).

   moment estimator
      A closed-form estimator built from low-order moments of the
      increments — used to initialise the parametric
      :math:`(\mathbf{D}, \Lambda)` profile, and, in the linear
      estimators, selected by the :term:`M_mode` convention.

   Vestergaard
      The covariance-based constant-diffusion estimator (after
      Vestergaard *et al.*), which fits the diffusion and the
      localization-error covariance jointly; selected by
      ``compute_diffusion_constant(method="noisy")`` — the noise-robust
      choice of
      :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant`
      and its ``"auto"`` selection when
      noise is detected.  Vestergaard, C. L., Blainey, P. C. &
      Flyvbjerg, H., *Optimal estimation of diffusion coefficients from
      single-particle trajectories*, **Phys. Rev. E** 89, 022726 (2014).

   WeakNoise
      The clean-data constant-diffusion estimator of
      ``compute_diffusion_constant``; assumes negligible localization
      error.

   M_mode
      The moment/kinematics convention of the linear estimators.
      Overdamped: ``"auto"`` (noise-aware selection), ``"Ito"``,
      ``"Ito-shift"``, ``"Strato"``.  Underdamped: ``"symmetric"``
      (the ``"auto"`` resolution), ``"early"``, ``"anticipated"``.

   G_mode
      The Gram-matrix construction mode of the linear estimators:
      ``"rectangle"``, ``"trapeze"``, ``"shift"`` (overdamped), plus
      ``"doubleshift"`` (underdamped).

   trapeze
      The trapezoidal Gram construction (``G_mode="trapeze"``), which
      symmetrises basis evaluations across each interval and removes
      the leading finite-Δt bias of the rectangle rule.  Amri *et
      al.*, *Phys. Rev. Research* **6**, 043030 (2024).

   Basis
      A parameter-free dictionary of state functions
      (:class:`~SFI.statefunc.Basis`) — the model class of the linear
      estimators, and the linear-in-θ fast path of the parametric
      estimators.  See :doc:`/bases/user_guide`.

   PSF
      Parametric State Function (:class:`~SFI.statefunc.PSF`) — a
      model family :math:`F(x;\theta)` with a named parameter tree;
      the model class of nonlinear parametric inference.  See
      :doc:`/statefunc/user_guide`.

   SF
      State Function with frozen parameters
      (:class:`~SFI.statefunc.SF`) — the evaluable object produced by
      a fit, ready for Langevin simulation.

   rank
      The tensor rank of a state-function output: 0 = scalar, 1 =
      vector (forces), 2 = matrix (diffusion tensors).

   mask
      The boolean validity array (shape ``(T, N)``) attached to each
      dataset, encoding missing frames and particles entering or
      leaving.  Honoured automatically by state functions and
      estimators.

   degradation
      Standardised synthetic data imperfections — added measurement
      noise, downsampling, frame loss, motion blur — applied via
      :mod:`SFI.trajectory.degrade` to quantify estimator sensitivity.

   bootstrapped trajectory
      A trajectory simulated from the *inferred* force and diffusion
      (:meth:`simulate_bootstrapped_trajectory`), used as a
      qualitative validation and for error propagation.

   held-out NMSE
      The residual-based normalised mean-square error of a fitted
      force on an independent test collection
      (``inf.holdout_score(test)`` after
      ``coll.split_time(...)``), with the diffusion noise floor
      subtracted.  A side feature for data-abundant scenarios — SFI's
      default validation (``force_predicted_MSE`` + diagnostics) costs
      no data; the held-out score is a bias detector whose resolution
      is set by χ² fluctuations.

   Pareto front
      The error-vs-sparsity frontier explored by
      :meth:`sparsify_force`; the returned
      :class:`~SFI.inference.sparse.SparsityResult` stores it and can
      be re-queried under any criterion without re-running the search.

   beam search
      The default sparse-search strategy of :meth:`sparsify_force`
      (the PASTIS original): a beam of candidate supports is grown and
      pruned by the information criterion.

   information criterion
      A penalised-likelihood score used to compare sparse supports:
      :term:`PASTIS` (recommended), :term:`AIC`, :term:`BIC`.

   velocity reconstruction
      The underdamped engine's internal estimation of unobserved
      velocities from positions (secant differences with
      bias-corrected moments).  You never supply velocities; see
      :doc:`/inference/underdamped`.

   ULI
      Underdamped Langevin Inference — the position-only inference
      scheme for inertial systems (Brückner, Ronceray & Broedersz,
      *Phys. Rev. Lett.* **125**, 058103 (2020)), implemented by
      :class:`~SFI.inference.UnderdampedLangevinInference`.

   Layout
      The grid declaration of the experimental SPDE toolbox
      (``GridLayout``): named field sectors on a regular grid with
      boundary conditions, providing differential operators and
      symmetry-aware embedding.  See :doc:`/spde/layout_guide`.

   Sector
      A named component group within a :term:`Layout` (e.g. a scalar
      field ``U``, a Q-tensor), addressed when building SPDE bases.

   per-dataset parameter
      A model parameter taking an independent inferred value per
      dataset of a pooled multi-experiment collection, selected through
      the reserved ``dataset_index`` extra:
      :func:`~SFI.bases.per_dataset_scalar` (parametric estimators) or
      :func:`~SFI.bases.dataset_indicator` one-hot features (linear
      estimators).  The per-particle analogue lives inside
      :func:`~SFI.statefunc.make_interactor` kernels via the reserved
      ``particle_index`` extra.  To reproduce a single experiment, fold
      the model at one index with
      :meth:`~SFI.statefunc.StateExpr.specialize`, which removes the
      ``dataset_index`` dependence (see :term:`specialize`).

   specialize
      Collapse a pooled model to one experiment's standalone
      single-condition form: :meth:`~SFI.statefunc.StateExpr.specialize`
      folds every :term:`per-dataset parameter` at a chosen
      ``dataset_index`` (per-dataset arrays reduce to that index's slice;
      one-hot indicators become constant) so the result does not read
      ``dataset_index``.  Used by :meth:`~SFI.inference.OverdampedLangevinInference.simulate_bootstrapped_trajectory`
      to export a clean single-trajectory model.

   weights (multi-experiment)
      Per-dataset **unnormalised multipliers** of a
      :class:`~SFI.trajectory.TrajectoryCollection`, applied to every
      estimator (force, diffusion, parametric): ``"pool"`` (default — pool
      all increments on equal footing), ``"per_dataset"`` (each experiment
      counts equally), or an explicit array.  Within-dataset weighting is
      intrinsic to each estimator (force per-Δt, diffusion per-point).  See
      :doc:`/trajectory/user_guide`.
