.. _parametric-algorithm:

Parametric estimators — algorithm and parameters
==========================================================

This page gives a precise, implementation-level description of the
parametric estimator behind
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
:meth:`~SFI.inference.UnderdampedLangevinInference.infer_force`, and the
corresponding :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` methods.  The implementation lives in
:mod:`SFI.inference.parametric_core`.

See :ref:`parametric-concept` for the mathematical motivation.


Force inference: ``infer_force(F, ...)``
----------------------------------------

Signature (both engines)::

   infer_force(F, theta0=None, *,
               D=None, Lambda=None,
               integrator="rk4", n_substeps=1,
               inner="auto", eiv="auto",
               max_outer=5, inner_maxiter=80)

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Parameter
     - Meaning
   * - ``F``
     - :class:`~SFI.statefunc.Basis` (linear-in-θ; converted with ``to_psf()``, PASTIS
       scorer wired) or :class:`~SFI.statefunc.PSF` (general differentiable family).
   * - ``theta0``
     - Initial parameters (dict or flat array; default zeros).
   * - ``D``, ``Lambda``
     - When *both* are given they are held fixed and profiling is
       skipped (the fast path).  Otherwise profiled — see below.
   * - ``integrator``
     - ``"rk4"`` (default) or ``"euler"`` flow predictor.
   * - ``n_substeps``
     - Integrator micro-steps per observation interval (default 1 —
       the single-step minimal estimator).  Raise to 2 at coarse
       effective sampling: the RK4 flow model saturates at
       ``n_substeps=2`` on typical chaotic benchmarks.  With per-step
       ``dt`` the substep is ``h_k = dt_k / n_substeps``.
   * - ``inner``
     - ``"auto"`` → ``"gn"`` (direct for a linear ``Basis``, damped
       for a nonlinear ``PSF``).  ``"lbfgs"`` (frozen-precision
       quadratic) can be forced on the overdamped engine; underdamped
       ``"lbfgs"`` redirects to ``"gn"`` with a warning.
   * - ``eiv``
     - Skip-trick errors-in-variables instrument: ``"auto"`` (default)
       resolves to ``True`` for every model — the estimating equation
       that stays consistent under measurement noise.  Interacting
       models build the instrument from the same N-body flow as the
       residual.  Explicit ``True`` / ``False`` / float blend always
       wins.  See the coarse-sampling warning below.
   * - ``max_outer``
     - Outer Gauss–Newton / IRLS iterations.
   * - ``inner_maxiter``
     - L-BFGS iterations per outer step on the L-BFGS path (raise for
       NN-scale families).

.. warning::

   **Coarse sampling on (near-)clean data.**  ``eiv="auto"`` does not
   inspect the noise level.  A Δt-independent measurement noise
   :math:`\Lambda` is only identifiable against process noise
   :math:`\propto \Delta t` when the sampling resolves the dynamics; at
   coarse effective sampling on nonlinear systems the ``(D, Λ)``
   profile can fit a **spurious** :math:`\Lambda` on clean data,
   inflating/skewing :math:`\hat D` and degrading the force fit.  When
   the data is known (near-)clean and the sampling is coarse, pass
   ``eiv=False`` — which also holds :math:`\Lambda \equiv 0` in the
   profile — and ``n_substeps=2``.


Solve orchestration
-------------------

The shared driver (``parametric_core.solve._orchestrate``) executes:

1. **Noise init** (skipped when ``D`` and ``Lambda`` are fixed):
   closed-form moment estimators — Vestergaard diffusion + lag-1
   increment-anticorrelation :math:`\Lambda` (overdamped), the ULI
   ``noisy`` diffusion + ULI :math:`\Lambda` (underdamped).  One
   trajectory pass each, eigenvalue-floored to SPD, exact under
   non-uniform sampling (the generalised stencils reduce bitwise to
   the classical ones on uniform grids).  For the explicit
   overdamped MLE (``eiv=False``) :math:`\Lambda` is held at zero —
   profiling it against clean stiff data is unidentifiable.

2. **Inner solve** at the initial noise levels:

   * ``inner="gn"`` — the exact core returns the whitened
     normal-equation pieces :math:`(G, f)` directly (no nested
     autodiff through the flow), so each step is the closed-form
     :math:`\delta\theta = -(G + \lambda I)^{-1} f` with a
     Levenberg–Marquardt line search.  The condition number of
     :math:`G` is capped by a Tikhonov ridge
     (:math:`\operatorname{cond} \le 10^{10}`), which bounds θ along
     unidentified directions instead of letting them diverge.  With
     ``eiv`` active the left factor of :math:`G` and :math:`f` is the
     η-clean instrument and the merit function is the estimating
     equation residual :math:`\|f\|` (the IV root does not minimise
     the NLL).  Nonlinear-in-θ PSFs run the same loop as damped
     Gauss–Newton (the sensitivities come from the θ-recursion with an
     autodiff fallback for ``∂F/∂θ``), keeping the instrument active.
   * ``inner="lbfgs"`` — overdamped: scipy L-BFGS on a
     frozen-precision quadratic (iteratively reweighted), with the
     same condition-cap ridge as a Tikhonov term.  Underdamped
     ``"lbfgs"`` redirects to the damped-GN path with a
     ``RuntimeWarning``: under exact whitening the underdamped
     frozen-precision quadratic is unbounded along the damping
     direction (the shooting velocity at θ_live absorbs over-damped
     misfit).

3. **Reprofile once**: minimise the exact banded NLL over the
   Cholesky-parameterised noise matrices :math:`(\mathbf{D},
   \Lambda)` at the fitted θ, warm-started from the moment init
   (L-BFGS, ``profile_maxiter=None`` → 200).  The profile runs on
   **cached fixed-θ tensors**: residuals, flow Jacobians and RK4 stage
   Jacobians are computed once at :math:`\hat\theta`; every profile
   iteration rebuilds the covariance blocks for the candidate
   :math:`(\mathbf{D}, \Lambda)` from the cache — zero force or basis
   evaluations (budget-gated, with a transparent recompute fallback).
   Then θ is re-solved at the refined precision.  For linear-in-θ
   models this is the iteratively-reweighted fixed point — profiling
   at every outer iteration would repeat the dominant-cost step for
   no change in :math:`\hat\theta`.

4. **Final Gram and covariance**: :math:`(G, f, H)` are recomputed at
   the optimum, with :math:`H = \psi_{\rm left}^\top P \psi_{\rm left}`
   the model-based variance of the estimating function.
   ``compute_force_error()`` uses the sandwich
   :math:`\operatorname{Cov}(\hat\theta) = G^{-1} H G^{-\top}` — on the
   symmetric path :math:`H = G` and this collapses to the familiar
   :math:`G^{-1}`; on the skip-trick path the asymmetric estimating
   equation does not obey the information identity and the sandwich is
   required for correct error bars.

Residual programs
~~~~~~~~~~~~~~~~~

*Overdamped*: 2-point flow residuals, block-tridiagonal covariance
(bandwidth 1), whitened exactly by the reverse-time innovations
recursion.

*Underdamped*: only positions are observed; each residual resolves the
velocity by one-Newton-step *shooting*, the 3-point residuals have
pentadiagonal covariance (bandwidth 2), and the process noise enters
at :math:`O(\Delta t^3)`.

Masked frames and tracking gaps restart the innovations recursion:
only ``bandwidth`` residuals are lost per gap, so fragmented data
keeps nearly all of its information.  Per-particle masks follow the dynamic-at-centre /
static-at-neighbours validity rule, and the instrument's lagged base
points are additionally required to be unmasked.

.. warning::

   The underdamped skip-trick is structurally unavailable for a single
   Euler step: the Euler position update :math:`x + v\,\Delta t` does
   not depend on the force, so the instrument
   :math:`\partial\Phi^x/\partial\theta` vanishes identically and the
   asymmetric Gram is rank-zero.  ``solve_force_ud`` detects
   ``integrator="euler", n_substeps=1`` with ``eiv`` enabled, emits a
   ``RuntimeWarning``, and falls back to ``eiv=False``.  The RK4
   default carries the force into the position within one step and has
   no such degeneracy.


Non-uniform sampling
--------------------

Datasets sampled at irregular times fit directly — force and
diffusion, both engines.  Per-dataset step streams are resolved in
precedence order: an absolute time vector ``t`` (finite differences) →
a per-step ``dt`` array → the scalar ``dt`` broadcast.  Each interval
gets its own flow step (``h_k = dt_k / n_substeps``) and its own
process covariance; nothing is interpolated or resampled, and
different datasets in a collection may carry different (even mixed
uniform/irregular) sampling.

- The moment initialisers are exactly unbiased under non-uniform
  sampling: the overdamped stencils normalise per sample by the
  two-interval mean, and the underdamped ULI stencil subtracts the
  ballistic-leakage term that appears when neighbouring intervals
  differ (all corrections vanish bitwise on uniform grids).
- The underdamped covariance blocks require the Lyapunov-exact process
  covariance (the default); with ``SFI_EXACT_UPGRADES=0`` an irregular
  underdamped fit raises a clear error rather than using leading-order
  blocks that assume equal neighbouring intervals.


Block-level refinements (``SFI_EXACT_UPGRADES``)
------------------------------------------------

Three refinements ship default-on; set ``SFI_EXACT_UPGRADES=0`` to
recover the leading-order blocks:

- **Lyapunov-exact process covariance** — :math:`\dot Q = JQ + QJ^\top
  + 2D` integrated on the same RK4 stages as the flow (exact for
  linear drift), replacing the endpoint trapezoid.  Worth 10–60× in
  diffusion NMSE at coarse sampling; force-neutral.
- **Convexity correction** — cancels the :math:`\tfrac12
  \nabla^2\Phi:\Lambda` residual-mean bias that measurement noise
  induces through drift curvature (noise/EIV path, non-interacting
  overdamped models).
- **Huberized whitened score** — bounded-influence variant of the
  estimating equation for outlier-contaminated data (internal knob).


Diffusion inference: ``infer_diffusion(basis, ...)``
----------------------------------------------------

Signature (both engines)::

   infer_diffusion(basis=None, *, theta_D0=None,
                   integrator="rk4", n_substeps=1, maxiter=100)

Requires a prior parametric :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`.  The force is held fixed
at :math:`\hat\theta` and the exact banded NLL is minimised over
:math:`\theta_D` with L-BFGS (the log-determinant term makes the
diffusion level identifiable); :math:`\Lambda` from the force solve is
held fixed.  The solve runs on the same cached fixed-θ tensors as the
``(D, Λ)`` profile: per-point process covariances for any candidate
:math:`D(x;\theta_D)` are rebuilt from the cached RK4 stage Jacobians,
so each iteration costs linear algebra only — zero force or basis
evaluations.  ``basis`` is a rank-2 :class:`~SFI.statefunc.Basis` (linear
parameterisation; default ``symmetric_matrix_basis(d)``) or a :class:`~SFI.statefunc.PSF`
(direct :math:`D(x)`, or :math:`D(x, v)` in the underdamped case,
evaluated at the shooting velocity).

When ``theta_D0`` is not given, the start point is the least-squares
projection of the moment-estimated constant :math:`\hat{\mathbf{D}}`
onto the model — :math:`\theta_D = 0` means :math:`D \equiv 0`, a
singular covariance whose NLL gradient is non-finite, so L-BFGS could
never leave a zero start.


Multi-particle systems
----------------------

Interacting PSFs (``particles_input=True``) use the full
multi-particle force for the state update and propagate the tangent
per particle from the same-particle derivatives
(``F.d_x(same_particle=True)`` / ``F.d_v(same_particle=True)``) — the
frozen-background approximation, implemented in
``parametric_core.jacobians``.  The flow Jacobian stays block-diagonal
in the particle axis and the cost is O(N) per residual.
Non-interacting multi-particle data simply ``vmap`` the
single-particle programs, with per-particle masking.


Memory and precision
--------------------

The parametric path honours the engine-level ``max_memory_gb``: chunk
planning accounts for the real working set (ψ/J/r buffers, whitening
transients, basis memory hints), so wide or expensive bases stream
through bounded memory.  Chunking is exact — per-particle whitening
carries thread across chunk boundaries, so results are independent of
the chunk size.  All parametric solves run in float64 internally
(scoped ``jax.enable_x64``) whatever the session dtype; results are
returned in float64.


Cost model
----------

For trajectory length :math:`T`, basis size :math:`n`, dimension
:math:`d`:

- the Gauss–Newton path evaluates the whitened Gram
  (:math:`O(T n d^3)`) once per outer iteration plus once per
  line-search trial; linear-in-θ problems converge in 2–5 iterations;
- the sensitivities ψ come from the per-stage θ-recursion — one force
  / ``∂F/∂x`` / ``∂F/∂θ`` evaluation per RK4 stage, independent of
  :math:`n` (no per-parameter forward tangents), with a zero-autodiff
  fast path for linear bases;
- the :math:`(\mathbf{D}, \Lambda)` profile and ``infer_diffusion``
  iterate on cached fixed-θ tensors — linear-algebra cost only, zero
  basis evaluations per iteration;
- passing ``D=`` and ``Lambda=`` removes the profile cost entirely.

Small jobs are typically bound by the XLA compile floor (a few tens of
seconds); set a persistent compilation cache (``SFI_JAX_CACHE_DIR``)
when running many small solves.


See also
--------

- :ref:`parametric-concept` — mathematical foundations.
- :doc:`/inference/user_guide` — workflow and estimator choice.
