.. _parametric-algorithm:

Parametric windowed estimators — algorithm and parameters
==========================================================

This page gives a precise, implementation-level description of the
parametric estimator behind
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`,
:meth:`~SFI.inference.UnderdampedLangevinInference.infer_force`, and the
corresponding :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` methods.  The implementation lives in
:mod:`SFI.inference.parametric_core`; everything runs through the
:mod:`SFI.integrate` engine (chunking, masking, JIT) — there are no
hand-rolled trajectory loops.

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
       the single-step minimal estimator).
   * - ``inner``
     - ``"auto"`` → ``"gn"`` for a ``Basis``, ``"lbfgs"`` for a
       ``PSF``; can be forced.
   * - ``eiv``
     - Skip-trick errors-in-variables instrument: ``"auto"`` (default)
       resolves to ``True`` for every model — the estimating equation
       that stays consistent under measurement noise.  Interacting
       models build the instrument from the same N-body flow as the
       residual.  Explicit ``True`` / ``False`` / float blend always
       wins.  Gauss–Newton path only.
   * - ``max_outer``
     - Outer Gauss–Newton / IRLS iterations.
   * - ``inner_maxiter``
     - L-BFGS iterations per outer step on the PSF path (raise for
       NN-scale families).


Solve orchestration
-------------------

The shared driver (``parametric_core.solve._orchestrate``) executes:

1. **Noise init** (skipped when ``D`` and ``Lambda`` are fixed):
   closed-form moment estimators — Vestergaard diffusion + lag-1
   increment-anticorrelation :math:`\Lambda` (overdamped), the ULI
   ``noisy`` diffusion + ULI :math:`\Lambda` (underdamped).  One
   trajectory pass each, eigenvalue-floored to SPD.  For the explicit
   overdamped MLE (``eiv=False``) :math:`\Lambda` is held at zero —
   profiling it against clean stiff data is unidentifiable.

2. **Inner solve** at the initial noise levels:

   * ``inner="gn"`` — the windowed estimator returns the
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
     the NLL).
   * ``inner="lbfgs"`` — frozen-precision L-BFGS over θ
     (iteratively-reweighted), with the same condition-cap ridge as a
     Tikhonov term.

3. **Reprofile once**: minimise the windowed *conditional* NLL (local
   Schur log-determinant — non-degenerate in :math:`(\mathbf{D},
   \Lambda)`) over the Cholesky-parameterised noise matrices at
   the fitted θ, warm-started from the moment init (L-BFGS,
   ``profile_maxiter=20``); then re-solve for θ at the refined
   precision.  For linear-in-θ models this is the
   iteratively-reweighted fixed point — profiling at every outer
   iteration would repeat the dominant-cost step for no change in
   :math:`\hat\theta`.

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
(bandwidth 1), windowed precision with ``n_cond=3`` past residuals
conditioned on.

*Underdamped*: only positions are observed; each window resolves the
velocity by one-Newton-step *shooting*, the 3-point residuals have
pentadiagonal covariance (bandwidth 2, ``n_cond=4``), and the process
noise enters at :math:`O(\Delta t^3)`.

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


Diffusion inference: ``infer_diffusion(basis, ...)``
----------------------------------------------------

Signature (both engines)::

   infer_diffusion(basis=None, *, theta_D0=None,
                   integrator="rk4", n_substeps=1, maxiter=100)

Requires a prior parametric :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`.  The force is held fixed
at :math:`\hat\theta` and the windowed conditional NLL is minimised
over :math:`\theta_D` with L-BFGS (the log-determinant term makes the
diffusion level identifiable); :math:`\Lambda` from the force solve
is held fixed.  ``basis`` is a rank-2 :class:`~SFI.statefunc.Basis` (linear
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
in the particle axis and the cost is O(N) per window.  Non-interacting
multi-particle data simply ``vmap`` the single-particle programs; the
integrate engine reduces and masks over the particle axis.


Cost model
----------

For trajectory length :math:`T`, basis size :math:`n`, dimension
:math:`d`:

- the Gauss–Newton path evaluates the windowed Gram
  (:math:`O(T n d^3)`) once per outer iteration plus once per
  line-search trial; linear-in-θ problems converge in 2–5 iterations;
- the :math:`(\mathbf{D}, \Lambda)` profile costs one trajectory
  pass per L-BFGS iteration (``profile_maxiter=20``), paid once;
- the skip-trick instrument widens the Gram window (one extra flow
  sensitivity per window) — the price of noise consistency;
- passing ``D=`` and ``Lambda=`` removes the profile cost entirely.

As a reference point, an underdamped solve at :math:`T \approx 10^4`,
:math:`n = 8` runs in ~20 s on a laptop CPU core (the linear
estimator: ~10 s).


See also
--------

- :ref:`parametric-concept` — mathematical foundations.
- :doc:`/inference/user_guide` — workflow and estimator choice.
