.. _parametric-concept:

Parametric windowed estimators — concepts
==========================================

This page presents the mathematical foundations of the parametric windowed
estimators used by
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force` and
:meth:`~SFI.inference.UnderdampedLangevinInference.infer_force`.

Throughout we use the notation established in the :doc:`../physics_reference`.


Motivation
----------

The linear estimators (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`) solve a
*linear* regression:

.. math::

   M_a = \sum_b G_{ab}\, c_b

where the force moments :math:`M_a` and Gram matrix :math:`G_{ab}`
are computed at each time step from finite-difference velocities and
user-chosen basis functions.  This estimator is fast, non-iterative,
and unbiased at leading order in :math:`\Delta t`.  However:

1. **Finite-:math:`\Delta t` bias:** the linear regression is exact only
   in the :math:`\Delta t \to 0` limit.  At finite sampling interval the
   force moments carry systematic :math:`O(\Delta t)` errors, especially
   in regions of high curvature.

2. **Measurement noise:** when positions are corrupted by localization
   error :math:`\Lambda`, the finite-difference velocity
   :math:`v = \Delta x / \Delta t` is dominated by :math:`\Lambda /
   \Delta t` rather than by the true velocity.

3. **Nonlinear models:** the regression is restricted to forces that are
   linear in their parameters.  Parametric models such as neural
   networks or rational functions cannot be fitted directly.

The parametric windowed estimator addresses all three limitations.
It replaces the finite-difference velocity by an *exact numerical
flow*, accounts for measurement noise through a *joint Gaussian
likelihood*, and iterates via *Gauss–Newton* to handle nonlinear
parameter dependence.


The observation model
---------------------

We assume the observed trajectory :math:`y_0, y_1, \dots, y_{T-1}` is
related to the true latent state :math:`x_t` by

.. math::
   :label: obs-model

   y_t = x_t + \eta_t, \qquad
   \eta_t \sim \mathcal{N}(0, \Lambda)

where :math:`\Lambda` is the measurement-noise covariance
(localization error).  The latent state evolves according to the
Langevin SDE.

**Overdamped** (position only):

.. math::

   \mathrm{d}x
   = F(x;\,\theta)\,\mathrm{d}t
   + \sqrt{2\,D(x)}\;\mathrm{d}W_t .

**Underdamped** (position + velocity):

.. math::

   \dot{x} = v, \qquad
   \mathrm{d}v
   = F(x, v;\,\theta)\,\mathrm{d}t
   + \sqrt{2\,D}\;\mathrm{d}W_t .


Deterministic-flow linearisation
---------------------------------

Instead of the Euler–Maruyama secant, we linearise each observation
interval around the **deterministic flow** of the drift.  Given the
current state :math:`x_t`, we integrate the noise-free ODE
:math:`\dot{z} = F(z;\theta)`, :math:`z(0) = x_t`, over a full step
:math:`\Delta t` — solved with :math:`n_{\text{sub}}` RK4 micro-steps
(default :math:`n_{\text{sub}} = 1`, a single 4th-order step per
observation interval) — and model the stochastic part as a **single
Gaussian increment** accumulated over that interval.  There is no
operator splitting: one deterministic flow step and one noise
increment per observation.

The resulting **flow map** :math:`\Phi(x;\,\theta)` is the
RK4-integrated displacement:

.. math::
   :label: flow-map

   \Phi(x;\,\theta) = z(\Delta t) - x, \qquad
   \dot{z} = F(z;\,\theta), \quad z(0) = x.

Under this linearisation the *transition distribution* over one step is

.. math::
   :label: parametric-transition

   x_{t+1} \;\big|\; x_t
   \;\sim\;
   \mathcal{N}\!\Big(
      x_t + \Phi(x_t;\,\theta), \;
      V_t
   \Big)

where the *process-noise covariance* :math:`V_t` depends on the
diffusion and on the flow Jacobian
:math:`J_t = \partial \Phi / \partial x \big|_{x_t} + I`:

.. math::
   :label: process-cov

   V_t = \Delta t \bigl(
      J_t \, D(x_t) \, J_t^{\!\top} + D(x_t + \Phi_t)
   \bigr) .

In the leading-order (LO) approximation used by the Gauss–Newton
solver, the log-determinant contribution of :math:`V_t` is treated
as :math:`\theta`-independent.  This is justified because the
:math:`\theta`-dependent part of :math:`\log |V_t|` enters only at
:math:`O(\Delta t^3)`.


Residuals and the joint Gaussian
---------------------------------

Combining :eq:`obs-model` and :eq:`parametric-transition` yields
**residuals**

.. math::
   :label: residuals

   r_t = y_{t+1} - y_t - \Phi(x_t;\,\theta)

that follow a *block-tridiagonal* multivariate Gaussian.  The
diagonal blocks of the joint covariance are

.. math::

   A_t = V_t + J_t \, \Lambda \, J_t^{\!\top} + \Lambda

and the off-diagonal blocks (from measurement noise coupling
consecutive residuals) are

.. math::

   C_t = -\Lambda \, J_t^{\!\top} .

The **pseudo-NLL** (negative log-likelihood up to constants) is:

.. math::
   :label: nll

   -\log p(y \mid \theta)
   = \tfrac{1}{2}
     \bigl[
        \log\det \Sigma
        + r^{\!\top} \Sigma^{-1} r
     \bigr]

where :math:`\Sigma` is the block-tridiagonal matrix with blocks
:math:`(A_t, C_t)`.  Both the log-determinant and the quadratic
form can be evaluated in :math:`O(T d^3)` time via a
block-Cholesky factorisation in a single ``jax.lax.scan`` pass.


Gauss–Newton iteration
-----------------------

The force :math:`F(x;\,\theta) = \sum_\alpha \theta_\alpha \,
\phi_\alpha(x)` (or any differentiable parametric model) enters
nonlinearly through the flow map :math:`\Phi`.  Rather than
optimising the NLL directly (which would require second-order
derivatives of the flow), the parametric estimator uses a
**Gauss–Newton** (GN) linearisation.

Define the **test functions** (parameter sensitivities):

.. math::

   \psi_\alpha(x_t)
   = \frac{\partial \Phi(x_t;\,\theta)}{\partial \theta_\alpha} ,

computed exactly through the flow with ``jax.jacfwd``.

**Errors-in-variables: the skip-trick instrument.**  Under measurement
noise the regressor :math:`\psi(y_t)` is evaluated at a *noisy* point,
which correlates it with the residual and biases the symmetric normal
equations (the classical errors-in-variables effect).  The estimator
therefore replaces the *left* factor of the normal equations by an
η-clean **instrument**: the sensitivity evaluated at a lagged point
whose noise is independent of the residual's,

.. math::

   \psi_{\rm inst}
   = \frac{\partial \Phi}{\partial \theta}
     \Big|_{\text{stop-grad clean point}} ,

giving an asymmetric estimating equation whose root remains consistent
under noise (``eiv=True``, the default on the Gauss–Newton path;
``eiv=False`` recovers the plain MLE).

The **whitened Gram matrix** and **whitened right-hand side** are

.. math::
   :label: gram-rhs

   G_{\alpha\beta}
   &= \sum_t
      \psi_\alpha^{\!\top} \, \Sigma_{\rm loc}^{-1} \, \psi_\beta, \\
   f_\alpha
   &= \sum_t
      \psi_\alpha^{\!\top} \, \Sigma_{\rm loc}^{-1} \, \rho_t,

where :math:`\rho_t = r_t + \psi \cdot \theta` is the augmented
residual and :math:`\Sigma_{\rm loc}` is the local-precision
covariance window (see :ref:`parametric-algorithm` for details).

One GN step updates

.. math::

   \theta^{(k+1)} = G^{-1} f

(not an increment — :math:`f` already contains the shifted residual).
The iteration converges in 3–10 steps for typical problems.

**Regularisation.**  When :math:`\operatorname{cond}(G) >
\texttt{gram\_cond\_max}` the system is Tikhonov-damped:
:math:`G_{\rm reg} = G + \lambda I` with
:math:`\lambda = \sigma_1(G) / \texttt{gram\_cond\_max}`.


Comparison with previous SFI estimators
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`
     - :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` (parametric)
   * - Discretisation
     - Euler (:math:`O(\Delta t)`)
     - RK4 parametric (:math:`O(\Delta t^2)`)
   * - Measurement noise
     - Heuristic subtraction
     - Joint likelihood :eq:`nll`
   * - Nonlinear models
     - No (linear in :math:`\theta`)
     - Yes (Gauss–Newton)
   * - (D, Λ) estimation
     - Pre-computed constant
     - Moment init + profiled conditional NLL
   * - Cost per iteration
     - :math:`O(T n^2 d)`
     - :math:`O(T n d^3)` (windowed)
   * - Iterations needed
     - 1 (exact solve)
     - 2–5 GN steps

The parametric windowed approach is **more accurate at larger**
:math:`\Delta t`, **handles measurement noise natively**, and can
fit **nonlinear parametric models**.  The cost is a modest iteration
loop.  For small :math:`\Delta t` and clean data, the linear
regression is often sufficient.


Overdamped vs. underdamped specifics
-------------------------------------

Overdamped parametric
^^^^^^^^^^^^^^^^^^^^^

The overdamped estimator uses a **tridiagonal** covariance structure
(bandwidth 1): each residual couples only to its immediate neighbours
through the off-diagonal block :math:`C_t = -\Lambda J_t^\top`.

The residuals are:

.. math::

   r_t = y_{t+1} - y_t - \Phi(y_t;\,\theta) .

The solve proceeds in three steps:

1. **Moment init:** :math:`(\mathbf{D}, \Lambda)` from closed-form
   noise-robust moment estimators (Vestergaard diffusion + lag-1
   increment anticorrelation) — one trajectory pass, no optimiser.
2. **Gauss–Newton loop:** iterated whitened least-squares at the
   fixed noise levels.
3. **Reprofile:** one windowed conditional-NLL refinement of
   :math:`(\mathbf{D}, \Lambda)` at the fitted :math:`\theta`,
   followed by a warm re-solve (the iteratively-reweighted fixed
   point for linear-in-:math:`\theta` models).


Underdamped parametric (ULI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The underdamped estimator works in :math:`(x, v)` phase space — only
positions are observed, so the velocity at each window is resolved by
*shooting*: one Newton step finds the initial velocity whose flow hits
the next observed position.  The 3-point residuals have a
**pentadiagonal** covariance structure (bandwidth 2), and the process
noise enters the position at :math:`O(\Delta t^3)`.

The skip-trick instrument is built from the clean lagged position pair
and is particularly effective here: the shooting velocity divides the
position noise by :math:`\Delta t`, so the velocity errors-in-variables
bias of the plain MLE is strong.

.. note::

   A single **Euler** step cannot support the underdamped skip-trick:
   the Euler position update :math:`x + v\,\Delta t` does not depend on
   the force, so the instrument is identically zero.  The solver
   detects ``integrator="euler", n_substeps=1`` with ``eiv`` enabled,
   warns, and falls back to the plain MLE.  The RK4 default does not
   have this degeneracy.


State-dependent diffusion inference
------------------------------------

After force inference, ``infer_diffusion()`` fits a state-dependent
diffusion tensor :math:`D(x;\,\theta_D)` by optimising the windowed
local-precision NLL with the force held fixed.

Two parametrisations are available:

- a rank-2 **Basis**: :math:`D(x) = \sum_j \theta_j \, d_j(x)`
  (linear in :math:`\theta_D`; the default
  ``symmetric_matrix_basis(d)`` spans all constant symmetric
  matrices);
- a **PSF**: :math:`D(x) = \text{PSF}(x;\,\theta_D)` directly — in the
  underdamped case the model may be velocity-dependent,
  :math:`D(x, v)`, evaluated at the shooting velocity.

The optimisation uses L-BFGS with exact JAX gradients through the
integrate framework, initialised by projecting the moment-estimated
constant :math:`\hat{\mathbf{D}}` onto the model (a zero start is a
singular covariance and cannot be escaped).


Key references
--------------

1. Frishman, A. & Ronceray, P., *Learning force fields from stochastic
   trajectories*, **Phys. Rev. X** 10, 021009 (2020).
   — The original SFI linear regression estimator.

2. Vestergaard, C. L., Blainey, P. C. & Flyvbjerg, H., *Optimal
   estimation of diffusion coefficients from single-particle
   trajectories*, **Phys. Rev. E** 89, 022726 (2014).
   — Measurement-noise-aware diffusion estimation.

3. Brückner, D. B., Ronceray, P. & Broedersz, C. P., *Inferring the
   dynamics of underdamped stochastic systems*, **Phys. Rev. Lett.**
   125, 058103 (2020).
   — Underdamped Langevin Inference (ULI), extended here to the parametric
   windowed estimator.

4. Amri, S. *et al.*, *Inferring geometrical dynamics of cell nucleus
   translocation*, **Phys. Rev. Res.** 6, 043030 (2024).
   — Trapezoidal integration scheme.


.. seealso::

   :ref:`parametric-algorithm` — algorithmic implementation details
   (parameters, stages, data flow).

   :ref:`inference-user-guide` — how to use the inference methods
   in practice.
