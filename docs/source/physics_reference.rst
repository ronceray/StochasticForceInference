Physics Reference
=================

This page collects the key mathematical formulas and physical equations
implemented in the SFI library.  Notation follows the conventions of the
PASTIS paper (Ronceray, 2025) and is used consistently throughout the
documentation.


Notation
--------

Throughout SFI the following symbols are used consistently.
Bold face denotes vectors and matrices; italic denotes scalars.

.. list-table::
   :widths: 20 80

   * - :math:`\mathbf{x}_t`
     - State (position) vector at time :math:`t`.
       Shape ``(N_particles, d)`` or ``(d,)`` for a single particle.
   * - :math:`\mathbf{v}_t`
     - Velocity, either known or reconstructed via finite differences
       (secant velocities).
   * - :math:`\Delta\mathbf{x}_t`
     - Displacement :math:`\mathbf{x}_{t+1} - \mathbf{x}_t`.
   * - :math:`\mathbf{F}(\mathbf{x})`
     - Drift / force field (Itô convention unless stated otherwise).
   * - :math:`\mathbf{D}(\mathbf{x})`
     - Diffusion tensor (Itô convention).
   * - :math:`\bar{\mathbf{D}}`
     - Empirical (time-averaged) diffusion tensor.
       In code: ``inf.diffusion_average``.
   * - :math:`\mathbf{B}`
     - Noise amplitude: :math:`\mathbf{B} = \sqrt{2\mathbf{D}}`.
   * - :math:`\mathrm{d}W_t`
     - Wiener increment (Gaussian white noise),
       :math:`\mathrm{d}W \sim \mathcal{N}(0,\,I\,\mathrm{d}t)`.
   * - :math:`b_i(\mathbf{x})`
     - Basis function :math:`i` evaluated at :math:`\mathbf{x}`.
       Scalar by default; bold :math:`\mathbf{b}_i` when vector-valued.
   * - :math:`\hat{F}_i`
     - Inferred coefficient for basis function :math:`i`.
   * - :math:`n`
     - Library (basis) size — number of basis functions.
   * - :math:`G_{ij}`
     - Gram matrix
       :math:`G_{ij} = \langle \mathbf{b}_i \cdot \bar{\mathbf{D}}^{-1} \cdot \mathbf{b}_j \rangle`.
   * - :math:`M_i`
     - Force moment
       :math:`M_i = \langle \mathbf{v}_t \cdot \bar{\mathbf{D}}^{-1} \cdot \mathbf{b}_i \rangle`.
   * - :math:`\Lambda`
     - Measurement / localization noise covariance.
   * - :math:`p_{\text{PASTIS}}`
     - PASTIS significance threshold (model selection parameter).

.. note::

   **Code mapping:** the code attribute ``diffusion_average`` corresponds
   to :math:`\bar{\mathbf{D}}`.  Older versions of SFI used the symbol
   :math:`A = 2\bar{\mathbf{D}}`; throughout these docs we normalise to
   :math:`\bar{\mathbf{D}}`.


Dynamical equations
-------------------

Overdamped Langevin SDE
~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :class:`SFI.inference.overdamped.OverdampedLangevinInference`,
:class:`SFI.langevin.overdamped.OverdampedProcess`

.. math::

   \mathrm{d}\mathbf{x}_t
   = \mathbf{F}(\mathbf{x}_t)\,\mathrm{d}t
   + \sqrt{2\,\mathbf{D}(\mathbf{x}_t)}\;\mathrm{d}W_t

:math:`\mathbf{F}(\mathbf{x})` is the Itô drift,
:math:`\mathbf{D}(\mathbf{x})` the diffusion tensor (Itô convention),
and :math:`\mathrm{d}W_t` a Wiener increment.


Underdamped Langevin SDE
~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :class:`SFI.inference.underdamped.UnderdampedLangevinInference`,
:class:`SFI.langevin.underdamped.UnderdampedProcess`

.. math::

   \mathrm{d}\mathbf{x}_t = \mathbf{v}_t\,\mathrm{d}t, \qquad
   \mathrm{d}\mathbf{v}_t
   = \mathbf{F}(\mathbf{x},\mathbf{v})\,\mathrm{d}t
   + \sqrt{2\,\mathbf{D}(\mathbf{x},\mathbf{v})}\;\mathrm{d}W_t

Only positions :math:`\mathbf{x}(t)` are observed; velocities are
reconstructed from finite differences (secant velocities).


Simulation
----------

Euler–Maruyama integrator (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.langevin.overdamped` — ``_make_step``

.. math::

   \mathbf{x}_{t+\mathrm{d}t}
   = \mathbf{x}_t + \mathrm{d}t\,\mathbf{F}(\mathbf{x}_t)
     + \sqrt{2\,\mathrm{d}t}\;\mathbf{B}(\mathbf{x}_t)\,\boldsymbol{\xi}_t

where :math:`\mathbf{B} = \sqrt{2\mathbf{D}}` and
:math:`\boldsymbol{\xi}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`.


Velocity-Verlet integrator (underdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.langevin.underdamped` — ``_make_step``

Stochastic splitting (kick–drift–kick):

.. math::

   \mathbf{v}_{1/2} &= \mathbf{v}
     + \tfrac{1}{2}\mathrm{d}t\,\mathbf{F}(\mathbf{x},\mathbf{v})
     + \sqrt{\mathrm{d}t/2}\;\mathbf{B}(\mathbf{x},\mathbf{v})\,
       \boldsymbol{\xi}_1 \\
   \mathbf{x}' &= \mathbf{x} + \mathrm{d}t\,\mathbf{v}_{1/2} \\
   \mathbf{v}' &= \mathbf{v}_{1/2}
     + \tfrac{1}{2}\mathrm{d}t\,\mathbf{F}(\mathbf{x}',\mathbf{v}_{1/2})
     + \sqrt{\mathrm{d}t/2}\;\mathbf{B}(\mathbf{x}',\mathbf{v}_{1/2})\,
       \boldsymbol{\xi}_2

Preserves the symplectic structure of the deterministic part.


Noise amplitude from diffusion tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.langevin.base` — ``_setup_diffusion``

.. math::

   \mathbf{B} = \sqrt{2\,\mathbf{D}}

For scalar :math:`\sigma`: :math:`\mathbf{B} = \sqrt{2\sigma}\,\mathbf{I}_d`.
For constant matrix :math:`\mathbf{D}`: PSD matrix square root.
For state-dependent :math:`\mathbf{D}(\mathbf{x})`: evaluated at each step.


Diffusion estimators
--------------------

MSD diffusion estimator (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_D_msd``

.. math::

   \hat{\mathbf{D}}_{\text{MSD}}(t)
   = \frac{1}{2\,\mathrm{d}t}\,
     \Delta\mathbf{x}_t \otimes \Delta\mathbf{x}_t

Simplest estimator; biased by measurement noise.


Vestergaard–Blainey–Flyvbjerg estimator (the ``"noisy"`` estimator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_D_noisy``

.. math::

   \hat{\mathbf{D}}_{\text{V}}(t) = \tfrac{1}{4}\bigl[
       \Delta\mathbf{x}_t \otimes \mathbf{v}_t
     + 2\,\Delta\mathbf{x}_t \otimes \mathbf{v}_{t-1}
     + 2\,\Delta\mathbf{x}_{t-1} \otimes \mathbf{v}_t
     + \Delta\mathbf{x}_{t-1} \otimes \mathbf{v}_{t-1}
   \bigr]

Two-point estimator robust to measurement noise
(Vestergaard, Blainey, & Flyvbjerg, *Phys. Rev. E*, 2014).


Weak-noise diffusion estimator (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_D_weaknoise``

.. math::

   \hat{\mathbf{D}}_{\text{WN}}(t)
   = \tfrac{1}{4}\bigl(\Delta\mathbf{x}_t - \Delta\mathbf{x}_{t-1}\bigr)
     \otimes \bigl(\mathbf{v}_t - \mathbf{v}_{t-1}\bigr)

Uses successive-displacement differences; suitable when localization
noise is negligible.


Measurement noise estimator (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_Lambda``

.. math::

   \hat{\Lambda}_t
   = -\,\tfrac{1}{2}\bigl[
       \Delta\mathbf{x}_t \otimes \Delta\mathbf{x}_{t-1}
     + \Delta\mathbf{x}_{t-1} \otimes \Delta\mathbf{x}_t
   \bigr]

Estimates localization / measurement noise from anti-correlation
of successive displacements.


MSD diffusion estimator (underdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.underdamped` — ``_D_msd_uli``

.. math::

   \hat{\mathbf{D}}_{\text{MSD}}^{\text{ULI}}(t)
   = \frac{3}{4\,\mathrm{d}t^3}\,
     (\Delta\mathbf{x}_t - \Delta\mathbf{x}_{t-1})
     \otimes (\Delta\mathbf{x}_t - \Delta\mathbf{x}_{t-1})


Weak-noise diffusion estimator (underdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.underdamped` — ``_D_weaknoise_uli``

.. math::

   \hat{\mathbf{D}}_{\text{WN}}^{\text{ULI}}(t)
   = \frac{1}{2\,\mathrm{d}t^3}\,
     (2\,\Delta\mathbf{x}_t - \Delta\mathbf{x}_{t-1}
     - \Delta\mathbf{x}_{t+1})
     \otimes (\cdots)


Noisy diffusion estimator (underdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.underdamped` — ``_D_noisy_uli``

.. math::

   \hat{\mathbf{D}}_{\text{noisy}}^{\text{ULI}}(t)
   = \frac{3}{11\,\mathrm{d}t^3}\,\operatorname{sym}\!
     \bigl[-a + b + c - 3d + e + f\bigr]

where :math:`a = \Delta\mathbf{x}_t \otimes \Delta\mathbf{x}_t`,
:math:`b = \Delta\mathbf{x}_{t-1} \otimes \Delta\mathbf{x}_{t-1}`, etc.
Optimally handles both signal and localization noise.


Measurement noise estimator (underdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.underdamped` — ``_Lambda_uli``

.. math::

   \hat{\Lambda}^{\text{ULI}}
   = \frac{1}{44}\,\operatorname{sym}\!
     \bigl[10a + b + c + 8d - 10e - 10f\bigr]

Same displacement products :math:`(a\ldots f)` as the noisy
diffusion estimator.  Extracts localization noise for
underdamped systems.


Force inference
---------------

Linear force regression (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :meth:`SFI.inference.overdamped.OverdampedLangevinInference.infer_force_linear`

.. math::

   \hat{\mathbf{F}}(\mathbf{x})
   = \sum_{i=1}^{n} \hat{F}_i\, \mathbf{b}_i(\mathbf{x})
   \qquad\text{where}\qquad
   G\,\hat{F} = M

:math:`G_{ij} = \langle \mathbf{b}_i \cdot \bar{\mathbf{D}}^{-1}
\cdot \mathbf{b}_j \rangle` is the :math:`\bar{\mathbf{D}}`-weighted
Gram matrix, :math:`M_i = \langle \mathbf{v}_t \cdot
\bar{\mathbf{D}}^{-1} \cdot \mathbf{b}_i \rangle` are the force
moments, and :math:`n` is the library size (number of basis functions).


Overdamped force moments
~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_force_moments``

**Itô moments:**

.. math::

   M_i = \bigl\langle \mathbf{v}_t \cdot \bar{\mathbf{D}}^{-1}
   \cdot \mathbf{b}_i(\mathbf{x}_t) \bigr\rangle

**Stratonovich moments** (trapezoid + gradient correction):

.. math::

   M_i^{\text{S}} = \tfrac{1}{2}\bigl\langle \mathbf{v}_t \cdot
   \bar{\mathbf{D}}^{-1}
   \cdot \bigl[\mathbf{b}_i(\mathbf{x}_t)
   + \mathbf{b}_i(\mathbf{x}_{t+1})\bigr] \bigr\rangle
   \;-\; \bigl\langle \mathbf{D}_{\text{inst}} :
   (\bar{\mathbf{D}}^{-1} \cdot \nabla_{\mathbf{x}} \mathbf{b}_i)
   \bigr\rangle


Underdamped force moments (ULI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.underdamped` — ``_force_moments``

.. math::

   M_i = \bigl\langle \hat{\mathbf{a}}_t \cdot \bar{\mathbf{D}}^{-1}
   \cdot \mathbf{b}_i(\hat{\mathbf{x}}_t, \hat{\mathbf{v}}_t)
   \bigr\rangle
   \;+\; w \,\bigl\langle -\mathbf{D}_{\text{inst}} :
   (\bar{\mathbf{D}}^{-1}\cdot\partial_{\mathbf{v}} \mathbf{b}_i)
   \bigr\rangle

where :math:`w = (1+2\ell)/3`, with :math:`\ell=1` (symmetric),
:math:`\ell=0` (early), :math:`\ell=-\tfrac{1}{2}` (anticipated).


Itô quasi-likelihood loss (nonlinear force)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :mod:`SFI.inference.overdamped` — ``_build_force_loss_psf``

.. math::

   \mathcal{L}(\theta)
   = \tfrac{1}{4}\bigl\langle \mathbf{F}(\mathbf{x};\theta)^\top
     \bar{\mathbf{D}}^{-1}\,\mathbf{F}(\mathbf{x};\theta)
     \bigr\rangle
   - \tfrac{1}{2}\bigl\langle \mathbf{F}(\mathbf{x};\theta)^\top
     \bar{\mathbf{D}}^{-1}\,\mathbf{v} \bigr\rangle

Negative log-quasi-likelihood for parametric drift estimation.


Kinematic reconstructions (ULI)
-------------------------------

*Source:* :mod:`SFI.inference.underdamped` — ``_X_sym_uli``, ``_V_sym_uli``, ``_A_sym_uli``, etc.

Three reconstruction modes for the unobserved velocity:

**Symmetric:**

.. math::

   \hat{\mathbf{x}}(t)
   = \tfrac{1}{3}\bigl[\mathbf{x}_{t-1}+\mathbf{x}_t+\mathbf{x}_{t+1}\bigr],
   \quad
   \hat{\mathbf{v}}(t)
   = \frac{\Delta\mathbf{x}_t+\Delta\mathbf{x}_{t-1}}{2\,\mathrm{d}t},
   \quad
   \hat{\mathbf{a}}(t)
   = \frac{\Delta\mathbf{x}_t - \Delta\mathbf{x}_{t-1}}{\mathrm{d}t^2}

**Early:**

.. math::

   \hat{\mathbf{x}}(t) = \mathbf{x}_t,
   \quad
   \hat{\mathbf{v}}(t) = \frac{\Delta\mathbf{x}_{t-1}}{\mathrm{d}t}

**Anticipated:**

.. math::

   \hat{\mathbf{x}}(t)
   = \tfrac{1}{3}\bigl[\mathbf{x}_t + \mathbf{x}_{t+1}
     + \mathbf{x}_{t+2}\bigr],
   \quad
   \hat{\mathbf{a}}(t)
   = \frac{\Delta\mathbf{x}_{t+1} - \Delta\mathbf{x}_t}{\mathrm{d}t^2}


Error analysis
--------------

Force coefficient covariance & predicted error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :meth:`SFI.inference.base.BaseLangevinInference.compute_force_error`

.. math::

   \operatorname{Cov}(\hat{F}) = 2\,G^{-1},
   \qquad
   I_F = \tfrac{1}{2}\,\hat{F}^\top M,
   \qquad
   \text{NMSE}_{F,\text{pred}}
   = \frac{\operatorname{Tr}(G\cdot\operatorname{Cov}(\hat{F}))}{I_F}

Assumes the sampling error dominates; measurement noise and
discretization biases are not addressed.


Normalized MSE metrics (force & diffusion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :meth:`SFI.inference.base.BaseLangevinInference.compare_to_exact`

.. math::

   \text{NMSE}_F
   = \frac{\langle (\mathbf{F}_{\text{exact}} - \hat{\mathbf{F}})^\top
     \bar{\mathbf{D}}^{-1}
     (\mathbf{F}_{\text{exact}} - \hat{\mathbf{F}}) \rangle}
   {\langle \hat{\mathbf{F}}^\top\,\bar{\mathbf{D}}^{-1}\,
     \hat{\mathbf{F}} \rangle}

.. math::

   \text{NMSE}_D
   = \frac{\langle \operatorname{tr}(\bar{\mathbf{D}}^{-1}\,\mathbf{E}\,
     \bar{\mathbf{D}}^{-1}\,\mathbf{E}) \rangle}
   {\langle \operatorname{tr}(\bar{\mathbf{D}}^{-1}\,\hat{\mathbf{D}}\,
     \bar{\mathbf{D}}^{-1}\,\hat{\mathbf{D}}) \rangle}

where :math:`\mathbf{E} = \mathbf{D}_{\text{exact}} - \hat{\mathbf{D}}`.


Model selection
---------------

*Source:* :mod:`SFI.inference.sparsity` — :class:`~SFI.inference.SparseScorer`, :meth:`~SFI.inference.SparsityResult.select_by_ic`

.. math::

   \text{AIC}(k) &= \mathcal{I}(k) - k \\
   \text{BIC}(k) &= \mathcal{I}(k) - \tfrac{1}{2}\,k\,\ln n \\
   \text{PASTIS}(k) &= \mathcal{I}(k) - k\,\ln(n / n_0)

where :math:`\mathcal{I}(k) = \tfrac{1}{2}\,\hat{F}_B^\top M_B` is the
log-likelihood gain with :math:`k` basis terms selected from a library
of size :math:`n`, and :math:`n_0` is the PASTIS prior scale.

The PASTIS significance :math:`p_{\text{PASTIS}}` corresponds to the
threshold above which a basis term is considered supported by the data.


Observables
-----------

Information functional & entropy production (overdamped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :meth:`SFI.langevin.overdamped.OverdampedProcess.simulate`

.. math::

   I \approx \tfrac{1}{4}\sum_t
     \Delta\mathbf{x}_t^\top\,\mathbf{D}^{-1}(\mathbf{x}_t)\,
     \mathbf{F}(\mathbf{x}_t)

.. math::

   S \approx \sum_t
     \Delta\mathbf{x}_t^\top\,\mathbf{D}^{-1}(\mathbf{x}_{\text{mid}})\,
     \tfrac{1}{2}\bigl[\mathbf{F}(\mathbf{x}_t)
     +\mathbf{F}(\mathbf{x}_{t+1})\bigr]

:math:`I` estimates the information content; :math:`S` the
entropy production (time-reversal asymmetry).


Basis functions
---------------

Multivariate polynomial basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :func:`SFI.bases.monomials_up_to`

.. math::

   b_\alpha(\mathbf{x}) = \prod_{k=1}^{d} x_k^{\alpha_k},
   \qquad |\alpha| \le \texttt{order}

Full polynomial dictionary up to a given total degree, optionally
including velocity monomials and lifted to vector or matrix rank.


Radial pair interaction basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :func:`SFI.bases.pairs.radial_pair_basis`

.. math::

   \mathbf{b}_\alpha(\mathbf{r}_{ij})
   = \phi_\alpha(r_{ij})\;\hat{\mathbf{r}}_{ij}

Scalar radial kernel :math:`\phi_\alpha` times the unit displacement
vector.  Available kernel families: exponential-polynomial,
Gaussian, power-law, and compactly supported.


Discrete Laplacian on regular grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Source:* :class:`SFI.bases.spde.Laplacian`

.. math::

   \nabla^2 u \approx \sum_{\alpha=1}^{n_{\dim}}
   \frac{u_{+\alpha} + u_{-\alpha} - 2\,u_0}{\Delta x_\alpha^2}

Finite-difference Laplacian using a cross stencil on a Cartesian grid.
Available as a composable operator via the :class:`~SFI.bases.spde.Laplacian`
class.  See :doc:`/spde/user_guide` for a full introduction to SPDE operators.


.. rubric:: Additional formulas from API documentation

The following entries are collected automatically from ``.. physics::``
directives in the API documentation:

.. physicslist::
