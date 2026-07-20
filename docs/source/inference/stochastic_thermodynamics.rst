.. _stochastic-thermodynamics:

Stochastic thermodynamics with SFI
==================================

SFI infers the two fields that define an overdamped stochastic system —
the drift :math:`\mathbf{F}(\mathbf{x})` and the diffusion
:math:`\mathbf{D}(\mathbf{x})`.  For a physicist these are more than a
model of the dynamics: they carry the system's **thermodynamics**.  From
a fitted model one can reconstruct an energy landscape, exhibit the
probability currents that break detailed balance, and *measure the rate
at which the system dissipates free energy* — the entropy production —
directly from trajectory data, with error bars and no ground truth.

This page collects the concepts behind these measurements and routes to
the executable demos.  The worked example is
:doc:`/gallery/entropy_production_demo` — a driven optical trap whose
dissipation is invisible to every static observable.

Throughout, units are natural: :math:`k_B T` is absorbed so that
energies are in units of :math:`k_B T`, entropies in units of
:math:`k_B` (nats), and the mobility is unity so that
:math:`D = k_B T`.


Equilibrium: forces from energies
---------------------------------

A system is in **equilibrium** when its drift derives from a potential,
:math:`\mathbf{F} = -\nabla U` (constant :math:`D`; state-dependent
diffusion modifies this relation — see
:ref:`multiplicative-noise-thermo`).  Then the dynamics
obeys *detailed balance*: every trajectory and its time-reverse are
equally probable in the steady state, probability currents vanish, and
the density is the Boltzmann weight

.. math::

   \rho(\mathbf{x}) \;\propto\; e^{-U(\mathbf{x})/D} .

Three SFI workflows are natural here:

- **Reconstruct the potential from a generic fit.**  Fit the force with
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`,
  then integrate (or, for a linear force, symmetrize) the result.  The
  Boltzmann weight of the reconstructed potential can be compared to the
  observed density — a strong self-consistency check.
- **Fit a gradient basis — still linear.**  Pick scalar functions
  :math:`u_i` and use their (negative) gradients as the vector basis,
  :math:`\mathbf{b}_i = -\nabla u_i` — one
  :meth:`~SFI.statefunc.StateExpr.d_x` call on a scalar
  :class:`~SFI.statefunc.Basis` builds it.  The fit stays linear in the
  coefficients, so
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`
  applies unchanged, the fitted force is *exactly* conservative, and the
  potential comes out linearly, :math:`U = \sum_i c_i u_i`, with error
  bars inherited from the coefficients.  (Drop the constant function:
  its gradient is the zero feature.)
- **Parametrize by the energy.**  With the parametric estimator
  (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`) and
  JAX automatic differentiation, the model can be written as
  :math:`\mathbf{F}_\theta = -\nabla U_\theta` for *any* differentiable
  energy family — the fit returns the energy parameters directly, with
  error bars.  See the energy-recovery section of
  :doc:`/gallery/entropy_production_demo` for a three-parameter example,
  and :doc:`/gallery/advanced/nn_force_demo` for a neural-network
  landscape (Müller–Brown) fitted through exactly this
  ``-U.d_x()`` pattern.

.. tip::

   Fitting :math:`-\nabla U_\theta` instead of a generic vector field
   *builds the physics in*: the fitted model is exactly conservative,
   with typically far fewer parameters — and what the data cannot
   express as a gradient shows up as residual, not as spurious force
   terms.


Breaking detailed balance: currents
-----------------------------------

Out of equilibrium, the steady state carries a non-zero **probability
current** :math:`\mathbf{j} = \rho\,\mathbf{v}`, with mean local
(phase-space) velocity

.. math::

   \mathbf{v}(\mathbf{x})
   \;=\; \mathbf{F}(\mathbf{x}) - D\,\nabla \ln \rho(\mathbf{x})
   \qquad\text{(constant } D\text{)}.

The velocity field is the *order parameter of irreversibility*: it
vanishes identically in equilibrium and traces the circulation loops of
a non-equilibrium steady state.  Crucially, the density itself can be
blind to it — in the driven optical trap of
:doc:`/gallery/entropy_production_demo`, :math:`\rho` remains exactly
Boltzmann while :math:`\mathbf{v}` circulates.  For a strongly driven
example with :math:`O(1)` currents, see :doc:`/gallery/limitcycle_demo`.

**The observed-phase-space caveat.**  SFI sees currents only in the
coordinates you record.  Hidden degrees of freedom — an unobserved
coordinate, a chemical state, fast solvent modes — can hide part (or
all) of the circulation: projecting a non-equilibrium process onto fewer
coordinates can only *decrease* the measured irreversibility.  Every
current-based quantity below is therefore a **lower bound** on the
system's dissipation, attached to the phase space you observe.  This is
a feature as much as a limitation: a *positive* measurement is decisive
evidence of broken detailed balance, while a null one only bounds it.


Measuring entropy production
----------------------------

The dissipation of an overdamped steady state is the **entropy
production rate**

.. math::

   \sigma \;=\; \bigl\langle \mathbf{v} \cdot D^{-1} \mathbf{v}
   \bigr\rangle \;\ge\; 0 ,

in units of :math:`k_B` per unit time (heat dissipated into the bath
divided by :math:`k_B T`).  SFI estimates it with
:meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
(after any linear force fit), by projecting the phase-space velocity
onto the force basis — the estimator introduced in Frishman & Ronceray
(2020):

.. math::

   \Delta\hat S \;=\; 2\, M_v^\top G^{-1} M_v ,
   \qquad
   M_{v,a} = \sum_t \mathrm{d}\mathbf{X}_t \circ A^{-1} \mathbf{b}_a ,
   \qquad A = 2\bar D ,

where :math:`\circ` denotes the Stratonovich (mid-point) discretization.
The estimate comes with two qualifiers, both reported:

- a **fluctuation bias** :math:`\dot S_{\rm bias} = 2N_b/\tau_N`: with
  :math:`N_b` basis functions and total observation time
  :math:`\tau_N`, even a perfectly reversible trajectory yields this
  much apparent dissipation from noise alone (reported, and also
  subtracted in the ``Sdot_debiased`` / ``DeltaS_debiased`` outputs);
- a **statistical error**
  :math:`\delta\Delta S = \sqrt{2\,\Delta\hat S + (2N_b)^2}`.

The fluctuation bias is nothing exotic — it is the **AIC complexity
correction** in thermodynamic clothing.  :math:`\Delta\hat S` is four
times the plug-in information carried by the projected velocity field
(see *two integrals* below), and a maximized log-likelihood — or
information — exceeds its expectation by :math:`\tfrac12` per fitted
degree of freedom: the same counting that penalizes model complexity in
:meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force`
(``criterion="AIC"``) removes :math:`2N_b` from :math:`\Delta\hat S`.
Use the raw value as the conservative plug-in, the debiased one for
point estimates — near equilibrium it fluctuates around zero and may
come out negative, which is honesty, not pathology.

Together, bias and error set a *thermodynamic limit of detection*:
irreversibility becomes measurable only once the trajectory has
dissipated a few :math:`k_B` beyond the bias.  Near equilibrium this is
expensive — currents are linear in the driving, so :math:`\sigma` is
*quadratic*: a twice-fainter drive takes four times the data.  The
detection-limit sweep in :doc:`/gallery/entropy_production_demo` shows
both effects quantitatively.

**Two integrals, two meanings.**  The same trajectory functional read in
the two discretizations measures two different things (see the
observables of :class:`~SFI.langevin.OverdampedProcess` and
:doc:`/physics_reference`):

.. math::

   I \;=\; \tfrac14 \textstyle\sum_t
   \mathrm{d}\mathbf{X}_t \cdot D^{-1} \mathbf{F}(\mathbf{x}_t)
   \qquad\text{(Itô)}

is the **information** the trajectory carries about the force field —
the quantity that controls SFI's error bars
(``force_information``) — while the Stratonovich reading

.. math::

   \Delta S \;=\; \textstyle\sum_t
   \mathrm{d}\mathbf{X}_t \circ D^{-1} \mathbf{F}(\mathbf{x})

is the log-ratio of forward to time-reversed path probabilities: the
**entropy production**.  The two are tightly related — the entropy
production rate is four times the information rate carried by the
velocity field, :math:`\sigma = 4\,\dot I_v` — so *measuring dissipation
is exactly accumulating information about the irreversible part of the
dynamics*.

**Practical caveats.**

- The estimator assumes constant (or averaged) diffusion; a warning is
  emitted when a state-dependent :math:`D(\mathbf{x})` field was fitted
  (see :ref:`multiplicative-noise-thermo`).
- **Measurement (localization) noise is benign at leading order** — a
  parity gift.  The Stratonovich moments are *odd* under time reversal,
  while independent localization noise is *even*: the naive
  :math:`O(\Lambda/\Delta t)` noise–increment correlation cancels
  exactly, telescoping to a boundary term.  (This is special to the
  entropy estimator: the Itô force moments enjoy no such cancellation.)
  What survives is second order — an :math:`O(\Lambda)` smoothing of
  the basis functions and a noise-inflated variance that raises the
  effective fluctuation bias — and is *not* included in the error bar.
  The exception is noise **correlated with the motion**: motion blur
  from finite exposure breaks the parity argument and does bias the
  estimate.  Denoise or subsample for precision, not out of fear of a
  diverging bias (see :doc:`/inference/noise_and_sampling`).


Underdamped systems: dissipation from positions only (experimental)
--------------------------------------------------------------------

.. warning::

   **Experimental.**  The underdamped estimator is new and its error
   theory is provisional: the bias and error formulas mirror the
   overdamped derivation and are pinned empirically (a
   :math:`\gamma\Delta t` calibration sweep), not derived from first
   principles.  Expect the outputs and error bars to be refined.

With inertia the state is :math:`(\mathbf{x}, \mathbf{v})` and time
reversal flips the velocity: :math:`\mathbf{x}` is *even*,
:math:`\mathbf{v}` is *odd*.  The fitted acceleration field then splits
by parity,

.. math::

   F^{\pm}(\mathbf{x},\mathbf{v})
   = \tfrac12\bigl[F(\mathbf{x},\mathbf{v})
     \pm F(\mathbf{x},-\mathbf{v})\bigr] ,

into a *reversible* part :math:`F^{+}` (conservative and confining
forces) and an *irreversible* part :math:`F^{-}` (friction and
velocity-odd driving) — the decomposition that organises stochastic
thermodynamics with odd variables (Spinney & Ford 2012; Lee, Kwon &
Park 2013).  :meth:`~SFI.inference.UnderdampedLangevinInference.time_reversal_split`
returns the two parts of a fitted model, and
:meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`
evaluates the Stratonovich log path-probability ratio

.. math::

   \Delta \hat S = \sum_t \bigl(\mathrm{d}\hat v_t \circ
   - \hat F^{+}\mathrm{d}t\bigr)^\top \bar D^{-1} \hat F^{-} ,

with the velocities :math:`\hat v` **reconstructed from positions**
(the ULI kinematics) — no velocity measurements are needed.  In a
steady state its mean is the entropy production rate whenever
:math:`\nabla_v \cdot F^{+} = 0`; the heat dissipated into the bath
follows as :math:`\dot Q = T\,\sigma` with :math:`T = \bar D/\hat\gamma`
(Sekimoto 1998).  For any *linear* underdamped model the exact rate is
available in closed form from the phase-space Ornstein–Uhlenbeck
formula with the parity-aware irreversible drift (see Qian 2001;
Godrèche & Luck 2019) — the validation route used by SFI's test suite.

Underdamped-specific caveats:

- **Resolution prerequisite.**  Velocity reconstruction requires
  :math:`\Delta t \lesssim \tau_v` (the velocity correlation time); a
  warning is emitted when :math:`\hat\gamma\,\Delta t > 0.5`.  Beyond
  that the measurement degrades toward a coarse-grained lower bound.
- **Fluctuation scale.**  Friction makes the odd sector large even at
  equilibrium: the log ratio's contributions cancel in the mean but not
  in variance, so error bars are set by the odd quadratic
  :math:`\langle F^-\!\cdot \bar D^{-1} F^-\rangle` (reported, with an
  empirical block-variance estimate).
- **Same-sample plug-in bias.**  Evaluating the functional with
  coefficients fitted on the same trajectory produces a positive
  :math:`O(1/\tau_N)` bias; it is removed by **cross-fitting** (fit on
  one half, evaluate on the other — the ``coefficients`` argument), and
  it decays with trajectory length.
- **Overdamped limits are singular.**  With temperature gradients or
  state-dependent friction, the small-inertia limit carries an
  *entropic anomaly* absent from the overdamped description (Celani et
  al. 2012): analysing inertial data with an overdamped model can be
  systematically wrong — infer at the underdamped level when inertia is
  resolvable (the :func:`~SFI.diagnostics.classify_dynamics` verdict
  helps decide).


.. _multiplicative-noise-thermo:

Multiplicative noise: equilibrium, spurious drifts, blowtorches
---------------------------------------------------------------

When the noise amplitude depends on the state — multiplicative noise,
:math:`D = D(\mathbf{x})` — equilibrium intuition breaks in a subtler
way.  The Itô drift that SFI infers is then *not* the mechanical force:
for a system with mobility :math:`\mu(\mathbf{x})` and physical force
:math:`\mathbf{f}`,

.. math::

   \mathbf{F}_{\rm It\hat o}(\mathbf{x})
   \;=\; \mu(\mathbf{x})\,\mathbf{f}(\mathbf{x})
   \;+\; \nabla\!\cdot\! \mathbf{D}(\mathbf{x}) ,

and the divergence term — the **spurious drift** — pushes probability
around even with no force at all.  Interpreting an inferred drift
thermodynamically therefore requires the diffusion *field*, not just
its average — infer it with
:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`
or :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`
(see :doc:`/gallery/anisotropic_diffusion_demo` for tensor-valued
fields).

**Equilibrium with multiplicative noise.**  State-dependent noise does
*not* by itself break detailed balance.  When the diffusion field
reflects a state-dependent *mobility* at uniform temperature —
fluctuation–dissipation, :math:`\mathbf{D}(\mathbf{x}) =
\mu(\mathbf{x})\,k_BT` — a conservative force
:math:`\mathbf{f} = -\nabla U` gives (in our :math:`k_BT = 1` units)

.. math::

   \mathbf{F}_{\rm It\hat o}(\mathbf{x})
   \;=\; -\,\mathbf{D}(\mathbf{x})\,\nabla U(\mathbf{x})
   \;+\; \nabla\!\cdot\!\mathbf{D}(\mathbf{x}) ,

and the stationary density remains exactly Boltzmann,
:math:`\rho \propto e^{-U}` — the spurious drift is precisely what
keeps it so (plug :math:`\rho = e^{-U}` into the zero-current condition
:math:`\mathbf{F}\rho = \nabla\!\cdot\!(\mathbf{D}\rho)` to check).
Read as an inference statement, this is a sharp **equilibrium test**:
a fitted pair :math:`(\hat{\mathbf{F}}, \hat{\mathbf{D}})` is
compatible with equilibrium if and only if

.. math::

   \boldsymbol{\Phi}(\mathbf{x}) \;=\;
   \hat{\mathbf{D}}^{-1}\bigl(\hat{\mathbf{F}}
   - \nabla\!\cdot\!\hat{\mathbf{D}}\bigr)

is a gradient field, :math:`\boldsymbol{\Phi} = -\nabla U`; any
rotational part of :math:`\boldsymbol{\Phi}` is a genuine
detailed-balance violation in the observed coordinates.

**Joint** :math:`(U, D)` **inference in practice.**  Recovering the
energy landscape under multiplicative noise therefore takes *both*
fields, and SFI composes them rather than adding a dedicated mode:

1. fit the diffusion field
   (:meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`
   or the noise-robust
   :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`);
2. fit the drift
   (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`
   or :meth:`~SFI.inference.OverdampedLangevinInference.infer_force`);
3. form :math:`\boldsymbol{\Phi}` from the two fits — the divergence
   :math:`\nabla\!\cdot\!\hat{\mathbf{D}}` is one
   :meth:`~SFI.statefunc.StateExpr.d_x` call away — and test it for
   gradientness / line-integrate it for :math:`U`.

Alternatively, *build the equilibrium ansatz in*: with the fitted
:math:`\hat{\mathbf{D}}(\mathbf{x})` frozen, write the drift family

.. math::

   \mathbf{F}_\theta \;=\;
   -\,\hat{\mathbf{D}}(\mathbf{x})\,\nabla U_\theta(\mathbf{x})
   \;+\; \nabla\!\cdot\!\hat{\mathbf{D}}(\mathbf{x})

(autodiff supplies both derivative terms) and fit :math:`U_\theta`
directly with
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force` — the
state-dependent-\ :math:`D` generalization of the gradient-basis and
autodiff-energy workflows of the equilibrium section.  Misfit of this
constrained family relative to a free drift fit is then itself
evidence of broken detailed balance.

**Blowtorches.**  If instead the noise field reflects a genuine
*temperature* gradient, no potential :math:`U` exists at all: the
steady state is not the naive Boltzmann weight :math:`e^{-U/\bar D}`,
and occupancy is re-shaped by the temperature landscape — the
*Landauer blowtorch* effect, with heat flowing even in a static
system.  See :doc:`/gallery/multiplicative_diffusion_demo` for the
blowtorch worked example.  Entropy production with a state-dependent
:math:`\mathbf{D}` is where the current estimator stops: it contracts
with the constant :math:`\bar D` and warns (the natural extension — a
trajectory integral against :math:`\mathbf{D}(\mathbf{x})^{-1}` — is
noted in
:meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`).


Which tool for which question
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Question
     - Tool / demo
   * - Is this system out of equilibrium?
     - :meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`
       — positive beyond bias and error bar is decisive;
       :doc:`/gallery/entropy_production_demo`
   * - What drives it — which terms break detailed balance?
     - :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force`
       (PASTIS) + symmetric/antisymmetric decomposition of the fit
   * - What is the energy landscape?
     - Parametric energy fit via autodiff
       (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`);
       :doc:`/gallery/advanced/nn_force_demo`
   * - Where does probability circulate?
     - :func:`~SFI.utils.plotting.stream_field` on the non-gradient part
       of the fit; :doc:`/gallery/limitcycle_demo`
   * - Is the noise state-dependent, and what does it do to occupancy?
     - :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`;
       :doc:`/gallery/multiplicative_diffusion_demo`
   * - Equilibrium despite multiplicative noise?
     - Gradientness of :math:`\hat{\mathbf{D}}^{-1}(\hat{\mathbf{F}} -
       \nabla\!\cdot\!\hat{\mathbf{D}})` — see
       :ref:`multiplicative-noise-thermo`
   * - Driven, or inertial?
     - :func:`~SFI.diagnostics.classify_dynamics`;
       :doc:`/gallery/dynamics_order_demo`


References
----------

Method:

   Frishman, A. & Ronceray, P., *Learning force fields from stochastic
   trajectories*, **Physical Review X** 10, 021009 (2020).
   `DOI: 10.1103/PhysRevX.10.021009
   <https://doi.org/10.1103/PhysRevX.10.021009>`_ — introduces both the
   inference framework and the entropy-production estimator used here.

   Brückner, D. B., Ronceray, P. & Broedersz, C. P., *Inferring the
   dynamics of underdamped stochastic systems*, **Physical Review
   Letters** 125, 058103 (2020).
   `DOI: 10.1103/PhysRevLett.125.058103
   <https://doi.org/10.1103/PhysRevLett.125.058103>`_ — the
   velocity-reconstruction engine behind the underdamped estimator.

Stochastic thermodynamics:

   Seifert, U., *Stochastic thermodynamics, fluctuation theorems and
   molecular machines*, **Reports on Progress in Physics** 75, 126001
   (2012).
   `DOI: 10.1088/0034-4885/75/12/126001
   <https://doi.org/10.1088/0034-4885/75/12/126001>`_

   Sekimoto, K., *Langevin equation and thermodynamics*, **Progress of
   Theoretical Physics Supplement** 130, 17 (1998).
   `DOI: 10.1143/PTPS.130.17 <https://doi.org/10.1143/PTPS.130.17>`_

Odd-parity (underdamped) entropy production:

   Spinney, R. E. & Ford, I. J., *Nonequilibrium thermodynamics of
   stochastic systems with odd and even variables*, **Physical Review
   Letters** 108, 170603 (2012).
   `DOI: 10.1103/PhysRevLett.108.170603
   <https://doi.org/10.1103/PhysRevLett.108.170603>`_; and *Entropy
   production in full phase space for continuous stochastic dynamics*,
   **Physical Review E** 85, 051113 (2012).
   `DOI: 10.1103/PhysRevE.85.051113
   <https://doi.org/10.1103/PhysRevE.85.051113>`_

   Lee, H. K., Kwon, C. & Park, H., *Fluctuation theorems and entropy
   production with odd-parity variables*, **Physical Review Letters**
   110, 050602 (2013).
   `DOI: 10.1103/PhysRevLett.110.050602
   <https://doi.org/10.1103/PhysRevLett.110.050602>`_

   Celani, A., Bo, S., Eichhorn, R. & Aurell, E., *Anomalous
   thermodynamics at the microscale*, **Physical Review Letters** 109,
   260603 (2012).
   `DOI: 10.1103/PhysRevLett.109.260603
   <https://doi.org/10.1103/PhysRevLett.109.260603>`_

Exact linear (phase-space Ornstein–Uhlenbeck) benchmarks:

   Qian, H., *Mathematical formalism for isothermal linear
   irreversibility*, **Proceedings of the Royal Society A** 457, 2645
   (2001).
   `DOI: 10.1098/rspa.2001.0811
   <https://doi.org/10.1098/rspa.2001.0811>`_

   Godrèche, C. & Luck, J.-M., *Characterising the nonequilibrium
   stationary states of Ornstein–Uhlenbeck processes*, **Journal of
   Physics A** 52, 035002 (2019).
   `DOI: 10.1088/1751-8121/aaf190
   <https://doi.org/10.1088/1751-8121/aaf190>`_

Broken detailed balance in living and driven systems:

   Battle, C. et al., *Broken detailed balance at mesoscopic scales in
   active biological systems*, **Science** 352, 604 (2016).
   `DOI: 10.1126/science.aac8167
   <https://doi.org/10.1126/science.aac8167>`_

   Gnesotto, F. S., Mura, F., Gladrow, J. & Broedersz, C. P., *Broken
   detailed balance and non-equilibrium dynamics in living systems: a
   review*, **Reports on Progress in Physics** 81, 066601 (2018).
   `DOI: 10.1088/1361-6633/aab3ed
   <https://doi.org/10.1088/1361-6633/aab3ed>`_

Non-conservative forces in optical traps:

   Roichman, Y., Sun, B., Stolarski, A. & Grier, D. G., *Influence of
   nonconservative optical forces on the dynamics of optically trapped
   colloidal spheres: the fountain of probability*, **Physical Review
   Letters** 101, 128301 (2008).
   `DOI: 10.1103/PhysRevLett.101.128301
   <https://doi.org/10.1103/PhysRevLett.101.128301>`_

   Wu, P., Huang, R., Tischer, C., Jonas, A. & Florin, E.-L., *Direct
   measurement of the nonconservative force field generated by optical
   tweezers*, **Physical Review Letters** 103, 108101 (2009).
   `DOI: 10.1103/PhysRevLett.103.108101
   <https://doi.org/10.1103/PhysRevLett.103.108101>`_
