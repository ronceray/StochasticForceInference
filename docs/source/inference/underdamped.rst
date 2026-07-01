.. _underdamped-guide:

Underdamped systems
===================

When inertia matters — vibrated grains, swimming or flying organisms,
underdamped colloids, oscillators — positions and velocities evolve
jointly:

.. math::

   \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v}, \qquad
   \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t}
   = \mathbf{F}(\mathbf{x},\mathbf{v})
   + \sqrt{2\,\mathbf{D}(\mathbf{x},\mathbf{v})}\;\mathrm{d}W_t .

:class:`~SFI.inference.UnderdampedLangevinInference` (ULI) infers
:math:`\mathbf{F}(\mathbf{x},\mathbf{v})` and
:math:`\mathbf{D}(\mathbf{x},\mathbf{v})` from **positions only**:
velocities are reconstructed internally from finite differences
("secant velocities"), and the estimators are built to correct for the
biases this reconstruction introduces.  You never supply velocities.


When to use the underdamped engine
----------------------------------

Use ULI rather than the overdamped engine when:

- trajectories oscillate or overshoot (ballistic short-time behaviour
  rather than diffusive),
- the velocity autocorrelation time is resolved by your sampling
  (several frames per relaxation time),
- the force you care about depends on velocity (friction, drag,
  alignment, motility).

If relaxation to the steady drift happens *within* one frame, the
system is effectively overdamped at your resolution — use
:class:`~SFI.inference.OverdampedLangevinInference` instead.

Unsure which regime your data is in?  The **experimental**
:ref:`overdamped/underdamped classifier <dynamics-order-classifier>`
reads raw positions and returns an ``"OD"`` / ``"UD"`` /
``"inconclusive"`` verdict, robust to localization noise and coarse
sampling.


Workflow
--------

The API mirrors the overdamped engine; only the physics of the moment
estimators differs.  The basis may now depend on velocity:

.. code-block:: python

   from SFI import UnderdampedLangevinInference
   from SFI.bases import monomials_up_to

   inf = UnderdampedLangevinInference(coll)     # positions-only data
   inf.compute_diffusion_constant()             # constant D + measurement-noise (Λ) estimate

   # F(x, v): build a basis over the (x, v) phase space
   B = monomials_up_to(order=3, dim=1, rank='vector', include_v=True)
   inf.infer_force_linear(B)                     # preset="auto" (the default)

   inf.compute_force_error()
   inf.sparsify_force(criterion="PASTIS")
   inf.print_report()

For compositional bases, the velocity primitives are first-class:
``v_components(dim)`` and ``frame(dim, velocity=True)`` — see
:doc:`/bases/user_guide`.

Presets (linear estimators)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The linear force estimator
(:meth:`~SFI.inference.UnderdampedLangevinInference.infer_force_linear`)
has three knobs — the kinematics convention (``M_mode``), the Gram
construction (``G_mode``), and the instantaneous diffusion estimator used
in the Itô correction (``diffusion_method``).  In practice only the
diffusion estimator matters, and it is governed by **measurement noise**.
A single ``preset`` sets all three; the default ``"auto"`` chooses for
you:

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - ``preset``
     - Resolves to (``M_mode`` / ``G_mode`` / ``diffusion_method``)
     - Use
   * - ``"auto"`` *(default)*
     - ``robust`` if measurement noise is detected, else ``clean``
     - the safe default
   * - ``"robust"``
     - ``symmetric`` / ``trapeze`` / ``noisy``
     - measurement noise present
   * - ``"clean"``
     - ``symmetric`` / ``trapeze`` / ``WeakNoise``
     - verified noise-free data
   * - ``"legacy-clean-v1.0"``
     - ``early`` / ``rectangle`` / ``MSD``
     - superseded (SFI v1.0)
   * - ``"legacy-noisy-v1.0"``
     - ``symmetric`` / ``rectangle`` / ``noisy``
     - superseded (SFI v1.0)

``"robust"`` is broadly applicable: accuracy is flat across measurement
noise within its usable band and it never fails catastrophically.
``"clean"`` is a sharper variant for **verified** noise-free data — it
helps mainly when sampling is coarse and noise is negligible, and
degrades quickly once any localization noise is present.  The two
``legacy-*`` presets reproduce the published SFI v1.0 conventions
(rectangle Gram); they are kept **for reproducibility only** and are
superseded by the trapeze-Gram presets above.

**How** ``"auto"`` **decides.**  It switches on the sign of the
localization-noise estimate ``Lambda_trace`` that
:meth:`~SFI.inference.UnderdampedLangevinInference.compute_diffusion_constant`
already produces (so call it first): ``> 0`` means noise dominates the
velocity-increment autocorrelation → ``"robust"``; ``< 0`` means the
force persistence of the dynamics dominates (effectively clean) →
``"clean"``.  Because that zero-crossing tracks the ``noisy`` ↔
``WeakNoise`` transition, ``"auto"`` selects the right estimator across
near-ideal, coarsely-sampled, and high-noise data without the sharp
failures ``"clean"`` can suffer once noise appears.

Power users can override any axis explicitly — e.g.
``infer_force_linear(B, diffusion_method="MSD")`` or
``infer_force_linear(B, M_mode="anticipated", G_mode="rectangle")`` — and
the explicit value wins while the preset fills the rest.  The available
modes are ``M_mode`` ∈ {``symmetric`` (default), ``early``,
``anticipated``}; ``G_mode`` ∈ {``rectangle``, ``trapeze`` (default),
``shift``, ``doubleshift``}; ``diffusion_method`` ∈ {``noisy``,
``WeakNoise``, ``MSD``}.

State-dependent diffusion :math:`\mathbf{D}(\mathbf{x},\mathbf{v})` is
inferred with
:meth:`~SFI.inference.UnderdampedLangevinInference.infer_diffusion_linear`
(rank-2 basis), or with the parametric
:meth:`~SFI.inference.UnderdampedLangevinInference.infer_diffusion`.

.. note::

   Each inference object supports **one** fit: re-running
   ``infer_force_linear`` on the same instance raises an error.
   Create a fresh :class:`~SFI.inference.UnderdampedLangevinInference` per model.


Velocity-reconstruction pitfalls
--------------------------------

Reconstructing :math:`\mathbf{v}` from positions is what makes
underdamped inference harder than overdamped, and it concentrates the
failure modes:

**Measurement noise is amplified twice.**  A localization error of
covariance :math:`\Lambda` adds variance
:math:`\sim 2\Lambda/\Delta t^2` to secant velocities and even
more to finite-difference accelerations.  At video rates this easily
dominates: noise that is invisible in the trajectory plot can wreck a
naive velocity histogram.  The ULI moment estimators are constructed
to cancel the leading noise-induced biases, and the diagnostics
(below) tell you when that correction is no longer enough — at which
point switch to the parametric estimator, which models
:math:`\Lambda` explicitly (see
:doc:`/inference/noise_and_sampling`).

**Coarse sampling aliases oscillations.**  You need several frames per
oscillation/relaxation period for the secant velocity to track the
true one; below that the dynamics alias and no estimator can recover
the force.  If :math:`\Delta t` is marginal (but above the aliasing
threshold), the parametric estimator's RK4 flow step extends the
usable range compared to the linear estimators.

**Masked frames cost three-point stencils.**  Velocity and
acceleration estimates need two or three consecutive valid frames; a
masked frame invalidates the stencils that straddle it.  The
trajectory layer handles this bookkeeping automatically — but heavily
gapped tracks lose proportionally more usable samples in underdamped
than in overdamped inference.


Parametric underdamped inference
--------------------------------

The parametric estimators work in the underdamped regime with the same
call signature, including multi-particle interacting systems (see
:ref:`choosing-an-estimator`):

.. code-block:: python

   inf = UnderdampedLangevinInference(coll)
   inf.infer_force(B_xv)          # Basis or PSF over (x, v); profiles (D, Λ)
   inf.infer_diffusion(B_D)       # optional: D(x, v)

Reach for it when measurement noise is significant, sampling is
coarse, or the model is nonlinear in its parameters — the same regime
table as for overdamped data applies.


Underdamped diagnostics
-----------------------

:func:`SFI.diagnostics.assess` works unchanged on a ULI fit.  Under
the hood the residual is the symmetric finite-difference acceleration
minus the fitted force, whitened by the proper :math:`\tfrac23\,(2\bar
D)/\Delta t` variance, with every second index kept so the pooled
innovations are serially independent — see :doc:`/diagnostics` for the
construction and the interpretation guide.  The flags mean the same
thing as in the overdamped case: autocorrelation → missing dynamics;
variance mismatch → wrong diffusion or unmodelled noise; MSE
inconsistency → a bias floor (usually measurement noise).


Worked examples
---------------

- :doc:`/gallery/van_der_pol_demo` — 1D Van der Pol oscillator from
  position-only data; nonlinear velocity-dependent force.
- :doc:`/gallery/velocity_dependent_noise_demo` — velocity-dependent
  noise amplitude :math:`D(v)` recovered jointly with the force.
- :doc:`/gallery/advanced/flocking_3d_demo` — 3D underdamped flocking:
  multi-particle parametric inference with trap, cohesion, and
  velocity-alignment forces.


.. seealso::

   - :ref:`dynamics-order-classifier` — the experimental
     overdamped/underdamped detector.
   - :ref:`choosing-an-estimator` — linear vs. parametric regime table.
   - :doc:`/inference/noise_and_sampling` — measurement noise and
     coarse sampling, where the parametric estimators take over.
   - :doc:`/physics_reference` — SDE conventions and estimator
     definitions.
