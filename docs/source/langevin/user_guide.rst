.. _langevin-user-guide:

Simulation
==========

The :mod:`SFI.langevin` subpackage provides Langevin simulators that
use the same state-function objects (:class:`~SFI.statefunc.PSF` / :class:`~SFI.statefunc.SF`) as the inference
engines.  This closes the **inference → simulation → validation** loop:
you infer a model from data, simulate from it, and compare the synthetic
trajectory back to the original.

Two simulators are available:

- :class:`~SFI.langevin.OverdampedProcess` — stochastic Heun (default,
  Euler–Maruyama optional) for
  :math:`\mathrm{d}\mathbf{x} = \mathbf{F}(\mathbf{x})\,\mathrm{d}t + \sqrt{2\mathbf{D}(\mathbf{x})}\,\mathrm{d}W_t`
- :class:`~SFI.langevin.UnderdampedProcess` — velocity-Verlet-like for
  :math:`\mathrm{d}\mathbf{x} = \mathbf{v}\,\mathrm{d}t,\;\mathrm{d}\mathbf{v} = \mathbf{F}(\mathbf{x},\mathbf{v})\,\mathrm{d}t + \sqrt{2\mathbf{D}}\,\mathrm{d}W_t`


Quick example
-------------

.. code-block:: python

   import jax.numpy as jnp
   from jax import random
   from SFI.langevin import OverdampedProcess
   from SFI import make_sf

   # Define force as a simple function, wrap as SF
   F_sf = make_sf(lambda x, *, mask=None: -x, dim=2, rank=1)
   proc = OverdampedProcess(F_sf, D=jnp.eye(2) * 0.5)
   proc.initialize(jnp.zeros(2))

   coll = proc.simulate(
       dt=0.01,
       Nsteps=10_000,
       key=random.PRNGKey(0),
       prerun=100,
       oversampling=10,
   )


Workflow
--------

1. **Construct** the process with a force model ``F`` (:class:`~SFI.statefunc.PSF` or
   :class:`~SFI.statefunc.SF`) and a diffusion ``D`` (scalar, matrix, or ``PSF``/``SF``).

2. **Bind parameters** via :meth:`set_params` if ``F`` or ``D`` are
   unbound ``PSF`` objects.

3. **Initialize** the state with :meth:`initialize` (position for
   overdamped; position + velocity for underdamped).

4. **Simulate** via :meth:`simulate`, which returns a
   :class:`~SFI.trajectory.TrajectoryCollection`.


Diffusion specification
-----------------------

The ``D`` argument accepts multiple forms:

.. list-table::
   :header-rows: 1

   * - Input
     - Interpretation
   * - ``float`` (scalar)
     - Isotropic: :math:`D = \sigma \cdot I_d`
   * - ``(d, d)`` array
     - Constant diffusion matrix
   * - :class:`~SFI.statefunc.PSF` / :class:`~SFI.statefunc.SF` with ``rank=2``
     - State-dependent :math:`D(x)` (or :math:`D(x, v)`)


Particle systems
----------------

Both simulators respect the ``pdepth`` contract of the state
functions:

- ``pdepth=0``: single particle, ``x0`` has shape ``(d,)``
- ``pdepth=1``: interacting particles, ``x0`` has shape ``(P, d)``

For interacting-particle systems, attach ``extras_global`` and/or
``extras_local`` with :meth:`set_extras` before simulating, to pass
system metadata (box size, species labels, neighbor lists, …) through to
the models at every time step.

For a comprehensive guide to setting up particle systems, see
:doc:`/particles/user_guide`.


Choosing ``dt`` and ``oversampling``
------------------------------------

The time step ``dt`` is the interval between **recorded** frames.
The ``oversampling`` parameter controls how many internal sub-steps
are taken per recorded frame:

- **Larger** ``oversampling`` → more accurate integration, but slower.
- A rule of thumb: ``oversampling`` should be large enough that
  :math:`\mathrm{d}t_{\text{internal}} = \mathrm{d}t / \text{oversampling}`
  is small compared to the fastest timescale in the dynamics.
- For stiff systems (strong gradients, large forces), increase
  ``oversampling`` to 10–100.
- For diffusion-dominated systems with gentle forces, ``oversampling=1``
  may suffice.


Time-dependent extras (protocols)
---------------------------------

Drives, ramps, and switching protocols enter a simulation as
**time-dependent extras** through the unchanged ``set_extras`` API:

.. code-block:: python

   from SFI.trajectory import time_series_extra

   k_t = (np.arange(Nsteps) // 1000 % 2).astype(float)   # square wave
   proc.set_extras(extras_global={"k_drive": time_series_extra(k_t)})
   coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=key)

Conventions:

- A :class:`~SFI.trajectory.TimeSeriesExtra` must carry **one value per
  recorded frame** (leading axis ``== Nsteps``); the value is held
  constant across the ``oversampling`` substeps of its frame
  (zeroth-order hold), and the prerun uses the frame-0 value.
- A plain callable is interpreted as ``f(t)`` of *physical time* and
  materialized at the frame times ``t = k\,\mathrm{d}t`` before the run.
- The schedule is **attached to the returned collection** (as a
  :class:`~SFI.trajectory.TimeSeriesExtra`), aligned so that the increment
  ``X[k+1] - X[k]`` was generated under the frame-``k`` value — exactly
  the pairing the inference layer assumes.  The round-trip idiom is
  therefore one line on each side: simulate with the protocol, then
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear` on a basis containing
  :func:`~SFI.bases.extra_scalar` terms
  (:doc:`/gallery/time_dependent_forcing_demo`).
- ``simulate_chunked`` does not support time-dependent extras.

Simulation parameters
---------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``dt``
     - Time step between recorded frames
   * - ``Nsteps``
     - Number of recorded frames
   * - ``key``
     - JAX PRNG key
   * - ``prerun``
     - Warm-up steps (discarded)
   * - ``oversampling``
     - Internal sub-steps per recorded step (improves accuracy)


Observables (information and entropy production)
-------------------------------------------------

Both simulators can compute entropy- and information-production
estimates along the run, attached to the returned collection's dataset
metadata:

.. code-block:: python

   obs = coll.datasets[0].meta["observables"]
   I = obs["information"]   # I ≈ (1/4) Σ_t ⟨Δx_t, D⁻¹ F(x_t)⟩
   S = obs["entropy"]       # S ≈ Σ_t ⟨Δx_t, D⁻¹(x_mid) · F̄_t⟩

For :class:`~SFI.langevin.OverdampedProcess` this is on by default
(pass ``compute_observables=False`` to skip).  For
:class:`~SFI.langevin.UnderdampedProcess` pass
``compute_observables=True``: the accumulation runs *inside* the
integrator with the true internal velocities (which are otherwise never
stored), using the time-reversal parity split of the acceleration
field,

.. code-block:: python

   # S = Σ_t (Δv_t − F̄⁺ Δt)ᵀ D⁻¹ F̄⁻ ,   F±(x,v) = [F(x,v) ± F(x,−v)]/2
   coll = proc.simulate(dt=dt, Nsteps=N, key=key, compute_observables=True)

at the cost of three extra force evaluations per substep.  These
ground-truth observables are the natural cross-check for the
inference-side estimators
(:meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production` /
:meth:`~SFI.inference.UnderdampedLangevinInference.compute_entropy_production`).
