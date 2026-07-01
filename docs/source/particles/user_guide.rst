.. _particles-user-guide:

Particle systems
================

.. currentmodule:: SFI.statefunc

This guide explains how to set up SFI for systems of **many interacting
particles** — active colloids, migrating cells, coarse-grained proteins,
lattice fields — where forces arise from pairwise (or higher-order)
interactions and the state vector has shape ``(T, P, d)``.

For single-particle problems the standard workflow
(:doc:`/trajectory/user_guide` → :doc:`/inference/user_guide`) applies
unchanged.  This page covers the additional concepts needed when
``P > 1``.


When do you need this?
----------------------

Use the particle machinery whenever:

- you track **multiple interacting agents** and want to infer their
  interaction law,
- the force on particle *i* depends on the positions of its neighbours,
- you need periodic boundary conditions (PBC), neighbour lists, or
  species labels.

If your particles are **non-interacting** (e.g. many independent
diffusing colloids observed simultaneously), you can still store them
in a ``(T, P, d)`` collection, but a simple single-particle basis is
sufficient — the inference engine will process each particle
independently via the mask.


Trajectory layout
-----------------

A particle trajectory has shape ``(T, P, d)``:

.. code-block:: python

   import jax.numpy as jnp
   from SFI.trajectory import TrajectoryCollection

   T, P, d = 500, 20, 2
   X = jnp.zeros((T, P, d))       # your tracked positions
   mask = jnp.ones((T, P), bool)  # optional: False = missing

   coll = TrajectoryCollection.from_arrays(
       X=X, dt=0.01, mask=mask,
       extras_global={"box": jnp.array([10.0, 10.0])},
       extras_local={"radius": jnp.ones(P) * 0.5},
   )

**Extras** carry per-experiment (global) and per-particle (local) metadata
that your interaction models can access at every time step.  Common
examples:

- ``box`` — simulation-box size for periodic boundary conditions,
- ``species`` — integer labels for multi-species systems,
- ``radius`` — per-particle radii for polydisperse systems.


Local interaction laws
----------------------

A local **interactor** defines the force that particle *j* exerts on
particle *i*, purely as a function of a small group of particles
(pair, triplet, …).  For standard radial pair interactions, prefer
the pre-built kernels in :mod:`SFI.bases.pairs` — they handle
vectorisation, normalisation, and periodic-image shifting for you:

.. code-block:: python

   from SFI.bases.pairs import pair_direction

   inter = pair_direction(dim=2)  # rank-1 unit vector (pair repulsion)

When you need a custom kernel, use :func:`~SFI.statefunc.make_interactor` directly:

.. code-block:: python

   from SFI.statefunc import make_interactor
   import jax.numpy as jnp

   def pair_kernel(Xk, *, extras):
       xi, xj = Xk[0], Xk[1]
       dx = xj - xi
       r = jnp.sqrt(jnp.sum(dx**2) + 1e-12)
       return (dx / r)[..., None]   # shape (d, 1)

   inter = make_interactor(
       pair_kernel,
       dim=2, rank=1, K=2,
       n_features=1,
       extras_keys=(),
       labels=("pair_kernel",),
   )

Key points:

- ``K=2`` means this is a **pair** interaction; the input ``Xk`` has
  shape ``(2, d)``.
- The output has shape ``(d, n_features)`` — one rank-1 feature.
- The function is written for a **single pair**; SFI dispatches it
  over all neighbour pairs automatically.

For pre-built radial, angular, and dyadic pair kernels, see
:mod:`SFI.bases.pairs`.


Dispatching over neighbours
----------------------------

The interactor becomes a global basis (or PSF) by calling a dispatcher:

.. code-block:: python

   B_pairs = inter.dispatch_pairs()
   # → Basis(rank=1, pdepth=1, n_features=1)

The result is an ordinary :class:`Basis` with ``pdepth=1``, meaning it
produces per-particle outputs.  You can combine it with single-particle
terms using the ``&`` operator:

.. code-block:: python

   from SFI.bases import monomials_up_to

   B_ext = monomials_up_to(order=2, dim=2, rank='vector')
   B_total = B_ext & B_pairs  # external + interaction features

The dispatcher handles:

- **all-pairs** enumeration (default for small *P*),
- **neighbour lists** with a radius cutoff for large *P*,
- **periodic boundary conditions** via the ``box`` extra.


Using pre-built pair bases
--------------------------

For common interaction kernels you rarely need to write a custom
interactor.  :mod:`SFI.bases.pairs` provides ready-made pair bases:

.. code-block:: python

   from SFI.bases.pairs import radial_pair_basis, gaussian_kernels

   ks = gaussian_kernels([0.5, 1.0, 2.0, 4.0])
   B_pairs = radial_pair_basis(ks, dim=2).dispatch_pairs()

Available families:

.. list-table::
   :header-rows: 1

   * - Builder
     - Output
   * - :func:`~SFI.bases.pairs.radial_pair_basis`
     - :math:`\phi(r)\,\hat{\mathbf{r}}` (rank-1)
   * - :func:`~SFI.bases.pairs.scalar_pair_basis`
     - :math:`\phi(r)` (rank-0)
   * - :func:`~SFI.bases.pairs.dyadic_pair_basis`
     - :math:`\phi(r)\,\hat{\mathbf{r}}\otimes\hat{\mathbf{r}}` (rank-2)
   * - :func:`~SFI.bases.pairs.angular_pair_basis`
     - :math:`\phi(r)\,g(\Delta\theta)` (rank-1, heading coupling)

Kernel families: ``gaussian_kernels``, ``exp_poly_kernels``,
``power_kernels``, ``compact_kernels`` — choose length scales that span
the observed range of inter-particle distances.


Simulation with particles
--------------------------

Both :class:`~SFI.langevin.OverdampedProcess` and
:class:`~SFI.langevin.UnderdampedProcess` respect the ``pdepth``
contract of state functions:

.. code-block:: python

   from SFI.langevin import OverdampedProcess
   from jax import random

   F_sf = ...  # SF with pdepth=1
   proc = OverdampedProcess(F_sf, D=0.1)
   proc.set_extras(extras_global={"box": jnp.array([10.0, 10.0])})
   proc.initialize(x0)  # x0 shape: (P, d)

   coll = proc.simulate(
       dt=0.01, Nsteps=5000,
       key=random.PRNGKey(0),
       oversampling=10,
   )

Attach ``extras_global`` (and ``extras_local``) to the process with
``proc.set_extras(...)`` before simulating, so they are available to the
force model at every step.


Inference on particle data
---------------------------

The inference workflow is the same as for single particles:

.. code-block:: python

   inf = SFI.OverdampedLangevinInference(coll)
   inf.compute_diffusion_constant()
   inf.infer_force_linear(B_total)
   inf.sparsify_force(criterion="PASTIS")
   inf.print_report()

The difference is purely in the basis: because ``B_total`` has
``pdepth=1`` features, the Gram matrix and force moments are computed
over all particles and time steps jointly.


Practical tips
--------------

**Start simple.**
Begin with a small radial pair basis + low-order polynomial external
basis, and let PASTIS select.  Add angular or dyadic terms only if
the residuals suggest anisotropic interactions.

**Choose kernel length scales wisely.**
Cover the range of observed inter-particle distances with a few
(4–8) kernels at roughly logarithmically spaced scales.

**Watch the feature count.**
Pair bases combined with external bases can grow quickly.  The
sparsification step is essential to avoid overfitting.

**Periodic boundary conditions.**
Pass ``extras_global={"box": box_size}`` and use PBC-aware pair
bases.  See :func:`~SFI.bases.pairs.pbc_displacement`.


.. seealso::

   - :doc:`/bases/user_guide` — pair-interaction bases, plus guidance on
     selecting and combining bases.
   - :doc:`/statefunc/user_guide` — interactor API and dispatch mechanics.
   - Gallery: :doc:`/gallery/abp_align_demo` — active Brownian particles
     with alignment interactions.
   - Gallery: :doc:`/gallery/advanced/flocking_3d_demo` — 3D underdamped
     flocking (trap + cohesion + velocity alignment); compares the
     Parametric GN and loss backends on the same multi-particle dataset.
