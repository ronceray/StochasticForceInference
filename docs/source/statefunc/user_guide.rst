.. _statefunc-user-guide:

Models and state functions
==========================

.. currentmodule:: SFI.statefunc

Why state functions?
--------------------

In SFI, every quantity that depends on the system state — forces, diffusion
tensors, basis dictionaries, observables — is represented as a **state
function**.  This uniform representation means you can:

- compose models from reusable building blocks (``+``, ``*``, ``&``),
- switch between linear inference (exact) and nonlinear optimisation
  with the same object,
- pass inferred models directly to Langevin simulators without glue code.

.. tip::

   **Basis → PSF → SF** is the central progression in SFI.
   A :class:`Basis` is a parameter-free dictionary for linear inference;
   :class:`PSF` adds named parameters for nonlinear models;
   :class:`SF` freezes parameters for simulation.
   The progression, with recipes, is detailed below.

The :mod:`SFI.statefunc` module provides a common language for
"state-dependent functions" used across SFI: basis dictionaries, parametric
forces, diffusion tensors, and interacting-particle operators.  Instead of
writing ad-hoc JAX functions for each model, SFI uses a small set of
composable objects with explicit shape contracts, names, and differentiation
rules.

The main entry points are the factory helpers :func:`make_basis` and
:func:`make_psf`, which turn small JAX functions into rich objects that can be
used across SFI for both linear and nonlinear inference, and for simulation.

At a glance
-----------

High-level objects and helpers:

* :func:`~SFI.statefunc.make_basis` – wrap a "single-sample" JAX function into a
  :class:`Basis`. This is the usual entry point for building *linear*
  dictionaries of features.
* :func:`~SFI.statefunc.make_psf` – wrap a single-sample function with
  parameters into a :class:`PSF` (parametric state function) with a named
  parameter tree.
* :func:`~SFI.statefunc.make_interactor` – define a local K-body rule for
  interacting particle systems, to be dispatched into a global state function.
* :class:`~SFI.statefunc.Basis` – a finite dictionary of parameter-free
  features, typically used in exact solutions of linear inference problems.
* :class:`~SFI.statefunc.PSF` – a parametric state function :math:`F(x;\theta)`,
  used in both linear and truly nonlinear inference and as a bridge to
  simulation. Parameter names encode sharing-by-name.
* :class:`~SFI.statefunc.SF` – a state function with fixed parameters, suitable
  for direct evaluation and Langevin simulation.
* :class:`~SFI.statefunc.StateExpr` – the underlying expression graph with a
  static contract (dimension, tensor rank, particle depth, number of features).
* :class:`~SFI.statefunc.Interactor` – a local K-body operator for interacting
  particle systems, to be dispatched into Basis/PSF/SF objects.
* Global helpers such as :func:`set_jit`, and memory accounting facilities used
  by :mod:`SFI.integrate` for adaptive chunking.


Overview: what a state function is
----------------------------------

A *state function* in SFI is a JAX-traceable map that takes as input one
or more trajectories :math:`x(t)` (and optionally velocities :math:`v(t)`)
and returns, for each time and particle, a scalar, vector, or tensor
quantity built from the state. The same object is used consistently in:

* bases for linear inference (exact solution of a linear regression problem),
* parametric models for nonlinear inference (e.g. gradient-based optimisation),
* drift and diffusion functions in Langevin simulators. 

A state-function family :class:`PSF` represents :math:`F(x; \theta)`, where
:math:`\theta` is a structured collection of parameters (a :class:`~SFI.statefunc.ParamSuite`).
The wrapper :class:`SF` then fixes :math:`\theta` to a concrete value and
can be passed directly to the simulation and integration modules. 


Single-sample functions and the factories
-----------------------------------------

The basic pattern is:

1. Write a **single-sample** JAX function that takes a single state vector
   (and optionally a single velocity, a mask, and extras) and returns a
   feature array of shape ``(..., n_features)`` with features on the last
   axis.
2. Wrap it with :func:`make_basis` or :func:`make_psf` to obtain a
   :class:`Basis` or :class:`PSF` that can be called on full trajectories. 


Minimal example: scalar features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a 2D example with three scalar features:

.. code-block:: python

    import jax.numpy as jnp
    from SFI.statefunc import make_basis

    def scalar_feats(x):
        # x has shape (dim,) = (2,)
        x0, x1 = x[0], x[1]
        f0 = x0
        f1 = x0 * x1
        f2 = x1**2
        return jnp.stack([f0, f1, f2], axis=-1)  # (n_features,) = (3,)

    B = make_basis(
        scalar_feats,
        dim=2,
        rank=0,           # scalar
        n_features=3,
        labels=["x0", "x0*x1", "x1^2"],
    )

At call time, this scalar :class:`Basis` expects an array ``x`` with shape

* ``(T, dim)`` for a single trajectory (no particles),
* ``(T, P, dim)`` for :math:`P` particles, if you use interacting objects later.

The output then has shape:

* ``(T, 3)`` (no particles),
* ``(T, P, 3)`` (per-particle features). 

You never need to explicitly handle masks or extras here. By default the
factory generates nodes that accept optional ``mask`` and ``extras`` and
enforce that these are consistent if they are provided by the rest of SFI
(e.g. by :class:`SFI.trajectory.TrajectoryCollection`). As a user, you can
usually ignore them and just implement the pure single-sample map.


Vector and tensor outputs; controlling symmetries
-------------------------------------------------

The static contract of a state function includes:

* ``rank`` – tensor rank (0 = scalar, 1 = vector, 2 = matrix, …),
* ``dim`` – spatial dimension,
* ``pdepth`` – how many particle axes are carried in the output,
* ``n_features`` – number of features, always on the last axis. 

For a **vector-valued** basis in 3D, the single-sample function should
return shape ``(3, n_features)``:

.. code-block:: python

    def vec_feats(x):
        # x: (3,)
        x0, x1, x2 = x[0], x[1], x[2]
        f0 = jnp.array([-x0, 0.0, 0.0])
        f1 = jnp.array([0.0, -x1, 0.0])
        f2 = jnp.array([0.0, 0.0, -x2])
        return jnp.stack([f0, f1, f2], axis=-1)  # (3, 3) → 3 vector features

    B_vec = make_basis(
        vec_feats,
        dim=3,
        rank=1,          # vector
        n_features=3,
        labels=["-x0 e0", "-x1 e1", "-x2 e2"],
    )

On trajectories with shape ``x.shape == (T, 3)``, this yields
``y.shape == (T, 3, 3)`` with layout

.. math::

    y_{t, m, a}

where :math:`m` indexes the spatial component and :math:`a` the feature.

You can combine scalar and vector/tensor objects using operations such as
``+``, scalar multiplication, ``.einsum`` and ``.dot``, to build up forces
and diffusion tensors with **precise symmetry control**. For instance,
radial projectors can be built by combining unit vectors and scalar
radial bases. This is a key difference to the previous SFI implementation,
where all dimensions shared the same scalar basis; here you can explicitly
encode rotational, reflection, or anisotropic symmetries at the level of
the state-function graph. 


Pre-built bases and basis algebra
---------------------------------

The :mod:`SFI.bases` package provides commonly used polynomial, structural,
linear, and pair-interaction bases, together with an algebra for combining
them (``*``, ``&``, slicing, ``.vectorize()``, ``.tensorize()``).

.. seealso::

   :doc:`/bases/user_guide` for the complete bases reference and
   guidance on selecting a basis adapted to your problem.


.. _building-models:

From Basis → PSF → SF
---------------------

The three-level hierarchy at a glance:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Object
     - Parameters
     - Typical use
   * - :class:`Basis`
     - None (parameter-free)
     - Linear inference: the dictionary :math:`\{b_i(\mathbf{x})\}`.
       The inference engine solves for coefficients
       :math:`\hat{F}_i` exactly.
   * - :class:`PSF`
     - Named parameter tree :math:`\theta`
     - Nonlinear inference: parametric force
       :math:`\mathbf{F}(\mathbf{x};\theta)`.  Optimised via
       JAX gradients.
   * - :class:`SF`
     - Frozen (fixed :math:`\theta`)
     - Simulation and evaluation: a pure function
       :math:`\mathbf{F}(\mathbf{x})` passed to Langevin integrators.

Linear inference
~~~~~~~~~~~~~~~~

A :class:`Basis` represents a deterministic dictionary of features
:math:`f_j(x)`. It is the natural object for **linear inference**, where
you want to solve for coefficients :math:`\theta_j` in

.. math::

    F(x; \theta) = \sum_j \theta_j f_j(x). 

Pass the basis directly to the inference engine:

.. code-block:: python

    from SFI.bases import monomials_up_to

    B = monomials_up_to(order=3, dim=2, rank='vector')

    inf.infer_force_linear(B)
    inf.sparsify_force(criterion="PASTIS")

The engine handles the Basis → PSF conversion internally.

The inference modules use :class:`Basis` when they can solve the linear
problem exactly, and :class:`PSF` when they want to include nonlinear
parameters and rely on generic optimisers.  Both share the same underlying
expression tree and contracts, so switching between linear and nonlinear
treatments is cheap.


Nonlinear models and PSF
~~~~~~~~~~~~~~~~~~~~~~~~

For fully **nonlinear** parametric models, you can build a :class:`PSF`
directly with :func:`make_psf` by providing a single-sample function that
depends explicitly on a parameter dict. 

Minimal example:

.. code-block:: python

    from SFI.statefunc import make_psf

    def force_local(x, *, params):
        k = params["k"]     # scalar stiffness
        return -k * x       # vector in R^dim

    F = make_psf(
        force_local,
        dim=3,
        rank=1,            # vector force
        n_features=1,      # single feature
        params={"k": ()},  # scalar parameter
        labels=["-k x"],
    )

    theta = {"k": jnp.array(10.0)}
    y = F(x, params=theta)       # y has shape batch · dim

Name reuse for parameter sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter names are *global* within a PSF tree: using the same name in
multiple bricks means the parameter is shared, and both branches of the
model see the same array. The factory uses these names to build a
:class:`ParamSuite` that specifies shapes and dtypes. 

This makes it easy to express constraints such as equal coefficients in
different directions, or shared length scales, at the level of the state
function: just reuse the parameter name and give it a consistent shape.


Fixing θ: SF for simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a PSF is calibrated, you often want to freeze its parameters and use
it as a pure state-function, e.g. for Langevin simulation.  This is the
role of :class:`SF` (State Function).

After linear inference, the engine produces a ready-to-use SF:

.. code-block:: python

    F_sf = inf.force_inferred      # already an SF
    proc = OverdampedProcess(F_sf, D=inf.diffusion_average)

You can also freeze a PSF manually:

.. code-block:: python

    F_fixed = psf.bind(params=theta)   # PSF → SF

Here ``F_fixed(x)`` has the same contract as the underlying PSF, but
no longer accepts a ``params`` argument.  For fixed, parameter-free
functions (e.g. an exact model for comparison), use :func:`make_sf`
directly:

.. code-block:: python

    from SFI.statefunc import make_sf

    F_exact = make_sf(lambda x: -x, dim=2, rank=1)


Strategy: linear baseline, then a nonlinear model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common strategy is to fit a cheap linear model first to capture the
dominant low-order structure, then fit a neural-network PSF and keep it
only if it improves on that baseline.  Give each model family its own
inference object (one ``inf`` per fit):

.. code-block:: python

   from SFI import OverdampedLangevinInference
   from SFI.bases import monomials_up_to

   # Stage 1: linear backbone
   inf = OverdampedLangevinInference(coll)
   B = monomials_up_to(order=3, dim=2, rank='vector')
   inf.infer_force_linear(B)
   inf.sparsify_force(criterion="PASTIS")

   # Stage 2: a fresh inference object for the nonlinear model
   # … define F_nn — see the neural-network section of /bases/user_guide …
   inf_nn = OverdampedLangevinInference(coll)
   inf_nn.infer_force(F_nn, theta0=...)

Compare the held-out error of the two fits to decide whether the network
is worth its extra parameters.


From inference to simulation
----------------------------

Every inferred model can be turned into a simulator:

.. code-block:: python

   # Quick route: bootstrapped simulation
   coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(
       key=jax.random.PRNGKey(42), oversampling=10,
   )

   # Manual route: extract SF and build process
   from SFI.langevin import OverdampedProcess

   F_sf = inf.force_inferred    # already an SF
   proc = OverdampedProcess(F_sf, D=inf.diffusion_average)
   proc.initialize(x0)
   coll_sim = proc.simulate(dt=0.01, Nsteps=10_000, key=key)

This round-trip — data → inference → simulation → comparison — is
central to the SFI workflow and appears in most gallery examples.


Grid-based SPDE models (experimental)
-------------------------------------

For spatially-extended problems (reaction–diffusion, active matter,
phase fields), the Layout/Sector/Embed paradigm provides differential
operators and symmetry-aware basis construction on grids.  This is part
of the experimental SPDE toolbox — see :doc:`/spde/index` and the
layout guide at :doc:`/spde/layout_guide`.


Inputs: x, v, mask, extras
---------------------------

At call time, all Basis/PSF/SF objects follow the same signature:

.. code-block:: python

    y = expr(
        x,
        v=None,
        mask=None,
        extras=None,
        params=None,   # only for PSF
    )

Shapes and contracts
~~~~~~~~~~~~~~~~~~~~

* ``x``: state, always last axis is ``dim``; prefix is batch and possibly
  particle indices. For single-particle problems, you can think of
  ``x.shape == (T, dim)``. For multi-particle problems with
  ``particles_input=True`` nodes, ``x.shape == (T, P, dim)``. 
* ``v``: velocity, optional and only required if the dictionary was built
  with ``needs_v=True``. It must have the same shape as ``x``.
* ``mask``: optional array that broadcasts to the batch/particle prefix of
  ``x``. It is used extensively by the trajectory layer to encode missing
  data (particles entering or leaving, dropped frames, etc.) and is
  honoured automatically by state functions. In typical user-defined
  functions you do not need to mention it: the factories inject the right
  masking logic.
* ``extras``: a JAX-compatible pytree of auxiliary data (e.g. periodic
  box size, experiment-level parameters, adjacency or neighbour lists).
  Factories enforce that declared extras keys are present but do not
  constrain their shape; individual leaves and dispatchers interpret them.
  

The output ``y`` has layout:

.. math::

    \text{batch} \cdot [P]^{\text{pdepth}} \cdot (\text{dim})^{\text{rank}} \cdot \text{features},

where :

* ``pdepth`` controls how many particle axes remain in the output,
* ``rank`` is the tensor rank (0, 1, 2, …),
* ``features`` is usually dropped if it equals 1 and
  ``drop_features=True``. 


Interacting particle systems (optional)
---------------------------------------

For systems of many interacting particles (e.g. active matter, coarse-grained
fluids, lattice fields), :mod:`statefunc` provides a clean separation
between:

* **Local interaction laws** on small groups of particles (pairs, triplets,…),
* **Dispatchers** that apply these laws over all relevant neighbours,
  optionally enforcing symmetries and neighbourhood structures.

The user-facing entry point here is :class:`Interactor` and
:func:`make_interactor`. 


Local interactors
~~~~~~~~~~~~~~~~~

A local interactor is built from a single-sample function on a *tuple* of
particles:

.. code-block:: python

    from SFI.statefunc import make_interactor

    def local_pair_force(Xk, *, extras):
        # Xk: (K, dim) with K=2 → Xk[0] = xi, Xk[1] = xj
        xi, xj = Xk[0], Xk[1]
        dx = xj - xi
        r2 = jnp.sum(dx**2)
        # simple pairwise repulsion
        f = dx / (r2 + 1e-6)          # vector in R^dim
        return f[..., None]           # shape (dim, 1) → one feature

    inter = make_interactor(
        local_pair_force,
        dim=3,
        rank=1,                       # vector-valued
        K=2,                          # pair interaction
        n_features=1,
        extras_keys=("box",),         # require extras["box"], but do not parse it here
        labels=("pair_repulsion",),
    )

The :class:`~SFI.statefunc.Interactor` is still a local expression; it expects inputs of shape
``(K, dim)`` per *sample*, and has ``particles_input=True, pdepth=0`` in
its contract. 


Dispatching over neighbours
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain a global dictionary over all particles, you call a dispatcher
method on the interactor, which conceptually mirrors graph neural network
layers: for each edge in a particle graph, apply the local kernel, then
aggregate into per-particle outputs. The dispatcher can work with all
pairs, radius cutoffs, or explicit neighbour lists.

Sketch:

.. code-block:: python

    # x has shape (T, P, dim)
    # neighbours is some dispatcher object built from the interactions backend
    B_pairs = inter.dispatch(
        neighbours,
        owners="focal",       # reduce onto each focal particle
        reducer="sum",        # sum over neighbours
        return_as="basis",    # or "psf" if local nodes have params
    )

Here ``B_pairs`` is an ordinary :class:`Basis` (or :class:`PSF`) with a
contract that now has ``pdepth=1`` (a per-particle vector field). It can
be concatenated with single-particle bases, combined with scalar radial
kernels through einsums, and used directly in inference and simulation,
just like any other state function. 


Performance and memory
----------------------

The entire state-function tree keeps track of the *single-sample* memory
footprint (an internal ``MemHint``). This is used by the
integration routines to choose vectorisation and chunking strategies that
fit in the available device memory. From the user side this is mostly
transparent: as long as you stick to the factories and basic composition
operations, the integrators can automatically adjust chunk sizes. 

You can globally enable or disable JIT compilation of Basis/PSF/SF
``__call__`` using :func:`SFI.statefunc.set_jit`, which is useful when
profiling or debugging. 
