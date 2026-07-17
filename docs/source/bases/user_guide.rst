.. _bases-user-guide:

Building bases
==============

A **basis** in SFI is a finite dictionary of known functions
:math:`\{b_1(\mathbf{x}),\ldots,b_n(\mathbf{x})\}` used to represent an
unknown force or diffusion field as a linear combination
:math:`\hat{\mathbf{F}}(\mathbf{x}) = \sum_i \hat{F}_i\,
\mathbf{b}_i(\mathbf{x})`.  Both estimator families consume the same
:class:`~SFI.statefunc.Basis` objects: the linear estimators project onto
the basis in closed form, and the parametric estimators treat it as a
linear-in-:math:`\theta` model family (with sparsification wired in).
Build the basis once; switch estimators freely.

.. tip::

   **Tensor rank** controls what the basis produces per feature:
   ``rank=0`` → a scalar, ``rank=1`` → a vector (force component),
   ``rank=2`` → a matrix (diffusion component).  Most force inference
   uses ``rank='vector'``.

This page covers the full basis-building workflow: how to construct a
basis (compositional primitives, pre-built factories, custom functions),
the algebra for combining building blocks, and how to choose a basis
adapted to your problem.  For a general introduction to state functions
— the objects bases are made of — see :doc:`/statefunc/user_guide`.


.. _basis-construction:

Decision tree
-------------

SFI exposes three ways to construct a basis ``B`` or a parametric state
function ``F`` (PSF) that represents a force field.  Ask, in order:

1. *Can I write the force as a symbolic expression in coordinate
   variables, velocities, and named parameters?*  If yes → use
   **compositional primitives** (:func:`~SFI.bases.x_components`, :func:`~SFI.bases.v_components`,
   :func:`~SFI.bases.unit_axes`, ``named_scalar(s)``, ``frame``, ``+``, ``-``, ``*``).
2. *Is the force a standard polynomial, pair kernel, or angular
   feature?*  If yes → use a **pre-built factory**
   (:func:`~SFI.bases.monomials_up_to`,
   :mod:`SFI.bases.pairs`, :func:`~SFI.bases.ones_basis`).
3. *Is the force a neural network or an opaque, non-compositional
   closure?* → use **make_sf / make_psf / make_interactor**.


Compositional primitives
------------------------

:mod:`SFI.bases` exposes a small algebra of primitive bases that compose
cleanly with ``+``, ``-``, ``*``, and ``**``:

.. code-block:: python

   from SFI.bases import (
       x_components, v_components, unit_axes, frame,
       named_scalar, named_scalars, ones_basis,
   )

- ``x_components(dim)`` returns per-axis coordinate primitives
  (``x, y, z`` for ``dim ≤ 3``, ``x0, x1, …`` otherwise).
- ``v_components(dim)`` returns the velocity analogues (``vx, vy, vz``).
- ``unit_axes(dim)`` returns rank-1 unit-vector primitives (``ex, ey, ez``).
- ``frame(dim, velocity=False)`` is a convenience returning
  ``(components, axes)`` in one call (``(x, v, axes)`` with
  ``velocity=True``).
- ``named_scalar(name, default=...)`` / ``named_scalars(*names)`` /
  ``named_scalars(**{name: default})`` produce parametric scalar PSFs
  that carry their own :class:`~SFI.statefunc.ParamSpec` and an optional ground-truth
  default.
- ``extra_scalar(name)`` is the data-carried counterpart: a rank-0
  Basis symbol whose value is read from ``extras[name]`` at evaluation
  time — experiment conditions, drive protocols (time-dependent
  extras), or per-particle properties become composable terms (e.g.
  ``extra_scalar("k_drive") * X(dim)``).
- ``per_dataset_scalar(name, n_datasets)`` is its *inferred*
  per-experiment sibling: a parameter array with one entry per dataset,
  selected through the reserved ``dataset_index`` extra (parametric
  estimators, L-BFGS path).  ``dataset_indicator(n_datasets)`` is the
  linear-estimator route: one-hot features giving an independent linear
  coefficient per experiment.  Both fold to a single experiment via
  :meth:`~SFI.statefunc.StateExpr.specialize` (``model.specialize(dataset=k)``),
  which drops the ``dataset_index`` dependence — used when reproducing one
  experiment of a pooled fit (see :term:`specialize`).

**Multiplication rule.**  ``basis * basis`` performs einsum on same-rank
operands (scalar × scalar → scalar, scalar × vector → vector,
vector · vector → scalar).  A ``Basis * PSF`` multiplication
auto-promotes to a PSF carrying the PSF's parameters, so you can mix
parametric and non-parametric primitives freely.


Canonical patterns
~~~~~~~~~~~~~~~~~~

**Lorenz attractor (3D, three named parameters).**

.. code-block:: python

   from SFI.bases import x_components, unit_axes, named_scalars

   x, y, z = x_components(3)
   ex, ey, ez = unit_axes(3)
   sigma, rho, beta = named_scalars(sigma=20.0, rho=8.0, beta=2.0)

   F = (sigma * (y - x)) * ex \
     + (x * (rho - z) - y) * ey \
     + (x * y - beta * z) * ez

**Damped harmonic oscillator (underdamped, 1D).**

.. code-block:: python

   from SFI.bases import x_components, v_components, unit_axes, named_scalars

   (x,)  = x_components(1)
   (v,)  = v_components(1)
   (ex,) = unit_axes(1)
   k, gamma = named_scalars(k=1.0, gamma=0.5)

   F = (-k * x - gamma * v) * ex

**Stuart–Landau / limit cycle (2D).**

.. code-block:: python

   from SFI.bases import x_components, unit_axes, named_scalar, ones_basis

   x, y   = x_components(2)
   ex, ey = unit_axes(2)
   omega  = named_scalar("omega", default=2.0)
   one    = ones_basis(2)
   r2     = x * x + y * y
   F = (one - r2) * (x * ex + y * ey) + omega * (x * ey - y * ex)

**Double-well potential (1D).**

.. code-block:: python

   from SFI.bases import x_components, unit_axes

   (x,)  = x_components(1)
   (ex,) = unit_axes(1)
   F = (x - x ** 3) * ex


Pre-built bases
---------------


Monomials
~~~~~~~~~

:func:`~SFI.bases.monomials_up_to` and
:func:`~SFI.bases.monomials_degree` build scalar polynomial
bases over the state vector.  The ``rank`` keyword lifts the result to
vector or tensor form in one step:

.. code-block:: python

   from SFI.bases import monomials_up_to

   B = monomials_up_to(order=3, dim=2, rank='vector')
   # → rank-1 Basis with 2 × 10 = 20 features

Supported ``rank`` values: ``'scalar'`` (default), ``'vector'``,
``'symmetric_matrix'`` (full d(d+1)/2 tensor components), and
``'identity_matrix'`` (isotropic, 1 tensor component per scalar feature).


Structural constants
~~~~~~~~~~~~~~~~~~~~

:mod:`SFI.bases.constants` provides structural "skeleton" bases used to
build tensorial expressions:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Description
   * - ``ones_basis(dim)``
     - Scalar constant (1 feature)
   * - ``unit_vector_basis(dim, axes)``
     - Unit vectors along each axis (rank-1, *dim* features)
   * - ``identity_matrix_basis(dim)``
     - :math:`\delta_{ij}` (rank-2, 1 feature)
   * - ``symmetric_matrix_basis(dim)``
     - All symmetric unit matrices (rank-2, d(d+1)/2 features)
   * - ``constant_array(A)``
     - Wrap an arbitrary constant array ``A`` as a basis feature


Linear shorthand
~~~~~~~~~~~~~~~~

:mod:`SFI.bases.linear` exports coordinate selectors:

.. code-block:: python

   from SFI.bases import X, V, x_coordinate, linear_basis

   B = X(dim=3)                 # rank-1, the position vector itself
   c = x_coordinate(1, dim=3)   # rank-0, just x[1]


Custom user-defined bases
~~~~~~~~~~~~~~~~~~~~~~~~~

For basis functions that cannot be expressed by composing the pre-built
building blocks, use :func:`~SFI.statefunc.make_basis`.  The user function
receives a single sample ``x`` of shape ``(dim,)`` and returns an array
whose shape encodes rank and features:

.. code-block:: python

   from SFI.statefunc import make_basis

   def radial_bumps(x):
       """Gaussian bumps at fixed centres (rank 0, 3 features)."""
       centres = jnp.array([[0., 0.], [1., 0.], [0., 1.]])
       d2 = jnp.sum((x[None, :] - centres) ** 2, axis=-1)
       return jnp.exp(-d2 / 0.5)        # shape (3,) → 3 scalar features

   B = make_basis(radial_bumps, dim=2, rank=0, n_features=3)
   B_vec = B.vectorize(2)               # lift to vector for force inference

Only declare ``v``, ``mask``, or ``extras`` as keyword arguments when your
function **actually uses them**.

**Per-particle extras.**  By default an ``extras`` value is a shared
constant.  Naming a key in ``particle_extras=`` instead vmaps it
alongside ``x``, so the single-sample function sees *its own* particle's
value — the route to per-particle terms in single-particle bases.  This
is how individual "home range" stiffnesses, mobilities, or labels become
basis features:

.. code-block:: python

   def home(x, *, extras):                       # one anchor per particle
       pull = -(x - extras["x0"])                 # extras["x0"] is this particle's anchor
       onehot = (jnp.arange(N) == extras["home_id"]).astype(x.dtype)
       return pull[:, None] * onehot[None, :]     # (dim, N): feature i ≠ 0 only for particle i

   B = make_basis(home, dim=2, rank=1, n_features=N,
                  extras_keys=("x0", "home_id"),
                  particle_extras=("x0", "home_id"))   # vmapped per particle

The anchors and ids ride along as ``extras_local`` on the dataset; a
single linear fit then returns one coefficient per particle.  See the
``home_range_demo`` gallery example.  Inside *interacting* kernels, the
reserved ``particle_index`` extra plays the same role — see
:func:`~SFI.statefunc.make_interactor`.
If your basis depends on per-experiment metadata (trap centres, box
sizes, …), declare ``extras`` and ``extras_keys``:

.. code-block:: python

   def centred_disp(x, *, extras):
       x0 = extras["trap_centre"]
       return (x - x0)[:, None]          # shape (dim, 1)

   B = make_basis(centred_disp, dim=2, rank=1, n_features=1,
                  extras_keys=("trap_centre",))

See the ``custom_basis_demo`` gallery example for a complete multi-experiment
workflow using extras.


Basis algebra
-------------

State expressions support a small algebra for combining features.


Feature multiplication (``*``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiplying two expressions forms their **Cartesian product** over the
feature axis and an **einsum contraction** over supporting indices:

.. code-block:: python

   S = monomials_up_to(2, dim=2)          # scalar, 6 features
   V = unit_vector_basis(2)               # vector, 2 features
   B = S * V                              # vector, 12 features

If both operands are multi-feature (F > 1), the result has
``F_left × F_right`` features — this is the Cartesian product.  A warning
is emitted when this grows quickly.


Feature concatenation (``&``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``&`` operator **stacks** features (both operands must share rank):

.. code-block:: python

   A = monomials_degree(1, dim=3)   # scalar, 3 features
   B = monomials_degree(2, dim=3)   # scalar, 6 features
   C = A & B                        # scalar, 9 features


Feature slicing (``[]``)
~~~~~~~~~~~~~~~~~~~~~~~~

Index or slice the **feature** axis:

.. code-block:: python

   C[:3]     # first 3 features
   C[0:6:2]  # every other feature


Lifting: ``.vectorize()`` and ``.tensorize()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any rank-0 expression can be promoted to higher rank:

.. code-block:: python

   S = monomials_up_to(3, dim=2)      # scalar, 10 features

   Bv = S.vectorize(2)                # rank-1, 20 features
   Bt = S.tensorize(2, 'symmetric')   # rank-2, 30 features (d(d+1)/2 = 3)
   Bi = S.tensorize(2, 'identity')    # rank-2, 10 features (isotropic)

``.vectorize(dim, axes)`` is equivalent to
``S * unit_vector_basis(dim, axes)``.
``.tensorize(dim, 'symmetric')`` uses :func:`~SFI.bases.symmetric_matrix_basis`;
``'identity'`` uses :func:`~SFI.bases.identity_matrix_basis`.

.. note::

   These methods cannot be used on undispatched
   :class:`~SFI.statefunc.Interactor` objects.  For interacting pairs,
   use the dedicated pair bases in :mod:`SFI.bases.pairs` (see below),
   or dispatch first and then call ``.vectorize()`` on the resulting
   :class:`~SFI.statefunc.Basis`.


Pair-interaction bases: :mod:`SFI.bases.pairs`
----------------------------------------------

For multi-particle systems, :mod:`SFI.bases.pairs` provides generic
building blocks that replace the need to write custom
:func:`~SFI.statefunc.make_interactor` calls for common patterns.


Kernel families
~~~~~~~~~~~~~~~

Pre-built 1-D kernel functions, returned as lists of ``(callable, label)``
pairs:

.. code-block:: python

   from SFI.bases.pairs import gaussian_kernels, exp_poly_kernels

   ks = gaussian_kernels([0.5, 1.0, 2.0])
   ks = exp_poly_kernels(degrees=[0, 1], lengths=[1.0, 2.0])

Also available: :func:`~SFI.bases.pairs.power_kernels` and
:func:`~SFI.bases.pairs.compact_kernels`.


Pair Interactors
~~~~~~~~~~~~~~~~

Build :class:`~SFI.statefunc.Interactor` objects ready for
``.dispatch_pairs()``:

.. code-block:: python

   from SFI.bases.pairs import (
       scalar_pair_basis,     # rank-0: φ(r)
       radial_pair_basis,     # rank-1: φ(r) r̂
       dyadic_pair_basis,     # rank-2: φ(r) r̂⊗r̂
       angular_pair_basis,    # rank-1: φ(r) g(Δθ), for orientation coupling
   )

   inter = radial_pair_basis(ks, dim=2)
   B_pairs = inter.dispatch_pairs()   # → Basis(rank=1, pdepth=1)

All pair bases accept an optional ``box`` parameter for periodic boundary
conditions, and ``spatial_dims`` to select which state-vector components
are spatial coordinates.


Heading vector
~~~~~~~~~~~~~~

For active particles with a heading angle :math:`\theta`:

.. code-block:: python

   from SFI.bases.pairs import heading_vector

   e_th = heading_vector(dim=3, angle_index=2)  # (cos θ, sin θ, 0)


PBC utilities
~~~~~~~~~~~~~

Low-level functions for minimum-image displacements:

.. code-block:: python

   from SFI.bases.pairs import pbc_displacement, wrap_angle

   dx = pbc_displacement(xj, xi, box)   # minimum-image Δx
   a = wrap_angle(angle)                 # map to (-π, π]


.. _bases-choosing:

Choosing a basis
----------------

Picking the right basis is one of the most important modelling decisions
in stochastic force inference.  The basis determines what functional forms
the inferred force and diffusion can represent, how well the method
generalises to unseen data, and how tractable the inference problem
remains.

*Completeness vs. parsimony.*
A basis that is too small will miss genuine features of the dynamics; one
that is too large wastes statistical power and can overfit.  SFI's PASTIS
criterion (see :doc:`/physics_reference`) quantifies this trade-off and
automatically selects the best-supported subset.

*Symmetry encoding.*
Whenever the system has known symmetries (isotropy, periodicity, exchange
symmetry between particles, …), encoding them in the basis — rather than
hoping the inference will discover them — dramatically reduces the number
of parameters and improves the quality of the fit.

*Tensor structure.*
In SFI, forces are vector-valued and diffusion tensors are matrix-valued.
Using the basis algebra above to build the correct tensorial structure
from scalar building blocks lets you control which spatial components
share parameters and which are independent.


When to use polynomial bases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomial bases (:func:`~SFI.bases.monomials_up_to`) are the
default starting point.  They work well when:

- the dynamics vary smoothly over the sampled region,
- data covers a bounded domain (polynomials diverge at large distances),
- you expect forces that are polynomial or well-approximated by
  low-order polynomials (harmonic traps, cubic bistable potentials, …).

Typical recipe:

.. code-block:: python

   from SFI.bases import monomials_up_to

   B = monomials_up_to(order=4, dim=2, rank='vector')

Start with a moderate order (3–5) and use :meth:`sparsify_force` to prune
unnecessary terms.


When to use pair-interaction bases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-particle systems where forces arise from pairwise interactions,
use the radial pair bases from :mod:`SFI.bases.pairs`:

.. code-block:: python

   from SFI.bases.pairs import radial_pair_basis, gaussian_kernels

   ks = gaussian_kernels([0.5, 1.0, 2.0, 4.0])
   B = radial_pair_basis(ks, dim=2).dispatch_pairs()

This is appropriate when:

- forces between particles depend on inter-particle distance,
- the system has many particles and a polynomial basis in full state
  space would be prohibitively large,
- you want to infer an effective pair potential or interaction kernel.

The choice of kernel family (Gaussian, exponential-polynomial, power-law,
compact) and their characteristic length scales should reflect the expected
range and shape of the interaction.  A practical approach is to cover the
range of observed inter-particle distances with a few kernels at logarithmically
spaced length scales, and let PASTIS select the relevant ones.


Combining single-particle and pair bases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many experimental systems, both external (single-particle) and
interaction (pair) forces coexist.  You can concatenate them:

.. code-block:: python

   from SFI.bases import monomials_up_to
   from SFI.bases.pairs import radial_pair_basis, gaussian_kernels

   B_ext = monomials_up_to(order=2, dim=2, rank='vector')
   B_int = radial_pair_basis(
       gaussian_kernels([1.0, 3.0]),
       dim=2,
   ).dispatch_pairs()

   B_total = B_ext & B_int   # concatenate features

The sparsification step will then identify which terms (external,
interaction, or both) are supported by the data.


Encoding anisotropy and broken symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the system is *not* isotropic — for instance, particles in a channel
or cells migrating in a gradient — you can break spatial symmetry
explicitly by using different scalar bases for different spatial
components:

.. code-block:: python

   from SFI.bases import monomials_up_to, unit_vector_basis

   # Strong confinement along y, weaker along x
   S_x = monomials_up_to(4, dim=2)   # high order in x
   S_y = monomials_up_to(2, dim=2)   # low order in y (captures confinement)

   e_x = unit_vector_basis(2, axes=[0])
   e_y = unit_vector_basis(2, axes=[1])

   B = (S_x * e_x) & (S_y * e_y)

This gives the inference more flexibility along x than along y, matching
the expected physics.


High-dimensional and neural-network approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the state dimension is large or the functional form of the force is
highly nonlinear, fixed polynomial or kernel bases may not be practical.
Neural-network models can be wrapped as :class:`~SFI.statefunc.PSF` objects
and trained via
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`
(the nonlinear-in-θ L-BFGS path).

The recommended approach is SFI's **compositional NN API**, which requires
no external library and integrates seamlessly with the basis algebra:

.. code-block:: python

   from SFI.bases import X
   import jax.numpy as jnp

   F_nn = (
       X(dim=2)
       .rank_to_features()
       .dense(64, weight="W1", bias="b1")
       .elementwisemap(jnp.tanh)
       .dense(64, weight="W2", bias="b2")
       .elementwisemap(jnp.tanh)
       .dense(2, weight="W3", bias="b3")
       .features_to_rank(1)
   )

   inf.infer_force(F_nn, theta0=..., inner_maxiter=500)

See the ``nn_force_demo`` gallery example for a complete workflow.

.. tip::

   You can also wrap models from external JAX libraries such as
   `equinox <https://docs.kidger.site/equinox/>`_ as
   :class:`~SFI.statefunc.PSF` objects via :func:`~SFI.statefunc.make_psf`:

   .. code-block:: python

      import jax
      import equinox as eqx
      from SFI.statefunc import make_psf

      class ForceNet(eqx.Module):
          layers: list

          def __init__(self, key):
              k1, k2, k3 = jax.random.split(key, 3)
              self.layers = [
                  eqx.nn.Linear(2, 64, key=k1),
                  eqx.nn.Linear(64, 64, key=k2),
                  eqx.nn.Linear(64, 2, key=k3),
              ]

          def __call__(self, x):
              h = x
              for layer in self.layers[:-1]:
                  h = jax.nn.tanh(layer(h))
              return self.layers[-1](h)

      net = ForceNet(jax.random.PRNGKey(0))

      # ``make_psf`` expects the trainable parameters as a dict of arrays,
      # so split the equinox module into its array leaves (the parameters)
      # and its static structure, and recombine inside the callable.
      arrays, static = eqx.partition(net, eqx.is_array)
      leaves, treedef = jax.tree_util.tree_flatten(arrays)
      init = {f"w{i}": leaf for i, leaf in enumerate(leaves)}

      def nn_force(x, *, params):
          leaves = [params[f"w{i}"] for i in range(len(init))]
          net = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)
          return net(x)[..., None]

      F_nn = make_psf(nn_force, dim=2, rank=1, n_features=1, params=init)

Strategies:

- use a small linear basis for the dominant low-order structure, then
  train a neural-network PSF on the residual (two-stage approach);
- start from a pre-trained network and fine-tune with SFI's
  quasi-likelihood loss.

See :doc:`/statefunc/user_guide` for the full model-composition
workflow and the gallery for worked examples.


Rules of thumb for basis sizing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Problem type
     - Typical basis size :math:`n`
     - Guidance
   * - 2D, smooth potential
     - 10–30
     - polynomials order 3–5, ``rank='vector'``
   * - 2D, bistable / limit cycle
     - 20–50
     - order 4–6; PASTIS prunes to ~10
   * - 3D, isotropic interactions
     - 5–15 pair kernels
     - radial pair basis at log-spaced lengths
   * - High-dimensional (d > 5)
     - neural network
     - polynomial bases grow as :math:`\binom{d + k}{k}`; prefer NN
   * - Mixed (external + pair)
     - 15–40 total
     - concatenate with ``&``; let PASTIS select
   * - Spatial field (SPDE)
     - 2–20 operator terms
     - composable stencil operators; see :doc:`/spde/user_guide`


Anti-patterns
-------------

**Don't** re-implement coordinate projections, unit axes, or scalar
parameter containers from scratch — compose with the primitives above.

**Don't** thread the same :class:`~SFI.inference.OverdampedLangevinInference` through
multiple distinct fits; allocate one per model (``inf``, ``inf_nn``,
``inf_parametric``, …).  Each distinct force model (linear basis, PSF,
NN, parametric-refined) gets its **own** inference instance.

**Don't** pass ``set_params(theta_F=...)`` to a process whose PSF was
built from primitives with defaults; the process resolves the
:class:`~SFI.statefunc.ParamSpec` defaults automatically on ``initialize``.


See also
--------

- :doc:`/statefunc/user_guide` — building and composing state functions,
  including the Basis → PSF → SF progression.
- :doc:`/particles/user_guide` — pair interactors and multi-particle
  data layout.
- :doc:`/physics_reference` — mathematical background on polynomial,
  radial, and discrete-Laplacian bases.
- :doc:`/spde/user_guide` — spatial field inference with composable
  stencil operators.
