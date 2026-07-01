.. _statefunc-overview:

State function design
=====================

What is a state function?
-------------------------

A *state function* in SFI is any object that maps a dynamical state
(positions, optional velocities, internal coordinates, …) plus optional extras
(masks, neighbor lists, experimental tags) to tensor-valued outputs with a
well-defined *rank* and *feature axis*.

Formally, a state function has a static contract, implemented by a common
mixin :class:`_ContractMixin`:

* ``rank`` – tensor rank as a :class:`Rank` enum (SCALAR, VECTOR, MATRIX,
  …);
* ``dim`` – number of spatial coordinates per particle (or ``None`` to remain
  agnostic);
* ``needs_v`` – whether the function requires velocity input;
* ``pdepth`` – number of particle axes in the *output* (0, 1 or 2);
* ``n_features`` – number of feature channels on the last axis;
* ``param_suite`` – optional :class:`ParamSuite` describing parameters, or
  ``None`` for deterministic nodes;
* ``extras_required`` – a tuple of extras keys that must be present;
* ``particles_input`` – whether the input ``x`` is expected to carry a particle
  axis.

Velocities are optional throughout: for overdamped systems, state functions
typically depend only on positions; for underdamped systems, objects with
``needs_v=True`` additionally take velocities and can expose derivatives with
respect to both.

The contract is enforced at node level by two main guards:

* ``_assert_inputs`` checks the shape and consistency of ``x``, ``v``, ``mask``
  and ``extras``;
* ``_assert_outputs`` checks that the output ``y`` matches the declared
  feature, rank, and particle-depth structure.

Masking and missing data
------------------------

Inference in SFI must deal with complex patterns of missing data:

* particles appear and disappear over time;
* different coordinates of the same particle may be masked;
* estimators often evaluate derivatives at shifted time points.

The contract enforces a simple rule for masks:

* a mask, if provided, must broadcast to the prefix of ``x`` excluding the
  spatial dimension, i.e. to ``x.shape[:-1]``;
* with ``particles_input=True``, this prefix is ``(…, N_particles)`` and the
  mask has the natural shape "batch · particles".

Masks can be boolean or numeric; broadcasting follows NumPy rules. In typical
use, masks are constructed automatically from missing data in
:class:`TrajectoryCollection` (trajectory module). User-defined single-sample
functions almost never manipulate masks explicitly:

* factories accept a keyword-only argument ``mask=None``, but most user
  functions do not need to mention it;
* the surrounding SFI code constructs and applies masks as needed.

Differentiation and custom rules
--------------------------------

Inference modules need derivatives with respect to:

* positions and velocities (for drift and diffusion estimators);
* parameters (for likelihood-based inference and gradient-based optimisation).

Naive autodiff through raw code is not enough: it must interact correctly with
masking, particle indexing, and multi-particle interactions. The
:class:`StateExpr` graph carries explicit derivative nodes, and the API
provides consistent methods such as ``d_x()``, ``d_v()`` and ``d_theta()`` on
:class:`PSF` and :class:`SF`. Custom differentiation rules for multi-particle
systems are implemented at node level while still leveraging JAX under the
hood.

Modularity and algebra of operators
-----------------------------------

Realistic models are built from reusable bricks: monomials, radial functions,
local tensors, external fields, etc. The state function abstraction provides
an algebra of operators on :class:`StateExpr`:

* concatenation of feature dictionaries;
* products with scalar fields;
* Einstein summations (``einsum``) to contract features with unit vectors,
  projectors, and tensors;
* feature slicing and reshaping;
* mapping over extra axes (e.g. species or feature groups).

The static contract for composite nodes is obtained by merging children via
``merge_contract`` with mode-specific rules:

* ``mode="concat"`` – concatenate features; requires equal rank, sums
  ``n_features``;
* ``mode="map"`` – elementwise maps; requires equal rank and ``n_features``;
* ``mode="einsum"`` – spatial contraction; the Einstein spec determines the
  output rank and the features are the Cartesian product of the inputs.
  Particle depth and ``particles_input`` are merged using a strict broadcast
  rule (either identical, or scalars combined with particle-wise nodes).

Because each operator preserves the static contract, complex bases can be built
safely by combining simpler ones. In particular, it is natural to:

* start from scalar features such as monomials or radial kernels;
* contract them with unit vectors to obtain vector forces;
* assemble higher-rank tensors (diffusion matrices, projectors, stress
  tensors) with prescribed symmetry.

Identity, naming and model selection
------------------------------------

Inference modules need to keep track of *which* feature is which:

* coefficients must be associated with meaningful labels;
* model selection routines need stable feature identities for pruning;
* pretty-printing and diagnostics require readable names.

Each state function carries labels and descriptors for its feature axis, and
these are preserved under composition (with appropriate concatenation or
prefixing rules). This traceability is what allows higher-level code to talk
about “turning off a specific kernel” or “selecting the best radial terms”
without losing track of which contribution is which.

Extras: pass-through pytrees
----------------------------

Many models depend on *extras* that are not part of the core state:

* boundary conditions (e.g. periodic box size);
* neighbor or adjacency lists;
* experiment-specific parameters or tags;
* particle identities and types.

State functions declare which extras keys they require, via
``extras_required``. At runtime:

* ``_assert_inputs`` guarantees that required keys are present if any extras
  mapping is provided;
* extras themselves can be any JAX-compatible pytree (arrays, nested dicts,
  tuples, …) and are passed through to leaves and dispatchers without shape
  validation at this level.

This is essential for interacting-particle and spatial models, where local
kernels need access to box sizes, neighbor lists, or other contextual data.

Named parameters and parametric trees
-------------------------------------

Parametric state functions (:class:`PSF`) organise parameters into named
blocks described by :class:`ParamSpec` and grouped into a :class:`ParamSuite`:

* each :class:`ParamSpec` has a ``name``, ``shape``, ``dtype``, and an
  initialisation rule;
* :class:`ParamSuite` is an immutable container of specs; it can map between
  flat vectors and parameter trees and can coerce user-provided parameter
  dicts to the expected shapes and dtypes.

Crucially, parameter *names* encode sharing:

* merging parameter suites (e.g. when composing PSFs) uses name-based union;
* if a name appears in multiple suites, the corresponding specs must be
  compatible (same shape and dtype) and are merged into a single shared
  parameter; otherwise an error is raised.

Thus, reusing a parameter name in different parts of a PSF implies that the
parameter is **shared** and must have the same array structure. This is how
weight tying and shared kernels are expressed in the architecture.

Factory entry points: ``make_basis`` and ``make_psf``
-----------------------------------------------------

The factories :func:`make_basis` and :func:`make_psf` are the primary
user-facing entry points. They take *single-sample* functions written in
natural Python / :mod:`jax.numpy` and turn them into full :class:`Basis` or
:class:`PSF` objects with explicit metadata.

Both take a *single-sample* function plus declared metadata (``dim``,
``rank``, ``n_features``, ``labels``, parameter specs) and return the
corresponding rich object; worked examples live in
:doc:`/statefunc/user_guide`.  In both cases:

* the function is written for a *single* sample; SFI handles batching over time
  and particles;
* ``mask`` is an optional keyword that most users can ignore;
* metadata (dimension, rank, particle depth, feature labels, extras keys,
  parameter spec) is declared once at construction and then enforced.

Single-feature PSFs are common and natural: they represent a single parametric
field or force. Bases, in contrast, are typically multi-feature dictionaries
used for linear combinations; if you conceptually have a single feature with a
coefficient, a PSF is usually a better fit than a single-feature Basis.

Linear and nonlinear inference: Basis vs PSF
--------------------------------------------

The distinction between :class:`Basis` and :class:`PSF` reflects the two main
inference regimes in SFI:

* **Linear inference** uses :class:`Basis` objects. Given a basis matrix and
  data, the inference module solves the linear regression problem exactly (or
  via sparse / regularised variants) to obtain optimal coefficients.
* **Nonlinear inference** uses :class:`PSF` objects. Parameters are updated
  through nonlinear optimisers (gradient descent, quasi-Newton, etc.), using
  derivatives with respect to parameters exposed by the PSF.

Both are important:

* linear inference is fast, robust, and often sufficient when a rich basis is
  available;
* nonlinear PSFs are needed for models with genuinely nonlinear parameter
  dependencies or shared parameters across many features.

Once parameters are inferred, a PSF can be frozen into an :class:`SF` for use
in simulation and downstream analysis.

Interacting particle systems (optional)
---------------------------------------

For systems of many interacting particles (or lattice sites), SFI takes a
“local rule + dispatcher” approach:

1. A local :class:`Interactor` describes how a small group of particles
   (typically a pair) contributes to the quantity of interest (force, torque,
   current, …). It operates on a local configuration ``Xk`` with shape
   ``(K, dim)`` and optional extras and parameters.
2. Calling ``Interactor.dispatch(...)`` turns this local rule into a global
   state function (Basis/PSF/SF), specifying:
   * how to select and combine the K-tuples (all pairs, finite range, given
   neighbor list, …);
   * how to reduce them (sum, average, owner conventions).

The underlying dispatcher and interaction-node machinery lives in backend
modules; :class:`Interactor` and its ``.dispatch`` method are the main
front-end objects. Conceptually, this design is close to graph neural
networks: the interactor plays the role of a message function on edges, and
the dispatcher specifies the graph and aggregation on nodes.

Users who do not work with interacting-particle or spatial systems can skip
this part; the rest of :mod:`statefunc` is sufficient for single-particle or
well-mixed models.

Particles input vs particle depth
---------------------------------

Two related but distinct concepts control how particle axes are handled:

* ``particles_input`` indicates whether the state function expects a particle
  axis in its input. Many single-particle bricks operate on a single state
  vector and are vmapped over particles by the surrounding code, while
  interactors expect a fixed number ``K`` of particles as input.
* ``pdepth`` counts how many particle axes the *output* carries (0, 1 or 2).
  A single-particle force typically has ``pdepth = 1``; a purely local scalar
  kernel might have ``pdepth = 0``; a pairwise matrix could have ``pdepth = 2``.

The contract for outputs enforces:

* if ``particles_input=True``, an input ``x`` with prefix
  ``batch · P_in · dim`` must produce outputs whose prefix is
  ``batch · P_in^pdepth`` (i.e. one particle axis per depth level);
* if ``particles_input=False``, a node with ``pdepth=0`` preserves the batch
  prefix of ``x``, and other combinations are rejected at runtime.

This ensures that the same operators (concatenation, derivatives, etc.) work
uniformly across single-particle functions, multi-particle interactions, and
spatial fields.

Symmetries and tensor structure
-------------------------------

A key motivation for this architecture is precise control over symmetries:

* scalar features can encode radial dependence, invariants, or other symmetry
  constraints;
* unit vectors and projectors can be composed into tensors with prescribed
  symmetries (e.g. longitudinal vs transverse components, radial vs tangential
  directions);
* dimension-dependent behaviour is explicit: basis functions can differ
  between components, not just be copies of the same scalar applied to each
  dimension.

This contrasts with earlier SFI architectures, where basis functions were
often implicitly “the same for all dimensions”. In the new design, the
structure of forces and diffusion tensors is expressed directly at the level
of state functions, making symmetries transparent and easier to enforce.

Feature axis and ``drop_features``
----------------------------------

All state functions place the feature axis last. Some operations can “drop”
features and return a featureless tensor (for example, turning a single-feature
PSF into an :class:`SF` that represents a single field rather than a
dictionary).

To avoid ambiguity:

* operations that would need to drop more than one feature
  (``n_features > 1``) are forbidden and raise an error;
* single-feature PSFs are the intended target for such operations;
* Bases are typically multi-feature; users are expected to select individual
  features explicitly before applying operations that drop the feature axis.

Backend architecture: leaves, nodes, operations
-----------------------------------------------

Internally, :mod:`statefunc` organises computations as a tree of nodes:

* leaf nodes implement primitive bricks (single-sample functions, interactors,
  parameter blocks);
* operation nodes implement composition (concat, map, einsum, derivatives,
  slicing, interaction dispatch, …);
* every node inherits the same static contract and participates in memory
  accounting.

This node-level backend (including the interaction dispatchers) is considered
an implementation detail. The front-end abstractions (:class:`Basis`,
:class:`PSF`, :class:`SF`, :class:`Interactor`, and the factory functions) are
the stable public surface.

Performance and memory-aware evaluation
---------------------------------------

Each node also contributes to a global estimate of memory footprint. This
allows the integration layer (:mod:`SFI.integrate`) to:

* plan vectorised evaluations over time and particles;
* adaptively chunk computations to stay within a target memory budget.

This memory-aware design is what lets large systems (many particles, long
trajectories, rich bases) be evaluated and differentiated efficiently without
manual chunking logic in user code.

.. seealso::

   The :doc:`/gallery/index` demo gallery and the
   :doc:`/inference/user_guide` section showcase the evaluation
   engine on concrete inference problems.
