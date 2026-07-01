.. _statefunc-reference:

State function API
==================

.. currentmodule:: SFI.statefunc

This page is a curated map of the public API in :mod:`SFI.statefunc`.  It
summarises the main classes and functions and links to the full
auto-generated API documentation.

For a conceptual introduction, see :doc:`/statefunc/overview` and
:doc:`/statefunc/user_guide`.  For canonical recipes (Lorenz, harmonic
underdamped, limit cycle, double-well) see
:doc:`/bases/user_guide`.

What is a state function?
-------------------------

The package is built on three closely related abstractions:

.. list-table::
   :widths: 18 40 42
   :header-rows: 1

   * - Object
     - Purpose
     - Use when…
   * - :class:`Basis`
     - Dictionary of parameter-free vector features
       :math:`\{\phi_\alpha(x)\}` used in linear inference.
     - You want closed-form regression: :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear` /
       :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`.
   * - :class:`PSF`
     - Parametric family :math:`F(x; \theta)` — linear or nonlinear in
       :math:`\theta`.
     - You want parametric inference (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`), or a typed
       handle for an MLP / gated drift.
   * - :class:`SF`
     - Fitted state function with frozen parameters; callable on data.
     - You want to evaluate / simulate / serialise a result, or define a
       ground-truth field for an :mod:`SFI.langevin` process.

The factories below are the three corresponding entry points.

Mini-examples
~~~~~~~~~~~~~

.. code-block:: python

   from SFI import make_sf
   from SFI.statefunc import make_basis, make_psf

   # rank is an int: 0 = scalar, 1 = vector, 2 = matrix

   # Basis — used by linear inference
   B = make_basis(lambda x: x, dim=2, rank=1)

   # PSF — used by parametric / nonlinear inference.
   # The parameters arrive in a keyword named ``params``; declare each
   # block by shape (``()`` = scalar).
   def spring(x, *, params):
       return -params["k"] * x

   F_psf = make_psf(spring, dim=2, rank=1, n_features=1, params={"k": ()})

   # SF — frozen ground truth for a simulator
   F_sf = make_sf(lambda x: -1.33 * x, dim=2, rank=1)

Core classes
------------

These are the main user-facing objects:

* :class:`Basis` –
  dictionary of parameter-free features used in linear inference.
* :class:`PSF` –
  parametric state-function family :math:`F(x; \theta)` used in nonlinear
  inference and as a bridge to simulation.
* :class:`SF` –
  state function with fixed parameters, used for evaluation and Langevin
  processes.
* :class:`StateExpr` –
  underlying expression graph; most users see it indirectly through Basis/PSF/SF.
* :class:`Interactor` –
  local K-body interaction rule, dispatched to build global state functions.

Factories and helpers
---------------------

High-level constructors:

* :func:`make_basis` –
  wrap a single-sample function into a :class:`Basis`.
* :func:`make_sf` –
  wrap a single-sample callable into a fixed-parameter :class:`SF`,
  typically to define a ground-truth force or diffusion field for a
  :mod:`SFI.langevin` simulator.
* :func:`make_psf` –
  wrap a single-sample function with parameters into a :class:`PSF`.
* :func:`make_interactor` –
  build an :class:`Interactor` from a local K-body rule.

Parameter handling:

* :class:`ParamSpec` –
  describes a single named parameter block (shape, dtype, init).
* :class:`ParamSuite` –
  immutable container of parameter specs, used internally by PSFs.

Execution, differentiation, and memory
--------------------------------------

Common methods shared by :class:`Basis`, :class:`PSF`, and :class:`SF`:

* :meth:`StateExpr.__call__` –
  evaluate on trajectories; usually accessed via Basis/PSF/SF.
* :meth:`PSF.d_x` / :meth:`SF.d_x` –
  derivative with respect to state coordinates.
* :meth:`PSF.d_v` / :meth:`SF.d_v` –
  derivative with respect to velocities (for underdamped models).
* :meth:`PSF.d_theta` –
  derivative with respect to parameters :math:`\theta`.

Runtime / performance controls:

* :func:`set_jit` –
  enable/disable JIT compilation of Basis/PSF/SF calls.
* :mod:`SFI.statefunc.memhint` (``MemHint``, ``SampleMeta``) –
  internal memory-usage estimates used by :mod:`SFI.integrate` for
  adaptive chunking.

Interacting particle systems (optional)
---------------------------------------

These are only relevant if you work with interacting particles or lattice
fields. For a high-level explanation see the “Interacting particle systems”
section in :ref:`statefunc-overview`.

Front-end:

* :class:`Interactor` –
  defines a local K-body rule on ``Xk``.
* :meth:`Interactor.dispatch` –
  apply the rule over a neighbour graph to build a global Basis/PSF/SF.

Back-end interaction graph specs (expert use, stable subset only):

* :mod:`SFI.statefunc.nodes.interactions` –
  interaction graph specifications and dispatchers.

Layout and sectors
------------------

For building structured multi-field models on regular grids or with
heterogeneous tensor components.  See :doc:`/spde/layout_guide`
for the user guide (part of the experimental SPDE toolbox).

.. currentmodule:: SFI.statefunc.layout

* :class:`GridLayout` –
  declares named sectors on a regular grid; provides differential operators
  and ``embed()`` compilation.
* :class:`ScalarSector` –
  a scalar field (one index per grid site).
* :class:`VectorSector` –
  a vector field (``sdim`` indices per grid site).
* :class:`SymTensorSector` –
  a symmetric rank-2 tensor (Voigt-packed).
* :class:`TensorSector` –
  a general rank-2 tensor.
* :class:`StateLayout` –
  protocol shared by all layout types.

.. currentmodule:: SFI.statefunc.structexpr

* :class:`StructuredExpr` –
  symbolic expression node in the *inner world* of layouts; supports
  ``+``, ``*``, ``&``, ``dot``, ``einsum``, ``stack``, ``eye``.

.. currentmodule:: SFI.statefunc

Contracts
---------

Every node carries the same static *contract* — shape and requirement
metadata (rank, dim, pdepth, n_features, required extras keys, …) — which
is what lets nodes be composed safely. See :doc:`/statefunc/overview` for
the meaning of each field. The one public handle you touch directly is:

* :class:`Rank` –
  enumeration of tensor ranks (SCALAR, VECTOR, MATRIX, …).

Full API
--------

For complete signatures and member lists, run ``SFI_DOCS_RUN_APIDOC=1 make html``
to generate the full auto-documented API from source.
