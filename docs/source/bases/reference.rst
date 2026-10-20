.. _bases-reference:

Bases API
=========

.. currentmodule:: SFI.bases

This page documents the public API of the :mod:`SFI.bases` package.
For usage examples, the construction decision tree, canonical patterns
(Lorenz, harmonic UD, limit cycle, double-well), and guidance on
choosing a basis, see :doc:`/bases/user_guide`.

Reference card
--------------

All builders are importable from :mod:`SFI.bases` directly
(``from SFI.bases import monomials_up_to``):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Builder
     - One-liner
   * - ``monomials_up_to(order, dim, rank=...)``
     - Polynomials up to total order; ``rank`` lifts to vector/tensor
   * - ``monomials_degree(degree, dim, ...)``
     - Polynomials of exact total degree
   * - ``ones_basis(dim)``
     - Scalar constant 1
   * - ``unit_vector_basis(dim, axes)``
     - Cartesian unit vectors (rank-1)
   * - ``identity_matrix_basis(dim)``
     - Isotropic :math:`\delta_{ij}` (rank-2)
   * - ``symmetric_matrix_basis(dim)``
     - All symmetric unit matrices (rank-2)
   * - ``constant_array(A)``
     - Wrap a constant array as one feature
   * - ``X(dim)`` / ``V(dim)``
     - The position / velocity vector itself (rank-1)
   * - ``x_coordinate(i, dim)``
     - Single coordinate :math:`x_i` (rank-0)
   * - ``linear_basis(dim)``
     - Coordinate-extraction identity
   * - :func:`~SFI.bases.x_components` / :func:`~SFI.bases.v_components` / :func:`~SFI.bases.unit_axes` / ``frame``
     - Compositional per-axis primitives (see :doc:`/bases/user_guide`)
   * - ``named_scalar(s)``
     - Parametric scalar PSFs with optional defaults
   * - ``SFI.bases.pairs.*``
     - Pair-interaction builders and kernel families (below)
   * - ``SFI.bases.spde.*``
     - PDE differential operators (:doc:`/spde/reference`, experimental)


Two ways to build a basis
-------------------------

Both yield a :class:`~SFI.statefunc.Basis` and are freely combinable
with ``+``, ``*``, and ``&`` (see :doc:`/bases/user_guide`).

* **Factory path** (this page).  Pre-built builders return a complete
  :class:`Basis` in one call: :func:`monomials_up_to`,
  :func:`monomials_degree`, :func:`linear_basis`, the kernel families
  under :mod:`SFI.bases.pairs`, and the SPDE operators under
  :mod:`SFI.bases.spde`.  Right tool when the structure of the basis is
  already standard (polynomials, pair potentials, finite-difference
  operators).

* **Compositional path** — primitives :func:`~SFI.bases.x_components`, :func:`~SFI.bases.v_components`,
  :func:`~SFI.bases.unit_axes`, ``frame``, ``named_scalar(s)`` plus the algebra
  operators ``+`` / ``*`` / ``&``.  Right tool when the basis follows
  the symmetry of the problem rather than a generic family
  (e.g. ``ex * x[0] + ey * x[1]`` for a 2D radial force).


Monomials
---------

.. autofunction:: SFI.bases.monomials_up_to
   :no-index:
.. autofunction:: SFI.bases.monomials_degree
   :no-index:


Structural constants
--------------------

.. autofunction:: SFI.bases.ones_basis
   :no-index:
.. autofunction:: SFI.bases.unit_vector_basis
   :no-index:
.. autofunction:: SFI.bases.identity_matrix_basis
   :no-index:
.. autofunction:: SFI.bases.symmetric_matrix_basis
   :no-index:
.. autofunction:: SFI.bases.constant_array
   :no-index:


Linear shorthand
----------------

.. autofunction:: SFI.bases.X
   :no-index:
.. autofunction:: SFI.bases.V
   :no-index:
.. autofunction:: SFI.bases.x_coordinate
   :no-index:
.. autofunction:: SFI.bases.linear_basis
   :no-index:


Pair-interaction bases
----------------------

.. autofunction:: SFI.bases.pairs.scalar_pair_basis
   :no-index:
.. autofunction:: SFI.bases.pairs.radial_pair_basis
   :no-index:
.. autofunction:: SFI.bases.pairs.dyadic_pair_basis
   :no-index:
.. autofunction:: SFI.bases.pairs.angular_pair_basis
   :no-index:

Kernel families
~~~~~~~~~~~~~~~

Each family parametrises a different radial profile for pair potentials.
Choose the one that best matches the expected interaction range and
short-distance behaviour:

* :func:`~SFI.bases.pairs.gaussian_kernels` — smooth bumps
  :math:`\phi_\sigma(r) = \exp(-r^2/2\sigma^2)`; good general-purpose
  default for soft, finite-range interactions.
* :func:`~SFI.bases.pairs.exp_poly_kernels` —
  :math:`\phi_{k,L}(r) = r^k \exp(-r/L)`; exponentially-decaying with a
  polynomial prefactor (e.g. screened forces).
* :func:`~SFI.bases.pairs.power_kernels` — pure power laws
  :math:`\phi_k(r) = r^k` (no decay); use with care, requires a finite
  neighbour cutoff.
* :func:`~SFI.bases.pairs.compact_kernels` — compactly supported
  :math:`r^k (1 - r/r_c)^2` for :math:`r < r_c`; identically zero past
  ``r_c``, ideal with a :term:`neighbour list`.

.. autofunction:: SFI.bases.pairs.gaussian_kernels
   :no-index:
.. autofunction:: SFI.bases.pairs.exp_poly_kernels
   :no-index:
.. autofunction:: SFI.bases.pairs.power_kernels
   :no-index:
.. autofunction:: SFI.bases.pairs.compact_kernels
   :no-index:

Pair utilities
~~~~~~~~~~~~~~

.. autofunction:: SFI.bases.pairs.heading_vector
   :no-index:
.. autofunction:: SFI.bases.pairs.pbc_displacement
   :no-index:
.. autofunction:: SFI.bases.pairs.wrap_angle
   :no-index:


SPDE operators
--------------

Composable spatial differential operators for grid-based field inference
are documented in a dedicated section:

.. seealso::

   :doc:`/spde/reference` — Laplacian, Gradient, Biharmonic and more.
   :doc:`/spde/user_guide` — tutorial and concepts.
