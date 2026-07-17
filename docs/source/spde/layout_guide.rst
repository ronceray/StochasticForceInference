.. _layout-guide:

Structured fields: Layout, Sectors, and Embed
===============================================

.. currentmodule:: SFI.statefunc.layout

Many physical systems have **internal structure** beyond a flat
state vector — grid fields with spatial derivatives, Q-tensors with
nematic symmetry, or particles with distinct position and orientation
degrees of freedom.  The **Layout / Sector / Embed** paradigm lets you
declare this structure once, then compose differential operators and
pointwise algebra in a *symbolic inner world* before compiling to an
inference-ready :class:`~SFI.statefunc.Basis`.


Why layouts?
------------

Standard SFI bases work on flat state vectors :math:`\mathbf{x} \in \mathbb{R}^d`.
But for a reaction–diffusion PDE on a 64×64 grid with two fields, the
state is :math:`\mathbf{x} \in \mathbb{R}^{4096 \times 2}`, and the force
depends on **spatial derivatives** (Laplacian, gradient) that couple
neighbouring grid sites.  Building the correct stencil-based basis by hand
is tedious and error-prone.

Layouts solve this by providing:

1. **Sectors** that name each field and declare its tensor type
   (scalar, vector, symmetric tensor, …).
2. **Differential operators** (``lap``, ``grad``, ``div``, …) that act on
   symbolic expressions and automatically compute the correct stencil
   footprints.
3. **Pointwise algebra** (``+``, ``*``, ``&``, ``einsum``, …) that
   composes freely with differential operators.
4. **Embed**, the compilation boundary that produces an outer-world
   :class:`~SFI.statefunc.Basis` (or :class:`~SFI.statefunc.PSF`) ready for SFI's inference engine.


The two-world architecture
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Inner world (symbolic)
     - Outer world (inference)
   * - Objects
     - :class:`~SFI.statefunc.structexpr.StructuredExpr`
     - :class:`~SFI.statefunc.Basis` / :class:`~SFI.statefunc.PSF`
   * - Indexing
     - ``sdims`` = per-axis structured dimension sizes
     - ``n_features`` × ``dim``
   * - Composition
     - ``+``, ``*``, ``&``, ``einsum``, ``dot``, ``stack``, ``eye``
     - ``&`` (concat), ``*`` (product), ``[]`` (slice)
   * - Boundary
     - —
     - ``layout.embed(rank=1, ...)``

Users build expressions in the inner world using layout methods and
algebra, then call :meth:`GridLayout.embed` once to cross the boundary.


Sectors
-------

A **sector** names a group of state-vector indices and declares their
tensor structure:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Sector type
     - ``sdims``
     - Use case
   * - :class:`ScalarSector`
     - ``()``
     - A single scalar field (concentration, phase, …)
   * - :class:`VectorSector`
     - ``(sdim,)``
     - A vector field (velocity, displacement, …).
       Set ``spatial=True`` if the vector lives in the grid's physical space
       (enables ``div``, ``curl``).
   * - :class:`SymTensorSector`
     - ``(sdim, sdim)``
     - A symmetric tensor (Q-tensor, strain, stress, …).
       Only the independent Voigt components are stored; the full tensor
       is reconstructed automatically.
       Set ``traceless=True`` for traceless tensors (e.g. 2D nematic
       Q-tensor: stores 2 components, reconstructs 3).
   * - :class:`TensorSector`
     - ``(sdim, sdim)``
     - A general (non-symmetric) rank-2 tensor.

Each sector maps to a contiguous range of columns in the flat
state vector.  For example, a traceless symmetric 2D Q-tensor with
``indices=[0, 1]`` stores :math:`Q_{xx}` at column 0 and :math:`Q_{xy}`
at column 1, while :math:`Q_{yy} = -Q_{xx}` is reconstructed on the fly.


GridLayout: differential operators on regular grids
----------------------------------------------------

:class:`GridLayout` is the main layout class for spatial
finite-difference models.  It takes named sectors and grid parameters:

.. code-block:: python

   from SFI.statefunc.layout import GridLayout, ScalarSector

   layout = GridLayout(
       U=ScalarSector([0]),
       V=ScalarSector([1]),
       dim=2, ndim=2, bc="pbc",
   )
   U = layout.U   # symbolic StructuredExpr for the U field
   V = layout.V   # symbolic StructuredExpr for the V field

Parameters:

- ``dim`` — total number of field components per grid site (= sum of all
  sector sizes).
- ``ndim`` — number of spatial dimensions (2 for a 2D grid).
- ``bc`` — boundary condition: ``"pbc"`` (periodic) or ``"noflux"``.
- ``**sectors`` — keyword arguments naming each sector.

Once the layout is constructed, each sector is available as a named
attribute (``layout.U``, ``layout.V``, …) returning a
:class:`~SFI.statefunc.structexpr.StructuredExpr`.


Differential operators
~~~~~~~~~~~~~~~~~~~~~~

All operators return symbolic :class:`~SFI.statefunc.structexpr.StructuredExpr`
objects — no numerical computation happens until ``embed()`` is called.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Method
     - sdims transform
     - Description
   * - ``layout.grad(expr)``
     - ``S → S + (ndim,)``
     - Spatial gradient (2nd-order central FD)
   * - ``layout.lap(expr)``
     - ``S → S``
     - Laplacian ∇² (rank-preserving)
   * - ``layout.div(expr)``
     - ``(*S, ndim) → S``
     - Divergence (contracts last axis)
   * - ``layout.biharmonic(expr)``
     - ``S → S``
     - Biharmonic ∇⁴ (13-point stencil in 2D)
   * - ``layout.lap_of_grad_sq(expr)``
     - ``S → S``  (scalar input)
     - :math:`\nabla^2(|\nabla f|^2)` — Active Model B+ term
   * - ``layout.advection_by(v, φ)``
     - sdims of ``φ``
     - :math:`(\mathbf{v}\cdot\nabla)\phi`
   * - ``layout.strain_rate(v)``
     - ``(ndim,) → (ndim, ndim)``
     - Symmetric strain rate :math:`(\nabla v + \nabla v^T)/2`
   * - ``layout.vorticity(v)``
     - ``(ndim,) → (ndim, ndim)``
     - Vorticity :math:`(\nabla v - \nabla v^T)/2`


Pointwise algebra
~~~~~~~~~~~~~~~~~

Symbolic expressions support rich algebra *before* embedding:

- **Addition / subtraction**: ``expr1 + expr2``, ``expr1 - expr2``
  (requires matching sdims).
- **Scalar multiplication**: ``2.0 * expr`` or ``expr * scalar_expr``
  (broadcasts scalars to any tensor shape).
- **Feature concatenation**: ``expr1 & expr2`` (requires matching sdims;
  concatenates along the feature axis).
- **Contraction**: ``expr1.dot(expr2)`` — contracts the last axis of each
  operand (like a batched inner product).
- **Einsum**: ``StructuredExpr.einsum("ij,jk->ik", A, B)`` — arbitrary
  index contractions, following NumPy ``einsum`` conventions.
- **Stack**: ``StructuredExpr.stack([e1, e2], sdim=2)`` — build a vector
  from scalars by stacking along a new leading axis.
- **Identity**: ``StructuredExpr.eye(sdim, layout=...)`` — the Kronecker
  delta :math:`\delta_{ij}`.
- **Negation**: ``-expr``.


Compiling to a Basis: ``embed()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`GridLayout.embed` crosses from the inner world to the outer world.
Each sector's expression tree is compiled into stencil-based evaluation
functions:

.. code-block:: python

   BASIS = layout.embed(rank=1, U=U_force, V=V_force)

- ``rank=1`` for force models (rank-1 output = vector per site).
- Keyword arguments map **sector names** to their expression trees.
- Features are ordered: all features of the first sector, then all features
  of the second sector, etc.

The resulting :class:`~SFI.statefunc.Basis` is a standard SFI state function — pass it to
``inf.infer_force_linear(BASIS, ...)`` as usual.


Example: Gray-Scott reaction-diffusion
----------------------------------------

A reaction-diffusion system with two scalar fields on a 2D periodic grid:

.. code-block:: python

   from SFI.statefunc.layout import GridLayout, ScalarSector

   layout = GridLayout(
       U=ScalarSector([0]),
       V=ScalarSector([1]),
       dim=2, ndim=2, bc="pbc",
   )
   U = layout.U
   V = layout.V
   UVV = U * V * V

   # 7-term basis: 4 for U, 3 for V
   U_force = (U * 0 + 1) & U & UVV & layout.lap(U)
   V_force = UVV & V & layout.lap(V)
   BASIS = layout.embed(rank=1, U=U_force, V=V_force)

   # theta order follows sector order: U features first, then V
   theta = jnp.array([F, -F, -1.0, DU, +1.0, -(F+K), DV])

The stencil composition engine automatically determines that
``layout.lap(U)`` requires a 5-point cross stencil, while pointwise
terms like ``U * V * V`` need only the local site.  The compiled
:class:`~SFI.statefunc.Basis` gathers the minimal neighbour set for each feature.

.. seealso:: :ref:`sphx_glr_gallery_gray_scott_demo.py`


Example: active nematic Q-tensor
----------------------------------

A traceless symmetric Q-tensor field with active self-advection:

.. code-block:: python

   from SFI.statefunc.layout import GridLayout, SymTensorSector
   from SFI.statefunc.structexpr import StructuredExpr

   layout = GridLayout(
       Q=SymTensorSector([0, 1], sdim=2, traceless=True),
       dim=2, ndim=2, bc="pbc",
   )
   Q = layout.Q  # sdims=(2, 2) — full Q tensor

   # tr(Q²) = Q_ij Q_ij — scalar contraction
   S2 = StructuredExpr.einsum("ij,ij->", Q, Q)

   # Active self-advection: (div Q · ∇) Q
   v_active = layout.div(Q)                       # sdims=(2,)
   grad_Q = layout.grad(Q)                         # sdims=(2,2,2)
   advect = StructuredExpr.einsum("ijk,k->ij",
                                   grad_Q, v_active)  # sdims=(2,2)

   Q_force = layout.lap(Q) & Q & (S2 * Q) & advect
   BASIS = layout.embed(rank=1, Q=Q_force)

   # Only 4 coefficients — nematic symmetry is structural
   theta = jnp.array([K, a, -b, zeta])

Because the Q-tensor is a single ``SymTensorSector``, the same
coefficient multiplies both :math:`Q_{xx}` and :math:`Q_{xy}` — nematic
symmetry is enforced by construction, giving 4 coefficients instead of 8.


Symmetry-based layouts beyond SPDEs
-------------------------------------

The sector/embed paradigm is not limited to spatial PDEs.  Any system
where the state vector has **heterogeneous structure** — distinct
physical roles for different components — benefits from layouts.

Active Brownian particles
~~~~~~~~~~~~~~~~~~~~~~~~~

Each particle has position :math:`(x, y)` and heading angle :math:`\theta`.
The ABP force is naturally expressed with sectors:

.. code-block:: python

   from SFI.statefunc.layout import GridLayout, VectorSector, ScalarSector

   # Position (x, y) = vector sector; angle θ = scalar sector
   layout = GridLayout(
       pos=VectorSector([0, 1], sdim=2, spatial=True),
       angle=ScalarSector([2]),
       dim=3, ndim=2, bc="pbc",
   )
   pos = layout.pos     # sdims=(2,) — position vector
   angle = layout.angle  # sdims=() — heading angle

This declares that columns 0–1 are a spatial 2-vector and column 2 is a
scalar angle.  A model can then use ``layout.grad(...)`` for spatial
derivatives (e.g. density-dependent propulsion) while keeping the angle
sector separate.

Coupled vector–scalar systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any problem mixing fields of different tensor types fits naturally:

- **Viscoelastic fluids**: velocity ``VectorSector`` + conformation tensor
  ``SymTensorSector``
- **Magnetohydrodynamics**: velocity + magnetic field (two ``VectorSector``
  entries)
- **Phase-field crystals**: density ``ScalarSector`` + orientation
  ``ScalarSector``

The key benefit is that ``embed()`` respects the tensor structure of each
sector: a vector basis function produces a vector force, and coefficients
are correctly shared across tensor components.


The stencil composition engine
-------------------------------

When you compose differential operators (e.g. ``layout.lap(φ**3)`` or
``einsum(grad(Q), div(Q))``), the engine must determine the **minimal set
of neighbour offsets** needed to evaluate the expression at a given grid
point.

The rules are:

- **Pointwise** nodes (``+``, ``*``, ``einsum``, ``stack``, ``&``) use the
  **union** of their children's footprints — they only combine values that
  are already available.
- **Differential operator** nodes (``grad``, ``lap``, ``div``, …) use the
  **Minkowski sum** of the operator's template with the child's footprint —
  they need to evaluate the child at shifted grid points.

For example:

- ``lap(φ)`` needs the cross stencil (5 points in 2D).
- ``lap(φ**3)`` also needs only the cross — ``φ**3`` is pointwise.
- ``einsum(grad(Q), div(Q))`` needs the cross ∪ cross = cross (5 points).
- ``biharmonic(φ) = lap(lap(φ))`` needs the biharmonic stencil (13 points).

This automatic footprint computation means you can freely compose
operators without worrying about stencil sizes or double-counting.


Extras and grid setup
----------------------

Grid-based layouts need **extras** that describe the grid geometry at
runtime (grid shape and spacing).  These are provided by
:func:`~SFI.bases.spde.square_grid_extras`:

.. code-block:: python

   from SFI.bases.spde import square_grid_extras

   extras = square_grid_extras(grid_shape=(64, 64), dx=1.0)

   proc.set_extras(extras_global=extras)

For conserved (divergence-form) noise, use
:func:`~SFI.bases.spde.conserved_noise_pbc`:

.. code-block:: python

   from SFI.bases.spde import conserved_noise_pbc

   noise = conserved_noise_pbc(sigma=0.3, grid_shape=(64, 64), dx=1.0, n_fields=1)
   proc = OverdampedProcess(BASIS, D=noise)


.. seealso::

   - :doc:`/statefunc/user_guide` — state function basics and model
     composition recipes (Basis, PSF, SF)
   - :doc:`/bases/user_guide` — pre-built bases (polynomials, pairs, …)
     and guidance on which basis to use for your problem
   - :ref:`sphx_glr_gallery_gray_scott_demo.py` — full Gray-Scott example
