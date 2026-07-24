.. _spde-reference:

SPDE API reference
==================

.. currentmodule:: SFI.bases.spde

This page documents the public API for grid-based SPDE operators and
helpers.  For narrative examples, see :doc:`/spde/user_guide`.


Grid extras
-----------

.. autofunction:: square_grid_extras
   :no-index:


Composable operators
--------------------

.. autoclass:: StencilOp
   :members: __call__, __repr__, visualize_stencil
   :show-inheritance:
   :no-index:

Scalar operators
~~~~~~~~~~~~~~~~

.. autoclass:: Laplacian
   :show-inheritance:
   :no-index:

.. autoclass:: Biharmonic
   :show-inheritance:
   :no-index:

.. autoclass:: LaplacianOfGradientSquared
   :show-inheritance:
   :no-index:

Vector operators
~~~~~~~~~~~~~~~~

.. autoclass:: Gradient
   :show-inheritance:
   :no-index:

.. autoclass:: Divergence
   :show-inheritance:
   :no-index:

.. autoclass:: Curl
   :show-inheritance:
   :no-index:

.. autoclass:: SymGrad
   :show-inheritance:
   :no-index:

.. autoclass:: SkewGrad
   :show-inheritance:
   :no-index:

Advection
~~~~~~~~~

.. autoclass:: AdvectionBy
   :show-inheritance:
   :no-index:

Convenience functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vector_field
   :no-index:

Stencil composition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: minkowski_sum_offsets
   :no-index:


Noise models
------------

.. currentmodule:: SFI.langevin.noise

.. autoclass:: ConservedNoise
   :members: sample, effective_D_per_site
   :show-inheritance:
   :no-index:

.. autoclass:: CompositeNoise
   :members: sample
   :show-inheritance:
   :no-index:

Factory
~~~~~~~

.. currentmodule:: SFI.bases.spde

.. autofunction:: conserved_noise_pbc
   :no-index:
