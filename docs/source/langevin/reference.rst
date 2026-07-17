.. _langevin-reference:

Simulation API
==============

.. currentmodule:: SFI.langevin

For a narrative introduction and workflow examples, see
:doc:`/langevin/user_guide`.

.. autoclass:: SFI.langevin.OverdampedProcess
   :members: initialize, set_params, set_extras, simulate
   :show-inheritance:
   :no-index:

.. autoclass:: SFI.langevin.UnderdampedProcess
   :members: initialize, set_params, set_extras, simulate
   :show-inheritance:
   :no-index:


Noise models
------------

.. currentmodule:: SFI.langevin.noise

.. autoclass:: WhiteNoise
   :members: sample, effective_D_per_site
   :show-inheritance:
   :no-index:

.. autoclass:: ConservedNoise
   :members: sample, effective_D_per_site
   :show-inheritance:
   :no-index:

.. autoclass:: CompositeNoise
   :members: sample
   :show-inheritance:
   :no-index:

.. seealso::

   :doc:`/spde/user_guide` — conserved and composite noise for SPDE
   simulations.
