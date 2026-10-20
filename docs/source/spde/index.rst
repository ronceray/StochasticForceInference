.. _spde-index:

SPDE (spatial fields) — experimental
====================================

SFI extends stochastic force inference to **spatial field data** governed
by stochastic partial differential equations (SPDEs).  The SPDE toolbox
provides composable finite-difference operators, grid-aware noise models,
and integration with the standard SFI inference pipeline.

.. admonition:: Experimental
   :class: warning

   The SPDE toolbox is functional but **experimental** in this release:
   its API may change, and only the linear estimators
   (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`, :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`) are validated on
   grid layouts.

.. toctree::
   :maxdepth: 2

   user_guide
   layout_guide
