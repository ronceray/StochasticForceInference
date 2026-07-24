Inference API
=============

What this module does
---------------------

:mod:`SFI.inference` exposes the engines that take a
:class:`~SFI.trajectory.TrajectoryCollection` and return a fitted force
and (optionally) diffusion model.  Every inference path starts the same
way — instantiate the engine with the collection — and then chooses one
of the three paths below.

Two estimator families
----------------------

The choice depends on time-step size, measurement noise, and how
non-linear the model is in its parameters.  Pick *one* per task;
:doc:`/inference/user_guide` has the regime table.

* **Linear estimators** (fast, closed-form). Methods
  :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant`,
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`,
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`.
  Closed-form projection onto a :class:`~SFI.statefunc.Basis`.  Use when
  :math:`\Delta t` is small and measurement noise is negligible.

* **Parametric estimators** (robust to measurement noise and finite
  :math:`\Delta t`). Methods
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` and
  :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`.
  Takes a :class:`~SFI.statefunc.Basis` or a
  :class:`~SFI.statefunc.PSF` (parametric family) and fits it by a
  flow-residual likelihood: one or more :term:`RK4` flow
  steps per observation interval, a :term:`local-precision NLL`, and
  native :math:`(D, \Lambda)` profiling.  Two inner solvers,
  selected automatically: ``inner="gn"`` (direct bandwidth-limited
  :term:`Gauss–Newton` with the skip-trick errors-in-variables
  instrument — the linear-in-:math:`\theta` fast path) and
  ``inner="lbfgs"`` (frozen-precision :term:`L-BFGS`, for
  non-linear-in-:math:`\theta` families such as neural-net drifts).

Linear-estimator sequence, using :term:`PASTIS` for sparse selection:

.. code-block:: python

   from SFI import OverdampedLangevinInference
   from SFI.bases import monomials_up_to

   inf = OverdampedLangevinInference(collection)
   inf.compute_diffusion_constant(method="auto")          # 1. constant D
   inf.infer_force_linear(monomials_up_to(2, dim=2, rank="vector"),
                          M_mode="Strato")                # 2. force regression
   inf.compute_force_error()                              # 3. error analysis
   inf.sparsify_force(criterion="PASTIS", p=0.1)          # 4. optional sparsity
   inf.print_report()                                     # 5. summary

Parametric-estimator sequence (:class:`~SFI.statefunc.Basis` or :class:`~SFI.statefunc.PSF`; profiles
:math:`(D, \Lambda)` natively):

.. code-block:: python

   inf = OverdampedLangevinInference(collection)
   inf.infer_force(monomials_up_to(2, dim=2, rank="vector"))
   inf.infer_diffusion()                # no-arg: symmetric-matrix basis
   inf.compute_force_error()
   inf.print_report()

.. seealso::

   :doc:`/inference/user_guide` for choosing between the linear and
   parametric estimators;
   :doc:`/physics_reference` for SDE notation and conventions;
   :doc:`/bases/user_guide` for how to build the basis fed to
   :meth:`infer_force_linear`;
   :doc:`/gallery/ou_demo`, :doc:`/gallery/lorenz_demo`,
   :doc:`/gallery/advanced/nn_force_demo` — one worked example per path.


Inference engines
-----------------

.. autosummary::
   :nosignatures:

   ~SFI.inference.OverdampedLangevinInference
   ~SFI.inference.UnderdampedLangevinInference

.. autoclass:: SFI.inference.OverdampedLangevinInference
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: SFI.inference.UnderdampedLangevinInference
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Results
-------

.. autosummary::
   :nosignatures:

   ~SFI.inference.InferenceResultSF

.. autoclass:: SFI.inference.InferenceResultSF
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Sparse model selection
----------------------

.. autosummary::
   :nosignatures:

   ~SFI.inference.sparse.SparseScorer
   ~SFI.inference.sparse.SparsityResult
   ~SFI.inference.sparse.SparsityStrategy
   ~SFI.inference.sparse.BeamSearchStrategy
   ~SFI.inference.sparse.GreedyStepwiseStrategy
   ~SFI.inference.sparse.HillClimbStrategy
   ~SFI.inference.sparse.STLSQStrategy
   ~SFI.inference.sparse.LassoStrategy

.. autoclass:: SFI.inference.sparse.SparseScorer
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.SparsityResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.SparsityStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.BeamSearchStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.GreedyStepwiseStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.HillClimbStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.STLSQStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: SFI.inference.sparse.LassoStrategy
   :members:
   :undoc-members:
   :no-index:

Model-comparison metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: SFI.inference.sparse.overlap_metrics
   :no-index:

.. autofunction:: SFI.inference.sparse.predictive_nmse
   :no-index:


Serialization helpers
---------------------

.. autofunction:: SFI.inference.save_results
   :no-index:
.. autofunction:: SFI.inference.load_results
   :no-index:
.. autofunction:: SFI.inference.save_model
   :no-index:
.. autofunction:: SFI.inference.load_model
   :no-index:
