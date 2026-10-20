API overview
============

User-facing modules and classes.  Each API reference page is also
accessible from the sidebar under **API reference**.

Inference
   :doc:`inference/reference` —
   :class:`~SFI.inference.OverdampedLangevinInference`,
   :class:`~SFI.inference.UnderdampedLangevinInference`,
   :class:`~SFI.inference.InferenceResultSF`,
   :class:`~SFI.inference.SparseScorer`,
   :class:`~SFI.inference.SparsityResult`,
   serialization helpers.

State functions
   :doc:`statefunc/reference` —
   :class:`~SFI.statefunc.Basis`, :class:`~SFI.statefunc.PSF`,
   :class:`~SFI.statefunc.SF`, :class:`~SFI.statefunc.Interactor`,
   factories and parameter handling.

Bases
   :doc:`bases/reference` —
   pre-built polynomial, structural, linear and pair-interaction
   bases from :mod:`SFI.bases`.

Trajectory handling
   :doc:`trajectory/reference` —
   :class:`~SFI.trajectory.TrajectoryCollection` constructors, I/O,
   collection operations, and streaming interface.  On-disk file
   formats are specified in :doc:`trajectory/data_formats`.

Simulation
   :doc:`langevin/reference` —
   :class:`~SFI.langevin.OverdampedProcess`,
   :class:`~SFI.langevin.UnderdampedProcess`.

Diagnostics
   :doc:`diagnostics` —
   :func:`~SFI.diagnostics.assess` and the standardised-innovation test
   suite for validating a fitted model on real data.

Integration backend
   :mod:`SFI.integrate` provides the JIT-compiled time-averaging engine
   used internally by the inference classes (:func:`~SFI.integrate.integrate`,
   :func:`~SFI.integrate.make_parametric_integrator`, :class:`~SFI.integrate.Integrand`, :class:`~SFI.integrate.Term`,
   :func:`~SFI.integrate.stream` / :func:`~SFI.integrate.timeop` / :func:`~SFI.integrate.velocity`).  **Backend module — not
   intended for user code.**  See the *Integration backend* row of the
   API map (``_project_index/api_map.md#sfiintegrate``) for the symbol
   inventory.

Conceptual guides
^^^^^^^^^^^^^^^^^

Not reference pages, but you will land on them often from the API:

* :doc:`physics_reference` — all SDEs, drift / diffusion conventions,
  and the mapping between symbols in the math and identifiers in the
  source.
* :doc:`bases/user_guide` — decision tree for choosing between
  compositional primitives and factory builders, with canonical
  patterns (Lorenz, harmonic UD, limit cycle, double-well).
