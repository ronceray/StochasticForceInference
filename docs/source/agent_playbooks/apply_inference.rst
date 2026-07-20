Playbook — apply inference to a dataset
=======================================

.. note::

   Prerequisite: read ``AGENTS.md`` at the repository root for the
   canonical imports and the "do not re-implement" rule.

This playbook is the standard recipe for running SFI on a dataset
(experimental or synthetic). Every step names the canonical class or
function to use — **do not re-implement any of them**.

1. Decide overdamped vs underdamped
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Data
     - Model
     - Class
   * - Only positions observed
     - Overdamped
     - :class:`~SFI.inference.OverdampedLangevinInference`
   * - Only positions observed, but inertia matters (ballistic regime)
     - Underdamped
     - :class:`~SFI.inference.UnderdampedLangevinInference`
   * - Positions *and* velocities observed
     - Underdamped (pass ``V``)
     - ``UnderdampedLangevinInference``

Heuristic: if ``<Δx² / (2 dim · D · Δt)>`` is far from 1, the overdamped
assumption is violated; switch to underdamped.

2. Load the data
----------------

Always use the trajectory containers. **Do not build mask or increment
arrays by hand.**

.. code-block:: python

   import numpy as np
   from SFI.trajectory import TrajectoryCollection

   X = np.load("positions.npy")          # shape (T, d) or (T, N, d)
   coll = TrajectoryCollection.from_arrays(X=X, dt=0.01)

Multi-experiment data: pass a list of arrays, or build individual
:class:`~SFI.trajectory.TrajectoryDataset` objects with :meth:`~SFI.trajectory.TrajectoryDataset.from_arrays` and
combine them. For masked data (missing frames) pass ``mask=`` — the
collection handles all mask-aware arithmetic downstream. For CSV /
Parquet / HDF5 input, use ``TrajectoryCollection.load`` (format spec:
:doc:`../../trajectory/data_formats`); for
synthetic degradation (noise, downsampling, motion blur) see
:mod:`SFI.trajectory.degrade`.

3. Choose a basis
-----------------

Use the ready-made builders in :mod:`SFI.bases`. Only fall back to
``make_basis(func, ...)`` for truly custom functional forms.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - You want…
     - Use
   * - Polynomial in position up to order *n*
     - ``monomials_up_to(order=n, dim=d, rank="vector")``
   * - Polynomial in position *and* velocity
     - ``monomials_up_to(order=n, dim=d, include_v=True, rank="vector")``
   * - Coordinate-wise linear map
     - ``linear_basis(dim=d)``
   * - Constant (isotropic) diffusion
     - ``identity_matrix_basis(dim=d)``
   * - Symmetric-matrix diffusion
     - ``symmetric_matrix_basis(dim=d)``
   * - Pair interactions
     - :mod:`SFI.bases.pairs` builders
   * - Spatial differential operators (SPDE)
     - :mod:`SFI.bases.spde` builders
   * - Fully custom
     - ``SFI.statefunc.make_basis(func, dim, rank, n_features, labels)``

.. note::

   For ABP / active-matter models, compose ``SFI.bases.pairs``
   primitives (``heading_vector``, ``pbc_displacement``, ``wrap_angle``).
   See ``examples/_gallery_utils/abp.py`` for a worked example.

Rule of thumb: start with ``monomials_up_to(order=3, dim=d,
rank="vector")`` for the force, ``identity_matrix_basis(dim=d)`` for a
constant-*D* fit, and sparsify downstream.

4. Run inference — pick the estimator family
--------------------------------------------

Two first-class estimator families; route by data regime (full regime
table: :doc:`../../inference/user_guide`):

Linear estimators (fast, closed-form)
   :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_constant` → :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear` →
   :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion_linear`. Use when measurement noise is negligible
   and data are well-sampled — exact in that limit, seconds even on
   large datasets.

Parametric estimators (robust; compute-intensive)
   :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` (RK4-integrated flow + Gauss–Newton) →
   :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion` (state-dependent *D* via local-precision NLL).
   Use when positions are noisy (:math:`y = x + \eta`), when
   :math:`F\,\Delta t` is not small, or when the model is nonlinear in
   its parameters.

Linear example:

.. code-block:: python

   import SFI
   from SFI.bases import monomials_up_to, identity_matrix_basis

   inf = SFI.OverdampedLangevinInference(coll)
   inf.compute_diffusion_constant(method="auto")

   B_force = monomials_up_to(order=3, dim=coll.d, rank="vector")
   inf.infer_force_linear(B_force, M_mode="Strato")

   B_diff = identity_matrix_basis(dim=coll.d)  # or symmetric_matrix_basis
   inf.infer_diffusion_linear(B_diff)

   inf.compute_force_error()
   inf.compute_diffusion_error()

Parametric example:

.. code-block:: python

   # Build a *parametric* force F(x; θ) instead of a linear basis
   from SFI.statefunc import make_psf

   def force_fn(x, *, params):
       return -params["k"] * x

   F = make_psf(force_fn, dim=coll.d, rank=1,
                params={"k": ()}, labels=["-k x"])

   inf = SFI.OverdampedLangevinInference(coll)
   inf.infer_force(F)
   inf.infer_diffusion(B_diff)

5. (Optional) Sparsify
----------------------

After :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`, call :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force` to identify which
basis terms the data actually supports.

.. code-block:: python

   result = inf.sparsify_force(criterion="PASTIS", p=0.1)
   # or: criterion="AIC" / "BIC"
   # method kwargs: "beam" (default), "greedy", "stlsq", "lasso"

   result.all_ic(verbose=True)   # summary table across all criteria

For held-out scoring and precision/recall against a known ground truth:

.. code-block:: python

   from SFI.inference.sparse import overlap_metrics, predictive_nmse

6. Report and validate
----------------------

.. code-block:: python

   inf.print_report()                 # console summary
   report = inf.report_dict()         # serialisable dict

   # Compare to ground truth (simulations only)
   inf.compare_to_exact(model_exact=proc)

   # Bootstrapped trajectory from the inferred model
   coll_boot, proc_boot = inf.simulate_bootstrapped_trajectory(key)

7. Save / load
--------------

.. code-block:: python

   from SFI.inference import save_model, load_model
   save_model(inf.force_inferred, "F.npz")
   # reloading needs a template supplying the basis / PSF structure
   F_loaded = load_model("F.npz", template=inf.force_inferred)

   # Full inference object
   inf.save_results("run.json")

8. Plot
-------

If you produce figures (gallery demo, paper figure, diagnostic), follow
``GALLERY_STYLE_GUIDE.md`` at the repository root: build dark-theme
figures with ``dark_fig()`` and the ``SFI_COLORS`` palette; never use
pure black.  (Gallery demos call the ``apply_style()`` helper from
``examples/_gallery_utils/helpers.py`` instead, which is only importable
inside the examples tree.)

.. code-block:: python

   from SFI.utils.plotting import dark_fig, SFI_COLORS
   fig, ax = dark_fig()

9. Anti-patterns — do not do this
---------------------------------

- ``np.diff(X, axis=0) / dt`` for increments — use the collection's
  increments; it is mask-aware.
- Writing your own Euler-Maruyama loop — use
  :class:`SFI.langevin.OverdampedProcess`.
- Hand-rolling polynomial features — use :func:`SFI.bases.monomials_up_to`.
- Manually assembling Gram matrices or running lstsq — the inference
  engines already do this and track the covariance.
- Thresholding coefficients by hand — use :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force`.

10. See also
------------

- Worked examples: :doc:`../../gallery/index`
  (``examples/gallery/ou_demo.py``, ``lorenz_demo.py``,
  ``experimental_workflow_demo.py`` are the closest templates).
- User guides: :doc:`../../inference/user_guide`,
  :doc:`../../trajectory/user_guide`, :doc:`../../bases/user_guide`.
