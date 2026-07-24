.. _trajectory-user-guide:

Trajectory data
===============

.. currentmodule:: SFI.trajectory

The :mod:`SFI.trajectory` submodule is the **data handling layer** of SFI.  It
provides a single user-facing container,
:class:`SFI.trajectory.TrajectoryCollection`, which represents trajectories
together with masks and extras in a JAX-friendly form.

This guide shows how to construct, save, load and combine
:class:`TrajectoryCollection` objects, and how they integrate into the
simulation and inference workflow.

Why a dedicated container, not a DataFrame?
-------------------------------------------

A tracking table (one row per detection) is the natural *interchange*
format, and SFI ingests it directly (:meth:`from_dataframe`, below).
But inference does not run on rows — it runs on **streams** derived from
the trajectory: positions ``X``, increments ``dX``, finite-difference
velocities, and basis functions evaluated along the path.  The
collection exists to serve those streams, which a row-oriented DataFrame
cannot:

- **Increment / stream views.**  ``dX``, secant velocities, and the
  skipped errors-in-variables instrument are built lazily from a sliding
  time window with the correct per-stream alignment — not stored as
  columns.
- **Masking.**  Missing frames and particles entering or leaving are
  tracked per ``(time, particle)`` so each stream only ever yields rows
  where its whole finite-difference window is valid (see *Working with
  masking*).
- **Normalisation and weighting.**  Pooled experiments of different
  length and quality are reweighted into one estimator (per-dataset
  weights, effective observation time).
- **JAX, above all.**  The data are held as dense, rectangular
  ``(T, N, d)`` arrays so the whole pipeline is JIT-compiled, ``vmap``-ed
  and GPU-ready; a dynamically-shaped table of Python objects would
  defeat that.

So: bring your data as a DataFrame or a tracking file, and let the
collection turn it into the JAX-native, mask-aware streams the
estimators consume.

Basic construction from arrays
------------------------------

The main constructor when you already have tensors in memory is
:meth:`TrajectoryCollection.from_arrays`:

.. code-block:: python

   import jax.numpy as jnp
   from SFI.trajectory import TrajectoryCollection

   # Toy data: T time steps, N particles, d dimensions
   T, N, d = 1000, 5, 2
   X = jnp.zeros((T, N, d))   # your tracked positions

   # Uniform timestep
   dt = 0.1

   # Optional boolean mask (T, N): False = missing sample
   mask = jnp.ones((T, N), dtype=bool)

   coll = TrajectoryCollection.from_arrays(
       X=X,
       dt=dt,
       mask=mask,
       extras_global={"temperature": 300.0},
       extras_local={"radius": jnp.ones((N,))},
   )

Key points:

- ``X`` has shape ``(T, N, d)`` or ``(T, d)``; in the latter case,
  a single particle is assumed.
- You may specify either:

  - a scalar or per-step ``dt`` (shape ``(T,)``), or
  - an absolute time vector ``t`` (shape ``(T,)``), or
  - both ``t`` and ``dt=None``; the dataset will derive step sizes
    from ``t`` when needed.

- If ``mask`` is omitted, all samples are treated as valid.

The resulting ``coll`` is a single-dataset collection; you can pass it
directly to inference routines.

.. note::

   The middle axis ``N`` is the **particle** axis (see
   :term:`particles`).  At ``pdepth=0`` a single law is applied
   independently to each particle; interacting models (``pdepth=1``) see
   all particles at once.  The particle count may change over time — the
   mask records who is present at each frame — so ``N`` is the maximum
   concurrent count.  See :doc:`/particles/user_guide`.

Constructing from tracking tables
---------------------------------

Many experimental pipelines produce per-detection tables with columns
such as:

- particle identifier,
- frame or time index,
- coordinates (x, y, z, …),
- optional per-row metadata.

There are two ways to turn such tables into a collection.

From a saved tracking file
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your data are already in a CSV / parquet / HDF5 file that follows the
SFI trajectory format — specified in :doc:`/trajectory/data_formats` —
you can load it directly:

.. code-block:: python

   from SFI.trajectory import TrajectoryCollection

   coll = TrajectoryCollection.load("tracks.parquet")

This will:

- parse the state vectors and (particle, time) indices,
- reconstruct masked ``(T, N, d)`` arrays,
- read extras and metadata from the file header and columns,
- return a single-dataset collection.

If the file contains raw tracking output in a different layout, you can:

1. use pandas (or your preferred tool) to reshape it into the standard
   column layout, then
2. feed the columns to :meth:`TrajectoryCollection.from_columns`.

From a pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~

For raw tracking tables (trackpy, TrackMate, custom pipelines), the
most direct route is :meth:`TrajectoryCollection.from_dataframe` —
columns are addressed by name, in any order, and junk columns are
dropped:

.. code-block:: python

   import pandas as pd
   from SFI.trajectory import TrajectoryCollection

   tracks = pd.read_csv("raw_tracks.csv")   # e.g. track_id, frame, x, y, quality

   coll = TrajectoryCollection.from_dataframe(
       tracks,
       particle="track_id",
       time="frame",
       coords=("x", "y"),
       dt=0.05,
   )

Common particle/time column names (``particle_id``, ``particle``,
``track_id``, ``frame``, ``time``, …) are auto-detected when the
keywords are omitted; an ambiguous table raises with the candidates
listed.  A float-valued time column is factorized into frame indices
and its unique values become the absolute time axis.  Columns with the
extras prefixes (``G_``/``TG_``/``P_``/``TP_``, see
:doc:`/trajectory/data_formats`) are parsed into extras exactly as
when loading files.

From in-memory columns
~~~~~~~~~~~~~~~~~~~~~~

For custom pipelines that already expose columns, use
:meth:`TrajectoryCollection.from_columns` (the lower-level form that
:meth:`from_dataframe` delegates to):

.. code-block:: python

   import numpy as np
   from SFI.trajectory import TrajectoryCollection

   # Example columns
   particle_idx = np.array([0, 0, 1, 1, 0, 0, 1, 1])   # (L,)
   time_idx     = np.array([0, 1, 0, 1, 2, 3, 2, 3])   # (L,)
   state_vecs   = np.random.randn(8, 2)                # (L, d)

   coll = TrajectoryCollection.from_columns(
       particle_idx=particle_idx,
       time_idx=time_idx,
       state_vectors=state_vecs,
       dt=0.1,                # or t=..., or use extras_global["t"]
   )

The constructor:

- reconstructs the dense ``(T, N, d)`` state array,
- infers the number of particles and time steps,
- builds the internal mask (dropping any missing rows),
- attaches any provided extras or metadata.

Working with masking
--------------------

The mask is a boolean array of shape ``(T, N)`` attached to each
dataset. It encodes which **positions** are available at each time and
for each particle.

**How missing data is handled.**  Dropped detections, blinking labels,
tracking gaps, and particles that enter or leave the field of view are
all represented the same way — the corresponding mask entries are set to
``False``.  Nothing is removed from the array and nothing is
interpolated: a masked frame simply does not contribute the increments
that would have used it.  You can therefore keep one clean rectangular
``(T, N, d)`` array even when each particle is observed over a different,
gappy subset of frames.  When ``mask`` is ``None`` the dataset is treated
as fully observed.

Internally, when an inference program requests streams such as ``"X"``
and ``"dX"``, the dataset:

1. computes the smallest time window needed to provide all requested
   streams (e.g. ``t`` and ``t+1`` for ``"dX"``),
2. defines the set of valid indices where this window is in-bounds and
   all required positions are unmasked,
3. only ever produces rows for those valid indices.

This means:

- you can freely mask out arbitrary times and particles without worrying
  about the exact finite-difference scheme used later,
- the same collection can be reused for different inference programs
  that request different combinations of streams.

You typically do not interact with valid indices directly; the
integration runtime requests them via internal streaming methods on the
collection.

Using extras
------------

Extras are user-defined fields attached to a dataset, accessible during
integration through the ``"extras"`` stream. They are divided into:

- ``extras_global``: shared by all particles in the dataset,
- ``extras_local``: per-particle quantities.

Example: per-experiment and per-particle parameters:

.. code-block:: python

   import jax.numpy as jnp
   from SFI.trajectory import TrajectoryCollection

   X = jnp.zeros((T, N, d))

   coll = TrajectoryCollection.from_arrays(
       X=X,
       dt=0.1,
       extras_global={
           "temperature": 300.0,
           "salt_concentration": 0.2,
       },
       extras_local={
           "radius": jnp.linspace(0.5, 1.0, N),
       },
   )

Semantics:

- Global extras are visible to all particles; local extras are aligned
  with the particle index axis.
- If the same key appears in both, the local version wins.

**Time-dependent extras** are first-class: wrap an array with a leading
time axis as a :class:`~SFI.trajectory.TimeSeriesExtra`
(``time_series_extra(values)``, shape ``(T, ...)`` for globals or
``(T, N, ...)`` per particle), and the integration runtime slices it to
the value at each frame.  ``TG_`` / ``TP_`` file columns load this way
automatically (:doc:`/trajectory/data_formats`).

During integration, when a program asks for the ``"extras"`` stream, it
receives a dictionary, for each time index, built by:

- slicing any recognised time-series extras at that index,
- evaluating any callable extras at that index, passing along an
  optional context string supplied by the integrator,
- merging global and local dictionaries.

This makes it natural to fit force fields that depend not only on the
state, but also on experiment conditions, drive protocols, or particle
labels.  The compositional entry point is
:func:`~SFI.bases.extra_scalar`, which turns an extras key into a basis
symbol:

.. code-block:: python

   from SFI.bases import X, extra_scalar

   # trap + protocol-modulated trap: F = θ₁·x + θ₂·k(t)·x
   B = X(dim=1) & (extra_scalar("k_drive", dim=1) * X(dim=1))
   inf.infer_force_linear(B)

See :doc:`/gallery/time_dependent_forcing_demo` for the full round
trip (simulate with a protocol, infer it back), and
:doc:`/langevin/user_guide` for the simulation-side conventions.  For
multi-experiment collections (see below), you can attach different
extras to each dataset and use them as input features in inference.

Combining multiple experiments
------------------------------

A :class:`TrajectoryCollection` can hold several independent datasets.
They may differ in:

- number of particles and duration,
- masks and time axes,
- extras and metadata.

You can build them separately and then concatenate:

.. code-block:: python

   coll1 = TrajectoryCollection.from_arrays(X=X1, dt=0.1,
                                            extras_global={"temperature": 280.0})
   coll2 = TrajectoryCollection.from_arrays(X=X2, dt=0.1,
                                            extras_global={"temperature": 300.0})

   coll = coll1 & coll2          # quick merge; default "pool" policy

   # …or make each experiment count equally regardless of length:
   coll = coll1.concat([coll2], weights="per_dataset")

The resulting collection:

- has two datasets, each with its own extras and mask,
- carries per-dataset weights — an **unnormalised multiplier** that the
  runtime applies to every estimator (force, diffusion, parametric).

The ``weights`` policy sets how much each dataset contributes to the
pooled inference:

- ``"pool"`` (default): multiplier ``1`` for every dataset — pool all the
  increments on equal footing.  Combined with each estimator's intrinsic
  within-dataset weighting (the force is per-:math:`\Delta t`, the
  diffusion per-point), each dataset then contributes in proportion to its
  effective observation time (force) or its number of points (diffusion).
- ``"per_dataset"``: each experiment contributes equally regardless of
  length (multiplier :math:`\overline{T_\mathrm{eff}}/T_{\mathrm{eff},d}`);
  exact for the force, and for the diffusion when ``dt`` is uniform.
- an explicit array: manual per-dataset multipliers.

A single inference on the concatenated collection then fits one shared
model to all experiments at once — see
:doc:`/gallery/advanced/multi_experiment_demo` for a worked example.

**Shared vs. experiment-specific parameters.**  Every produced row
carries the reserved extra ``dataset_index`` (the dataset's position in
the collection — injected at integration time, never stored on disk),
which lets part of the model be experiment-specific while the rest is
shared:

.. code-block:: python

   from SFI.bases import X, named_scalar, per_dataset_scalar, unit_vector_basis

   k = named_scalar("k", default=1.0)            # shared stiffness
   a = per_dataset_scalar("a", n_datasets=2)     # one drift per experiment
   F = a * unit_vector_basis(1) - k * X(dim=1)
   inf.infer_force(F)                            # parametric (L-BFGS path)

For the **linear estimators**, use one-hot indicator features instead:
``dataset_indicator(n) * X(dim)`` gives an independent linear
coefficient per experiment (block-diagonal Gram, PASTIS-prunable).
Per-particle inferred parameters in interacting models use the same
idea with the reserved ``particle_index`` extra inside
:func:`~SFI.statefunc.make_interactor` kernels (declare it via
``particle_extras=...``).

**Reproducing one experiment.**  ``dataset_index`` is purely an
inference-time coordinate — it has no meaning in a simulation.  To
reproduce a single experiment from a pooled fit, collapse the model to
that condition with
:meth:`~SFI.statefunc.StateExpr.specialize`: per-dataset parameters are
folded at the chosen index and ``dataset_index`` drops out, leaving a
standalone single-condition model.  :meth:`~SFI.inference.OverdampedLangevinInference.simulate_bootstrapped_trajectory`
does this for you when given ``dataset=k`` (``k`` is the experiment's
position in the pooled collection):

.. code-block:: python

   coll_k, proc_k = inf.simulate_bootstrapped_trajectory(key, dataset=k)
   # proc_k is experiment k's own model; coll_k is a plain single
   # trajectory (no dataset_index), so re-inference uses a plain basis.

Do **not** set ``dataset_index`` yourself as a user extra — it is
framework-reserved and will be rejected.

This structure is useful when:

- pooling multiple repeats of the same experiment,
- combining trajectories recorded at different conditions and letting
  the state expression depend symbolically on those conditions (e.g.
  learning a force field ``F(x, T)`` from datasets at different
  temperatures).

Saving and loading collections
------------------------------

Use :meth:`TrajectoryCollection.save` and
:meth:`TrajectoryCollection.load` for persistence and interchange.
The on-disk formats (CSV / Parquet / HDF5 table layout, extras-column
prefixes, metadata header) are specified in
:doc:`/trajectory/data_formats`.

Single-dataset case
~~~~~~~~~~~~~~~~~~~

For a single dataset, you can save and reload from a single file:

.. code-block:: python

   coll = TrajectoryCollection.from_arrays(X=X, dt=0.1)

   # Save to a parquet file (recommended)
   coll.save("traj.parquet")

   # Later…
   coll2 = TrajectoryCollection.load("traj.parquet")

Rules:

- The collection must contain exactly one dataset when saving to a
  single file.
- Masked samples are dropped; the file only contains valid rows.
- At load time, particle IDs are optionally relabelled to ``0..N-1`` and
  time indices shifted to start at 0 (configurable via ``relabel``).

Multi-dataset case
~~~~~~~~~~~~~~~~~~

For multiple datasets, save to a directory:

.. code-block:: python

   # coll contains several datasets
   coll.save("my_experiments/")

   # Directory "my_experiments" will contain:
   #   ds_000.parquet, ds_001.parquet, ...
   #   manifest.yaml

   coll2 = TrajectoryCollection.load("my_experiments/")

Each dataset is stored in its own file, and a ``manifest.yaml`` records
their names and filenames. Loading the directory reconstructs the full
collection with one dataset per file.

Round-trip guarantees
~~~~~~~~~~~~~~~~~~~~~

A collection that is saved and then reloaded:

- preserves the state arrays, masks, time axis and extras, up to the
  loss of masked samples (which are not written to disk),
- restores per-dataset metadata and I/O-level time information,
- recomputes reasonable default weights (by default equal weights for
  multi-dataset collections, or the chosen policy).

You can rely on this for reproducible pipelines where simulation,
storage, and inference may happen at different times or on different
machines.

Train/test splitting
--------------------

For held-out validation on data-abundant problems,
:meth:`TrajectoryCollection.split_time` cuts every dataset along time:

.. code-block:: python

   train, test = coll.split_time(0.8)        # optional: gap=, reweight=

Masks, time axes (kept absolute), and extras follow the split
(time-series extras are sliced; static extras are shared).  See the
held-out note in :doc:`/inference/user_guide` for when this is — and
is not — the right tool: SFI's default validation route
(``force_predicted_MSE`` + diagnostics) costs no data.

Optional: degrading synthetic data
----------------------------------

The :meth:`TrajectoryCollection.degrade` method is a convenience wrapper
for generating synthetic “experimental-like” trajectories from clean
simulations:

.. code-block:: python

   degraded = coll.degrade(
       downsample=4,            # keep every 4th frame (effective dt ×4)
       motion_blur=3,           # average over motion_blur+1 frames before
                                #   downsampling (exposure smear);
                                #   0 <= motion_blur < downsample
       data_loss_fraction=0.1,  # randomly mask out 10% of surviving points
       noise=0.01,              # add Gaussian localisation noise, std 0.01
       reweight="pool",         # recompute pooled weights after degrading
   )

Two further options not shown above: ``ROI=`` drops points outside a
region of interest (a radial cutoff, an axis-aligned box, or a
predicate), and ``seed=`` fixes the RNG for the added noise and the
drop-out so a degradation is reproducible.

Typical use cases:

- testing robustness of inference algorithms to motion blur and missing
  data,
- benchmarking against known synthetic ground truth.

Real experimental data are usually ingested directly from tracking
tables via :meth:`TrajectoryCollection.load` or
:meth:`TrajectoryCollection.from_columns`; degradation is not required
for those workflows.

Quick plotting
--------------

The :mod:`SFI.utils.plotting` module provides simple helpers that work
directly with collections. For example:

.. code-block:: python

   import matplotlib.pyplot as plt
   from SFI.utils import plotting

   # Time series of all coordinates for the first dataset
   plotting.timeseries(coll, dataset=0)
   plt.show()

   # 2D phase plot for coordinates 0 and 1
   plotting.phase2d(coll, dataset=0, dims=(0, 1))
   plt.gca().set_aspect("equal")
   plt.show()

These utilities internally extract ``(t, X, mask)`` from the chosen
dataset, respect the mask when drawing trajectories, and can be used as
building blocks for more elaborate visualisation routines.

Sanity-checking your data
-------------------------

A quick look before inference rarely hurts: plot the trajectory with
``plotting.timeseries`` / ``plotting.phase2d`` (above) and watch for
tracking artefacts (jumps, stuck particles, coordinate resets) or a
mismatched ``dt`` (displacements spanning a large fraction of the
explored range); ``ds.mask`` — ``None`` when fully observed — reports the
coverage.  None of this is essential, though: the rigorous, fit-aware
checks happen *after* inference, through the diagnostics suite
(:doc:`/diagnostics`).

Further reading
---------------

- :doc:`reference` for a curated reference of the public methods on
  :class:`TrajectoryCollection`.
- The autogenerated :mod:`SFI.trajectory` API page for full signatures
  and parameter documentation.
