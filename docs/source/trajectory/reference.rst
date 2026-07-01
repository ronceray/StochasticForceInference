.. _trajectory-reference:

Trajectory API
==============

.. currentmodule:: SFI.trajectory

Public container
----------------

The :class:`TrajectoryCollection` class is the single public entry point of
:mod:`SFI.trajectory`. It represents one or more experiments, each with a
rectangular tensor :math:`X(t, n, d)` plus masks and extras, and is used both
as:

* the output of Langevin simulators (synthetic benchmarks), and
* the input to inference pipelines (experimental data).

It is the canonical first argument of every inference engine —
:class:`~SFI.OverdampedLangevinInference(collection)` and
:class:`~SFI.UnderdampedLangevinInference(collection)`.

.. autosummary::
   :nosignatures:

   TrajectoryCollection


How to build one
~~~~~~~~~~~~~~~~

Pick the constructor that matches your data shape:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - You have…
     - Use this
     - One-liner
   * - A dense in-memory tensor with shape ``(T, N, d)`` or ``(T, d)``
       and a fixed ``N``
     - :meth:`TrajectoryCollection.from_arrays`
     - ``TrajectoryCollection.from_arrays(X=X, dt=0.01)``
   * - A tabular table — one row per ``(particle, time)`` observation,
       particles that enter/leave at different times
     - :meth:`TrajectoryCollection.from_columns`
     - ``TrajectoryCollection.from_columns(particle_idx, time_idx,
       state_vectors, dt=0.01)``
   * - One or more files on disk (CSV, Parquet, HDF5)
     - :meth:`TrajectoryCollection.load`
     - ``TrajectoryCollection.load("trajectory.parquet")``
   * - Several existing collections (e.g. multi-experiment)
     - :meth:`TrajectoryCollection.concat`
     - ``c1 & c2``  or  ``c1.concat([c2])``

.. note::

   For tracked-particle data where the number of particles changes over
   time, :meth:`~SFI.trajectory.TrajectoryCollection.from_columns` is the canonical path — **do not pre-pad
   missing rows with NaN and call** :meth:`~SFI.trajectory.TrajectoryCollection.from_arrays`.

The on-disk formats accepted by :meth:`TrajectoryCollection.load` and
written by :meth:`TrajectoryCollection.save` are specified in
:doc:`/trajectory/data_formats`.


Core constructors and I/O
-------------------------

High-level constructors and round-trip helpers. These are the usual "front
door" for getting data in and out of SFI:

.. autosummary::
   :nosignatures:

   TrajectoryCollection.from_arrays
   TrajectoryCollection.from_columns
   TrajectoryCollection.load
   TrajectoryCollection.save
   TrajectoryCollection.to_arrays


Collection operations
---------------------

Operations that change how experiments are grouped or weighted:

.. autosummary::
   :nosignatures:

   TrajectoryCollection.concat
   TrajectoryCollection.with_weights
   TrajectoryCollection.degrade


Streaming interface
-------------------

The streaming API exposes one-row-at-a-time access to the underlying datasets.
It is what :mod:`SFI.integrate` uses internally to build increments and masks
with controlled memory usage. Most users will only need these methods when
implementing custom inference loops:

.. autosummary::
   :nosignatures:

   TrajectoryCollection.iter_slices
   TrajectoryCollection.peek_row


Advanced: low-level helpers
---------------------------

The following modules are used internally by :class:`TrajectoryCollection`
to implement columnar I/O, degradation of synthetic data, and the dataset
streaming contract. They are documented for advanced users who need direct
access to these layers (for example, custom file formats or standalone
benchmark scripts):

.. autosummary::
   :nosignatures:

   io
   degrade
   dataset
