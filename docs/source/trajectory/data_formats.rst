.. _trajectory-data-formats:

Trajectory file formats
=======================

This page specifies the on-disk formats read by
:meth:`TrajectoryCollection.load <SFI.trajectory.TrajectoryCollection.load>`
and written by
:meth:`TrajectoryCollection.save <SFI.trajectory.TrajectoryCollection.save>`.
Three formats are supported — **CSV**, **Parquet**, and **HDF5** — all
sharing the same tabular layout: one row per observation.

.. code-block:: python

   from SFI.trajectory import TrajectoryCollection

   coll = TrajectoryCollection.load("tracks.csv")      # or .parquet / .h5
   coll.save("tracks.parquet")                          # format from suffix


Table layout
------------

Each row is one observation of one particle at one time step:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Column
     - Required
     - Content
   * - ``particle_id``
     - optional
     - Integer track identifier.  If absent, the file is a
       *single-trajectory* file (pass ``particle_column=None`` to the
       low-level loader).
   * - ``time_step``
     - yes
     - Integer time index (0-based after relabelling).
   * - ``x0, x1, …``
     - yes
     - State-vector components (positions, angles, concentrations, …).

**CSV files identify columns by position, not by name**: by default
column 0 is the particle identifier, column 1 the time index, and every
remaining column without an extras prefix (below) is a state component.
Your columns can therefore be named ``particle_id, frame, x, y`` or
anything else.  Parquet and HDF5 files identify columns **by name** and
must use the canonical names ``particle_id`` and ``time_step``.

Rows containing NaNs are dropped on load; masked samples are dropped on
save (only valid rows are written).


Extras columns
--------------

Per-observation metadata is carried in extra numeric columns,
classified by a name prefix:

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Prefix
     - Kind
     - Example
   * - ``G_``
     - Global scalar (constant for the whole dataset)
     - ``G_temperature`` — stored in the header on save
   * - ``TG_``
     - Time-dependent global (depends on :math:`t` only)
     - ``TG_field`` — an external drive protocol
   * - ``P_``
     - Per-particle constant (depends on particle only)
     - ``P_radius`` — individual particle sizes
   * - ``TP_``
     - Time- and particle-dependent
     - ``TP_intensity`` — per-detection fluorescence

On load these populate ``extras_global`` / ``extras_local`` of the
dataset and become available to state functions through the ``extras``
mechanism (see :doc:`/trajectory/user_guide`).


Metadata header
---------------

A file can carry a YAML metadata mapping — most importantly the time
step ``dt``:

- **CSV** — leading comment lines: a ``# ---`` opener followed by
  ``# key: value`` lines.

  .. code-block:: text

     # ---
     # dt: 0.01
     # description: 2D optical tweezer
     particle_id,frame,x,y
     0,0,-0.017995,-0.025163
     0,1,0.037124,-0.100932

  (Plain ``# key: value`` lines without the ``# ---`` opener are also
  accepted, as in ``examples/experimental_data/optical_tweezer.csv``.)

- **Parquet** — the same YAML string stored in the table schema
  metadata under the key ``sfi_yaml_header``.

- **HDF5** — one dataset per column inside a ``table`` group; the YAML
  string stored as the root attribute ``sfi_yaml_header``.

Recognised keys:

- ``dt`` — scalar sampling interval (seconds, or your time unit).
  Accepted both at the top level and inside ``extras_global`` (files
  written by :meth:`TrajectoryCollection.save` use the latter);
- ``extras_global`` — a mapping of arbitrary scalars or arrays.  The
  special key ``t`` (a length-``T`` vector) defines a non-uniform time
  axis and overrides ``dt``;
- anything else is kept as free-form dataset metadata (``coll.datasets[0].meta``).


Named columns
-------------

When a file does not follow the positional/canonical layout above,
select the columns explicitly — by **name** for any format, or by
index for CSV:

.. code-block:: python

   coll = TrajectoryCollection.load(
       "raw_tracks.csv",
       particle_column="particle",      # or an int index (CSV only)
       time_column="t",
       state_columns=("x", "y"),        # drops every other non-extras column
   )

For in-memory tables, :meth:`TrajectoryCollection.from_dataframe` is
the more convenient entry point (auto-detection of common column
names) — see :doc:`/trajectory/user_guide`.

Loading behaviour
-----------------

:meth:`TrajectoryCollection.load` accepts a single file or a directory
and takes two knobs:

- ``relabel=True`` (default) — particle IDs are compressed to
  ``0..N-1`` and time indices shifted to start at 0.  The original IDs
  are recorded in ``extras_local["original_particle_id"]``.
- ``compress_particles=False`` — when True, particles whose time
  supports do not overlap (with a 2-frame safety buffer) are packed
  into the same column slot.  Useful for open-boundary data where
  particles enter and leave the field of view, which otherwise makes
  the array width grow with the total number of unique tracks rather
  than the concurrent count.  The mapping is stored in
  ``dataset.meta["particle_column_map"]``.

Weights: every load initialises dataset weights with the default
``"pool"`` policy; call ``coll.with_weights(...)`` after loading if you
need a different policy.


Multi-dataset directories
-------------------------

A collection with several datasets saves to a directory:

.. code-block:: text

   my_experiments/
   ├── ds_000.parquet
   ├── ds_001.parquet
   └── manifest.yaml        # records dataset names and filenames

``TrajectoryCollection.load("my_experiments/")`` reconstructs the full
collection, one dataset per file.


Round trip
----------

.. code-block:: python

   import jax.numpy as jnp
   from SFI.trajectory import TrajectoryCollection

   coll = TrajectoryCollection.from_arrays(X=jnp.zeros((100, 3, 2)), dt=0.05)
   coll.save("run.parquet")
   coll2 = TrajectoryCollection.load("run.parquet")

State arrays, masks, time axis, extras, and metadata survive the round
trip, up to the loss of masked samples (which are never written).


.. seealso::

   - :doc:`/trajectory/user_guide` — constructing collections from
     arrays, columns, or files; masking; extras; combining experiments.
   - :doc:`/trajectory/reference` — full API of the trajectory layer.
