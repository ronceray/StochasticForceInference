Playbook — add a feature to SFI
===============================

.. note::

   Prerequisite: read ``AGENTS.md`` at the repository root — §3
   ("reuse, don't re-implement") and §4 (canonical imports).

1. Find the right subpackage
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature kind
     - Goes in
   * - New inference method / solver / diagnostic
     - ``SFI/inference/``
   * - New basis family (polynomial, pair, SPDE, …)
     - ``SFI/bases/``
   * - New simulator / integrator / noise model
     - ``SFI/langevin/``
   * - New trajectory I/O / degradation model
     - ``SFI/trajectory/``
   * - New state-function primitive (Basis/PSF/SF composition)
     - ``SFI/statefunc/``
   * - New time-averaging operand or program
     - ``SFI/integrate/``
   * - Math / formatting / plotting helper
     - ``SFI/utils/``

If nothing fits cleanly, **ask the user** before creating a new
subpackage.

2. Check you are not duplicating
--------------------------------

Before writing code:

1. Search the subpackage's ``__init__.py`` re-exports.
2. Grep for the concept name in ``SFI/`` and in
   ``design_notes_history/INDEX.md`` (there may be a prior attempt).
3. Check ``AGENTS.md`` §4 canonical-imports table.
4. Check ``_project_index/api_map.json`` (machine-readable API index).

If a prior implementation was abandoned, the relevant design note in
``design_notes_history/`` usually explains why — read it before
reintroducing a superseded approach.

3. Implementation conventions
-----------------------------

- **JAX-first.** Hot paths are ``jit``-able. Avoid Python-level
  ``for`` loops over time or particles. Use ``jax.vmap`` /
  ``jax.lax.scan``. Toggle globally via ``SFI.statefunc.set_jit(False)``
  when debugging.
- **Shape contracts.** Every public function documents its expected
  shapes in the docstring (numpy style). Use the ``rank`` vocabulary
  (scalar / vector / matrix) from :mod:`SFI.statefunc`.
- **Mask-awareness.** If the feature consumes trajectories, it must
  respect masks. Do not peek into ``coll._mask``; use the public
  accessors on :class:`~SFI.trajectory.TrajectoryCollection`.
- **No hidden global state.** Parameters flow explicitly through
  :class:`~SFI.statefunc.PSF`/:class:`~SFI.statefunc.SF` objects.

4. Re-exports
-------------

For every new public symbol:

1. Add to the subpackage ``__init__.py`` ``__all__`` and import list.
2. If the symbol belongs to the top-level user surface (think
   ``from SFI import ...``), also re-export from ``SFI/__init__.py``.
3. Update ``AGENTS.md`` §4 canonical-imports table.
4. Regenerate the API map: ``python scripts/gen_api_map.py``.

.. important::

   Steps 3 and 4 are **not optional**. See ``AGENTS.md`` §9
   ("Keeping these agent docs fresh") — CI runs
   ``scripts/gen_api_map.py --check`` and will reject PRs whose API
   map is stale.

5. Tests
--------

Mandatory for any new public behaviour:

- Unit test under ``tests/<subpackage>/test_<feature>.py``.
- If it replaces an existing method, add a regression test that pins
  the old and new outputs on a reference seed.
- If it affects inference accuracy, add a smoke test that runs a
  known problem (e.g. OU or double-well) and asserts the inferred
  force within a tolerance.

Run locally::

   pytest tests/ -v
   pytest tests/<subpackage>/ -v        # scoped

Files named ``benchmark_*.py`` / ``validate_*.py`` / ``audit_*.py``
under ``tests/`` are **not** collected by pytest; they are
manually-invoked scripts.

6. Documentation
----------------

If the feature is user-facing:

- Add or update the relevant user guide under ``docs/source/<sub>/``.
- Add a gallery demo under ``examples/gallery/`` following
  ``GALLERY_STYLE_GUIDE.md``.
- Docstring is rendered by Sphinx autodoc — make it complete
  (parameters, returns, examples, physics block if there is math).

7. Deprecation / renaming existing API
--------------------------------------

- Never rename a public symbol without a deprecation shim for at least
  one release.
- Emit ``DeprecationWarning`` pointing to the replacement.
- Log the change in a short note under ``design_notes_history/``
  with the banner format applied by
  ``scripts/tag_historical_notes.py``.

8. Design-ambiguity checklist — when to ask, not guess
------------------------------------------------------

Before making any of the following choices on your own, **stop and ask
the user**:

- Creating a new subpackage or top-level module.
- Introducing a new public top-level export on :mod:`SFI`.
- Changing a public signature or removing a public symbol.
- Adding a heavy optional dependency.
- Introducing a new result-file format.
- Changing the sparsity criterion default.
