Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   pip install StochasticForceInference

If you need GPU acceleration, install the matching JAX wheel first using the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_,
then install SFI.


From a source checkout
----------------------

From the repository root (or an unpacked source distribution):

.. code-block:: bash

   pip install .

Development install
-------------------

For editable development from a repository checkout:

.. code-block:: bash

   pip install -e ".[dev,io]"

The ``[dev]`` extra already includes ``[docs]`` plus testing and linting
tools (pytest, ruff); combine with ``[io]`` for the full developer setup
with HDF5/Parquet backends.  If you only need to build the documentation,
``[docs]`` alone (sphinx, furo, myst-parser, …) is sufficient.  The
``[io]`` extra adds I/O backends (pyarrow, h5py) and is required for
loading Parquet / HDF5 trajectory files.


What to read next
-----------------

- If you already have experimental trajectories, start with
  :doc:`start_here` and :doc:`gallery/experimental_workflow_demo`.
- If you want a synthetic end-to-end walkthrough first, read
  :doc:`gallery/ou_demo`.
- If a first fit is hard to interpret, use :doc:`diagnostics`.


Build documentation
-------------------

First install the documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Then build the docs:

.. code-block:: bash

   cd docs
   make html

If ``make`` is not available (common on Windows in VS Code), use one of:

.. code-block:: powershell

   cd docs
   .\make.bat html

.. code-block:: bash

   cd docs
   python -m sphinx -M html source build

The commands above use pre-generated gallery figures and build fast.
To **re-run gallery examples** and regenerate all figures:

.. code-block:: bash

   cd docs
   SFI_DOCS_RUN_GALLERY=1 python -m sphinx -M html source build

Additional environment flags (only with ``SFI_DOCS_RUN_GALLERY=1``):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Effect
   * - ``SFI_DOCS_RUN_BENCHMARKS=0``
     - Skip benchmark scripts (faster iteration on gallery only).
   * - ``SFI_DOCS_RUN_STALE=1``
     - Force re-run of **all** examples, even if their source is unchanged.

For example, to rebuild gallery figures only (no benchmarks), with
stale-figure regeneration:

.. code-block:: bash

   cd docs
   SFI_DOCS_RUN_GALLERY=1 SFI_DOCS_RUN_BENCHMARKS=0 SFI_DOCS_RUN_STALE=1 \
       python -m sphinx -M html source build


Requirements
------------

- Python ≥ 3.11
- JAX ≥ 0.6

JAX defaults to CPU.  For GPU acceleration, follow the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.


Core dependencies
^^^^^^^^^^^^^^^^^

numpy, scipy, jax, jaxlib, pandas, matplotlib, equinox, opt-einsum, pyyaml.

Optional dependencies (``[io]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyarrow (Parquet I/O), h5py (HDF5 I/O).


Running the tests
-----------------

After installing with the ``[dev]`` extra, run the test suite with:

.. code-block:: bash

   pytest tests/

For a quick smoke test that exercises each submodule without running the full
benchmark suite:

.. code-block:: bash

   pytest tests/ -m smoke


Contributing
------------

Contributions are welcome.  Please open a GitHub issue before submitting a
pull request so the change can be discussed first.

- **Bug reports:** include a minimal reproducing example, the full error
  traceback, and your Python / JAX versions
  (``python --version``, ``python -c "import jax; print(jax.__version__)"``).
- **Feature requests:** open an issue describing what you need and why.  If
  you already have a concrete proposal, feel free to open a pull request
  alongside the issue.
- **Questions about usage:** open a GitHub issue tagged *question*, or reach
  out to the maintainer directly.
