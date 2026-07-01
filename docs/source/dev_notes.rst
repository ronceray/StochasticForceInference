.. _dev-notes:

Developer notes
===============

Environment setup
-----------------

.. code-block:: bash

   # Create / activate the dev environment
   python -m venv sfi_env
   source sfi_env/bin/activate
   pip install -e ".[dev,io]"


Building documentation
----------------------

**Quick build** (docs only, no examples re-run):

.. code-block:: bash

   cd docs
   make clean && make html
   # open build/html/index.html

This uses pre-generated gallery ``.rst`` and images already committed
under ``docs/source/gallery/``.  Sphinx-gallery is *not* loaded; no
example scripts are executed.

**Full build** (re-run all gallery examples):

.. code-block:: bash

   cd docs
   SFI_DOCS_RUN_GALLERY=1 make html

This loads ``sphinx_gallery.gen_gallery``, executes every ``*_demo.py``
script, and regenerates the gallery pages from scratch.
It can take a long time (10+ min) and requires all optional dependencies
(matplotlib, joblib, …).

Animations are rendered as **HTML5 ``<video>``** via ``ffmpeg`` (h264).
Make sure ``ffmpeg`` is installed (``apt install ffmpeg`` / ``brew install
ffmpeg``) before running a full build.

**Regenerate API stubs** (sphinx-apidoc):

.. code-block:: bash

   cd docs
   SFI_DOCS_RUN_APIDOC=1 make html
   # or just:  make apidoc && make html

**Full regeneration** (gallery + apidoc):

.. code-block:: bash

   cd docs
   SFI_DOCS_RUN_GALLERY=1 SFI_DOCS_RUN_APIDOC=1 make html


Running tests
-------------

.. code-block:: bash

   # Full test suite
   pytest tests/ -v

   # Quick smoke test (double-well only)
   python tests/smoke_dw.py


Code style
----------

.. code-block:: bash

   ruff check SFI/ tests/ examples/
   ruff format SFI/ tests/ examples/


Package build and release
-------------------------

.. code-block:: bash

   pip install build twine
   python -m build
   twine check dist/*
   # twine upload dist/*


Documentation structure
-----------------------

The docs follow a `Diátaxis <https://diataxis.fr/>`_-inspired layout:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Sidebar section
     - Content type
   * - **Getting started**
     - Installation + introductory tutorials
   * - **User guides**
     - How-to pages for each module (statefunc, bases, trajectory, inference, simulation)
   * - **Theory & design**
     - Physics reference, design rationale, "choosing a basis" guide
   * - **API reference**
     - Auto-generated class/function docs (one page per module)
   * - **Gallery**
     - Sphinx-gallery worked examples

Key conventions:

- Each module has a ``user_guide.rst`` (narrative) and ``reference.rst``
  (API) page, both listed directly in the root ``index.rst`` toctree.
- ``api_frontend.rst`` is a link hub — it does *not* use ``toctree`` to
  avoid duplicate sidebar entries.
- Gallery ``.rst`` pages are committed to source control so
  that ``make html`` works without running examples.


Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``SFI_DOCS_RUN_GALLERY=1``
     - Load sphinx-gallery and re-run all example scripts during build
   * - ``SFI_DOCS_RUN_APIDOC=1``
     - Regenerate ``docs/source/api/*.rst`` stubs via sphinx-apidoc
   * - ``SFI_DOCS_ALLOW_RST=1``
     - Render *all* docstrings as RST (for auditing; default: un-audited
       modules are rendered as literal blocks)


Planned extensions
------------------

- **More tutorials**: simple system, experimental data, multiparticle
  system (add under ``Getting started``).
- **In-depth physics guides**: estimators, sparsity (add under
  ``Theory & design``).
- **PINNs / neural-net extensions**: add user guide + API reference pages
  when ready.
- **Choosing a basis**: expand with case studies as more applications are
  developed.
