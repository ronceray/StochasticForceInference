"""
SFI documentation configuration for Sphinx.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "SFI"
author = "Pierre Ronceray"
version = "2.0"  # short X.Y version
release = "2.0.0"  # full version string

# ---------------------------------------------------------------------------
# Paths: locate repo root and package, add to sys.path
# ---------------------------------------------------------------------------

DOCS_SRC = Path(__file__).resolve().parent  # .../docs/source
API_OUT = DOCS_SRC / "api"


def _find_repo_and_pkg(start: Path, pkg_name: str = "SFI") -> tuple[Path, Path]:
    """
    Walk upwards from ``start`` until a Python package named ``pkg_name``
    (with an ``__init__.py``) is found. Return (repo_root, pkg_dir).
    """
    for anc in (start, *start.parents):
        cand = anc / pkg_name
        if cand.is_dir() and (cand / "__init__.py").exists():
            return anc, cand
    raise RuntimeError(f"Package '{pkg_name}' not found upwards from {start}")


REPO_ROOT, PKG_DIR = _find_repo_and_pkg(DOCS_SRC)
sys.path.insert(0, str(REPO_ROOT))  # allow "import SFI"

# Add examples directory so sphinx-gallery scripts can import _gallery_utils
sys.path.insert(0, str(REPO_ROOT / "examples"))

# Add local extensions directory to path
sys.path.insert(0, str(DOCS_SRC / "_extensions"))

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.video",
    "myst_parser",
    "gallery_tags",
    "physics_ref",
]
extensions += ["sphinx.ext.todo"]
# Hide maintainer TODOs in public builds; opt in locally when needed.
todo_include_todos = os.environ.get("SFI_DOCS_SHOW_TODOS", "").strip() == "1"

# ---------------------------------------------------------------------------
# Mermaid diagrams — dark theme to match Furo
# ---------------------------------------------------------------------------
mermaid_init_js = (
    "mermaid.initialize({"
    "  theme: 'dark',"
    "  themeVariables: {"
    "    primaryColor: '#3B9EFF',"
    "    primaryTextColor: '#D0D0D0',"
    "    primaryBorderColor: '#808080',"
    "    lineColor: '#B0B0B0',"
    "    secondaryColor: '#5D3A9B',"
    "    tertiaryColor: '#1e2028',"
    "    background: '#131416',"
    "    mainBkg: '#1e2028',"
    "    nodeBorder: '#808080',"
    "    clusterBkg: '#1e2028',"
    "    clusterBorder: '#808080',"
    "    titleColor: '#D0D0D0',"
    "    edgeLabelBackground: '#1e2028',"
    "    nodeTextColor: '#D0D0D0'"
    "  }"
    "})"
)

# ---------------------------------------------------------------------------
# Sphinx-Gallery: example gallery from the examples/ directory
# ---------------------------------------------------------------------------
# Gallery execution is OPT-IN.  Set SFI_DOCS_RUN_GALLERY=1 to re-run all
# example scripts.  Without this flag the build uses the pre-generated
# .rst / .ipynb / images that already live under docs/source/gallery/.
#
# Benchmarks are NEVER run during Sphinx builds.  They are precomputed
# via ``run_benchmarks.py`` and rendered via ``render_benchmarks.py``.
# The docs build only assembles pre-rendered PNG figures.
#
# Additional toggle (only meaningful when SFI_DOCS_RUN_GALLERY=1):
#   SFI_DOCS_RUN_STALE=1       Force re-run of ALL examples, even unchanged.
# ---------------------------------------------------------------------------

_RUN_GALLERY = os.environ.get("SFI_DOCS_RUN_GALLERY") == "1"
_RUN_STALE = os.environ.get("SFI_DOCS_RUN_STALE", "0") == "1"

if _RUN_GALLERY:
    # Avoid GPU pre-allocation OOM when running many examples sequentially.
    # With "false", JAX allocates on-demand instead of grabbing all GPU RAM
    # at startup, so each example script only uses what it actually needs.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    extensions.append("sphinx_gallery.gen_gallery")
else:
    # Register a lightweight ``image-sg`` directive so the pre-generated
    # gallery / benchmark RST files render their images even when
    # sphinx_gallery.gen_gallery is not loaded (it normally provides it).
    from docutils.parsers.rst import directives as _directives
    from sphinx.directives.patches import Figure as _Figure

    class _ImageSgStub(_Figure):
        """Minimal stand-in for sphinx-gallery's ``image-sg`` directive.

        Accepts the same options but delegates to the standard ``figure``
        node so images render correctly without sphinx-gallery.
        """
        option_spec = {**_Figure.option_spec, "srcset": _directives.unchanged}

    _directives.register_directive("image-sg", _ImageSgStub)

if _RUN_GALLERY:
    # ── Gallery caching with SFI-source staleness detection ──
    # Sphinx-gallery only re-runs examples whose source file changed (md5).
    # We extend this: *also* invalidate the cache when the SFI library itself
    # changes (new commit, refactored API, etc.).  A hash of every SFI/*.py
    # file is stored in <builddir>/.sfi_hash.  If it differs, we ``touch``
    # all example files so sphinx-gallery sees them as modified.

    import hashlib as _hashlib

    def _sfi_source_hash() -> str:
        """MD5 digest of every ``*.py`` file under the SFI package."""
        h = _hashlib.md5()
        for py_file in sorted(PKG_DIR.rglob("*.py")):
            h.update(py_file.read_bytes())
        return h.hexdigest()

    def _invalidate_stale_gallery() -> None:
        """Touch all gallery scripts if the SFI library has changed."""
        if os.environ.get("SFI_DOCS_SKIP_STALE") == "1":
            return
        stamp = Path(DOCS_SRC).parent / "build" / ".sfi_hash"
        current = _sfi_source_hash()
        if stamp.exists() and stamp.read_text().strip() == current:
            return  # library unchanged — gallery cache still valid
        # Library changed → force re-run of gallery demos only.
        # Benchmarks are NEVER re-run during builds; they use their own
        # BenchmarkCache with SFI-hash staleness detection.
        gallery_dir = REPO_ROOT / "examples" / "gallery"
        if gallery_dir.is_dir():
            # rglob so demos in subsections (e.g. advanced/) are caught too
            for script in gallery_dir.rglob("*_demo.py"):
                script.touch()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(current + "\n")

    _invalidate_stale_gallery()

    from sphinx_gallery.sorting import ExplicitOrder

    _example_dirs = [str(REPO_ROOT / "examples" / "gallery")]
    _gallery_dirs = ["gallery"]

    _filename_pattern = os.environ.get(
        "SFI_DOCS_GALLERY_FILTER", r"/.*_demo\.py$"
    )

    def _reset_jax_between_examples(*args, **kwargs):
        """Free JAX compilation caches between gallery demos.

        Called by sphinx-gallery before and after every example.  Live
        arrays are left alone (helper modules may legitimately hold
        state); the compilation caches are the dominant leak.
        """
        import gc

        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    sphinx_gallery_conf = {
        "examples_dirs": _example_dirs,
        "gallery_dirs": _gallery_dirs,
        "filename_pattern": _filename_pattern,
        "ignore_pattern": r"__init__\.py",
        "within_subsection_order": ExplicitOrder([
            # --- main gallery ---
            "ou_demo.py",               # simplest — 1D hello-world
            "limitcycle_demo.py",       # 2D nonlinear, non-equilibrium
            "lorenz_demo.py",           # 3D, sparsity, chaotic
            "lotka_volterra_demo.py",   # 6D ecology, sparse network (parametric, labelled)
            "experimental_workflow_demo.py",  # real-data template + dirty data
            "custom_basis_demo.py",     # hand-crafted basis with make_basis
            "multiplicative_diffusion_demo.py",  # state-dependent D(x), blowtorch
            "anisotropic_diffusion_demo.py",     # anisotropic tensor field D(x)
            "van_der_pol_demo.py",      # underdamped
            "velocity_dependent_noise_demo.py",  # underdamped multiplicative D(v)
            "home_range_demo.py",       # underdamped multi-particle, per-agent params
            "time_dependent_forcing_demo.py",  # protocols as extras
            "time_fourier_demo.py",     # learning unknown F(x,t) via time-Fourier
            "diagnostics_demo.py",      # assessing fit quality
            "dynamics_order_demo.py",   # overdamped-vs-underdamped classifier
            "entropy_production_demo.py",  # stochastic thermodynamics, dissipation
            "entropy_production_underdamped_demo.py",  # inertial dissipation, positions only
            "abp_align_demo.py",        # ABP, interactions, movies
            "abp_nonreciprocal_demo.py", # large-scale nonreciprocal ABPs
            "abp_to_spde_demo.py",       # ABP -> Toner-Tu SPDE discovery
            "gray_scott_demo.py",        # SPDE + multi-regime recovery
            # --- advanced/ subsection ---
            "nn_force_demo.py",          # neural-net force field
            "multi_experiment_demo.py",  # multi-experiment ABP (nonlinear)
            "flocking_3d_demo.py",       # underdamped multi-particle parametric
            "*",                        # catch-all for benchmarks & unlisted
        ]),
        # Render subsections (e.g. advanced/) inline on the single gallery
        # index page rather than as separate nested index pages.
        "nested_sections": False,
        "default_thumb_file": None,
        "line_numbers": False,
        "download_all_examples": False,
        "plot_gallery": "True",
        "abort_on_example_error": False,
        "min_reported_time": 5,
        "expected_failing_examples": [],
        "matplotlib_animations": (True, "mp4"),
        "run_stale_examples": _RUN_STALE,
        "remove_config_comments": True,
        "parallel": 1,
        # All demos execute in ONE sphinx process; JAX compilation caches
        # and buffers otherwise accumulate across demos (RSS was observed
        # to ratchet past 16 GB on a full gallery run).  Reset between
        # examples.
        "reset_modules": ("matplotlib", "seaborn", _reset_jax_between_examples),
    }

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_title = "SFI documentation"
html_css_files = ["custom.css"]
# sg-tags.js renders the gallery tag-filter bar from the data-sgtags
# attributes baked into the committed index.  Full gallery builds get it
# from sphinx_gallery.gen_gallery itself; gallery-less builds (RTD serves
# the committed pages) need the committed _static copy registered here.
html_js_files = ["download_fix.js"] + ([] if _RUN_GALLERY else ["sg-tags.js"])

# Pygments syntax highlighting — use Furo defaults for light, monokai for dark
pygments_dark_style = "monokai"

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Autodoc / autosummary / napoleon
# ---------------------------------------------------------------------------

# Let autosummary render tables, but do NOT auto-generate stub .rst files.
# Stubs are generated explicitly via the _run_apidoc helper (see below)
# when SFI_DOCS_RUN_APIDOC=1 is set.
autosummary_generate = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # Document methods inherited from BaseLangevinInference under each engine
    # so the canonical ``OverdampedLangevinInference.<method>`` cross-refs
    # resolve.  Excluding ``Module`` and ``int`` stops the walk before
    # equinox.Module/object and the stdlib int (so IntEnum members like
    # ``Rank`` don't document ``int.from_bytes`` & co.).
    "inherited-members": "Module,int",
}

autodoc_typehints = "description"
set_type_checking_flag = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
napoleon_use_ivar = True
napoleon_custom_sections = [
    ("Key Features", "notes"),
    ("Workflow", "notes"),
    ("Indices Convention", "notes"),
    ("Logging", "notes"),
    ("Core Equation", "notes"),
    ("Observables", "notes"),
    ("Rules", "notes"),
    ("Contractions", "notes"),
    ("Modes", "notes"),
    ("Shape effect", "notes"),
    ("Side effects", "notes"),
    ("Outputs", "notes"),
    ("Contract rules", "notes"),
    ("Shapes produced", "notes"),
    ("Mask handling", "notes"),
    ("Extras policy", "notes"),
    ("Updates", "notes"),
]

# Suppress noisy cross-reference warnings for JAX/NumPy types, and the
# config-cache notice from the callable in sphinx_gallery_conf's
# reset_modules (the incremental config cache is irrelevant for the
# opt-in gallery build).
suppress_warnings = ["ref.python", "config.cache"]

# ---------------------------------------------------------------------------
# Optional: generate API stubs with sphinx-apidoc
# ---------------------------------------------------------------------------


def _run_apidoc(_app) -> None:
    """
    Regenerate ``source/api/*.rst`` from the SFI package using sphinx-apidoc.

    This is wired through an environment variable so you can run:

        SFI_DOCS_RUN_APIDOC=1 make html

    when you want to refresh the API pages, and plain ``make html`` for normal
    documentation builds.
    """
    from sphinx.ext.apidoc import main as apidoc_main

    shutil.rmtree(API_OUT, ignore_errors=True)
    API_OUT.mkdir(parents=True, exist_ok=True)

    apidoc_args = [
        "-f",  # --force: overwrite
        "-e",  # --separate: one file per module
        "-M",  # --module-first: put module before package in headings
        "-o",
        str(API_OUT),
        str(PKG_DIR),
        # Exclude legacy / internal-only modules:
        str(PKG_DIR / "statefunc" / "OLD_stateexpr.py"),
        str(PKG_DIR / "langevin" / "base.py"),  # re-exported in SFI.langevin
        # Exclude subpackages whose public API is re-exported via
        # __init__.py one level up — avoids duplicate-object warnings.
        str(PKG_DIR / "statefunc" / "nodes"),
        # Internal parametric-inference implementation (RK4 Jacobian kernels,
        # solvers): not part of the public surface — keep it out of the API.
        str(PKG_DIR / "inference" / "parametric_core"),
        # ``assess`` the function shares its name with ``assess`` the module;
        # drop the module stub so the re-exported function is the sole target.
        str(PKG_DIR / "diagnostics" / "assess.py"),
    ]
    apidoc_main(apidoc_args)

    # apidoc emits a top-level ``modules.rst`` TOC that no other document
    # includes; mark it :orphan: so it does not trip the "not in any toctree"
    # warning.  The per-module stubs stay reachable as cross-reference targets.
    modules_rst = API_OUT / "modules.rst"
    if modules_rst.exists():
        modules_rst.write_text(":orphan:\n\n" + modules_rst.read_text())


# ---------------------------------------------------------------------------
# Docstring post-processing
# ---------------------------------------------------------------------------

# Modules whose docstrings are known to be RST-safe (or have been audited).
# All other modules will have their docstrings rendered as literal blocks
# to prevent Sphinx build errors from unpolished RST markup.
_RST_SAFE_MODULES = {
    # top-level
    "SFI",
    # subpackage __init__
    "SFI.bases",
    "SFI.inference",
    "SFI.integrate",
    "SFI.langevin",
    "SFI.statefunc",
    "SFI.trajectory",
    "SFI.utils",
    # inference
    "SFI.inference.base",
    "SFI.inference.overdamped",
    "SFI.inference.underdamped",
    "SFI.inference.result",
    "SFI.inference.sparsity",
    "SFI.inference.serialization",
    # langevin
    "SFI.langevin.base",
    "SFI.langevin.overdamped",
    "SFI.langevin.underdamped",
    # statefunc (public façades)
    "SFI.statefunc.factory",
    "SFI.statefunc.basis",
    "SFI.statefunc.psf",
    "SFI.statefunc.sf",
    "SFI.statefunc.stateexpr",
    "SFI.statefunc.interactor",
    # trajectory
    "SFI.trajectory.collection",
    "SFI.trajectory.dataset",
    "SFI.trajectory.degrade",
    "SFI.trajectory.io",
    # bases
    "SFI.bases.monomials",
    "SFI.bases.constants",
    "SFI.bases.linear",
    "SFI.bases.pairs",
    "SFI.bases.spde",
    # utils
    "SFI.utils.maths",
    "SFI.utils.formatting",
    "SFI.utils.plotting",
}


def _literalize_docstrings(app, what, name, obj, options, lines) -> None:
    """
    Render docstrings from un-audited modules as literal blocks to prevent
    Sphinx build errors from unpolished RST markup.

    Modules listed in ``_RST_SAFE_MODULES`` are rendered normally.
    Set ``SFI_DOCS_ALLOW_RST=1`` in the environment to allow *all*
    docstrings to be interpreted as RST (useful for auditing).
    """
    if not lines:
        return
    if os.environ.get("SFI_DOCS_ALLOW_RST") == "1":
        return

    # Check if the object belongs to an RST-safe module
    module = name.rsplit(".", 1)[0] if "." in name else name
    while module:
        if module in _RST_SAFE_MODULES:
            return
        # Walk up: SFI.foo.bar.Baz -> SFI.foo.bar -> SFI.foo -> SFI
        if "." in module:
            module = module.rsplit(".", 1)[0]
        else:
            break

    # Prefix docstring with "::" and indent, so it is rendered verbatim.
    lines[:] = ["::", ""] + [f"    {ln}" for ln in lines]


# ---------------------------------------------------------------------------
# Sphinx setup hook
# ---------------------------------------------------------------------------


def setup(app) -> None:
    # Regenerate API stubs automatically on Read the Docs (it always sets
    # READTHEDOCS=True), or on demand locally with SFI_DOCS_RUN_APIDOC=1.
    # Without these stubs the curated reference pages carry ``:no-index:``,
    # so API cross-references (:class:/:meth:/:func:) have no target and
    # render as plain text — apidoc supplies the canonical link targets.
    if (
        os.environ.get("SFI_DOCS_RUN_APIDOC") == "1"
        or os.environ.get("READTHEDOCS") == "True"
    ):
        app.connect("builder-inited", _run_apidoc)

    # Guard docstrings from un-audited modules.
    app.connect("autodoc-process-docstring", _literalize_docstrings)

    # After sphinx-gallery generates its (stripped) .ipynb files,
    # overwrite them with complete standalone notebooks rebuilt from
    # the full .py sources.  Priority 900 ensures this runs AFTER
    # sphinx-gallery's builder-inited handler (default priority 500).
    if _RUN_GALLERY:
        app.connect("builder-inited", _regenerate_gallery_notebooks,
                     priority=900)


def _regenerate_gallery_notebooks(app) -> None:
    """Overwrite sphinx-gallery .ipynb with standalone notebooks."""
    import sys as _sys
    _sys.path.insert(0, str(DOCS_SRC.parent))
    from regenerate_notebooks import regenerate_gallery_notebooks
    _sys.path.pop(0)

    examples_dir = REPO_ROOT / "examples" / "gallery"
    # Write into the same directory sphinx-gallery uses for its .ipynb
    # so Sphinx's :download: role picks up our versions.
    output_dir = DOCS_SRC / "gallery"
    regenerate_gallery_notebooks(examples_dir, output_dir)
