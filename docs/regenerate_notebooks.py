#!/usr/bin/env python3
"""
Regenerate gallery Jupyter notebooks from the original Python scripts.

sphinx-gallery strips ``sphinx_gallery_start_ignore`` / ``end_ignore``
blocks from *both* the rendered HTML **and** the generated ``.ipynb``
files.  This is by design: the blocks hide boilerplate from the gallery
page, but it also removes the imports and plotting code that the
notebook needs.

This script rebuilds complete, standalone ``.ipynb`` notebooks from
the full ``examples/gallery/*_demo.py`` source files, including the
ignored blocks, so that users who download the notebooks can run them
directly.

Modifications applied during regeneration:

1. **All code is included** — ``sphinx_gallery_start_ignore`` /
   ``end_ignore`` markers are removed (they are sphinx-gallery-only
   directives).
2. **sys.path hacks are removed** — the ``sys.path.insert(...)`` line
   used for the gallery build is stripped.  Users are expected to have
   SFI installed via ``pip install``.
3. **``_gallery_utils`` imports are replaced**: names SFI provides are
   imported from ``SFI.utils.plotting``; any other gallery helper (e.g.
   ``_gallery_utils.abp``, ``coarse_grain``, ``helpers.plot_kernel_ci``) is
   inlined in a code cell after the imports, so the notebook is standalone.
   Subsection demos (``advanced/``) are regenerated too.
4. **sphinx_gallery_* config comments** are stripped.
5. **RST-specific markup** (``:math:`...```, ``:func:`...```) is
   converted to plain-text / MathJax equivalents for Jupyter.

Usage::

    python docs/regenerate_notebooks.py            # from repo root
    python docs/regenerate_notebooks.py -o /tmp/nb  # custom output dir

Can also be called as a library from ``conf.py``::

    from regenerate_notebooks import regenerate_gallery_notebooks
    regenerate_gallery_notebooks(examples_dir, output_dir)
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "gallery"
_OUTPUT_DIR = _REPO_ROOT / "docs" / "source" / "gallery"

# Lines we always strip from cell source
_STRIP_PATTERNS = [
    # sphinx-gallery config comments
    re.compile(r"^\s*#\s*sphinx_gallery_\w+\s*=.*$"),
    # sphinx_gallery_start/end_ignore markers
    re.compile(r"^\s*#\s*sphinx_gallery_start_ignore\s*$"),
    re.compile(r"^\s*#\s*sphinx_gallery_end_ignore\s*$"),
    # sys.path.insert hack for gallery builds
    re.compile(r"^\s*sys\.path\.insert\(.*\)\s*$"),
    # The if "__file__" guard around sys.path (keep other uses)
    re.compile(r"^\s*if\s+[\"']__file__[\"']\s+in\s+dir\(\).*:.*$"),
    # Standalone `import sys` (used only for the path hack in gallery scripts)
    re.compile(r"^\s*import\s+sys\s*$"),
]

# Import replacement: replace _gallery_utils imports with SFI.utils.plotting
_GALLERY_IMPORT_PAT = re.compile(
    r"^from\s+_gallery_utils\.helpers\s+import\s+(.+)$", re.MULTILINE
)

# apply_style() → we want to keep it but make it import from the style file
# We'll replace it with a local define if the style file path is unknown


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def _parse_py_to_cells(source: str) -> list[dict]:
    """Parse a sphinx-gallery-style .py file into notebook cells.

    The file format is:
    - A docstring at the top → first markdown cell
    - ``# %%`` markers separate subsequent cells
    - Comment blocks (``# text``) between ``# %%`` → markdown cells
    - Code between ``# %%`` markers → code cells
    """
    cells: list[dict] = []

    # --- Extract docstring as first markdown cell ---
    # Match both regular (""") and raw (r""") docstrings
    docstring_match = re.match(r'^r?("""|\'\'\')(.*?)\1', source, re.DOTALL)
    if docstring_match:
        doc_text = docstring_match.group(2).strip()
        doc_text = _rst_to_md(doc_text)
        cells.append(_md_cell(doc_text))
        # Remove the docstring from source
        source = source[docstring_match.end():]

    # --- Split remainder by # %% markers ---
    # The # %% at the start of a line marks a new cell boundary
    segments = re.split(r"^# %%[^\n]*\n", source, flags=re.MULTILINE)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check if this segment starts with comment lines (RST prose)
        # and may also contain code
        lines = segment.split("\n")

        # Find where the comment header ends and code begins
        comment_lines = []
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped == "":
                comment_lines.append(line)
                code_start = i + 1
            else:
                break

        # Extract the comment block as markdown
        if comment_lines:
            md_text = _comments_to_markdown(comment_lines)
            if md_text.strip():
                cells.append(_md_cell(md_text))

        # Extract the code block
        code_lines = lines[code_start:]
        if code_lines:
            code = "\n".join(code_lines).strip()
            if code:
                code = _clean_code(code)
                if code.strip():
                    cells.append(_code_cell(code))

    return cells


def _comments_to_markdown(lines: list[str]) -> str:
    """Convert RST comment lines to markdown text.

    Lines of the form ``# text`` become ``text``.
    Lines of the form ``#`` become empty lines.
    sphinx_gallery directives are silently dropped.
    """
    md_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip sphinx_gallery directive comments
        if re.match(r"^#\s*sphinx_gallery_", stripped):
            continue
        if stripped == "#":
            md_lines.append("")
        elif stripped.startswith("# "):
            md_lines.append(stripped[2:])
        elif stripped.startswith("#"):
            md_lines.append(stripped[1:])
        else:
            # blank line between comments
            md_lines.append("")
    text = "\n".join(md_lines).strip()
    # Convert RST directives to rough markdown equivalents
    text = _rst_to_md(text)
    return text


def _rst_to_md(text: str) -> str:
    """Best-effort RST → Markdown conversion for notebook cells."""
    # :math:`...` → $...$
    text = re.sub(r":math:`([^`]+)`", r"$\1$", text)
    # :func:`~Mod.func` → `func`
    text = re.sub(r":(func|class|meth|mod|attr):`~?[\w.]*?(\w+)`", r"`\2`", text)
    # .. math:: block → $$ block
    def _math_block(m):
        body = m.group(1)
        # Dedent the body
        body = textwrap.dedent(body).strip()
        return f"\n$$\n{body}\n$$\n"
    text = re.sub(
        r"^\.\.\s+math::\s*\n(?:\s*\n)*((?:[ \t]+\S.*(?:\n|$))+)",
        _math_block,
        text,
        flags=re.MULTILINE,
    )
    # .. rubric:: Title → **Title**
    text = re.sub(r"^\.\.\s+rubric::\s*(.+)$", r"**\1**", text, flags=re.MULTILINE)
    # .. note:: / .. admonition:: → **Note:**
    text = re.sub(
        r"^\.\.\s+(note|admonition|warning)::\s*(.*)$",
        r"> **\1:** \2",
        text,
        flags=re.MULTILINE,
    )
    # RST section underlines → markdown headers
    # =====  → ## (under a title)
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if (i + 1 < len(lines) and
                re.match(r"^[=\-~^]+$", lines[i + 1].strip()) and
                len(lines[i + 1].strip()) >= len(lines[i].strip()) and
                lines[i].strip()):
            char = lines[i + 1].strip()[0]
            level = {"=": "#", "-": "##", "~": "###", "^": "####"}.get(char, "##")
            out.append(f"{level} {lines[i].strip()}")
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _clean_code(code: str) -> str:
    """Apply code-level transformations for notebook standalone use."""
    lines = code.split("\n")
    cleaned = []
    skip_paren = False       # inside multiline import parentheses
    _paren_imports = []      # names collected during multiline import
    skip_if_depth = 0        # >0 when inside a stripped if/else block
    skip_if_indent = -1      # indentation of the stripped ``if`` line

    for line in lines:
        stripped_line = line.lstrip()
        indent = len(line) - len(stripped_line)

        # --- Skip children of a stripped ``if`` block ----------------------
        if skip_if_depth > 0:
            if stripped_line == "":
                # blank lines inside the block — skip
                continue
            if indent > skip_if_indent:
                # indented deeper → still inside the block
                continue
            if stripped_line.startswith(("else:", "elif ")):
                # same indent → sibling branch of the stripped ``if``
                continue
            # Back to the same or lesser indent → block ended
            skip_if_depth = 0
            skip_if_indent = -1

        # --- Pattern-based stripping ---------------------------------------
        should_strip = False
        for pat in _STRIP_PATTERNS:
            if pat.match(line):
                should_strip = True
                break
        if should_strip:
            # When stripping an ``if`` statement, also skip its body
            if re.match(r"^\s*if\s+", line):
                skip_if_depth = 1
                skip_if_indent = indent
            continue

        # --- Strip any remaining _gallery_utils references -----------------
        if "_gallery_utils" in line:
            # Replace ``from _gallery_utils.helpers import ...`` with SFI
            m = _GALLERY_IMPORT_PAT.match(stripped_line)
            if m:
                rest = m.group(1).strip()
                # Multiline import: ``from _gallery_utils.helpers import (``
                if "(" in rest and ")" not in rest:
                    skip_paren = True
                    # Collect any names on the opening line after ``(``
                    after_paren = rest.split("(", 1)[1]
                    _paren_imports = [
                        n.strip().rstrip(",").strip()
                        for n in after_paren.split(",")
                        if n.strip().rstrip(",").strip()
                    ]
                    continue
                # Single-line import (may contain parens around names)
                imports = [s.strip() for s in rest.strip("()").split(",")]
                sfi_imports = []
                for imp in imports:
                    imp = imp.strip().rstrip(",").strip()
                    if not imp:
                        continue
                    if imp in _HELPERS_FROM_SFI:
                        sfi_imports.append(imp)
                    # else: silently drop (apply_style, etc.)
                if sfi_imports:
                    cleaned.append(
                        "from SFI.utils.plotting import "
                        + ", ".join(sfi_imports)
                    )
                continue

            # Multiline import with parens (fallback — no regex match)
            if "import" in line and "(" in line and ")" not in line:
                skip_paren = True
                _paren_imports = []
                continue
            # Any other _gallery_utils reference → drop
            continue

        # --- Skip inside multiline parenthesised import -------------------
        if skip_paren:
            if ")" in line:
                skip_paren = False
                # Collect any final names before the closing paren
                before_paren = line.split(")")[0]
                for name in before_paren.split(","):
                    name = name.strip().rstrip(",").strip()
                    if name:
                        _paren_imports.append(name)
                # Now emit the replacement import
                sfi_imports = [
                    n for n in _paren_imports
                    if n in _HELPERS_FROM_SFI
                ]
                if sfi_imports:
                    cleaned.append(
                        "from SFI.utils.plotting import "
                        + ", ".join(sfi_imports)
                    )
                _paren_imports = []
            else:
                # Collect names from continuation lines
                for name in line.split(","):
                    name = name.strip().rstrip(",").strip()
                    if name and not name.startswith("#"):
                        _paren_imports.append(name)
            continue

        # --- Strip apply_style() call (gallery-only mplstyle) --------------
        if stripped_line == "apply_style()":
            continue

        cleaned.append(line)

    result = "\n".join(cleaned)

    # If `_here` is referenced but never assigned, inject a fallback.
    # This happens when the if/__file__ block that defined it was stripped.
    # In the gallery build, _here points to examples/gallery/.
    if re.search(r"\b_here\b", result) and not re.search(r"^_here\s*=", result, re.MULTILINE):
        fallback = (
            "from pathlib import Path\n"
            "# _here should point to examples/gallery/ in the SFI repository.\n"
            "# Adjust this path if running from a different location.\n"
            "_here = Path('examples/gallery') if Path('examples/gallery').exists() "
            "else Path('.')\n"
        )
        result = fallback + result

    # Collapse runs of 3+ blank lines down to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _handle_multiline_gallery_import(lines: list[str], start: int) -> tuple[list[str], int]:
    """Handle multiline _gallery_utils imports spanning multiple lines."""
    import_names = []
    i = start
    while i < len(lines):
        line = lines[i].strip().rstrip(",").strip()
        if ")" in line:
            # Last line of import
            name = line.replace(")", "").strip().rstrip(",").strip()
            if name:
                import_names.append(name)
            break
        if line and not line.startswith("#") and not line.startswith("from") and not line.startswith("import"):
            import_names.append(line)
        i += 1

    result = []
    sfi_names = [n for n in import_names
                 if n in ("SFI_COLORS", "apply_style", "plot_pareto_front",
                          "plot_recovery_bar", "run_degradation_sweep",
                          "plot_nmse_vs_param", "stamp_output")]
    if sfi_names:
        result.append("from SFI.utils.plotting import " + ", ".join(sfi_names))
    return result, i


# ---------------------------------------------------------------------------
# Notebook cell constructors
# ---------------------------------------------------------------------------


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [text],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": [code],
    }


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def _make_notebook(cells: list[dict]) -> dict:
    """Wrap cells in a minimal notebook-v4 structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Gallery helper code inlined into the notebook
# ---------------------------------------------------------------------------

_GALLERY_UTILS_DIR = _REPO_ROOT / "examples" / "_gallery_utils"

# ``_gallery_utils.helpers`` names that SFI itself provides (imported from
# ``SFI.utils.plotting`` by ``_clean_code``) or that are gallery-only styling.
_HELPERS_FROM_SFI = {"SFI_COLORS", "plot_pareto_front", "plot_recovery_bar", "stamp_output"}
_HELPERS_DROPPED = {"apply_style", "stamp_fig"}

_ANY_GALLERY_IMPORT = re.compile(
    r"^[ \t]*from[ \t]+_gallery_utils\.(\w+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.MULTILINE
)


def _needed_gallery_code(source: str) -> dict[str, set[str]]:
    """``{module: names}`` imported from ``_gallery_utils`` that SFI does not provide."""
    needed: dict[str, set[str]] = {}
    for mod, names in _ANY_GALLERY_IMPORT.findall(source):
        names = {n.strip() for n in names.strip("()").replace("\n", ",").split(",")}
        names = {n.split(" as ")[0].strip() for n in names if n.strip() and not n.strip().startswith("#")}
        if mod == "helpers":
            names -= _HELPERS_FROM_SFI | _HELPERS_DROPPED
        if names:
            needed.setdefault(mod, set()).update(names)
    return needed


def _module_code(mod: str, names: set[str]) -> str:
    """Source needed for ``names`` from ``examples/_gallery_utils/<mod>.py``.

    Self-contained modules are inlined whole (minus the module docstring and
    ``from __future__``); from ``helpers`` only the requested top-level
    definitions are extracted, since the rest is gallery build plumbing.
    """
    import ast

    path = _GALLERY_UTILS_DIR / f"{mod}.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    if mod != "helpers":
        body = tree.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        parts = [ast.get_source_segment(src, node) for node in body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
        return "\n\n".join(p for p in parts if p)
    parts = ["import numpy as np", "import matplotlib.pyplot as plt", "from SFI.utils.plotting import SFI_COLORS"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            parts.append(ast.get_source_segment(src, node))
    return "\n\n".join(parts)


def _helper_cells(source: str) -> list[dict]:
    needed = _needed_gallery_code(source)
    if not needed:
        return []
    code = "\n\n\n".join(
        f"# --- from examples/_gallery_utils/{mod}.py ---\n{_module_code(mod, names)}"
        for mod, names in sorted(needed.items())
    )
    note = ("Helper code from the SFI gallery (``examples/_gallery_utils/``), "
            "inlined so that this notebook runs on its own.")
    return [_md_cell(note), _code_cell(code)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def regenerate_notebook(py_path: Path, output_dir: Path) -> Path:
    """Convert a single gallery .py script to a standalone .ipynb.

    Parameters
    ----------
    py_path : Path
        Source script, e.g. ``examples/gallery/ou_demo.py``.
    output_dir : Path
        Directory to write the ``.ipynb`` file.

    Returns
    -------
    Path
        Path to the written notebook file.
    """
    source = py_path.read_text(encoding="utf-8")
    cells = _parse_py_to_cells(source)
    extra = _helper_cells(source)
    if extra:
        first_code = next((i for i, c in enumerate(cells) if c["cell_type"] == "code"), len(cells) - 1)
        cells[first_code + 1:first_code + 1] = extra
    # Deterministic cell ids (from the notebook name, position and content):
    # regenerating an unchanged demo then rewrites an identical notebook.
    import hashlib

    for i, cell in enumerate(cells):
        key = f"{py_path.name}:{i}:{''.join(cell['source'])}".encode()
        cell["id"] = hashlib.sha1(key).hexdigest()[:8]
    nb = _make_notebook(cells)

    nb_path = output_dir / py_path.with_suffix(".ipynb").name
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return nb_path


def regenerate_gallery_notebooks(
    examples_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    verbose: bool = True,
) -> list[Path]:
    """Regenerate all gallery notebooks from their .py sources.

    Parameters
    ----------
    examples_dir : path-like, optional
        Directory containing ``*_demo.py`` scripts.
        Defaults to ``<repo>/examples/gallery/``.
    output_dir : path-like, optional
        Where to write ``.ipynb`` files.
        Defaults to ``<repo>/docs/source/gallery/``.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list of Path
        Paths to all regenerated notebook files.
    """
    examples_dir = Path(examples_dir or _EXAMPLES_DIR)
    output_dir = Path(output_dir or _OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top-level demos and one level of subsections (e.g. ``advanced/``),
    # mirrored into the same sub-directory of the output.
    scripts = sorted(examples_dir.glob("*_demo.py")) + sorted(examples_dir.glob("*/*_demo.py"))
    if verbose:
        print(f"regenerate_notebooks: {len(scripts)} scripts in {examples_dir}")

    written = []
    for py_path in scripts:
        out = output_dir / py_path.parent.relative_to(examples_dir)
        out.mkdir(parents=True, exist_ok=True)
        nb_path = regenerate_notebook(py_path, out)
        written.append(nb_path)
        if verbose:
            print(f"  ✓ {py_path.name} → {nb_path.name}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate gallery .ipynb from .py sources"
    )
    parser.add_argument(
        "-e", "--examples-dir",
        type=Path,
        default=_EXAMPLES_DIR,
        help="Directory with *_demo.py scripts",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Directory to write .ipynb files",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()
    regenerate_gallery_notebooks(
        args.examples_dir, args.output_dir, verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
