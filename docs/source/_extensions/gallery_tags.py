"""
Gallery tags — Sphinx extension for SFI example tagging.

Reads ``sphinx_gallery_tags`` from each gallery example's module namespace
and injects colored badge pills into the generated RST.  Also generates a
tag index page at ``gallery/tags.rst``.

Usage in an example script::

    # sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "benchmark"]

Add to ``conf.py``::

    extensions = [
        ...
        "docs.source._extensions.gallery_tags",
    ]
"""

from __future__ import annotations

import re
from pathlib import Path

from docutils import nodes
from sphinx.application import Sphinx

# ---------------------------------------------------------------------------
# Tag vocabulary and colours (CSS-safe hex)
# ---------------------------------------------------------------------------

TAG_COLORS: dict[str, str] = {
    # data source
    "synthetic": "#6c757d",
    "real-data": "#198754",
    # dynamics
    "overdamped": "#0d6efd",
    "underdamped": "#6610f2",
    # inference
    "linear": "#0dcaf0",
    "nonlinear": "#fd7e14",
    "parametric": "#d63384",
    # features
    "sparsity": "#ffc107",
    "PASTIS": "#ffc107",
    "benchmark": "#dc3545",
    "multi-particle": "#20c997",
    "interactions": "#20c997",
    "state-dependent-diffusion": "#6f42c1",
    "non-equilibrium": "#e66100",
    # system dimension
    "1D": "#adb5bd",
    "2D": "#adb5bd",
    "3D": "#adb5bd",
    "SPDE": "#adb5bd",
}

DEFAULT_COLOR = "#6c757d"


def _badge_html(tag: str) -> str:
    """Return an inline HTML badge for *tag*."""
    color = TAG_COLORS.get(tag, DEFAULT_COLOR)
    # Choose white or dark text based on luminance
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = "#000" if lum > 140 else "#fff"
    return (
        f'<span style="display:inline-block;padding:2px 8px;margin:2px;'
        f"border-radius:10px;font-size:0.8em;font-weight:600;"
        f"background:{color};color:{text_color};"
        f'">{tag}</span>'
    )


# ---------------------------------------------------------------------------
# Sphinx event hooks
# ---------------------------------------------------------------------------

# Regex to find sphinx_gallery_tags = [...] in Python source
_TAGS_RE = re.compile(
    r"""^#?\s*sphinx_gallery_tags\s*=\s*\[([^\]]*)\]""",
    re.MULTILINE,
)


def _extract_tags(source_path: Path) -> list[str]:
    """Parse ``sphinx_gallery_tags`` from a Python file."""
    try:
        text = source_path.read_text(encoding="utf-8")
    except Exception:
        return []
    m = _TAGS_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    tags = [t.strip().strip("\"'") for t in raw.split(",") if t.strip().strip("\"'")]
    return tags


_ALL_TAGS: dict[str, list[str]] = {}  # tag -> list of example titles / paths


def _inject_tags_into_gallery(app: Sphinx, doctree, docname: str) -> None:
    """
    After sphinx-gallery generates RST, inject tag badges at the top of
    each gallery example page.  We detect gallery pages by looking for
    ``gallery/`` in the docname.
    """
    if "gallery/" not in docname:
        return

    # Try to find the source .py path from sphinx-gallery metadata
    # Gallery pages are named like gallery/lorenz/lorenz_demo
    # Corresponding source is in examples/.../*_demo.py
    # We look for the download reference node which has the .py path
    source_path = None
    sg_conf = getattr(app.config, "sphinx_gallery_conf", None)
    if sg_conf:
        for examples_dir, gallery_dir in zip(
            sg_conf.get("examples_dirs", []),
            sg_conf.get("gallery_dirs", []),
        ):
            if gallery_dir.rstrip("/") in docname:
                # Derive the .py file name from docname
                basename = docname.rsplit("/", 1)[-1]
                candidate = Path(examples_dir) / (basename + ".py")
                if candidate.exists():
                    source_path = candidate
                    break

    if source_path is None:
        return

    tags = _extract_tags(source_path)
    if not tags:
        return

    # Register in global index
    for tag in tags:
        _ALL_TAGS.setdefault(tag, []).append(docname)

    # Build HTML badges
    badges_html = " ".join(_badge_html(t) for t in tags)
    raw_node = nodes.raw(
        "",
        f'<div style="margin-bottom:12px;">{badges_html}</div>',
        format="html",
    )

    # Insert after the first title node
    for i, node in enumerate(doctree.children):
        if isinstance(node, nodes.section):
            # Insert badges after the section title
            for j, child in enumerate(node.children):
                if isinstance(child, nodes.title):
                    node.insert(j + 1, raw_node)
                    return
            break

    # Fallback: insert at the very top
    doctree.insert(0, raw_node)


def _write_tag_index(app: Sphinx, exception) -> None:
    """
    After build, write ``gallery/tags.rst`` listing all tags found.

    The index is rebuilt from the *source of truth* — the
    ``sphinx_gallery_tags`` in each demo ``.py`` plus the generated
    ``gallery/**/*_demo.rst`` pages — rather than from in-memory state
    populated only while pages are (re-)read.  This makes ``tags.rst``
    complete and identical on **every** build, including incremental and
    gallery-less ones, so it is never clobbered with a partial/empty file
    (which previously required a manual ``git checkout`` to recover).
    """
    if exception:
        return

    srcdir = Path(app.srcdir)
    gallery_root = srcdir / "gallery"
    # Demo sources live in <repo>/examples/gallery; docs/source -> repo root.
    examples_root = srcdir.parents[1] / "examples" / "gallery"

    src_by_stem: dict[str, Path] = {}
    if examples_root.is_dir():
        for py in examples_root.rglob("*_demo.py"):
            src_by_stem.setdefault(py.stem, py)

    all_tags: dict[str, list[str]] = {}
    if gallery_root.is_dir():
        for rst in gallery_root.rglob("*_demo.rst"):
            src = src_by_stem.get(rst.stem)
            if src is None:
                continue
            docname = rst.relative_to(srcdir).with_suffix("").as_posix()
            for tag in _extract_tags(src):
                all_tags.setdefault(tag, []).append(docname)

    if not all_tags:
        # Sources unavailable (e.g. examples/ absent) — never overwrite the
        # committed index with an empty file.
        return

    lines = [
        ":orphan:",
        "",
        "Gallery tags",
        "============",
        "",
        "Examples grouped by tag.  Each example can carry multiple tags.",
        "",
    ]
    for tag in sorted(all_tags):
        lines.append(f".. rubric:: {tag}")
        lines.append("")
        for docname in sorted(set(all_tags[tag])):
            title = docname.rsplit("/", 1)[-1].replace("_", " ").title()
            lines.append(f"- :doc:`/{docname}` — *{title}*")
        lines.append("")

    (gallery_root / "tags.rst").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict:
    app.connect("doctree-resolved", _inject_tags_into_gallery)
    app.connect("build-finished", _write_tag_index)
    return {"version": "0.1", "parallel_read_safe": True}
