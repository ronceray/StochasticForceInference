"""
physics_ref — Sphinx extension for physics formula flagging.

Provides a ``.. physics::`` directive that renders as a styled admonition
and collects all occurrences so they can be rendered on a single
"Physics Reference" index page via the ``.. physicslist::`` directive.

Usage in a Python docstring or .rst file::

    .. physics:: Euler–Maruyama integrator (overdamped)
       :label: euler-maruyama-overdamped
       :category: Simulation

       .. math::

          x_{t+\\mathrm{d}t} = x_t + \\mathrm{d}t\\, F(x_t)
          + \\sqrt{\\mathrm{d}t}\\, B(x_t)\\,\\xi_t

       where :math:`B = \\sqrt{2D}` and :math:`\\xi \\sim \\mathcal{N}(0,I)`.

Then, in ``physics_reference.rst``::

    .. physicslist::

Add to ``conf.py``::

    extensions = [..., "physics_ref"]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.docutils import SphinxDirective

# ─────────────────────── custom nodes ────────────────────────


class physics_node(nodes.Admonition, nodes.Element):
    """Node for a single ``.. physics::`` entry."""


class physicslist_node(nodes.General, nodes.Element):
    """Placeholder replaced at doctree-resolved time."""


# ─────────────────────── visit / depart ──────────────────────

_BADGE_CSS = (
    "display:inline-block; padding:2px 8px; margin:0 4px 0 0;"
    "border-radius:10px; font-size:0.75em; font-weight:600;"
)

_CAT_COLORS = {
    "Dynamical equations": "#0d6efd",
    "Simulation": "#6610f2",
    "Estimator": "#198754",
    "Inference": "#fd7e14",
    "Error analysis": "#dc3545",
    "Model selection": "#ffc107",
    "Observable": "#0dcaf0",
    "Basis functions": "#20c997",
    "Kinematics": "#6f42c1",
}
_DEFAULT_CAT_COLOR = "#6c757d"


def _cat_badge_html(cat: str) -> str:
    color = _CAT_COLORS.get(cat, _DEFAULT_CAT_COLOR)
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    text_color = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
    return (
        f'<span style="{_BADGE_CSS}background:{color};color:{text_color};">'
        f"{cat}</span>"
    )


def visit_physics_node(self, node):
    self.visit_admonition(node)


def depart_physics_node(self, node):
    self.depart_admonition(node)


# ─────────────────────── directives ──────────────────────────


class PhysicsDirective(SphinxDirective):
    """
    ``.. physics:: Title``

    Options
    -------
    :label:    unique cross-ref id (e.g. ``euler-maruyama``)
    :category: grouping key for the physicslist index
    """

    has_content = True
    required_arguments = 1  # title
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: ClassVar[dict] = {
        "label": directives.unchanged_required,
        "category": directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        env: BuildEnvironment = self.env

        # -- Parse content into nested nodes (supports .. math:: etc.)
        content_node = nodes.container()
        self.state.nested_parse(
            self.content, self.content_offset, content_node
        )

        # -- Build the admonition node
        title_text = self.arguments[0]
        label = self.options.get("label", "")
        category = self.options.get("category", "")

        node = physics_node("\n")
        node["classes"] += ["admonition", "note"]

        # Title with optional category badge
        title_prefix = ""
        if category:
            title_prefix = f"[{category}] "
        title_node = nodes.title(title_text, "", nodes.Text(title_prefix + title_text))
        node += title_node
        node += content_node.children

        # -- Make an inline target for cross-referencing
        target_id = f"physics-{label}" if label else f"physics-{env.new_serialno('physics')}"
        target_node = nodes.target("", "", ids=[target_id])

        # -- Store in the environment for physicslist
        if not hasattr(env, "physics_all"):
            env.physics_all = []

        entry = {
            "docname": env.docname,
            "target_id": target_id,
            "title": title_text,
            "label": label,
            "category": category,
            "content_node": node.deepcopy(),
        }
        env.physics_all.append(entry)  # type: ignore[attr-defined]

        return [target_node, node]


class PhysicsListDirective(SphinxDirective):
    """``.. physicslist::``  — replaced at resolve time with all collected entries."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: ClassVar[dict] = {}

    def run(self) -> List[nodes.Node]:
        return [physicslist_node("")]


# ─────────────────────── event handlers ──────────────────────


def _purge_physics(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    """Remove entries whose source file changed / was deleted."""
    if not hasattr(env, "physics_all"):
        return
    env.physics_all = [  # type: ignore[attr-defined]
        e for e in env.physics_all if e["docname"] != docname
    ]


def _merge_physics(
    app: Sphinx,
    env: BuildEnvironment,
    docnames: List[str],
    other: BuildEnvironment,
) -> None:
    """Merge parallel-build environments."""
    if not hasattr(env, "physics_all"):
        env.physics_all = []
    if hasattr(other, "physics_all"):
        env.physics_all.extend(other.physics_all)  # type: ignore[attr-defined]


def _resolve_physicslist(
    app: Sphinx,
    doctree: nodes.document,
    fromdocname: str,
) -> None:
    """Replace every ``physicslist_node`` with the collected entries."""
    env = app.builder.env
    if not hasattr(env, "physics_all"):
        env.physics_all = []

    entries: List[Dict[str, Any]] = env.physics_all  # type: ignore[attr-defined]

    for node in doctree.findall(physicslist_node):
        # Deduplicate entries by label (same directive collected multiple times
        # when autodoc processes module, class, and method pages).
        seen_labels: set = set()
        unique_entries: List[Dict[str, Any]] = []
        for entry in entries:
            label = entry.get("label", "")
            if label and label in seen_labels:
                continue
            if label:
                seen_labels.add(label)
            unique_entries.append(entry)

        # Group by category
        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in unique_entries:
            cat = entry.get("category") or "Uncategorized"
            by_cat[cat].append(entry)

        # Desired category ordering
        cat_order = [
            "Dynamical equations",
            "Simulation",
            "Estimator",
            "Inference",
            "Kinematics",
            "Error analysis",
            "Model selection",
            "Observable",
            "Basis functions",
        ]
        ordered_cats = [c for c in cat_order if c in by_cat]
        ordered_cats += [c for c in sorted(by_cat) if c not in ordered_cats]

        replacement = nodes.container()

        for cat in ordered_cats:
            cat_entries = by_cat[cat]

            # Section heading
            sec = nodes.section(ids=[f"physics-cat-{cat.lower().replace(' ', '-')}"])
            sec += nodes.title(cat, cat)

            for entry in cat_entries:
                para = nodes.paragraph()

                # Cross-ref link back to the original location
                ref = nodes.reference("", "")
                ref["refdocname"] = entry["docname"]
                ref["refuri"] = app.builder.get_relative_uri(
                    fromdocname, entry["docname"]
                )
                ref["refuri"] += "#" + entry["target_id"]

                ref += nodes.strong(entry["title"], entry["title"])
                para += ref

                # Source hint
                source_hint = f"  (in {entry['docname']})"
                para += nodes.emphasis(source_hint, source_hint)

                sec += para

                # Re-insert the content node (the formulas)
                content_copy = entry["content_node"].deepcopy()
                sec += content_copy

            replacement += sec

        node.replace_self([replacement])


# ─────────────────────── setup ───────────────────────────────


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_node(
        physics_node,
        html=(visit_physics_node, depart_physics_node),
        latex=(visit_physics_node, depart_physics_node),
        text=(visit_physics_node, depart_physics_node),
    )
    app.add_node(physicslist_node)

    app.add_directive("physics", PhysicsDirective)
    app.add_directive("physicslist", PhysicsListDirective)

    app.connect("env-purge-doc", _purge_physics)
    app.connect("env-merge-info", _merge_physics)
    app.connect("doctree-resolved", _resolve_physicslist)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
