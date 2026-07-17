# SFI Gallery & Documentation Style Guide

> **Audience:** AI agents and human contributors editing gallery demos,
> benchmark scripts, or documentation pages.

---

## 1. Plot styling

All gallery and benchmark scripts use a shared matplotlib style file:

```
examples/sfi_gallery.mplstyle
```

Load it at the top of every demo via:

```python
from _gallery_utils.helpers import apply_style, SFI_COLORS
apply_style()
```

### Key design choices

| Property | Value | Rationale |
|---|---|---|
| Figure background | `none` (transparent) | Plots float on the Furo theme background |
| Axes background | `none` | Same — no white rectangles |
| Text / labels | `#D0D0D0` (light grey) | Readable on dark (`#131416`) without glaring on light |
| Tick marks | `#B0B0B0` | Slightly dimmer than labels |
| Spines | `#808080` | Subtle frame, top+right hidden |
| `savefig.transparent` | `True` | PNG/SVG output has no background fill |

### Color palette

Use `SFI_COLORS` for semantic consistency across all demos.
`SFI_COLORS` is defined in both:
- `examples/_gallery_utils/helpers.py` (for gallery builds)
- `SFI.utils.plotting` (for standalone notebooks and user code)

| Key | Hex | Use for |
|---|---|---|
| `data` | `#3B9EFF` | Raw trajectory data, scatter |
| `inferred` | `#FFC20A` | Inferred quantities (gold) |
| `exact` | `#FF7A1A` | Ground-truth / exact model (orange) |
| `bootstrap` | `#5D3A9B` | Bootstrap / simulation (purple) |
| `highlight` | `#40B0A6` | Emphasis, secondary data (teal) |
| `error` | `#FF2D6F` | Error bars, residuals (pink) |
| `secondary` | `#1A85FF` | Additional series (blue variant) |
| `tertiary` | `#D35FB7` | Additional series (magenta) |

The `axes.prop_cycle` in the mplstyle follows the same order, so
auto-colored lines use these colors by default.

### Forbidden colors

**Never use pure black** (`"k"`, `"black"`, `"#000000"`) for any plot
element. Black is invisible on the dark theme background (`#131416`).

Use instead:

| Element | Color |
|---|---|
| Reference / zero lines | `#808080` |
| Quiver arrows, error bars | `#B0B0B0` |
| Annotation text | `#D0D0D0` |
| Pareto front / guide lines | `#B0B0B0` |

### Figure layout

- **Dedicated thumbnail figure**: Each demo includes a final "Thumbnail"
  section with a purpose-built single-panel figure that uses data from
  the script itself.  This figure is used as the gallery thumbnail.
  ```python
  # sphinx_gallery_thumbnail_number = N  # N = the thumbnail figure number
  ```
  The thumbnail figure should be:
  - Single panel (no subplots)
  - Compact (`figsize=(4, 3)` or `(4, 4)` for square)
  - Minimal labels (ticks/labels removed — too small in thumbnail)
  - Visually distinctive — showing the most characteristic data
- Use `constrained_layout` (enabled by default in the mplstyle) instead
  of `tight_layout()`.
- Target figure size: `7.5 × 4.5` inches (the mplstyle default).
- For multi-panel figures, use `plt.subplots(1, N)` rather than tall
  column layouts — wide aspect ratios display better in the gallery.

### Thumbnail pattern

Every demo ends with a thumbnail section:

```python
# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
# ... plot the most characteristic visualization ...
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.tight_layout()
plt.show()
```

Then set `sphinx_gallery_thumbnail_number` in the header to point to
this figure (count all `plt.subplots()` / `plt.figure()` calls in order).

---

## 2. CSS theming (docs)

Custom CSS lives at:

```
docs/source/_static/custom.css
```

### How it works

- **Dark mode** (default Furo): Plots float on the theme background
  (`#131416`) with no wrapper. The light-grey text on transparent
  backgrounds is directly legible.
- **Light mode**: A CSS dark card (`background: #1e2028`) wraps
  gallery images so that the light-grey text remains readable against
  a dark rectangle.

### Thumbnail grid

The gallery overview page displays thumbnails in a 3–4 column CSS grid:

```css
grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
```

Thumbnails have `border-radius: 4px` for rounded corners.

---

## 3. Mermaid diagrams (docs)

Conceptual diagrams in the documentation use
[sphinxcontrib-mermaid](https://sphinxcontrib-mermaid-demo.readthedocs.io/).

Insert diagrams in RST with the `.. mermaid::` directive:

```rst
.. mermaid::

   graph LR
     A[Trajectory data] --> B[Basis selection]
     B --> C[Force inference]
```

### Mermaid theme

Configured in `docs/source/conf.py` to use the **dark** theme matching
the Furo dark mode. Node colors use the SFI palette.

### When to use Mermaid vs. matplotlib

| Diagram type | Tool |
|---|---|
| Flowcharts, pipelines, decision trees | Mermaid |
| Data-driven visuals, actual plots | matplotlib |
| Class/type hierarchies | Mermaid |
| Architecture diagrams | Mermaid |

---

## 4. Documentation page conventions

### RST formatting

- Use `.. code-block:: python` for all code snippets.
- Use `.. math::` for equations (LaTeX via MathJax).
- Use `.. list-table::` for structured comparisons.
- Use `.. tip::`, `.. note::`, `.. warning::` admonitions sparingly.
- Cross-reference with `:doc:`, `:class:`, `:func:`, `:meth:`.

### Gallery demo structure

Every gallery demo (`examples/gallery/*_demo.py`) follows this pattern:

```python
"""
Title of the demo
==================

One-paragraph description.

.. rubric:: Tags

tag1 · tag2
"""
# sphinx_gallery_tags = ["tag1", "tag2"]
# sphinx_gallery_thumbnail_number = N

# sphinx_gallery_start_ignore
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
# ... all imports ...
if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
apply_style()
# sphinx_gallery_end_ignore
stamp_output()
# %% Section heading
# ---------------------
#
# Prose describing this step ...

# Non-plotting code (appears in HTML):
result = SFI.some_function(...)

# sphinx_gallery_start_ignore
# Plotting code (hidden from HTML, included in standalone notebook):
fig, ax = plt.subplots()
ax.plot(..., color=SFI_COLORS["data"])
plt.show()
# sphinx_gallery_end_ignore
```

#### Ignore-block strategy

Wrap all **non-essential boilerplate** in `sphinx_gallery_start_ignore` /
`sphinx_gallery_end_ignore` blocks. This includes:

| What to hide | Why |
|---|---|
| Imports | Gallery HTML should show only SFI-specific code |
| `sys.path` hack | Build machinery, not relevant to users |
| `apply_style()` / `_gallery_utils` imports | Gallery-specific setup |
| Plotting code (`plt.subplots`, labels, etc.) | Keeps HTML focused on the *inference* |

sphinx-gallery strips these from **both** the HTML and the `.ipynb` it
generates. To give users complete, runnable notebooks despite this:

1. The `docs/regenerate_notebooks.py` script rebuilds `.ipynb` files
   from the full `.py` sources, including all ignored blocks.
2. It replaces `_gallery_utils` imports with `SFI.utils.plotting`
   equivalents so notebooks are standalone (`pip install SFI` only).
3. It strips `sys.path` hacks, `apply_style()`, and sphinx-gallery
   directives.
4. RST markup (`:math:`, `:func:`, section underlines) is converted
   to Markdown/MathJax for Jupyter.

The regenerator runs automatically during a gallery build (hooked into
`conf.py` as a `builder-inited` handler with priority 900).

### Benchmark demo structure

Same as gallery demos but in `examples/benchmarks/*_bench.py`. These
typically use `bench_utils.py` for sweep/grid infrastructure.

---

## 5. Adding a new gallery demo

1. Create `examples/gallery/my_demo.py` following the structure above.
2. Add its filename to the `within_subsection_order` list in
   `docs/source/conf.py` → `sphinx_gallery_conf` at the appropriate
   position.
3. Add a "Thumbnail" section at the end of the script with a dedicated
   single-panel figure (see "Thumbnail pattern" above).
4. Count the total number of figures in the script and set
   `sphinx_gallery_thumbnail_number` in the header accordingly.
5. Run a gallery build to verify:
   ```bash
   cd docs
   SFI_DOCS_RUN_GALLERY=1 JAX_PLATFORMS=cpu python -m sphinx -M html source build
   ```
6. Verify the regenerated notebook:
   ```bash
   python docs/regenerate_notebooks.py -o /tmp/test_nb
   jupyter notebook /tmp/test_nb/my_demo.ipynb
   ```

---

## 6. Timestamps (figures & text output)

Every figure and every gallery page's text output carries a discreet
generation timestamp. This helps track staleness when results are cached.

### 6.1 Figure timestamps (`stamp_fig`)

`apply_style()` installs a one-time monkey-patch on
`matplotlib.figure.Figure.savefig` so that **every figure saved after
`apply_style()` is called** is automatically stamped before writing.
No per-figure code is needed.

The figure stamp is:
- **Position:** bottom-right corner (`x=0.99, y=0.005` in figure coords)
- **Font:** 5 pt, color `#808080`, alpha 0.4
- **Format:** `YYYY-MM-DD HH:MM` (local time when the figure was rendered)
- **Idempotent:** a figure is stamped at most once (guarded by a
  `_sfi_stamped` attribute)

#### For gallery demos

No action needed — `apply_style()` enables auto-stamping.  All figures
captured by sphinx-gallery (which calls `fig.savefig()` internally) will
carry the timestamp.

#### For benchmark renderers

No action needed — `render_benchmarks.py` calls `apply_style()` at
module level, so `_savefig()` and `_save_thumb()` both auto-stamp.

#### For standalone notebooks / user code

Import and call `stamp_fig` explicitly:

```python
from SFI.utils.plotting import stamp_fig

fig, ax = plt.subplots()
# ... plotting code ...
stamp_fig(fig)      # or stamp_fig() for current figure
fig.savefig("my_plot.png")
```

#### Disabling the figure timestamp

If you need a figure without a timestamp (e.g. for a publication
export), either:
1. Save the figure *before* calling `apply_style()`, or
2. Set `fig._sfi_stamped = True` before saving to suppress it.

### 6.2 Text output timestamps (`stamp_output`)

Every gallery demo calls `stamp_output()` once — right after the
initial `# sphinx_gallery_end_ignore` block — so that the rendered
gallery page includes a `[Generated: YYYY-MM-DD HH:MM]` line in its
first text output block.

#### Convention

Place `stamp_output()` **outside** the ignore block so its output is
visible on the rendered gallery page:

```python
# sphinx_gallery_start_ignore
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
apply_style()
# sphinx_gallery_end_ignore
stamp_output()
# %%
# First visible section
# ----------------------
```

The `stamp_output` import is inside the ignore block (hidden from
gallery HTML) but the call is outside, producing a visible output line.

#### For standalone notebooks

`stamp_output` is also available from the library:

```python
from SFI.utils.plotting import stamp_output
stamp_output()
```

`docs/regenerate_notebooks.py` automatically rewrites the
`_gallery_utils` import to `SFI.utils.plotting`, keeping `stamp_output`
in regenerated notebooks.

---

## 7. Quick reference: file locations

| Asset | Path |
|---|---|
| mplstyle | `examples/sfi_gallery.mplstyle` |
| Color palette + helpers (gallery) | `examples/_gallery_utils/helpers.py` |
| Color palette + helpers (library) | `SFI/utils/plotting.py` |
| Timestamp utility (`stamp_fig`) | `examples/_gallery_utils/helpers.py` and `SFI/utils/plotting.py` |
| Notebook regenerator | `docs/regenerate_notebooks.py` |
| Download-fix JS | `docs/source/_static/download_fix.js` |
| Custom CSS | `docs/source/_static/custom.css` |
| Sphinx config | `docs/source/conf.py` |
| Gallery demos | `examples/gallery/*_demo.py` |
| Benchmark demos | `examples/benchmarks/*_bench.py` |
| This guide | `GALLERY_STYLE_GUIDE.md` |
