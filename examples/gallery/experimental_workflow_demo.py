"""
Experimental-data workflow template
======================================

The recommended workflow for applying SFI to **real experimental data**.
This demo loads a 2D optical-tweezer trajectory from a CSV file,
infers the force and diffusion, sparsifies with PASTIS, and
validates with a bootstrap trajectory.

The data includes localization noise, a slow rotation (torque),
and a weak Duffing-type cubic anharmonicity in :math:`x`.
PASTIS automatically detects these non-trivial terms beyond the
harmonic restoring force.

.. admonition:: SFI's design philosophy

   Stochastic Force Inference is built for experimental trajectories
   where *no model pre-exists*.  The linear regression backbone
   requires no initial guess, works with arbitrary basis functions,
   and the PASTIS information criterion rigorously identifies which
   terms are supported by the data.

.. rubric:: Tags

real-data · overdamped · experimental-workflow
"""

# sphinx_gallery_tags = ["real-data", "overdamped", "experimental-workflow"]
# sphinx_gallery_thumbnail_number = 5

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent))
else:
    import _gallery_utils
    _here = Path(_gallery_utils.__file__).resolve().parent.parent / "gallery"
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils import plotting

apply_style()
# sphinx_gallery_end_ignore
# %%
# Step 1 — Load trajectory from CSV
# ------------------------------------
#
# ``TrajectoryCollection.load()`` handles CSV (with optional YAML
# header), Parquet, and HDF5.  The CSV file here uses a ``# dt: 0.01``
# YAML header so the timestep is automatically set.
#
# In your own workflow, replace the path with your data file.

from SFI.trajectory import TrajectoryCollection

csv_path = _here.parent / "experimental_data" / "optical_tweezer.csv"
coll = TrajectoryCollection.load(csv_path)

print(f"Loaded: {coll.T} frames, d={coll.d}, dt={coll.dt}")

# %%
# Step 2 — Visualize the data
# ------------------------------
#
# Always inspect your data first: check for outliers, missing frames,
# or obvious drift.

# sphinx_gallery_start_ignore
_, X_arr, _ = coll.to_arrays(dataset=0)            # (T, N, d) for the start marker
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
plotting.phase2d(coll, ax=ax, linewidth=0.4, alpha=0.7, plot_colorbar=True)
ax.scatter(X_arr[0, 0, 0], X_arr[0, 0, 1], s=40,
           color=SFI_COLORS["highlight"], zorder=5, label="start")
ax.set_title("Trajectory (2D)")
ax.legend()

ax = axes[1]
plotting.timeseries(coll, dims=[0], ax=ax, lw=0.4,
                    color=SFI_COLORS["data"], label="x")
plotting.timeseries(coll, dims=[1], ax=ax, lw=0.4,
                    color=SFI_COLORS["exact"], label="y")
ax.set_ylabel("Coordinate")
ax.set_title("Time series"); ax.legend()

fig.suptitle("Step 2 — Data inspection", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Step 3 — Basis, inference, and sparsification
# -------------------------------------------------
#
# Start with a polynomial basis (degree 3).  For experimental data
# you typically do not know the diffusion tensor, so we estimate it.
# PASTIS then selects the simplest model the data supports.
#
# The data contains a weak Duffing cubic (:math:`-\alpha\,x^3`) in one
# axis and a slow rotation; PASTIS should recover these beyond the
# dominant harmonic terms.

from SFI.bases import monomials_up_to
from SFI import OverdampedLangevinInference
from SFI.utils import plotting

degree = 3
B = monomials_up_to(order=degree, dim=2, rank='vector')

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()
inf.infer_force_linear(B)
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS")
support = np.asarray(inf.force_support)
k_sel = len(support)
if k_sel > 0:
    inf.compute_force_error()

inf.print_report()
# sphinx_gallery_start_ignore
coeffs_sel = np.asarray(inf.force_coefficients)
n_feat = B.n_features
labels = inf.force_basis_labels

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
plotting.plot_recovery_bar(coeffs_sel, support, labels=labels, ax=ax)
ax.set_title(f"PASTIS selected {k_sel} / {n_feat} features")

ax = axes[1]
plotting.phase2d(coll, ax=ax, color=SFI_COLORS["data"], linewidth=0.2, alpha=0.3)
plt.sca(ax)
plotting.plot_field(coll, inf.force_inferred, N=10,
                    color=SFI_COLORS["inferred"], autoscale=True, scale=0.2)
ax.set_title("Inferred force field (sparse)")
ax.set_aspect("equal")

fig.suptitle("Step 3 — Sparsified inference", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Step 4 — Bootstrap validation
# --------------------------------
#
# Simulate a trajectory from the inferred model and compare with
# the original data — a self-consistency check.

key = random.PRNGKey(123)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(key)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(6, 5))
lc_data = plotting.phase2d(coll, ax=ax, tmax=2000, color=SFI_COLORS["data"],
                           linewidth=0.3, alpha=0.5)
lc_data.set_label("Original data")
lc_boot = plotting.phase2d(coll_boot, ax=ax, tmax=2000, color=SFI_COLORS["inferred"],
                           linewidth=0.3, alpha=0.5)
lc_boot.set_label("Bootstrap trajectory")
ax.autoscale()
ax.set_aspect("equal"); ax.legend()
ax.set_title("Step 4 — Bootstrap vs data")
plt.show()
# sphinx_gallery_end_ignore
# %%
# Step 5 — Consistency diagnostics
# -----------------------------------
#
# The final check on real data: recompute the standardised residuals
# and test whether they behave like an i.i.d. :math:`\mathcal{N}(0, 1)`
# sample.  Localization noise biases the linear Itô estimator (an
# errors-in-variables effect), so a residual-consistency flag here is a
# useful signal — not a failure of the workflow, but a sign that
# measurement noise matters for this dataset.  When it appears, the
# parametric estimator (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`) profiles the noise level and
# removes the bias — see :doc:`/inference/noise_and_sampling`.  Each
# flag below carries its one-line action hint inline.

from SFI.diagnostics import assess, plot_summary

report = assess(inf, level="standard")
report.print_summary()

# sphinx_gallery_start_ignore
fig_diag = plot_summary(report)
fig_diag.suptitle("Step 5 — residual diagnostics", y=1.02)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
plotting.phase2d(coll, ax=ax_t, color=SFI_COLORS["data"], linewidth=0.2, alpha=0.3)
plt.sca(ax_t)
plotting.plot_field(coll, inf.force_inferred, N=10,
                    color=SFI_COLORS["inferred"], autoscale=True, scale=0.2)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
ax_t.set_aspect("equal")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
