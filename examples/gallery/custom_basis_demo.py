r"""
Custom basis with extras — traps in measured light fields
===========================================================

When does a force model need hand-written code?  Rarely: per-experiment
constants, recorded drives and dataset-specific coefficients are all
expressible with built-in symbols.  The exception is physics the symbol
algebra cannot express — typically a **measured** object such as an
image.  This page shows both, on one dataset.

Three experiments trap a colloid in an optical tweezer at **different
positions and temperatures**.  In each, the particle also feels a weak
speckle light pattern, **recorded as an image** and stored with the
experiment.  The force is

.. math::

   \mathbf F_e(\mathbf x) = -k\,(\mathbf x - \mathbf x^0_e)
                         - \alpha\,\nabla I_e(\mathbf x),

with a shared stiffness :math:`k` and light coupling :math:`\alpha`,
an unknown trap centre :math:`\mathbf x^0_e` per experiment, and the
measured intensity map :math:`I_e`.

- The **trap** is built from built-in symbols: a shared linear term plus
  a per-experiment constant via :func:`~SFI.bases.dataset_indicator`.
- The **light force** needs :func:`~SFI.statefunc.make_basis`: a
  function that interpolates the gradient of the experiment's image,
  read from ``extras``.

.. rubric:: Tags

synthetic · overdamped · 2D · custom-basis · extras · multi-experiment
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "2D", "custom-basis", "extras", "multi-experiment"]
# sphinx_gallery_thumbnail_number = 4

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from SFI.trajectory import TrajectoryCollection
from SFI.utils.plotting import phase2d, plot_pareto_front

apply_style()
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Three experiments, three recorded light fields
# ----------------------------------------------
#
# Each experiment stores its intensity gradient — computed once from the
# recorded image on a pixel grid — in ``extras_global``.  Images are
# smoothed random speckle patterns, normalised to :math:`[0, 1]`.

from scipy.ndimage import gaussian_filter

L, n_pix = 3.0, 64                       # image covers [-L, L]² with n_pix² pixels
h = 2 * L / (n_pix - 1)                  # pixel size


def speckle_image(seed):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=(n_pix, n_pix)), sigma=4, mode="wrap")
    return (img - img.min()) / (img.max() - img.min())


experiments = [   # trap centre, diffusion (temperature), image seed
    {"x0": np.array([1.0, 0.5]), "D": 0.2, "seed": 1},
    {"x0": np.array([-0.5, 1.0]), "D": 0.5, "seed": 2},
    {"x0": np.array([0.0, -0.8]), "D": 1.0, "seed": 3},
]
for exp in experiments:
    exp["image"] = speckle_image(exp["seed"])
    exp["grad"] = np.stack(np.gradient(exp["image"], h))   # (2, n_pix, n_pix): ∂I/∂x, ∂I/∂y

# %%
# The custom brick: interpolated image gradient
# ---------------------------------------------
#
# The basis function receives the state and the experiment's extras, and
# returns the vector feature :math:`-\nabla I_e(\mathbf x)`, bilinearly
# interpolated between pixels.  ``extras_keys`` declares what it reads;
# the inference engine delivers each dataset's own image.

from jax.scipy.ndimage import map_coordinates

from SFI.statefunc import make_basis


def light_force(x, *, extras):
    """-∇I at x, bilinear interpolation of the stored gradient images."""
    pix = (x + L) / h                                   # (2,) pixel coordinates
    coords = [pix[0:1], pix[1:2]]
    g = extras["light_grad"]
    gx = map_coordinates(g[0], coords, order=1, mode="nearest")
    gy = map_coordinates(g[1], coords, order=1, mode="nearest")
    return -jnp.concatenate([gx, gy])[:, None]          # (dim, 1 feature)


B_light = make_basis(light_force, dim=2, rank=1, n_features=1,
                     extras_keys=("light_grad",), labels=["-∇I"])

# %%
# Simulate
# --------
#
# Each experiment is simulated with its own centre, temperature and
# image; the shared parameters are :math:`k = 2` and :math:`\alpha = 0.5`.
# The collections are then pooled into one multi-experiment dataset.

from SFI.bases import X, unit_vector_basis
from SFI.langevin import OverdampedProcess

k_true, alpha_true = 2.0, 0.5
dt, Nsteps = 0.01, 50_000

key = random.PRNGKey(42)
runs = []
for i, exp in enumerate(experiments):
    # F = -k x + k x0 + alpha (-∇I): a combination of the three bricks
    proc = OverdampedProcess(
        F=X(dim=2) & unit_vector_basis(2) & B_light, D=exp["D"],
        theta_F=jnp.array([-k_true, *(k_true * exp["x0"]), alpha_true]),
    )
    proc.set_extras(extras_global={"light_grad": jnp.asarray(exp["grad"])})
    proc.initialize(jnp.asarray(exp["x0"]))
    key, sub = random.split(key)
    runs.append(proc.simulate(dt=dt, Nsteps=Nsteps, key=sub, prerun=200, oversampling=10))

coll = runs[0].concat(runs[1:], weights="pool")
print(f"{len(coll.datasets)} experiments × {runs[0].T} frames")

# %%
# The first 30 time units of each trajectory over its light field; crosses
# mark the trap centres.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for i, (ax, exp) in enumerate(zip(axes, experiments)):
    ax.imshow(exp["image"].T, origin="lower", extent=(-L, L, -L, L), cmap="gray", alpha=0.6,
              interpolation="bilinear")
    phase2d(coll, dataset=i, tmax=3000, color=SFI_COLORS["data"], linewidth=0.3, alpha=0.6, ax=ax)
    ax.scatter(*exp["x0"], marker="x", s=100, color=SFI_COLORS["exact"], zorder=5)
    ax.set_title(f"experiment {i + 1}: D = {exp['D']}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$" if i == 0 else "")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
fig.suptitle("Three experiments — different traps, temperatures and light fields")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infer: built-in trap + custom light brick
# -----------------------------------------
#
# ``dataset_indicator(3) * unit_vector_basis(2, axes=[a])`` gives each
# experiment its own constant force along axis ``a`` — the
# :math:`k\,\mathbf x^0_e` of its trap — while ``X(dim=2)`` carries the
# shared stiffness.  We add all quadratic monomials as distractors and
# let PASTIS decide what the data support.  The data are clean
# (simulated without measurement noise), so we use the clean-data
# estimators: ``"WeakNoise"`` diffusion, in which the drift cancels, and
# ``preset="clean"``.

from SFI import OverdampedLangevinInference
from SFI.bases import dataset_indicator, monomials_degree

offsets = [dataset_indicator(3) * unit_vector_basis(2, axes=[a]) for a in range(2)]
B_trap = X(dim=2) & offsets[0] & offsets[1]
B = B_trap & B_light & monomials_degree(2, dim=2, rank="vector")

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B, preset="clean")
inf.compute_force_error()
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# %%
# The Pareto front shows the selection: the information gained by the
# best model of each size rises until seven terms are in, then flattens.
# PASTIS stops there; AIC, less strict, would admit one more.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.5, 3.5))
plot_pareto_front(inf.force_sparsity_result, criteria=("PASTIS", "AIC"), ax=ax)
ax.set_title("Model selection")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The selected model is the true one: the shared stiffness, the
# per-experiment offsets and the light coupling, while the quadratic
# terms are pruned (so is the x-offset of experiment 3, whose trap sits
# at :math:`x = 0`).  Dividing each offset by :math:`\hat k` gives the
# trap centres.

labels = list(B.labels)
support = list(np.asarray(inf.force_support))
coef = dict(zip(support, np.asarray(inf.force_coefficients)))
err = dict(zip(support, np.asarray(inf.force_coefficients_stderr)))
i_k, i_light = labels.index(B_trap.labels[0]), labels.index("-∇I")
k_hat = -coef[i_k]
print(f"k     = {k_hat:.3f} ± {err[i_k]:.3f}   (true {k_true})")
print(f"alpha = {coef.get(i_light, 0.0):.3f} ± {err.get(i_light, 0.0):.3f}   (true {alpha_true})")
for e, exp in enumerate(experiments):
    c = [coef.get(labels.index(f"ds{e}·e{a}"), 0.0) for a in range(2)]
    print(f"experiment {e + 1}: x0 = ({c[0] / k_hat:+.3f}, {c[1] / k_hat:+.3f})"
          f"   (true ({exp['x0'][0]:+.2f}, {exp['x0'][1]:+.2f}))")

# %%
# Three temperatures, one force fit
# ---------------------------------
#
# The experiments run at different temperatures, but the force fit used a
# single pooled diffusion constant.  That does not bias the coefficients
# — a linear regression is unbiased whatever its weights — but the
# per-experiment temperatures are worth having.  They come out of a
# diffusion fit on the same indicator idea,
# ``dataset_indicator(3) * identity_matrix_basis(2)``, with the same
# clean-data local estimator as above.

from SFI.bases import identity_matrix_basis

inf.infer_diffusion_linear(dataset_indicator(3) * identity_matrix_basis(2), M_mode="WeakNoise")
inf.compute_diffusion_error()
D_hat = np.asarray(inf.diffusion_coefficients).ravel()
D_err = np.asarray(inf.diffusion_coefficients_stderr).ravel()
for e, exp in enumerate(experiments):
    print(f"experiment {e + 1}: D = {D_hat[e]:.3f} ± {D_err[e]:.3f}   (true {exp['D']})")

# %%
# Does the pooled model reproduce each experiment?
# ------------------------------------------------
#
# :meth:`~SFI.OverdampedLangevinInference.simulate_bootstrapped_trajectory`
# with ``dataset=e`` collapses the pooled model to experiment ``e`` — its
# trap centre, its temperature and its recorded light field — and
# simulates it for as long as the recording.

boots = [inf.simulate_bootstrapped_trajectory(random.PRNGKey(10 + e), dataset=e)[0]
         for e in range(len(experiments))]

# sphinx_gallery_start_ignore
def _density(c, bins=70):
    Xc = np.asarray(c.to_arrays(dataset=0)[1])[:, 0, :]
    H, xe, ye = np.histogram2d(Xc[:, 0], Xc[:, 1], bins=bins, range=[[-L, L], [-L, L]], density=True)
    return gaussian_filter(H.T, 1.2), 0.5 * (xe[1:] + xe[:-1])


fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for i, (ax, exp) in enumerate(zip(axes, experiments)):
    sub = TrajectoryCollection.from_dataset(coll.datasets[i])
    Hd, cc = _density(sub)
    Hb, _ = _density(boots[i])
    lev = np.linspace(0.1, 0.9, 5) * Hd.max()
    ax.imshow(exp["image"].T, origin="lower", extent=(-L, L, -L, L), cmap="gray", alpha=0.45,
              interpolation="bilinear")
    ax.contourf(cc, cc, Hd, levels=np.r_[lev, 1.01 * Hd.max()], cmap="Blues", alpha=0.75)
    ax.contour(cc, cc, Hb, levels=lev, colors=SFI_COLORS["bootstrap"], linewidths=1.5)
    ax.scatter(*exp["x0"], marker="x", s=100, color=SFI_COLORS["exact"], zorder=5)
    ax.set_title(f"experiment {i + 1}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$" if i == 0 else "")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
axes[0].legend(handles=[Patch(color="#6baed6", label="data"),
                        Line2D([], [], color=SFI_COLORS["bootstrap"], lw=1.5, label="bootstrap")],
               loc="lower left", fontsize=8)
fig.suptitle("Position density: data (shading) vs bootstrap (lines)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each bootstrap follows its recording, including the distortions the
# speckle imprints on the trap.
#
# Summary
# --------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Situation
#      - Building block
#    * - Per-experiment constant or coefficient (linear estimators)
#      - ``dataset_indicator(n) * feature``
#    * - Per-experiment parameter in a parametric model
#      - :func:`~SFI.bases.per_dataset_scalar`
#    * - Recorded scalar drive or metadata
#      - :func:`~SFI.bases.extra_scalar`
#    * - Measured field, lookup table, anything not expressible in symbols
#      - ``make_basis(f, extras_keys=(...))``

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
ax_t.imshow(experiments[0]["image"].T, origin="lower", extent=(-L, L, -L, L), cmap="gray", alpha=0.6,
            interpolation="bilinear")
phase2d(coll, dataset=0, tmax=3000, color=SFI_COLORS["data"], linewidth=0.3, alpha=0.6, ax=ax_t)
ax_t.scatter(*experiments[0]["x0"], marker="x", s=80, color=SFI_COLORS["exact"], zorder=5)
ax_t.set_aspect("equal", adjustable="box")
ax_t.set_xlim(-0.5, 2.5); ax_t.set_ylim(-1.0, 2.0)
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

