r"""
Gray-Scott: equations from a corrupted movie
============================================

.. note::
   Uses the **experimental** SPDE toolbox — see :doc:`/spde/index`.

A few drops of an autocatalyst :math:`V` in a bath of feed :math:`U` grow
into a colony of **self-replicating spots**: each spot swells, splits in
two, and the daughters push apart until they tile the dish.  The
dynamics is the stochastic Gray-Scott model

.. math::

   \partial_t U = D_U \nabla^2 U - U V^2 + F (1-U)
                  + \sigma\,\nabla\!\cdot\!\boldsymbol\eta_U

   \partial_t V = D_V \nabla^2 V + U V^2 - (F+K) V
                  + \sigma\,\nabla\!\cdot\!\boldsymbol\eta_V

with conserved (divergence-form) noise :math:`\nabla\cdot\boldsymbol\eta`,
which only moves material around.  This page films the colony with an
imperfect camera — **pixel noise and dead pixels** — hands SFI the movie
and a generic library of 34 candidate terms, and asks for the equations:

- although every pixel is noisy and 2% of them are dead in every frame,
  **PASTIS** keeps exactly the seven Gray-Scott terms, with error bars;
- the recovered equations, restarted from the initial drops, **regrow
  the colony**: same growth curve, same spot spacing;
- a corruption ladder shows **where it breaks**: dead pixels only cost
  data, while pixel noise of a few percent of the spot contrast biases
  the nonlinear terms until PASTIS keeps spurious ones.

.. rubric:: Tags

synthetic · overdamped · SPDE · reaction-diffusion · 2D · sparsity
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "SPDE", "reaction-diffusion", "2D", "sparsity"]
# sphinx_gallery_thumbnail_number = 7

# sphinx_gallery_start_ignore
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Avoid GPU pre-allocation OOM when running after other gallery examples
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import (
    SFI_COLORS,
    apply_style,
    stamp_output,
)
from SFI.utils.plotting import plot_pareto_front

apply_style()
plt.rcParams["savefig.facecolor"] = "auto"   # dark movie background survives in the mp4
BG = "#131416"                               # page background (Furo dark)
CMAP_V, CMAP_U = "inferno", "viridis"
# The dense-fit conditioning is printed and discussed below instead.
warnings.filterwarnings("ignore", message=r".*Gram condition number")
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A generic candidate library
# ---------------------------
#
# Two named ``ScalarSector`` fields ``U`` and ``V`` live on one periodic
# :class:`~SFI.statefunc.layout.GridLayout`; differential operators
# (``layout.lap``, ``layout.grad``, ``layout.biharmonic``) and pointwise
# algebra compose freely, and ``layout.embed`` compiles the expression
# tree into an inference-ready :class:`~SFI.statefunc.Basis`.
#
# We do not hand-code the Gray-Scott structure.  Each field gets the same
# **generic** library: every monomial of :math:`(U, V)` up to order 3, both
# Laplacians, the three gradient products and both biharmonics — 17
# candidates per equation, 34 in total.  Labels are generated from the
# expression tree.

from SFI.statefunc.layout import GridLayout, ScalarSector

layout = GridLayout(U=ScalarSector([0]), V=ScalarSector([1]), dim=2, ndim=2, bc="pbc")
U, V = layout.U, layout.V
gU, gV = layout.grad(U), layout.grad(V)

terms = [
    layout.const(1), U, V,                        # "1", "U", "V"
    U**2, U * V, V**2,                            # quadratic
    U**3, U**2 * V, U * V**2, V**3,               # cubic
    layout.lap(U), layout.lap(V),                 # "∇²U", "∇²V"
    gU.dot(gU), gV.dot(gV), gU.dot(gV),           # gradient products
    layout.biharmonic(U), layout.biharmonic(V),   # "∇⁴U", "∇⁴V"
]
library = terms[0]
for t in terms[1:]:
    library = library & t

BASIS = layout.embed(rank=1, U=library, V=library)   # same candidates for U̇ and V̇
n_per = len(library.labels)                          # 17
labels = [f"{lbl}→U̇" for lbl in library.labels] + [f"{lbl}→V̇" for lbl in library.labels]


def restrict(support):
    """The library restricted to a support, compiled on its own (fast to simulate)."""
    picked = {}
    for name, offset in (("U", 0), ("V", n_per)):
        sel = [terms[i - offset] for i in sorted(support) if offset <= i < offset + n_per]
        picked[name] = sel[0]
        for t in sel[1:]:
            picked[name] = picked[name] & t
    return layout.embed(rank=1, **picked)

# %%
# The experiment
# --------------
#
# The true model is the library restricted to its seven Gray-Scott terms,
# with the classic diffusion ratio :math:`D_U/D_V = 2` and :math:`(F, K)`
# in the self-replicating-spot regime.  A periodic **128×128 dish**
# (``dx = 1``) starts from four square drops of :math:`V`.
#
# Pattern formation is slow — the colony needs about 3000 time units to
# fill the dish — while the inference needs frames closer together than
# the fastest relaxation, grid-scale diffusion at rate
# :math:`8 D_U/\Delta x^2 \approx 1.3`.  So, like a microscope alternating
# high-speed bursts with a slow time-lapse, each quarter of the run opens
# with an **analysis clip of 100 frames at** :math:`\Delta t = 0.125`,
# then keeps one frame every 12.5 time units for viewing.  Successive
# ``simulate`` calls continue the same trajectory.

from SFI.bases.spde import conserved_noise_pbc, square_grid_extras
from SFI.langevin import OverdampedProcess
from SFI.trajectory import TrajectoryCollection

GRID = (128, 128)
DX = 1.0
DU, DV = 0.16, 0.08
F_gs, K_gs = 0.0367, 0.0649     # self-replicating spots
SIGMA = 0.01                    # conserved-noise amplitude (both fields)
gray_scott = {                  # the seven true terms and their coefficients
    "1→U̇": F_gs, "U→U̇": -F_gs, "UV²→U̇": -1.0, "∇²U→U̇": DU,
    "V→V̇": -(F_gs + K_gs), "UV²→V̇": 1.0, "∇²V→V̇": DV,
}
support_true = np.sort([labels.index(name) for name in gray_scott])
theta_true = np.zeros(len(labels))
theta_true[[labels.index(name) for name in gray_scott]] = list(gray_scott.values())

DT = 0.125         # frame interval of the analysis clips
DT_SHOW = 12.5     # = one 100-frame clip: viewing frames stay evenly spaced
N_CLIP, N_QUARTER = 100, 59
T_QUARTER = DT_SHOW * (N_QUARTER + 1)                # 750 time units
box_extras = square_grid_extras(grid_shape=GRID, dx=DX)


def drops(key, n=4, size=10):
    """Uniform bath U=1, V=0 with n square drops (U=1/2, V=1/4) and 2% noise."""
    Nx, Ny = GRID
    k_pos, k_u, k_v = random.split(key, 3)
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    in_drop = np.zeros(GRID, dtype=bool)
    for i0, j0 in np.asarray(random.randint(k_pos, (n, 2), 0, Nx)):
        in_drop |= ((ii - i0) % Nx < size) & ((jj - j0) % Ny < size)
    U0 = np.where(in_drop, 0.5, 1.0) + 0.02 * np.asarray(random.normal(k_u, GRID))
    V0 = np.where(in_drop, 0.25, 0.0) + 0.02 * np.asarray(random.normal(k_v, GRID))
    return jnp.asarray(np.stack([U0, V0], axis=-1).reshape(Nx * Ny, 2))


proc = OverdampedProcess(restrict(support_true),
                         D=conserved_noise_pbc(sigma=SIGMA, grid_shape=GRID, dx=DX, n_fields=2))
proc.set_params(theta_F=jnp.asarray(theta_true[support_true], dtype=jnp.float32))
proc.set_extras(extras_global=box_extras)

key = random.PRNGKey(0)
key, k_ic = random.split(key)
x0 = drops(k_ic)
proc.initialize(x0)

show_frames, clips = [np.asarray(x0)[None]], []
for quarter in range(4):
    key, k_clip, k_show = random.split(key, 3)
    clip = proc.simulate(dt=DT, Nsteps=N_CLIP, key=k_clip,
                         oversampling=2,      # integrator step Δt/2
                         compute_observables=False)
    seg = proc.simulate(dt=DT_SHOW, Nsteps=N_QUARTER, key=k_show,
                        oversampling=100,     # integrator step 0.125 between viewing frames
                        compute_observables=False)
    clips.append(clip)
    show_frames += [clip.to_array()[-1:], seg.to_array()]
truth = TrajectoryCollection.from_arrays(X=np.concatenate(show_frames), dt=DT_SHOW,
                                         extras_global=box_extras)
clean = clips[0].concat(clips[1:])
t_show = DT_SHOW * np.arange(truth.T)
t_clips = [T_QUARTER * q for q in range(4)]          # each clip starts there
print(f"{truth.T} viewing frames + {len(clips)} clips of {N_CLIP} frames "
      f"on {GRID[0]}×{GRID[1]} cells")

# %%
# :math:`U` (top) and :math:`V` (bottom) at five times; the first four are
# the moments the analysis clips begin.

# sphinx_gallery_start_ignore
Xs = truth.to_array()
Nx, Ny = GRID


def field(X, ti, c):
    return X[ti, :, c].reshape(GRID).T


snap = np.linspace(0, truth.T - 1, 5).astype(int)
fig, axes = plt.subplots(2, 5, figsize=(11.5, 4.9))
for j, ti in enumerate(snap):
    im_u = axes[0, j].imshow(field(Xs, ti, 0), origin="lower", cmap=CMAP_U, vmin=0.2, vmax=1.0)
    im_v = axes[1, j].imshow(field(Xs, ti, 1), origin="lower", cmap=CMAP_V, vmin=0, vmax=0.45)
    axes[0, j].set_title(f"t = {t_show[ti]:.0f}")
    for ax in axes[:, j]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
axes[0, 0].set_ylabel("U  (feed)")
axes[1, 0].set_ylabel("V  (autocatalyst)")
fig.colorbar(im_u, ax=axes[0, :].tolist(), shrink=0.85, pad=0.01)
fig.colorbar(im_v, ax=axes[1, :].tolist(), shrink=0.85, pad=0.01)
fig.suptitle("Self-replicating spots: four drops grow into a colony", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The recording
# -------------
#
# Now the camera.  :func:`~SFI.trajectory.degrade.degrade_spatial_data`
# adds **pixel noise** — independent from pixel to pixel and from frame to
# frame — and kills a random 2% of the pixels in **every frame**.  A
# missing pixel hides more than its own value: every finite-difference
# stencil that reads it is corrupted.  ``stencil_radius=2`` — the reach of
# :math:`\nabla^4`, the widest operator in the library — excludes from the
# fit every cell whose stencil touches a missing pixel in any of the five
# frames an estimator row reads, while the remaining cells keep all their
# neighbours.  That is a heavy toll: only about a quarter of the cells
# survive it.

from SFI.trajectory.degrade import degrade_spatial_data

PIXEL_NOISE = 0.005   # standard deviation, in concentration units
DEAD_PIXELS = 0.02    # fraction of pixels missing in each frame

recording = degrade_spatial_data(clean, downscale=1, noise=PIXEL_NOISE,
                                 data_loss_fraction=DEAD_PIXELS, stencil_radius=2,
                                 bc="pbc", seed=1)
kept = np.mean([np.mean(ds.dynamic_mask) for ds in recording.datasets])
v_peak = truth.to_array()[-1, :, 1].max()        # spots at the end of the run
print(f"pixel noise {PIXEL_NOISE} = {100 * PIXEL_NOISE / v_peak:.1f}% of the spot peak V = {v_peak:.2f}")
print(f"{len(recording.datasets)} clips × {N_CLIP} frames × {GRID[0] * GRID[1]} cells; "
      f"{100 * kept:.0f}% of the cells enter the fit")

# %%
# One frame of the last clip: what the camera recorded (dead pixels in
# cyan); a zoom on the clean field and on the recording; and their
# difference — the pixel noise, small next to the field's own grid-scale
# roughness, which is real (the conserved noise).

# sphinx_gallery_start_ignore
ds_clean, ds_rec = clean.datasets[-1], recording.datasets[-1]
ti = ds_rec.T // 2
v_clean = np.asarray(ds_clean.X[ti])[:, 1].reshape(GRID).T
cell_ok = np.asarray(ds_rec.mask[ti]).reshape(GRID).T
v_rec = np.ma.masked_array(np.asarray(ds_rec.X[ti])[:, 1].reshape(GRID).T, mask=~cell_ok)
cm_rec = plt.get_cmap(CMAP_V).copy()
cm_rec.set_bad("#40E0FF")
cm_diff = plt.get_cmap("RdBu_r").copy()
cm_diff.set_bad("#40E0FF")
z0, zw = 40, 32   # zoom region
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
axes[0].imshow(v_rec, origin="lower", cmap=cm_rec, vmin=0, vmax=0.45)
axes[0].add_patch(plt.Rectangle((z0, z0), zw, zw, fill=False, ec="#D0D0D0", lw=1.2))
axes[0].set_title("recorded V (full dish)", fontsize=10)
axes[1].imshow(v_clean, origin="lower", cmap=CMAP_V, vmin=0, vmax=0.45)
axes[1].set_title("clean V (zoom)", fontsize=10)
axes[2].imshow(v_rec, origin="lower", cmap=cm_rec, vmin=0, vmax=0.45)
axes[2].set_title("recorded V (zoom)", fontsize=10)
im_d = axes[3].imshow(v_rec - v_clean, origin="lower", cmap=cm_diff,
                      vmin=-3 * PIXEL_NOISE, vmax=3 * PIXEL_NOISE)
axes[3].set_title("recorded − clean (zoom)", fontsize=10)
fig.colorbar(im_d, ax=axes[3], shrink=0.8, pad=0.02)
for ax in axes[1:]:
    ax.set_xlim(z0 - 0.5, z0 + zw - 0.5)
    ax.set_ylim(z0 - 0.5, z0 + zw - 0.5)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"What the camera sees: pixel noise {PIXEL_NOISE}, "
             f"{100 * DEAD_PIXELS:.0f}% dead pixels per frame", fontsize=12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Infer and select
# ----------------
#
# Default settings throughout.  The diffusion estimator also measures the
# variance :math:`\hat\Lambda` of the measurement noise — its square root
# lands near the camera's 0.005, plus a small floor that is there even
# without pixel noise (see the ladder at the end) — and the force
# estimator, seeing noise, switches to its noise-robust variant, which
# pairs frames whose pixel noise is independent.  The dense 34-term fit is
# badly conditioned: on a spot pattern several monomials nearly coincide —
# :math:`V \approx 0` between the spots, :math:`U` and :math:`V` locked
# together inside them — and the Gram matrix is close to singular.  PASTIS
# resolves this by pruning, and the error bars come from the selected
# model.

from SFI import OverdampedLangevinInference
from SFI.utils.formatting import model_summary

inf = OverdampedLangevinInference(recording)
inf.compute_diffusion_constant()
inf.infer_force_linear(BASIS)
print(f"measurement noise seen by SFI: sqrt(Λ) = {np.sqrt(np.diag(np.asarray(inf.Lambda))).round(4)} "
      f"for (U, V); force estimator: '{inf.metadata['force_preset']}'")
print(f"dense 34-term fit: Gram condition number {inf.metadata['force_gram_cond']:.1e}")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()

order = np.argsort(inf.force_support)               # candidate order
support = np.asarray(inf.force_support)[order]
coeffs = np.asarray(inf.force_coefficients)[order]
stderr = np.asarray(inf.force_coefficients_stderr)[order]
print(f"PASTIS keeps {len(support)}/{len(labels)} terms; "
      f"exactly the Gray-Scott ones: {set(support) == set(support_true)}")
print(model_summary(
    labels, coeffs, stderr=stderr, support=support,
    coeffs_true=theta_true[support_true], support_true=support_true,
    title="Gray-Scott: inferred vs true",
))

# %%
# The recovered equations, with one-standard-error uncertainties (top);
# the Pareto front — the best information gain at each model size, with
# the size each criterion selects (bottom left); and every selected
# coefficient divided by its true value (bottom right).

# sphinx_gallery_start_ignore
TEX = {"1": "", "∇²U": r"\nabla^2 U", "∇²V": r"\nabla^2 V", "∇⁴U": r"\nabla^4 U",
       "∇⁴V": r"\nabla^4 V", "∇U·∇U": r"|\nabla U|^2", "∇V·∇V": r"|\nabla V|^2",
       "∇U·∇V": r"\nabla U\cdot\nabla V"}


def tex_term(lbl):
    base = lbl.split("→")[0]
    return TEX.get(base, base.replace("²", "^2").replace("³", "^3"))


def tex_label(i):
    return "$" + (tex_term(labels[i]) or "1") + r"\to\dot{" + labels[i][-2] + "}$"


def sig_digits(err):
    return max(1, int(-np.floor(np.log10(err))) + 1)


def equation(field_name, idx, vals, errs=None):
    out = []
    for i, c, e in zip(idx, vals, errs if errs is not None else [None] * len(idx)):
        sgn = "-" if c < 0 else "+"
        if e is None:
            mag = "" if (abs(abs(c) - 1) < 1e-12 and tex_term(labels[i])) else f"{abs(c):g}"
        else:
            nd = sig_digits(e)
            mag = rf"({abs(c):.{nd}f}\pm{e:.{nd}f})"
        out.append(rf"{sgn}\,{mag}\,{tex_term(labels[i])}" if mag else rf"{sgn}\,{tex_term(labels[i])}")
    s = " ".join(out).lstrip("+\\, ")
    return rf"$\partial_t {field_name} = {s}$"


def lap_first(idx):   # Laplacian first, then reaction terms
    return sorted(idx, key=lambda i: (not labels[i].startswith("∇²"), i))


sel = {i: (c, e) for i, c, e in zip(support, coeffs, stderr)}
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.1])
ax_eq = fig.add_subplot(gs[0, :])
ax_eq.axis("off")
y = 0.92
for fld, lo, hi in [("U", 0, n_per), ("V", n_per, 2 * n_per)]:
    idx_s = lap_first([i for i in support if lo <= i < hi])
    idx_t = lap_first([i for i in support_true if lo <= i < hi])
    ax_eq.text(0.0, y, "recovered", color=SFI_COLORS["inferred"], fontsize=10,
               transform=ax_eq.transAxes, va="center")
    ax_eq.text(0.11, y, equation(fld, idx_s, [sel[i][0] for i in idx_s], [sel[i][1] for i in idx_s]),
               fontsize=13, transform=ax_eq.transAxes, va="center", color="#E8E8E8")
    ax_eq.text(0.0, y - 0.2, "true", color=SFI_COLORS["exact"], fontsize=10,
               transform=ax_eq.transAxes, va="center")
    ax_eq.text(0.11, y - 0.2, equation(fld, idx_t, theta_true[idx_t]),
               fontsize=13, transform=ax_eq.transAxes, va="center", color="#B0B0B0")
    y -= 0.52
ax_eq.set_title(f"From a generic library of {len(labels)} terms, PASTIS keeps {len(support)}",
                fontsize=12)

ax_p = fig.add_subplot(gs[1, 0])
plot_pareto_front(inf.force_sparsity_result, ax=ax_p)
ax_p.set_title("Pareto front")
ax_p.xaxis.set_major_locator(MaxNLocator(integer=True))

ax_r = fig.add_subplot(gs[1, 1])
idx_o = lap_first(list(support))
ratio = np.array([sel[i][0] / theta_true[i] if theta_true[i] else np.nan for i in idx_o])
ratio_err = np.array([sel[i][1] / abs(theta_true[i]) if theta_true[i] else np.nan for i in idx_o])
xs = np.arange(len(idx_o))
ax_r.axhline(1.0, color="#808080", lw=0.8, ls="--")
ax_r.errorbar(xs, ratio, yerr=ratio_err, fmt="o", ms=6, capsize=4,
              color=SFI_COLORS["inferred"], ecolor="#B0B0B0")
ax_r.set_xticks(xs)
ax_r.set_xticklabels([tex_label(i) for i in idx_o], rotation=35, ha="right", fontsize=9)
ax_r.set_ylabel("inferred / true")
ax_r.set_title("Selected coefficients (±1 s.e.)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Bootstrap: regrow the colony
# ----------------------------
#
# Restart the recovered equations from the four initial drops, with an
# independent noise realisation, and let them run for the whole 3000 time
# units — far beyond the four 12.5-unit clips they were learned from.
#
# One subtlety: the estimator treats the noise as white from cell to
# cell, which conserved noise is not — its spectrum grows as
# :math:`|k|^2`.  Knowing that structure, the per-cell diffusion
# :math:`\hat D` converts into the continuum amplitude
# :math:`\hat\sigma = (\hat D / \langle |k|^2\rangle)^{1/2}`, and the
# bootstrap uses conserved noise of that amplitude.

k_grid = 2 * np.pi * np.fft.fftfreq(GRID[0], DX)
mean_k2 = 2 * np.mean(k_grid**2)                      # ⟨|k|²⟩ on the 2D grid
sigma_hat = float(np.sqrt(np.mean(np.diag(np.asarray(inf.diffusion_average))) / mean_k2))
print(f"conserved-noise amplitude: sigma_hat = {sigma_hat:.4f}  (true {SIGMA})")

boot = OverdampedProcess(restrict(support),
                         D=conserved_noise_pbc(sigma=sigma_hat, grid_shape=GRID, dx=DX, n_fields=2))
boot.set_params(theta_F=jnp.asarray(coeffs, dtype=jnp.float32))
boot.set_extras(extras_global=box_extras)
boot.initialize(x0)
key, k_boot = random.split(key)
model = boot.simulate(dt=DT_SHOW, Nsteps=truth.T - 1, key=k_boot,
                      oversampling=100,     # integrator step 0.125, as for the truth
                      compute_observables=False)
model = TrajectoryCollection.from_arrays(X=np.concatenate([np.asarray(x0)[None], model.to_array()]),
                                         dt=DT_SHOW, extras_global=box_extras)
print(f"Bootstrap: {model.T} frames")

# %%
# Side by side over the whole run: the hidden truth (left), the camera's
# view (centre) and the recovered equations (right).  The patterns differ
# in detail — the noise and the spot divisions are random — but the colony
# grows the same way.

# sphinx_gallery_start_ignore
camera = degrade_spatial_data(truth, downscale=1, noise=PIXEL_NOISE,
                              data_loss_fraction=DEAD_PIXELS, bc="pbc", seed=2)
Xc_cam = np.asarray(camera.datasets[0].X)
Mc_cam = np.asarray(camera.datasets[0].mask)
Xm = model.to_array()


def cam_field(ti):
    return np.ma.masked_array(Xc_cam[ti, :, 1].reshape(GRID).T, mask=~Mc_cam[ti].reshape(GRID).T)


fig_m, axes_m = plt.subplots(1, 3, figsize=(6.6, 2.75))
fig_m.patch.set_facecolor(BG)
panels = [
    (axes_m[0], lambda ti: field(Xs, ti, 1), CMAP_V, "truth (hidden)"),
    (axes_m[1], cam_field, cm_rec, "camera"),
    (axes_m[2], lambda ti: field(Xm, ti, 1), CMAP_V, "recovered equations"),
]
ims = []
for ax, getter, cmap, title in panels:
    ims.append(ax.imshow(getter(0), origin="lower", cmap=cmap, vmin=0, vmax=0.45,
                         interpolation="nearest"))
    ax.set_title(title, fontsize=9, color="#D0D0D0")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
stamp = fig_m.suptitle("t = 0", fontsize=10, color="#D0D0D0")


def update(ti):
    for im, (_, getter, _, _) in zip(ims, panels):
        im.set_data(getter(ti))
    stamp.set_text(f"t = {t_show[ti]:.0f}")
    return ims


plt.rcParams["animation.bitrate"] = 2000   # kbit/s: noisy frames, keep the mp4 near 5 MB
anim_gs = FuncAnimation(fig_m, update, frames=range(truth.T), interval=70)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Two pattern statistics make the comparison quantitative: the mean
# concentration of :math:`V`, which grows with the number of spots, and
# the radially averaged autocorrelation :math:`C(r)` of :math:`V`, whose
# first maximum past its first zero is the distance between neighbouring
# spots.

from SFI.utils.plotting import spatial_acorr2d

Xs, Xm = truth.to_array(), model.to_array()


def spot_spacing(X):
    r, C = spatial_acorr2d(X[-1, :, 1].reshape(GRID), dx=DX)
    beyond = r > r[np.argmax(C < 0)]                 # past the first zero crossing
    return r[beyond][np.argmax(C[beyond])]


for name, X in [("truth", Xs), ("recovered equations", Xm)]:
    print(f"{name:>20s}: <V> at t = {t_show[-1]:.0f}: {X[-1, :, 1].mean():.4f}, "
          f"spot spacing {spot_spacing(X):.1f}")
r_bins, _ = spatial_acorr2d(Xs[-1, :, 1].reshape(GRID), dx=DX)
print(f"(C(r) is binned in steps of {r_bins[1] - r_bins[0]:.1f} cells)")

# %%
# Left: colony growth, with the four analysis clips marked — the recovered
# equations track the truth over the whole run.  Right: :math:`C(r)` at
# the end of the run; the two spot spacings agree to within one radial
# bin.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
for tc in t_clips:
    ax.axvspan(tc, tc + N_CLIP * DT, color=SFI_COLORS["highlight"], alpha=0.5, lw=3)
ax.plot(t_show, Xs[..., 1].mean(axis=1), color=SFI_COLORS["exact"], lw=2.2, label="truth")
ax.plot(t_show, Xm[..., 1].mean(axis=1), color=SFI_COLORS["inferred"], lw=1.6, ls="--",
        label="recovered equations")
ax.plot([], [], color=SFI_COLORS["highlight"], lw=3, label="analysis clips")
ax.set_xlabel("time")
ax.set_ylabel(r"$\langle V\rangle$")
ax.set_title("Colony growth")
ax.legend(loc="upper left")
ax = axes[1]
for X, lbl, c, ls in [(Xs, "truth", SFI_COLORS["exact"], "-"),
                      (Xm, "recovered equations", SFI_COLORS["inferred"], "--")]:
    r, C = spatial_acorr2d(field(X, -1, 1), dx=DX)
    ax.plot(r, C, color=c, ls=ls, lw=2 if ls == "-" else 1.6, label=lbl)
ax.axhline(0, color="#808080", lw=0.6)
ax.set_xlim(0, 40)
ax.set_xlabel("distance r")
ax.set_ylabel("C(r)")
ax.set_title(f"Spot spacing at t = {t_show[-1]:.0f}")
ax.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Where it breaks
# ---------------
#
# How much corruption can the pipeline take?  We rerun inference and
# selection on the last clip alone — a dish full of spots — climbing a
# ladder of corruptions: first more dead pixels, then more pixel noise on
# top of 2% dead pixels.

LADDER = [  # (dead-pixel fraction, pixel noise)
    (0.02, 0.0), (0.05, 0.0),
    (0.02, 0.005), (0.02, 0.01), (0.02, 0.015), (0.02, 0.02),
]
sweep = []
for dead, noise in LADDER:
    rec = degrade_spatial_data(clips[-1], downscale=1, noise=noise or None,
                               data_loss_fraction=dead, stencil_radius=2,
                               bc="pbc", seed=3)
    inf_s = OverdampedLangevinInference(rec)
    inf_s.compute_diffusion_constant()
    inf_s.infer_force_linear(BASIS)
    inf_s.sparsify_force(criterion="PASTIS")
    inf_s.compute_force_error()
    sweep.append(inf_s)
    sup_s = set(np.asarray(inf_s.force_support))
    print(f"{100 * dead:.0f}% dead, pixel noise {noise:<5}: "
          f"{100 * np.mean(rec.datasets[0].dynamic_mask):4.1f}% of cells fit, "
          f"sqrt(Λ) = {np.sqrt(np.diag(np.asarray(inf_s.Lambda))).round(4)}; "
          f"{len(sup_s & set(support_true))}/7 true terms, "
          f"{len(sup_s - set(support_true))} spurious")

# %%
# Top: which of the 34 candidates PASTIS keeps at each rung — gold for
# the Gray-Scott terms, pink for spurious ones, an orange frame for a
# missed true term.  Bottom: the seven true coefficients, divided by their
# true values, with their error bars (values off the scale are drawn as
# arrows at the edge).
#
# **Dead pixels cost data, not the model.**  At 5% per frame the stencil
# exclusion leaves only 4% of the cells; the error bars widen (the two
# :math:`V`-equation terms :math:`V` and :math:`UV^2` land about two of
# them low), and the seven terms still come back, with no spurious one.
#
# **Pixel noise is the breaking factor.**  On this clip the edge lies
# between pixel noise 0.01 and 0.015 — 2 to 3% of the spot peak: up to
# 0.01 PASTIS returns exactly the seven terms, at 0.015 it adds a swarm of
# spurious ones, and at 0.02 it loses true terms as well.  Pixel noise
# enters *inside* the candidates: a noisy :math:`V` turns :math:`V^2` into
# :math:`(V+\varepsilon)^2`, and every nonlinear term and stencil picks up
# an errors-in-variables bias that grows with the noise variance.  The
# noise-robust estimator pairs frames whose noise is independent, which
# removes the noise's direct contribution, but it cannot undo this
# distortion of the candidates themselves.  On a well-conditioned library
# the bias would only nudge the coefficients.  Here several cubic
# combinations of :math:`U` and :math:`V` nearly coincide on the data (the
# condition number printed above), so past some noise level a wrong
# combination fits better than the true one — and with hundreds of
# thousands of rows the improvement is significant: PASTIS keeps the
# spurious terms, and the coefficients lose their meaning.
#
# More data cannot push this edge back: the bias stays while the error
# bars shrink, so with more rows the edge moves to *lower* noise.  This is
# a genuine limit of the linear estimators on nearly collinear polynomial
# libraries, and the reason the four-clip analysis above uses a camera at
# half the noise this single clip tolerates.

# sphinx_gallery_start_ignore
rung_labels = [f"{100 * d:.0f}% dead" + (f" + noise {n:g}" if n else "") for d, n in LADDER]
fig, (ax_m, ax_c) = plt.subplots(2, 1, figsize=(11, 7.0), height_ratios=[1.0, 1.15])
state = np.zeros((len(LADDER), len(labels)))
for r, inf_s in enumerate(sweep):
    for i in np.asarray(inf_s.force_support):
        state[r, i] = 1 if i in support_true else 2
ax_m.imshow(state, cmap=ListedColormap(["#26282e", SFI_COLORS["inferred"], SFI_COLORS["error"]]),
            vmin=0, vmax=2, aspect="auto", interpolation="nearest")
for r, inf_s in enumerate(sweep):
    for i in set(support_true) - set(np.asarray(inf_s.force_support)):
        ax_m.add_patch(plt.Rectangle((i - 0.5, r - 0.5), 1, 1, fill=False,
                                     ec=SFI_COLORS["exact"], lw=1.5))
ax_m.axvline(n_per - 0.5, color="#D0D0D0", lw=1)
ax_m.axhline(1.5, color="#808080", lw=0.8, ls=":")
ax_m.set_xticks(range(len(labels)))
ax_m.set_xticklabels(["$" + (tex_term(l) or "1") + "$" for l in labels], rotation=90, fontsize=7)
ax_m.set_yticks(range(len(LADDER)))
ax_m.set_yticklabels(rung_labels, fontsize=8)
ax_m.text(n_per / 2 - 0.5, -0.9, r"$\partial_t U$ candidates", ha="center", fontsize=9)
ax_m.text(1.5 * n_per - 0.5, -0.9, r"$\partial_t V$ candidates", ha="center", fontsize=9)
for sp in ax_m.spines.values():
    sp.set_visible(False)

palette = [SFI_COLORS[k] for k in ("data", "inferred", "exact", "highlight", "error",
                                   "tertiary", "secondary")]
YLIM = (0.5, 1.5)
for j, i in enumerate(lap_first(list(support_true))):
    vals, errs = [], []
    for inf_s in sweep:
        sup_s = list(np.asarray(inf_s.force_support))
        if i in sup_s:
            k = sup_s.index(i)
            vals.append(float(np.asarray(inf_s.force_coefficients)[k]) / theta_true[i])
            errs.append(float(np.asarray(inf_s.force_coefficients_stderr)[k]) / abs(theta_true[i]))
        else:
            vals.append(np.nan)
            errs.append(np.nan)
    vals, errs = np.array(vals), np.array(errs)
    x = np.arange(len(LADDER)) + (j - 3) * 0.07
    inside = (vals > YLIM[0]) & (vals < YLIM[1])
    ax_c.errorbar(x[inside], vals[inside], yerr=errs[inside], fmt="o", ms=4, capsize=2,
                  lw=1, color=palette[j], label=tex_label(i))
    for xx, vv in zip(x[~inside], vals[~inside]):
        if np.isfinite(vv):
            ax_c.plot(xx, YLIM[1] - 0.03 if vv > 1 else YLIM[0] + 0.03,
                      "^" if vv > 1 else "v", ms=7, color=palette[j])
ax_c.axhline(1.0, color="#808080", lw=0.8, ls="--")
ax_c.axvline(1.5, color="#808080", lw=0.8, ls=":")
ax_c.set_ylim(*YLIM)
ax_c.set_xlim(-0.5, len(LADDER) - 0.5)
ax_c.set_xticks(range(len(LADDER)))
ax_c.set_xticklabels([lab.replace(" + ", "\n+ ") for lab in rung_labels], fontsize=8)
ax_c.set_ylabel("inferred / true")
ax_c.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.suptitle("Corruption ladder (last clip): dead pixels, then pixel noise", fontsize=13)
fig.get_layout_engine().set(rect=(0, 0.02, 1, 0.98))   # keep the two-line labels off the edge
plt.show()
# sphinx_gallery_end_ignore

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
ax_t.imshow(field(Xs, -1, 1), origin="lower", cmap=CMAP_V, vmin=0, vmax=0.45)
ax_t.set_xticks([])
ax_t.set_yticks([])
for sp in ax_t.spines.values():
    sp.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore
