r"""
Nonreciprocal active particles at scale — 10 000 particles
===========================================================

Infer three pairwise interaction kernels — **repulsion**, **alignment**
and **pursuit** — from the tracks of 10 000 self-propelled particles
whose interactions are **nonreciprocal**: each particle reacts to what it
sees ahead of it, not to what is behind.  Pursuit and alignment organise
the swarm into worm-like streams of chevrons that sweep through the box —
the kind of structure seen in tracked swarms of cells, insects or robots.

At this scale a list of all :math:`N^2/2 = 5\times 10^7` pairs is out of
reach.  Both the simulation and the inference run on **neighbour lists**
(CSR format) rebuilt every few steps; because the interaction kernels
have a finite range, the lists capture every interacting pair *exactly*.

The page walks through:

1. composing the force from the geometric primitives of
   :mod:`SFI.bases.pairs` — :func:`~SFI.bases.pairs.vision_gate` for the
   nonreciprocal coupling, :func:`~SFI.bases.pairs.particle_heading` for
   pursuit;
2. simulation with Verlet neighbour lists
   (:func:`~SFI.langevin.chunked.simulate_chunked`);
3. an overcomplete inference basis — compactly supported radial kernels
   of several ranges for each interaction;
4. **linear** force inference on short stretches of the recording, each
   with its own neighbour list;
5. the recovered kernels, with confidence bands and the range of pair
   distances the data actually sample;
6. a **bootstrap**: the inferred model restarted from a frame of the data,
   played side by side with it, and compared on collective statistics the
   fit never used.

.. rubric:: Tags

synthetic · overdamped · multi-particle · linear · large-scale · nonreciprocal
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "linear", "large-scale", "nonreciprocal"]
# sphinx_gallery_thumbnail_number = 6

from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output, plot_kernel_ci
from SFI.utils.plotting import animate_particles, animate_particles_comparison, dark_fig

apply_style()

# Movies are encoded on the figure's own (black) background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"

# Print library warnings without the local source path of this script.
warnings.formatwarning = lambda msg, cat, *args, **kw: f"{cat.__name__}: {msg}\n"


def _heading_scatter(ax, X, box, *, s=1.0, region=None):
    """Particles of one frame, wrapped into the box, coloured by heading."""
    xy = np.asarray(X[:, :2]) % box
    th = (np.asarray(X[:, 2]) + np.pi) % (2 * np.pi) - np.pi
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=th, cmap="hsv", vmin=-np.pi, vmax=np.pi,
               edgecolors="none", rasterized=True)
    (x0, x1), (y0, y1) = region if region is not None else ((0, box[0]), (0, box[1]))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _movie_style(axes):
    """Borderless panels with filled (edge-free) markers."""
    for ax in np.atleast_1d(axes):
        for col in ax.collections:
            col.set_edgecolor("none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


def _densest_region(X, box, width):
    """Square region of side ``width`` centred on the densest spot of a frame."""
    xy = np.asarray(X[:, :2]) % box
    counts = cKDTree(xy, boxsize=box).query_ball_point(xy, width / 4, return_length=True)
    c = np.clip(xy[np.argmax(counts)], width / 2, box - width / 2)
    return (c[0] - width / 2, c[0] + width / 2), (c[1] - width / 2, c[1] + width / 2)
# sphinx_gallery_end_ignore
stamp_output()

# %%
# System: nonreciprocal, vision-gated active particles
# ------------------------------------------------------
#
# Each particle has state :math:`(x, y, \theta)` and moves at speed
# :math:`c_0` along its heading :math:`\hat e_\theta`.  Three pair
# interactions act on top of this self-propulsion; two of them are
# filtered by a **vision cone** :math:`v(\delta) = (1 + \cos\delta)/2`,
# where :math:`\delta` is the bearing of the neighbour relative to the
# focal particle's heading — so :math:`i` may react to :math:`j` while
# :math:`j` ignores :math:`i`:
#
# - **Repulsion**: :math:`-\varepsilon\, b_{R}(r)\,\hat r_{ij}`
# - **Alignment** (a torque on :math:`\theta_i`):
#   :math:`A\, v(\delta)\, \sin(\theta_j-\theta_i)\, b_{L_a}(r)`
# - **Pursuit**: :math:`P\, v(\delta)\, b_{L_p}(r)\, \hat e_{\theta_j}` — a
#   particle is carried along by the neighbours it sees ahead of it.
#
# The radial profiles are compact bumps
# :math:`b_\ell(r) = (1 - r/\ell)^2` for :math:`r < \ell`, zero beyond:
# every interaction vanishes smoothly at a finite range, the largest of
# which, :math:`L_a = 4`, is the interaction cutoff.  All three
# components of the state receive white noise of strength :math:`D`.

N_particles = 10_000
density = 0.15                                   # particles per unit area
Lx = Ly = float(np.sqrt(N_particles / density))  # periodic box, ≈ 258
dt = 0.02
D_iso = 0.02
box = jnp.array([Lx, Ly])
box_np = np.array([Lx, Ly])

theta_F_exact = dict(
    c0=1.0,              # self-propulsion speed
    eps=8.0, R0=1.0,     # repulsion strength, range
    A=1.0, La=4.0,       # alignment strength, range
    P=0.75, Lp=3.0,      # pursuit strength, range
)
cutoff = 4.0             # interaction range: every kernel vanishes beyond r = 4

# %%
# Building the simulation force
# -------------------------------
#
# Each primitive of :mod:`SFI.bases.pairs` is one geometric factor — a
# direction, a gate, a radial profile — and products of them are
# dispatched over the pairs of a CSR neighbour list stored in the extras
# (``dispatch_pairs_from_extras``).

from SFI.bases.pairs import (
    angle_coupling,
    compact_kernels,
    heading_vector,
    pair_direction,
    parametric_radial_kernel,
    particle_heading,
    scalar_pair_basis,
    vision_gate,
)
from SFI.statefunc import Basis

dim = 3                                    # (x, y, θ) per particle
xy = dict(box="extras", spatial_dims=slice(0, 2))

e_heading = heading_vector(dim=dim, angle_index=2)                      # ê_θ
e_ij = pair_direction(dim=dim, embed_dim=dim, embed_axes=[0, 1], **xy)  # r̂_ij
sin_dtheta = angle_coupling(jnp.sin, dim=dim, angle_index=2)            # sin(θ_j − θ_i), on θ
e_j = particle_heading(1, dim=dim, angle_index=2)                       # ê_θj
vision = vision_gate(lambda d: (1 + jnp.cos(d)) / 2, dim=dim, angle_index=2, **xy)


def bump(r, ell):
    return jnp.where(r < ell, (1 - r / ell) ** 2, 0.0)


k_repel = parametric_radial_kernel(lambda r, p: -p["eps"] * bump(r, p["R0"]),
                                   params={"eps": (), "R0": ()}, dim=dim, **xy)
k_align = parametric_radial_kernel(lambda r, p: p["A"] * bump(r, p["La"]),
                                   params={"A": (), "La": ()}, dim=dim, **xy)
k_pursuit = parametric_radial_kernel(lambda r, p: p["P"] * bump(r, p["Lp"]),
                                     params={"P": (), "Lp": ()}, dim=dim, **xy)

csr = dict(indptr_key="indptr", indices_key="indices")
F_sim = (
    e_heading.to_psf(coeff_key="c0")
    + (k_repel * e_ij).dispatch_pairs_from_extras(**csr, return_as="psf")
    + (k_align * vision * sin_dtheta).dispatch_pairs_from_extras(**csr, return_as="psf")
    + (k_pursuit * vision * e_j).dispatch_pairs_from_extras(**csr, return_as="psf")
)

# %%
# Simulation with Verlet neighbour lists
# ----------------------------------------
#
# :func:`~SFI.langevin.chunked.simulate_chunked` integrates
# ``rebuild_every`` steps with a fixed neighbour list, then rebuilds it
# (with :class:`scipy.spatial.cKDTree`) from the current positions.  The
# list holds every pair closer than ``cutoff + skin``.  Between two
# rebuilds a pair distance shrinks by at most twice the largest
# single-particle displacement, so as long as no particle moves farther
# than ``skin / 2``, every pair that comes within the cutoff is already on
# the list and the simulated force is exact — the **Verlet criterion**.
# Particles swim at :math:`c_0 = 1` and pursuit makes the streams faster;
# we rebuild every 10 steps (0.2 time units) with a skin of 3, i.e. we
# allow speeds up to 7.5.  The simulator checks the criterion at every
# rebuild and would warn if it failed.

from SFI.langevin import OverdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.utils.neighbors import make_neighbor_extras

rebuild_every = 10
skin = 3.0
n_relax, n_record = 1250, 750          # 25 time units to form streams, then 15 recorded

key = random.PRNGKey(0)
key, kx, kth = random.split(key, 3)
x0 = jnp.concatenate([random.uniform(kx, (N_particles, 2)) * box,
                      random.uniform(kth, (N_particles, 1), minval=-jnp.pi, maxval=jnp.pi)],
                     axis=1)

proc = OverdampedProcess(F_sim, D=D_iso, extras_global={
    "box": box, **make_neighbor_extras(np.asarray(x0[:, :2]), cutoff + skin, box_np)})
proc.set_params(theta_F=theta_F_exact)
proc.initialize(x0)

chunk_kw = dict(cutoff=cutoff, skin=skin, box=box_np, rebuild_every=rebuild_every,
                spatial_dims=slice(0, 2))
coll = simulate_chunked(proc, dt=dt, Nsteps=n_relax + n_record, key=random.PRNGKey(1), **chunk_kw)

X_all = np.concatenate([np.asarray(x0)[None], coll.to_array(axis="time")])   # frame k = step k
print(f"{N_particles} particles in a {Lx:.0f} × {Ly:.0f} periodic box, "
      f"{X_all.shape[0] - 1} steps of dt = {dt}")

# %%
# The swarm
# ----------
#
# Starting from random positions and headings, the particles organise
# within a few tens of time units into long-lived, worm-like streams made
# of chevron-shaped groups, each travelling along its stream.  The
# snapshot is the last frame of the run, coloured by heading angle; the
# dashed frame marks the region shown in the bootstrap movie below.

# sphinx_gallery_start_ignore
t_start = n_relax                              # first frame of the recorded stretch
region = _densest_region(X_all[t_start], box_np, 100.0)

fig_snap, ax_snap = dark_fig(figsize=(7, 7))
_heading_scatter(ax_snap, X_all[-1], box_np, s=2.5)
ax_snap.add_patch(Rectangle((region[0][0], region[1][0]), region[0][1] - region[0][0],
                            region[1][1] - region[1][0], fill=False, edgecolor="#D0D0D0",
                            lw=1.0, ls="--"))
ax_snap.set_title(f"N = {N_particles},  t = {(X_all.shape[0] - 1) * dt:.0f}  (colour = heading)",
                  color="#D0D0D0", fontsize=12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# How the streams form: the whole run, one frame every 0.16 time units.

# sphinx_gallery_start_ignore
from SFI import TrajectoryCollection

fig_m1, ax_m1 = dark_fig(figsize=(5.0, 5.0))
anim_formation = animate_particles(
    TrajectoryCollection.from_arrays(X=X_all, dt=dt), color_dim=2, cmap="hsv",
    vmin=-np.pi, vmax=np.pi, box=box_np, s=2.0, skip=8, interval=80, ax=ax_m1)
_movie_style(ax_m1)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Overcomplete inference basis
# ------------------------------
#
# For inference we keep the geometric primitives but replace each
# parametric profile by a generic dictionary of compact bumps
# :math:`(1 - r/\ell)^2` (:func:`~SFI.bases.pairs.compact_kernels`) at
# ranges :math:`\ell = 1, 2, 3, 4` — the largest equals the cutoff, so the
# same neighbour lists serve the inference.  The alignment dictionary
# starts at :math:`\ell = 2`: inside the streams, neighbours closer than 1
# are almost exactly parallel, so a range-1 alignment bump would rest on
# almost no signal (its column of the Gram matrix is tiny, and SFI warns
# that the fit is ill-conditioned).

ranges = (1.0, 2.0, 3.0, 4.0)
kernels = [k for ell in ranges for k in compact_kernels(degrees=[0], cutoff=ell)]
kernels_align = kernels[1:]                                        # ℓ = 2, 3, 4

B_repel = (scalar_pair_basis(kernels, dim=dim, **xy) * e_ij
           ).dispatch_pairs_from_extras(**csr, return_as="basis")
B_align = (scalar_pair_basis(kernels_align, dim=dim, **xy) * vision * sin_dtheta
           ).dispatch_pairs_from_extras(**csr, return_as="basis")
B_pursuit = (scalar_pair_basis(kernels, dim=dim, **xy) * vision * e_j
             ).dispatch_pairs_from_extras(**csr, return_as="basis")
B_full = Basis.stack([e_heading, B_repel, B_align, B_pursuit])
print(f"{B_full.n_features} features: self-propulsion + {B_repel.n_features} repulsion "
      f"+ {B_align.n_features} alignment + {B_pursuit.n_features} pursuit kernels")

# %%
# Fitting on short stretches, each with its own neighbour list
# --------------------------------------------------------------
#
# Real tracking data come without neighbour lists, so we build them from
# the data the way the simulator did: we cut the recorded stretch into
# intervals of 10 steps, build each interval's list from its first frame
# with radius ``cutoff + skin``, and check the Verlet criterion on the
# tracks themselves.  Because the kernels vanish at the cutoff and no pair
# can enter their range unseen, the pair features of every interval are
# continuous and exact — the lists are invisible to the estimator (see
# :ref:`bases-cutoffs`).  Lists are padded to a common length so that one
# compiled program serves every interval.

from SFI import OverdampedLangevinInference
from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr

starts = range(t_start, t_start + n_record, 5 * rebuild_every)    # one interval in five
lists = [build_neighbor_csr(X_all[s, :, :2], cutoff + skin, box_np) for s in starts]
max_nnz = max(len(indices) for _, indices in lists)

intervals, max_disp = [], 0.0
for s, (indptr, indices) in zip(starts, lists):
    X_k = X_all[s:s + rebuild_every + 1]                           # 11 frames, 10 steps
    disp = np.abs(X_k[:, :, :2] - X_k[0, :, :2])
    disp = np.minimum(disp, box_np - disp)                         # periodic wrap
    max_disp = max(max_disp, float(np.sqrt((disp ** 2).sum(-1)).max()))
    indptr, indices = pad_neighbor_csr(indptr, indices, max_nnz)
    intervals.append((X_k, {"box": box, "indptr": jnp.asarray(indptr),
                            "indices": jnp.asarray(indices)}))


def interval_collection(stride=1):
    """All intervals in one collection, keeping every ``stride``-th frame."""
    parts = [TrajectoryCollection.from_arrays(X=X_k[::stride], dt=stride * dt, extras_global=eg)
             for X_k, eg in intervals]
    return parts[0].concat(parts[1:])


coll_fit = interval_collection()
print(f"{len(intervals)} intervals × {rebuild_every} steps × {N_particles} particles")
print(f"largest displacement within an interval: {max_disp:.2f}  (skin / 2 = {skin / 2})")

# %%
# The tracks are noise-free synthetic data, so we use the ``"clean"``
# preset (Itô moments with the trapeze Gram matrix).  The drift is strong
# compared with the noise over one step, so the diffusion constant is
# measured with the ``"WeakNoise"`` estimator.

inf = OverdampedLangevinInference(coll_fit)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_full, preset="clean")
inf.compute_force_error()
inf.compare_to_exact(force_exact=F_sim.bind(theta_F_exact), maxpoints=4000)

c0_hat, c0_cov = inf.coeff_block(e_heading)
print(f"D (x, y, θ)    = {np.diag(np.asarray(inf.diffusion_average)).round(4)}   (true {D_iso})")
print(f"c0             = {float(np.ravel(c0_hat)[0]):.4f} ± "
      f"{float(np.sqrt(np.ravel(c0_cov)[0])):.4f}   (true {theta_F_exact['c0']})")
print(f"predicted NMSE = {inf.force_predicted_MSE:.1e}   (sampling error, from the data alone)")
print(f"realised NMSE  = {float(inf.NMSE_force):.1e}   (against the true force)")

# %%
# With 1.5 million particle displacements the *sampling* error is tiny —
# that is what the predicted NMSE measures, from the data alone.  The
# error against the true force is about twice as large: a small
# systematic part remains, because the linear estimator is exact only as
# :math:`\Delta t \to 0` and inside the dense chevrons the repulsion
# between neighbours changes appreciably within one frame.  Both are well
# below a percent of the force.
#
# Recovered interaction kernels
# --------------------------------
#
# Each kernel is reconstructed as :math:`\sum_k c_k \phi_k(r)` with a 95%
# confidence band from the coefficient covariance
# (:func:`~SFI.inference.kernel_predict_ci`).  The grey histogram shows
# how often each pair distance occurs in the data.  Repulsion keeps the
# particles apart, so almost no pair is closer than about 0.5: below that
# the kernels are an *extrapolation* of the dictionary.  The bands widen
# there, but they only account for sampling noise, not for how the
# dictionary chooses to extrapolate — read the kernels where the
# histogram has pairs.

from SFI.inference import kernel_predict_ci

r_eval = np.linspace(0.0, cutoff, 241)
ci = {
    "repel": kernel_predict_ci(r_eval, kernels, *inf.coeff_block(B_repel)),
    "align": kernel_predict_ci(r_eval, kernels_align, *inf.coeff_block(B_align)),
    "pursuit": kernel_predict_ci(r_eval, kernels, *inf.coeff_block(B_pursuit)),
}
p = theta_F_exact
true_kernels = {
    "repel": -p["eps"] * np.asarray(bump(r_eval, p["R0"])),
    "align": p["A"] * np.asarray(bump(r_eval, p["La"])),
    "pursuit": p["P"] * np.asarray(bump(r_eval, p["Lp"])),
}

# sphinx_gallery_start_ignore
# Distances of all pairs closer than the cutoff, in a few fitted frames.
r_pairs = []
for s in list(starts)[::5]:
    pairs = cKDTree(X_all[s, :, :2] % box_np, boxsize=box_np).query_pairs(cutoff, output_type="ndarray")
    d = X_all[s, pairs[:, 0], :2] - X_all[s, pairs[:, 1], :2]
    d -= box_np * np.round(d / box_np)
    r_pairs.append(np.hypot(d[:, 0], d[:, 1]))
r_pairs = np.concatenate(r_pairs)

fig_kern, axes_kern = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, name, title in zip(axes_kern, ["repel", "align", "pursuit"],
                           ["Repulsion", "Alignment (vision-gated)", "Pursuit (vision-gated)"]):
    ax_h = ax.twinx()
    ax_h.hist(r_pairs, bins=80, range=(0, cutoff), color="#808080", alpha=0.3)
    ax_h.set_yticks([]); ax_h.spines["right"].set_visible(False)
    ax.set_zorder(ax_h.get_zorder() + 1); ax.patch.set_visible(False)
    plot_kernel_ci(ax, r_eval, true_kernels[name], ci[name], label_true="true",
                   label_inferred="inferred", ci_label="95% CI")
    k_true = true_kernels[name] / np.max(np.abs(true_kernels[name]))
    ax.set_ylim(min(0.0, k_true.min()) - 0.3, max(0.0, k_true.max()) + 0.3)
    ax.set_xlim(0, cutoff)
    ax.set_xlabel("pair distance r")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right" if name == "repel" else "upper right")
axes_kern[0].set_ylabel(r"kernel / max $|$true$|$")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Residual diagnostics
# ----------------------
#
# Standardised residuals pooled over all particles and the three state
# components — a check of the fit that needs no ground truth.

from SFI.diagnostics import assess

report = assess(inf, level="standard")
report.print_summary()

# %%
# With 4.5 million residuals the tests resolve tiny departures.  Every
# residual moment is within 2% of :math:`\mathcal N(0, 1)`; the standard
# deviation sits slightly below 1, as expected at a finite frame interval
# — the χ² line's expectation accounts for that, the normality test,
# which compares with exactly :math:`\mathcal N(0, 1)`, does not.  The χ²
# line compares the residual variance with its expectation at the sampled
# :math:`\Delta t`, computed to first order in :math:`\Delta t`.  A
# fraction-of-a-percent excess is enough for a large z-score at this
# sample size, and the NMSE the report converts it to is a scale for the
# excess, not a measured error: the comparison with the true force above
# shows a realised error about twice the predicted one, not a thousand
# times.
#
# Is the excess a missing interaction or the finite frame interval?  A
# missing interaction biases the fit whatever the frame interval; a
# discretisation effect grows with it.  Refitting the same intervals on
# every other frame is a check that needs no ground truth — useful here
# because the report's automatic finite-:math:`\Delta t` scale is not
# available for interacting bases:

inf_2dt = OverdampedLangevinInference(interval_collection(stride=2))
inf_2dt.compute_diffusion_constant(method="WeakNoise")
inf_2dt.infer_force_linear(B_full, preset="clean")
inf_2dt.compute_force_error()
inf_2dt.compare_to_exact(force_exact=F_sim.bind(theta_F_exact), maxpoints=4000)
report_2dt = assess(inf_2dt, level="standard")

for label, fit, rep in [("Δt ", inf, report), ("2Δt", inf_2dt, report_2dt)]:
    chi2 = rep.residuals["mse_consistency"]
    excess = chi2["mean_mahalanobis_norm"] / chi2["expected_mahalanobis_norm"] - 1
    print(f"frame interval {label}:  χ² excess {100 * excess:+.2f}% (z = {chi2['excess_z']:+.0f})   "
          f"predicted NMSE {fit.force_predicted_MSE:.1e}   realised NMSE {float(fit.NMSE_force):.1e}")

# %%
# Doubling the frame interval more than triples the χ² excess, and the
# realised error grows while the predicted sampling error does not:
# the excess comes from the finite frame interval — inside the dense
# chevrons the steep repulsion between neighbours changes within one
# frame — not from a missing interaction.  At the recorded frame rate the
# resulting bias is a fraction of a percent of the force.

# %%
# Bootstrap: does the inferred model make the same swarm?
# ---------------------------------------------------------
#
# An inferred model is a simulator.  We restart it from the first frame
# of the recorded stretch — with the same neighbour-list machinery — and
# run it for the same 15 time units.

proc_boot = inf.simulate_bootstrapped_trajectory(random.PRNGKey(2), simulate=False)
proc_boot.set_extras(extras_global={
    "box": box, **make_neighbor_extras(X_all[t_start, :, :2], cutoff + skin, box_np)})
proc_boot.initialize(jnp.asarray(X_all[t_start]))
coll_boot = simulate_chunked(proc_boot, dt=dt, Nsteps=n_record, key=random.PRNGKey(3), **chunk_kw)

X_data = X_all[t_start:t_start + n_record + 1]                    # the same frames of the data
X_boot = np.concatenate([X_all[t_start][None], coll_boot.to_array(axis="time")])

# %%
# The data (left) and the inferred model (right), in the region framed in
# the snapshot above, one frame every 0.06 time units.  Both start from
# the same configuration.  The large streams persist in both runs while
# the details soon part ways — individual trajectories are random — but
# the two should look alike: the same chevrons, moving at the same
# speeds.

# sphinx_gallery_start_ignore
fig_cmp, axes_cmp = dark_fig(1, 2, figsize=(6.6, 3.6))
anim_bootstrap = animate_particles_comparison(
    [TrajectoryCollection.from_arrays(X=X_data, dt=dt),
     TrajectoryCollection.from_arrays(X=X_boot, dt=dt)],
    titles=["data", "inferred model"], color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
    box=box_np, s=2.5, skip=3, interval=70, axes=axes_cmp,
)
for ax in axes_cmp:
    ax.set_xlim(*region[0]); ax.set_ylim(*region[1])
_movie_style(axes_cmp)
plt.show()
# sphinx_gallery_end_ignore

# %%
# To make "alike" quantitative we compare, over the second half of both
# runs, three collective statistics that the fit never targeted: the pair
# correlation :math:`g(r)` — how particles pack inside the chevrons; the
# alignment :math:`\langle\cos(\theta_i-\theta_j)\rangle` of pairs at
# distance :math:`r` — how far orientational order reaches along the
# streams; and the distribution of particle speeds, measured over 0.2 time
# units — pursuit lets a particle swim faster than :math:`c_0` when it
# follows others.

# sphinx_gallery_start_ignore
def pair_stats(frames, r_g=8.0, r_cos=40.0):
    """Pair correlation g(r) (fine bins, r < r_g) and pair alignment <cos Δθ>(r)
    (unit bins, r < r_cos), averaged over frames."""
    e_g, e_c = np.linspace(0.0, r_g, 65), np.linspace(0.0, r_cos, 41)
    n_g, n_c, cos_c = np.zeros(64), np.zeros(40), np.zeros(40)
    for Xt in frames:
        pairs = cKDTree(Xt[:, :2] % box_np, boxsize=box_np).query_pairs(r_cos, output_type="ndarray")
        d = Xt[pairs[:, 0], :2] - Xt[pairs[:, 1], :2]
        d -= box_np * np.round(d / box_np)
        r = np.hypot(d[:, 0], d[:, 1])
        n_g += np.histogram(r, e_g)[0]
        n_c += np.histogram(r, e_c)[0]
        cos_c += np.histogram(r, e_c, weights=np.cos(Xt[pairs[:, 0], 2] - Xt[pairs[:, 1], 2]))[0]
    shell = np.pi * (e_g[1:] ** 2 - e_g[:-1] ** 2)
    g = n_g / (len(frames) * 0.5 * N_particles * density * shell)
    return 0.5 * (e_g[1:] + e_g[:-1]), g, 0.5 * (e_c[1:] + e_c[:-1]), cos_c / np.maximum(n_c, 1)


def speeds(X, lag):
    d = X[lag:, :, :2] - X[:-lag, :, :2]
    return np.hypot(d[..., 0], d[..., 1]).ravel() / (lag * dt)


late = np.arange(n_record // 2, n_record + 1, n_record // 12)
stats = {}
for name, X in [("data", X_data), ("inferred model", X_boot)]:
    stats[name] = (*pair_stats(X[late]), speeds(X[n_record // 2:], rebuild_every))

fig_bs, (ax_g, ax_a, ax_v) = plt.subplots(1, 3, figsize=(14, 4))
for name, color, ls in [("data", SFI_COLORS["data"], "-"),
                        ("inferred model", SFI_COLORS["inferred"], "--")]:
    r_g, g, r_c, cos_r, v = stats[name]
    ax_g.plot(r_g, g, ls, color=color, lw=2, label=name)
    ax_a.plot(r_c, cos_r, ls, color=color, lw=2, label=name)
    ax_v.hist(v, bins=80, range=(0, 6), density=True, histtype="step", color=color, lw=2,
              ls=ls, label=name)
ax_g.axhline(1, ls=":", lw=0.8, color="#808080")
ax_g.set_xlabel("pair distance r"); ax_g.set_ylabel(r"$g(r)$"); ax_g.set_title("Pair correlation")
ax_a.axhline(0, ls=":", lw=0.8, color="#808080")
ax_a.set_xlabel("pair distance r"); ax_a.set_ylabel(r"$\langle\cos(\theta_i-\theta_j)\rangle$")
ax_a.set_title("Pair alignment")
ax_v.axvline(theta_F_exact["c0"], ls=":", lw=0.8, color="#808080")
ax_v.set_xlabel("speed over 0.2 time units"); ax_v.set_ylabel("probability density")
ax_v.set_title("Particle speeds"); ax_v.set_yscale("log")
for ax in (ax_g, ax_a, ax_v):
    ax.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
# The inferred model reproduces all three: how neighbours pack inside the
# chevrons (the peaks of :math:`g(r)`), how far orientational order
# reaches along the streams, and the speeds.  Most particles move well
# faster than :math:`c_0 = 1` (dotted line; the measured speed also
# contains a small diffusive part) because they are carried along by the
# neighbours they pursue — and the inferred pursuit kernel reproduces the
# whole distribution, tail included.

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb, ax_t = dark_fig(figsize=(4, 4))
_heading_scatter(ax_t, X_all[-1], box_np, s=1.5)
plt.show()
# sphinx_gallery_end_ignore
