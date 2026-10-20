r"""
3D milling school — the interaction rules of 200 swimmers
=========================================================

Two hundred fish swim around a landmark and organise into a **mill**:
a rotating torus in which everyone circles the same axis in the same
direction, a state many fish schools adopt.  A 3D tracking rig records
only their positions.  From these tracks alone, can we learn the rules
each individual follows — how it swims, what holds it near the centre,
how it reacts to its neighbours — with error bars, and check that these
rules, simulated, rebuild the mill?

Each swimmer is an inertial, self-propelled particle:

.. math::

    \dot{\mathbf{v}}_p = -k\,\mathbf{x}_p
        + \bigl(\alpha - \beta\,|\mathbf{v}_p|^2\bigr)\,\mathbf{v}_p
        + \sum_{q \ne p} \Bigl[\phi(r_{pq})\,\hat{\mathbf{r}}_{pq}
        + \psi(r_{pq})\,(\mathbf{v}_q - \mathbf{v}_p)\Bigr]
        + \sqrt{2D}\,\boldsymbol{\xi}_p .

The self-propulsion term makes each fish cruise at the preferred speed
:math:`v_0 = \sqrt{\alpha/\beta}`; :math:`-k\,\mathbf{x}` is a weak pull
towards the centre (a landmark); :math:`\phi(r)` is the positional
interaction (short-range repulsion, weak mid-range attraction; positive
values attract), :math:`\psi(r)` the velocity alignment with
neighbours, and :math:`D` the strength of the random part of the
swimming.  With 200 individuals every frame contains 19 900 pairs.

The recipe: build a generic dictionary of one-body terms and pair
kernels from SFI's built-in bases, fit it to the positions, let PASTIS
choose the terms, read the interaction kernels with confidence bands,
and validate by simulating the inferred school.

.. note::

   This is an **advanced** example (underdamped, 200 interacting
   individuals).  Start with the main gallery if you are new to SFI.

.. rubric:: Tags

synthetic · underdamped · multi-particle · linear · sparsity · large-scale · 3D
"""

# sphinx_gallery_tags = ["synthetic", "underdamped", "multi-particle", "linear", "sparsity", "large-scale", "3D"]
# sphinx_gallery_thumbnail_number = 6

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

from SFI.utils.plotting import plot_pareto_front, plot_recovery_bar

apply_style()
# Movies are encoded on the figure's own dark background; static figures
# keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"

# Print library warnings without the local source path of this script.
warnings.formatwarning = lambda msg, cat, *args, **kw: f"{cat.__name__}: {msg}\n"

MOVIE_BG = "#06080c"
# Cyclic, bright-on-dark colour map for headings (SFI palette, closed loop).
HEADING_CMAP = LinearSegmentedColormap.from_list(
    "heading", [SFI_COLORS[c] for c in ("data", "highlight", "inferred", "exact", "error", "tertiary", "data")]
)


def mill_frame(pos, lag=4):
    """Rotation matrix whose third row is the school's mean angular-momentum axis."""
    vel = pos[lag:] - pos[:-lag]
    rel = pos[:-lag] - pos[:-lag].mean(axis=1, keepdims=True)
    n = np.cross(rel, vel).sum(axis=(0, 1))
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n])


def heading_angle(Y, lag=4):
    """Heading angle in the plane orthogonal to the mill axis (centred differences)."""
    vel = np.empty_like(Y)
    vel[lag:-lag] = Y[2 * lag:] - Y[:-2 * lag]
    vel[:lag], vel[-lag:] = vel[lag], vel[-lag - 1]
    return np.arctan2(vel[..., 1], vel[..., 0])


def swarm_movie(arrays, titles, *, dt, rot, frames, step=4, trail=10, elev=35.0, spin=0.45,
                size=12.0, lw=0.9, zoom=1.4, panel=(3.3, 3.6)):
    """Rotating-camera 3D movie of one or several schools, side by side.

    Each panel is centred on its school's centre of mass and expressed in the
    mill frame ``rot`` (third axis = rotation axis).  Dots are coloured by
    heading around that axis (cyclic map), their size and opacity cue the
    depth, and fading trails show the last ``trail`` movie frames.
    """
    data = []
    for pos in arrays:
        Y = (pos - pos.mean(axis=1, keepdims=True)) @ rot.T
        data.append((Y, heading_angle(Y)))
    lim = 1.02 * max(np.quantile(np.hypot(Y[..., 0], Y[..., 1]), 0.999) for Y, _ in data)
    zlim = max(1.1 * max(np.quantile(np.abs(Y[..., 2]), 0.999) for Y, _ in data), 0.3 * lim)
    n = len(arrays)
    fig = plt.figure(figsize=(panel[0] * n, panel[1]), facecolor=MOVIE_BG, layout="none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=-0.05)
    axes, scats, lines = [], [], []
    for k, title in enumerate(titles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d", facecolor=MOVIE_BG)
        ax.set_axis_off()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-zlim, zlim)
        ax.set_box_aspect((1, 1, zlim / lim), zoom=zoom)
        ax.text2D(0.5, 0.96, title, transform=ax.transAxes, ha="center",
                  color="#D0D0D0", fontsize=10)
        lc = Line3DCollection([np.zeros((2, 3))], linewidths=lw)
        ax.add_collection3d(lc, autolim=False)
        sc = ax.scatter([], [], [], s=size, c=[], cmap=HEADING_CMAP, vmin=-np.pi, vmax=np.pi,
                        depthshade=True, edgecolors="none")
        axes.append(ax); scats.append(sc); lines.append(lc)
    clock = fig.text(0.5, 0.03, "", ha="center", color="#B0B0B0", fontsize=8)
    ramp = np.linspace(0.0, 0.7, trail)

    def update(i):
        f = frames[i]
        azim = -60.0 + spin * i
        el = elev + 6.0 * np.sin(2 * np.pi * i / len(frames))
        eye = np.array([np.cos(np.radians(el)) * np.cos(np.radians(azim)),
                        np.cos(np.radians(el)) * np.sin(np.radians(azim)),
                        np.sin(np.radians(el))])
        for ax, sc, lc, (Y, hue) in zip(axes, scats, lines, data):
            ax.view_init(elev=el, azim=azim)
            P = Y[f]
            sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
            sc.set_array(hue[f])
            sc.set_sizes(size * (1.0 + 0.6 * np.clip(P @ eye / lim, -1, 1)))
            hist = Y[f - step * trail:f + 1:step]                      # (trail+1, N, 3)
            seg = np.stack([hist[:-1], hist[1:]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 3)
            col = np.repeat(HEADING_CMAP((hue[f] + np.pi) / (2 * np.pi))[:, None, :], trail, axis=1)
            col[..., 3] = ramp
            lc.set_segments(seg)
            lc.set_color(col.reshape(-1, 4))
        clock.set_text(f"t = {f * dt:5.1f}")
        return scats + lines

    anim = FuncAnimation(fig, update, frames=len(frames), interval=70, blit=False)
    return fig, anim, update


def nearest_neighbour_distance(pos, every=25):
    Xs = pos[::every]
    dist = np.linalg.norm(Xs[:, :, None] - Xs[:, None], axis=-1)
    dist[:, np.arange(pos.shape[1]), np.arange(pos.shape[1])] = np.inf
    return dist.min(axis=-1).ravel()


def pair_distances(pos, every=50):
    Xs = pos[::every]
    iu = np.triu_indices(pos.shape[1], 1)
    return np.linalg.norm(Xs[:, :, None] - Xs[:, None], axis=-1)[:, iu[0], iu[1]].ravel()


# sphinx_gallery_end_ignore
stamp_output()

# %%
# The school, and what the tracker gives us
# -----------------------------------------
#
# The ground truth is written with the same building blocks we will use for
# inference: one-body terms :func:`~SFI.bases.X` and
# :func:`~SFI.bases.V`, and pair kernels from :mod:`SFI.bases.pairs`.
# The parameters: centre pull :math:`k = 0.06`, propulsion
# :math:`\alpha = \beta = 0.4` (cruising speed :math:`v_0 = 1`), repulsion
# :math:`\phi_{\rm rep}(r) = -2\,e^{-r^2/2\cdot 0.5^2}`, attraction
# :math:`\phi_{\rm att}(r) = 0.01\,e^{-r^2/2\cdot 2^2}` (weak per pair, but
# summed over many neighbours), alignment
# :math:`\psi(r) = 0.05\,e^{-r^2/2}` and noise :math:`D = 0.01`.  The
# school starts on a ring; the first 10 time units are discarded, and the
# mill is still tightening when the recording begins.

from SFI.bases import V, X
from SFI.bases.pairs import gaussian_kernels, pair_velocity_difference, radial_pair_basis, scalar_pair_basis
from SFI.langevin import UnderdampedProcess

d, N = 3, 200
v = V(dim=d)
swim = v & (v.dot(v) * v)                        # (α - β|v|²) v : two terms


def alignment(kernels):
    return (scalar_pair_basis(kernels, dim=d) * pair_velocity_difference(dim=d)).dispatch_pairs()


B_true = (X(dim=d) & swim
          & radial_pair_basis(gaussian_kernels([0.5, 2.0]), dim=d).dispatch_pairs()
          & alignment(gaussian_kernels([1.0])))
k, alpha, eps, c_att, a_align, D_true = 0.06, 0.4, 2.0, 0.01, 0.05, 0.01
school = UnderdampedProcess(B_true, D=D_true,
                            theta_F=jnp.array([-k, alpha, -alpha, -eps, c_att, a_align]))

k1, k2 = random.split(random.PRNGKey(7))
angle = 2 * np.pi * np.arange(N) / N
ring = np.stack([np.cos(angle), np.sin(angle), 0 * angle], axis=1)
tangent = np.stack([-np.sin(angle), np.cos(angle), 0 * angle], axis=1)
school.initialize(jnp.asarray(ring / np.sqrt(k)) + 0.3 * random.normal(k1, (N, d)), v0=jnp.asarray(tangent))

dt, T = 0.025, 2000                              # 40 frames per time unit
# 20 integrator substeps per frame keep the simulator's own error on D
# below the per-cent level; the first 400 frames (10 time units) are discarded.
recording = school.simulate(dt=dt, Nsteps=T, key=k2, oversampling=20, prerun=400)

# %%
# From here on, the model is forgotten.  All we keep is what a 3D tracker
# would deliver: a ``(frames, individuals, 3)`` array of positions and the
# frame interval — velocities and accelerations are never observed.

from SFI.trajectory import TrajectoryCollection

positions = recording.to_array()                 # (T, N, 3) — your tracking output
data = TrajectoryCollection.from_arrays(X=positions, dt=dt)

print(f"{N} individuals x {data.T} frames ({data.T * dt:.0f} time units)")

# sphinx_gallery_start_ignore
step = np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean()
print(f"mean displacement per frame {step:.3f}, "
      f"mean nearest-neighbour distance {nearest_neighbour_distance(positions).mean():.2f}")
# sphinx_gallery_end_ignore

# %%
# The recording in 3D.  The camera follows the school's centre of mass and
# slowly circles it; each fish is coloured by its heading around the mill
# axis, so a coherent mill shows as a colour wheel.

# sphinx_gallery_start_ignore
rot = mill_frame(positions)
fig_rec, anim_rec, _ = swarm_movie([positions], ["recording"], dt=dt, rot=rot,
                                   frames=np.arange(40, 40 + 4 * 200, 4), panel=(4.6, 4.0), size=14.0, lw=1.0)
plt.show()
# sphinx_gallery_end_ignore

# %%
# A generic dictionary of rules
# -----------------------------
#
# We do not assume the shapes of the interactions.  The dictionary offers
# three one-body terms (centre pull :math:`\mathbf{x}`, and the
# propulsion pair :math:`\mathbf{v}`, :math:`|\mathbf{v}|^2\mathbf{v}`) and
# pair kernels at three ranges — Gaussians of width 0.5, 1 and 2 —
# both for the positional interaction (kernel times the unit vector
# :math:`\hat{\mathbf{r}}_{pq}`) and for alignment (kernel times the
# velocity difference).  Nine candidates; the truth uses six of them.

kernels = gaussian_kernels([0.5, 1.0, 2.0])
B_centre = X(dim=d)
B_position = radial_pair_basis(kernels, dim=d).dispatch_pairs()
B_align = alignment(kernels)
B = B_centre & swim & B_position & B_align
theta_true = np.array([-k, alpha, -alpha, -eps, 0.0, c_att, 0.0, a_align, 0.0])
print(f"{B.n_features} candidate terms: {list(B.labels)}")

# %%
# Inference
# ---------
#
# The underdamped estimator reconstructs velocities and accelerations
# from the positions and projects the dynamics on the dictionary; the
# 19 900 pair sums per frame are evaluated by the basis itself.  Then
# PASTIS selects the terms the data support.
#
# ``preset="clean"`` declares the positions noise-free, which they are
# here.  This matters: second differences of positions amplify
# localisation noise :math:`\sigma` by :math:`1/\Delta t^2`, so the linear
# underdamped estimator needs :math:`\sigma \ll \sqrt{D\,\Delta t^3}` — a
# scale printed below, under a thousandth of the distance between
# neighbours (about 1) here.  For noisy tracks, the parametric
# :meth:`~SFI.inference.UnderdampedLangevinInference.infer_force` models
# the noise (see :doc:`/gallery/van_der_pol_demo`); its cost grows with the
# number of pairs and is suited to small groups.

import SFI

inf = SFI.UnderdampedLangevinInference(data)
inf.compute_diffusion_constant()
inf.infer_force_linear(B, preset="clean")
inf.compute_force_error()

coeffs_full = np.asarray(inf.force_coefficients_full)
stderr_full = np.asarray(inf.force_coefficients_stderr)
print("full dictionary (z = deviation from the truth in standard errors):")
for label, c, s, true in zip(B.labels, coeffs_full, stderr_full, theta_true):
    print(f"  {label:28s} {c:+.4f} ± {s:.4f}   true {true:+.3f}   z = {(c - true) / s:+.1f}")

# %%
# Every coefficient lands within about two standard errors of the truth,
# including the three that should vanish.  Now PASTIS chooses the model:

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

sel = np.asarray(inf.force_support)
z_sel = (np.asarray(inf.force_coefficients) - theta_true[sel]) / np.asarray(inf.force_coefficients_stderr)
print(f"true support recovered: {set(sel.tolist()) == set(np.flatnonzero(theta_true).tolist())}; "
      f"largest |z| among the selected terms: {np.abs(z_sel).max():.1f}")
D_hat = float(np.mean(np.diag(np.asarray(inf.diffusion_average))))
print(f"diffusion D = {D_hat:.4f} (true {D_true}); noise scale sqrt(D dt^3) = {np.sqrt(D_hat * dt**3):.1e}")

# %%
# Left: the Pareto front — information gained against model size; the
# selected model sits where adding a term stops paying.  Right: the full
# nine-term fit with 1-σ error bars against the truth, on a symmetric-log
# scale so that the weak attraction and alignment terms show next to the
# repulsion (:math:`\phi_\ell`, :math:`\psi_\ell`: positional and alignment
# kernels of width :math:`\ell`).  The short-range alignment
# :math:`\psi_{0.5}` carries the largest error bar — close pairs are
# rare, so the data say little about alignment at contact — and trades
# weight with its overlapping neighbour :math:`\psi_1`; PASTIS keeps only
# the latter.

# sphinx_gallery_start_ignore
short_labels = [r"$\mathbf{x}$", r"$\mathbf{v}$", r"$|\mathbf{v}|^2\mathbf{v}$",
                r"$\phi_{0.5}$", r"$\phi_{1}$", r"$\phi_{2}$", r"$\psi_{0.5}$", r"$\psi_{1}$", r"$\psi_{2}$"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
plot_recovery_bar(
    coeffs_full, np.arange(B.n_features),
    coeffs_true=theta_true, support_true=np.arange(B.n_features),
    labels=short_labels, stderr=stderr_full, ax=axes[1],
)
axes[1].set_yscale("symlog", linthresh=0.02)
axes[1].tick_params(axis="x", labelsize=10, rotation=0)
axes[1].set_title(f"Full dictionary — PASTIS keeps {len(sel)} / {B.n_features}")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The recovered rules
# -------------------
#
# Summing the selected terms gives each rule as a function, with a 95 %
# band from the coefficient covariance
# (:meth:`~SFI.inference.UnderdampedLangevinInference.coeff_block` and
# :func:`~SFI.inference.kernel_predict_ci`).  The shaded histograms show
# where the data live — speeds for the propulsion, pair distances for the
# kernels: a rule is only measured where it is sampled.

from SFI.inference import kernel_predict_ci

speeds = np.linspace(0.0, 1.6, 200)
c_swim, cov_swim = inf.coeff_block(swim)
J = np.stack([speeds, speeds**3], axis=1)        # propulsion along v: c1 s + c2 s^3
swim_mean = J @ c_swim
swim_half = 1.96 * np.sqrt(np.einsum("ri,ij,rj->r", J, cov_swim, J))

r_eval = np.linspace(0.05, 8.0, 300)
ci_pos = kernel_predict_ci(r_eval, kernels, *inf.coeff_block(B_position))
ci_ali = kernel_predict_ci(r_eval, kernels, *inf.coeff_block(B_align))
true_pos = theta_true[3:6] @ ci_pos["phi"]
true_ali = theta_true[6:9] @ ci_ali["phi"]
v0_hat = np.sqrt(-c_swim[0] / c_swim[1])
print(f"cruising speed sqrt(alpha/beta) = {v0_hat:.3f} (true 1)")

# sphinx_gallery_start_ignore
vel = data.velocity_array(scheme="central")
spd = np.linalg.norm(vel, axis=-1).ravel()
r_pairs = pair_distances(positions)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))


def _band(ax, x, mean, lo, hi, truth, hist, bins, title, xlabel):
    axh = ax.twinx()
    axh.hist(hist, bins=bins, color=SFI_COLORS["data"], alpha=0.18)
    axh.set_yticks([])
    axh.spines["right"].set_visible(False)
    ax.set_zorder(axh.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.fill_between(x, lo, hi, color=SFI_COLORS["inferred"], alpha=0.3, label="95% band")
    ax.plot(x, truth, "--", color=SFI_COLORS["exact"], lw=2, label="true")
    ax.plot(x, mean, color=SFI_COLORS["inferred"], lw=1.6, label="inferred")
    ax.axhline(0, color="#808080", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


_band(axes[0], speeds, swim_mean, swim_mean - swim_half, swim_mean + swim_half,
      alpha * speeds - alpha * speeds**3, spd, np.linspace(0, 1.6, 60),
      r"Propulsion $(\alpha-\beta s^2)\,s$", "speed s")
axes[0].legend(loc="lower left")
_band(axes[1], r_eval, ci_pos["mean"], ci_pos["lower"], ci_pos["upper"], true_pos,
      r_pairs, np.linspace(0, 8, 80), r"Positional $\phi(r)$  (>0: attraction)", "pair distance r")
axes[1].set_ylim(-2.2, 0.3)
ins = axes[1].inset_axes([0.42, 0.18, 0.54, 0.45])
m = r_eval > 1.7
ins.fill_between(r_eval[m], ci_pos["lower"][m], ci_pos["upper"][m], color=SFI_COLORS["inferred"], alpha=0.3)
ins.plot(r_eval[m], true_pos[m], "--", color=SFI_COLORS["exact"], lw=1.5)
ins.plot(r_eval[m], ci_pos["mean"][m], color=SFI_COLORS["inferred"], lw=1.2)
ins.axhline(0, color="#808080", lw=0.5)
ins.set_title("zoom: weak attraction", fontsize=8)
ins.tick_params(labelsize=7)
ins.patch.set_alpha(0.0)
_band(axes[2], r_eval, ci_ali["mean"], ci_ali["lower"], ci_ali["upper"], true_ali,
      r_pairs, np.linspace(0, 8, 80), r"Alignment $\psi(r)$", "pair distance r")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Do the inferred rules rebuild the mill?
# ---------------------------------------
#
# The fit never looked at collective properties.  We simulate the inferred
# model (:meth:`~SFI.inference.UnderdampedLangevinInference.simulate_bootstrapped_trajectory`),
# starting from the first recorded frame, for as long as the recording.
# ``oversampling=10`` substeps per frame keep the integrator's error far
# below the collective fluctuations compared here.

boot, _ = inf.simulate_bootstrapped_trajectory(key=random.PRNGKey(11), oversampling=10)
positions_boot = boot.to_array()

# %%
# Side by side, from the same starting snapshot: the recording and the
# school simulated from the inferred rules (same camera path).

# sphinx_gallery_start_ignore
fig_cmp, anim_cmp, _ = swarm_movie([positions, positions_boot], ["recording", "inferred model"],
                                   dt=dt, rot=rot, frames=np.arange(40, 40 + 4 * 220, 4))
plt.show()
# sphinx_gallery_end_ignore

# %%
# Quantitatively, two order parameters: the polarisation
# :math:`P = |\langle\hat{\mathbf{v}}\rangle|` (a school swimming in one
# direction has :math:`P \approx 1`) and the milling order
# :math:`M = |\langle\hat{\mathbf{r}}\times\hat{\mathbf{v}}\rangle|`, with
# :math:`\hat{\mathbf{r}}` the direction from the centre of mass (a perfect
# mill has :math:`M = 1`).  Velocities come from the positions by finite
# differences.


def order_parameters(coll):
    x, u = coll.to_array(), coll.velocity_array(scheme="central")
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    r = x - x.mean(axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return np.linalg.norm(u.mean(axis=1), axis=-1), np.linalg.norm(np.cross(r, u).mean(axis=1), axis=-1)


for name, coll in [("recording", data), ("inferred model", boot)]:
    P, M = order_parameters(coll)
    print(f"{name:15s}: milling M = {M.mean():.2f} ± {M.std():.2f},  polarisation P = {P.mean():.2f} ± {P.std():.2f}")

# %%
# Below: the order parameters in time, and the distributions of speed,
# nearest-neighbour distance and distance from the centre — the
# thickness and the hole of the torus.

# sphinx_gallery_start_ignore
stats = {}
for name, coll in [("recording", data), ("inferred model", boot)]:
    Xa, Va = coll.to_array(), coll.velocity_array(scheme="central")
    P, M = order_parameters(coll)
    stats[name] = dict(P=P, M=M, speed=np.linalg.norm(Va, axis=-1).ravel(),
                       nn=nearest_neighbour_distance(Xa),
                       radius=np.linalg.norm(Xa - Xa.mean(axis=1, keepdims=True), axis=-1).ravel())
    print(f"{name:15s}: mean speed {stats[name]['speed'].mean():.3f}, nearest neighbour "
          f"{stats[name]['nn'].mean():.2f}, distance from centre {stats[name]['radius'].mean():.2f}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
colors = {"recording": SFI_COLORS["data"], "inferred model": SFI_COLORS["tertiary"]}
tt = np.arange(T) * dt
for name, s in stats.items():
    axes[0].plot(tt, s["M"], color=colors[name], lw=1.2, label=f"M, {name}")
    axes[0].plot(tt, s["P"], color=colors[name], lw=1.0, ls=":", label=f"P, {name}")
    for ax, key, bins in [(axes[1], "speed", np.linspace(0.4, 1.4, 50)),
                          (axes[2], "nn", np.linspace(0.3, 1.8, 50)),
                          (axes[3], "radius", np.linspace(0, 8, 50))]:
        ax.hist(s[key], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=colors[name])
        ax.hist(s[key], bins=bins, density=True, histtype="step", lw=1.2, color=colors[name], label=name)
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel("time")
axes[0].set_title("Milling M (solid), polarisation P (dotted)")
axes[0].legend(fontsize=7, ncol=2, loc="center right")
for ax, lab in zip(axes[1:], ["speed", "nearest-neighbour distance", "distance from centre"]):
    ax.set_xlabel(lab)
    ax.set_yticks([])
axes[1].legend(fontsize=7)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The school simulated from the inferred rules mills as strongly as the
# recorded one — including the tightening of the mill over the first ten
# time units — and neither translates (low polarisation in both).  Its
# speeds, spacing and torus profile match the recording: rules learned
# from individual accelerations reproduce the collective state they were
# never fitted to.

# %%
# Thumbnail
# ---------
# ---------

# sphinx_gallery_start_ignore
Y = (positions - positions.mean(axis=1, keepdims=True)) @ rot.T
hue = heading_angle(Y)
fig_t = plt.figure(figsize=(4, 4))
ax_t = fig_t.add_subplot(111, projection="3d")
f0 = T // 2
seg = np.stack([Y[f0 - 60:f0:4], Y[f0 - 56:f0 + 4:4]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 3)
cols = np.repeat(HEADING_CMAP((hue[f0] + np.pi) / (2 * np.pi))[:, None, :], 15, axis=1)
cols[..., 3] = np.linspace(0.05, 0.8, 15)
ax_t.add_collection3d(Line3DCollection(seg, colors=cols.reshape(-1, 4), linewidths=0.9), autolim=False)
ax_t.scatter(*Y[f0].T, c=hue[f0], cmap=HEADING_CMAP, vmin=-np.pi, vmax=np.pi, s=10, depthshade=True)
lim = np.quantile(np.abs(Y), 0.999)
ax_t.set_xlim(-lim, lim); ax_t.set_ylim(-lim, lim); ax_t.set_zlim(-lim, lim)
ax_t.set_box_aspect((1, 1, 1), zoom=1.3)
ax_t.view_init(elev=30, azim=-60)
ax_t.set_axis_off()
plt.show()
# sphinx_gallery_end_ignore
