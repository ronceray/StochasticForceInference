r"""
Discovering Toner–Tu hydrodynamics from agent-based flocking
==============================================================

.. note::
   Uses the **experimental** SPDE toolbox — see :doc:`/spde/index`.

**From particles to equations.**  We simulate 10 000 self-propelled
particles that align with their neighbours, coarse-grain their positions
and headings into two continuum fields — the density and the
polarisation density — and let SFI infer the stochastic
equations that drive these fields.  We then read the result as a physicist
would: which terms matter, what they mean, how they compare with the
microscopic model and with the Toner–Tu theory of flocking — and whether
the inferred equations, run on their own, make flocks like the particles
do.

The particles sit near the **flocking onset**: dense polar flocks form,
roam, collide and dissolve in a dilute gas.  This is where hydrodynamics
is interesting — a featureless, fully ordered flock would teach the
inference little.

.. rubric:: Tags

synthetic · overdamped · multi-particle · linear · SPDE · sparsity
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "multi-particle", "linear", "SPDE", "sparsity"]
# sphinx_gallery_thumbnail_number = 7

# sphinx_gallery_start_ignore
from __future__ import annotations

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.coarse_grain import coarse_grain_polar
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output
from SFI.utils.plotting import animate_particles, plot_particles

apply_style()
# Movies are encoded on their own dark background and at their own figure
# resolution; static figures keep the transparent 150-dpi gallery style.
plt.rcParams["savefig.facecolor"] = "auto"
plt.rcParams["savefig.dpi"] = "figure"
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Microscopic flocking model
# --------------------------
#
# Each agent carries a position :math:`(x_i, y_i)` and a heading
# :math:`\theta_i`.  The force on the state :math:`(x, y, \theta)` is
#
# .. math::
#    \mathbf F_i =
#    \underbrace{c_0\, \hat e_{\theta_i}}_{\text{propulsion}}
#    \;+\; \underbrace{\sum_j (-\varepsilon)\,e^{-r_{ij}/R_0}\,
#          \hat r_{ij}}_{\text{soft repulsion}}
#    \;+\; \underbrace{\hat z\sum_j A\,e^{-r_{ij}/\ell}\,
#          \sin(\theta_j - \theta_i)}_{\text{alignment}}
#
# plus rotational noise :math:`D_\theta` and weak translational noise.
# Both pair kernels are cut off at :math:`r_c = 3`, so that the model —
# and the mean-field rates below — do not depend on the neighbour list.
# In mean field,
# alignment builds polarisation at a rate :math:`a_1\rho` with
# :math:`a_1 = \tfrac12\int K\,d^2r = \pi A\ell^2[1-(1+r_c/\ell)e^{-r_c/\ell}]`
# (:math:`K` the alignment kernel), while rotational diffusion destroys
# it at rate :math:`D_\theta`: flocking sets in when
# :math:`a_1\bar\rho > D_\theta`.  We sit about a third above this
# mean-field threshold — close to the actual onset, which fluctuations
# push higher.

from SFI.bases.pairs import (
    angle_coupling,
    heading_vector,
    pair_direction,
    parametric_radial_kernel,
)
from SFI.langevin import OverdampedProcess
from SFI.langevin.chunked import simulate_chunked
from SFI.utils.neighbors import make_neighbor_extras

N_particles = 10_000
L = 143.0                  # periodic box side → mean density ≈ 0.49
dt = 0.1                   # integration step
seed = 7

c0_true = 1.0              # self-propulsion speed
eps_true, R0_true = 2.0, 0.5
A_true, La_true = 0.55, 1.0
r_cut = 3.0                # range of both pair interactions
D_xy, D_theta = 0.005, 0.5
D_matrix = jnp.diag(jnp.array([D_xy, D_xy, D_theta]))

dim = 3
B_heading = heading_vector(dim=dim, angle_index=2)
e_ij = pair_direction(dim=dim, box="extras", spatial_dims=slice(0, 2),
                      embed_dim=dim, embed_axes=[0, 1])
g_align = angle_coupling(jnp.sin, dim=dim, angle_index=2)
k_repel = parametric_radial_kernel(
    lambda r, p: -p["eps"] * jnp.exp(-r / p["R0"]) * (r < r_cut),
    params={"eps": (), "R0": ()}, dim=dim, box="extras", spatial_dims=slice(0, 2),
)
k_align = parametric_radial_kernel(
    lambda r, p: p["A"] * jnp.exp(-r / p["La"]) * (r < r_cut),
    params={"A": (), "La": ()}, dim=dim, box="extras", spatial_dims=slice(0, 2),
)
csr_kw = dict(indptr_key="indptr", indices_key="indices")
F_sim = (
    B_heading.to_psf(coeff_key="c0")
    + (k_repel * e_ij).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
    + (k_align * g_align).dispatch_pairs_from_extras(**csr_kw, return_as="psf")
)

rho_bar = N_particles / L**2
a1_mf = np.pi * A_true * La_true**2 * (1 - (1 + r_cut / La_true) * np.exp(-r_cut / La_true))
print(f"N = {N_particles}, box = {L:.0f}×{L:.0f}, mean density = {rho_bar:.2f}")
print(f"mean-field alignment rate a1 = {a1_mf:.2f};  onset ratio a1·(mean density)/D_θ = {a1_mf * rho_bar / D_theta:.2f}")

# %%
# Simulation with neighbour lists
# -------------------------------
#
# :func:`~SFI.langevin.chunked.simulate_chunked` integrates the
# interacting system with a periodically rebuilt neighbour list of radius
# :math:`r_c` + skin; the skin (4) covers how far particles move between
# rebuilds (about one unit).  We integrate with step 0.1 and record every
# second step (``oversampling=2``), so frames are 0.2 time units apart.
# The trajectory is cached on disk so that re-running the page is fast.

box_np = np.array([L, L])
dt_frame = 0.2                       # recording interval
t_burn, t_run = 150.0, 250.0         # burn-in, then the analysed run
n_frames = int(round((t_burn + t_run) / dt_frame))
skin, rebuild_every = 4.0, 5         # rebuild the list every 5 frames (1 time unit)

key = random.PRNGKey(seed)
key, kx, kth = random.split(key, 3)
x0 = jnp.concatenate([
    random.uniform(kx, (N_particles, 2)) * L,
    random.uniform(kth, (N_particles, 1), minval=-jnp.pi, maxval=jnp.pi),
], axis=1)

extras0 = {"box": jnp.asarray(box_np)}
extras0.update(make_neighbor_extras(np.asarray(x0[:, :2]), r_cut + skin, box_np))
proc = OverdampedProcess(F_sim, D=D_matrix, extras_global=extras0)
proc.set_params(theta_F=dict(c0=c0_true, eps=eps_true, R0=R0_true, A=A_true, La=La_true))
proc.initialize(x0)

cache_dir = Path(__file__).resolve().parent / "_cache" if "__file__" in dir() else Path("_cache")
cache = cache_dir / (f"abp_to_spde_N{N_particles}_L{L:g}_A{A_true}_Dth{D_theta}_rc{r_cut:g}"
                     f"_T{t_burn + t_run:g}_seed{seed}.npy")
if cache.exists():
    X_micro = np.load(cache)
else:
    key, sub = random.split(key)
    coll_sim = simulate_chunked(
        proc, dt=dt_frame, Nsteps=n_frames, key=sub, cutoff=r_cut, box=box_np,
        skin=skin, rebuild_every=rebuild_every, save_every=250,
        oversampling=int(round(dt_frame / dt)), spatial_dims=slice(0, 2),
    )
    X_micro = np.asarray(coll_sim.to_array(axis="time"), dtype=np.float32)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, X_micro)
t_micro = np.arange(X_micro.shape[0]) * dt_frame
print(f"Trajectory: {X_micro.shape[0]} frames × {N_particles} particles, Δt = {dt_frame}")

# %%
# A fluctuating steady state
# --------------------------
#
# Two order parameters over the whole run: the global polar order
# :math:`\varphi = |\langle \hat e_\theta\rangle|` and the density
# contrast (relative standard deviation of the particle count in cells of
# side ≈ 3, the cells used below).  After the burn-in (shaded) both
# fluctuate around steady values: the box holds several flocks heading in
# different directions, so the *global* order stays modest while the
# density stays strongly heterogeneous.  We analyse everything after the
# burn-in — no hand-picked stretch.

NCELL = 48
GRID = (NCELL, NCELL)
DX = L / NCELL
i0 = int(round(t_burn / dt_frame))
X_use = X_micro[i0:]

phi_t = np.hypot(np.cos(X_micro[..., 2]).mean(1), np.sin(X_micro[..., 2]).mean(1))
counts = np.stack([np.histogram2d(X[:, 0] % L, X[:, 1] % L, bins=NCELL,
                                  range=[[0, L], [0, L]])[0].ravel() for X in X_micro])
contrast_t = counts.std(1) / counts.mean(1)

half = len(X_use) // 2
for name, s in [("polar order φ", phi_t[i0:]), ("density contrast", contrast_t[i0:])]:
    print(f"{name:17s}: first half {s[:half].mean():.3f}, second half {s[half:].mean():.3f}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_micro, phi_t, color=SFI_COLORS["data"], lw=1, label=r"polar order $\varphi$")
ax.plot(t_micro, contrast_t, color=SFI_COLORS["highlight"], lw=1, label="density contrast")
ax.axvspan(0, t_burn, color="#808080", alpha=0.2, lw=0)
ax.set_xlabel("time")
ax.set_ylim(0, 1.05)
ax.legend(loc="center right")
ax.set_title("Order parameters over the run (burn-in shaded)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The flock in motion
# -------------------
#
# One frame per time unit over the analysed run; colour encodes the
# heading angle.  Watch flocks of common heading form, sweep across the
# periodic box, merge when they meet and evaporate back into the gas.

from SFI.trajectory import TrajectoryCollection

coll_micro = TrajectoryCollection.from_arrays(X=X_use, dt=dt_frame)
movie_skip = int(round(1.0 / dt_frame))       # one movie frame per time unit

# sphinx_gallery_start_ignore
plt.rcParams["animation.bitrate"] = 2500   # 10 000 moving dots barely compress
fig_anim, ax_anim = plt.subplots(figsize=(5.5, 5.5), dpi=100)
fig_anim.patch.set_facecolor("#131416")
ax_anim.set_facecolor("#131416")


def _clock(ax, ti, X_t):
    ax.set_title(f"t = {t_burn + ti * dt_frame:.0f}", color="#D0D0D0")


anim = animate_particles(coll_micro, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                         box=box_np, skip=movie_skip, s=1.5, ax=ax_anim, interval=90,
                         overlay_fn=_clock)
ax_anim.set_xlim(0, L); ax_anim.set_ylim(0, L)
ax_anim.set_xticks([]); ax_anim.set_yticks([])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Coarse-graining: particles → continuum fields
# ---------------------------------------------
#
# Each particle deposits :math:`(1, \cos\theta, \sin\theta)` bilinearly
# onto a periodic 48×48 grid (cells of side ≈ 3), giving the density
# :math:`\rho` and the polarisation density :math:`\mathbf m = \rho\langle
# \hat e_\theta\rangle`.  We deliberately do **not** smooth the fields
# further: SFI treats the noise of different cells as independent, and a
# smoothing kernel would correlate it across cells and make the data look
# more informative than it is.
#
# Below, each particle snapshot (top) is paired with its fields (bottom),
# drawn with the same colour wheel: hue is the direction of
# :math:`\mathbf m`, brightness the density, and saturation the local
# polar order :math:`|\mathbf m|/\rho` (grey where headings are random).

fields = np.asarray(coarse_grain_polar(jnp.asarray(X_use), box=jnp.asarray(box_np),
                                       grid_shape=GRID, sigma_cells=0.0, angle_index=2))

from SFI.bases.spde import square_grid_extras

coll_fields = TrajectoryCollection.from_arrays(
    X=fields, dt=dt_frame, extras_global=square_grid_extras(grid_shape=GRID, dx=DX),
)
print(f"Field trajectory: {coll_fields.T} frames × {coll_fields.N} cells × 3 channels;  "
      f"∫ρ dA = {fields[0, :, 0].sum() * DX**2:.0f}")

# sphinx_gallery_start_ignore
plt.rcParams["animation.bitrate"] = -1     # back to the encoder's default for smooth fields
rho_max = float(np.percentile(fields[..., 0], 99.5))


def flock_rgb(frame):
    """Colour-wheel rendering of one (ρ, m) frame: hue = arg m, value = ρ, saturation = |m|/ρ."""
    f = np.asarray(frame).reshape(*GRID, 3)
    rho_, mx, my = f[..., 0], f[..., 1], f[..., 2]
    hue = (np.arctan2(my, mx) + np.pi) / (2 * np.pi)
    sat = np.clip(np.hypot(mx, my) / np.maximum(rho_, 1e-9), 0.0, 1.0)
    val = np.clip(rho_ / rho_max, 0.0, 1.0)
    return hsv_to_rgb(np.stack([hue, sat, val], -1)).transpose(1, 0, 2)


snap_idx = np.linspace(0, len(X_use) - 1, 4).astype(int)
fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
for k, ti in enumerate(snap_idx):
    ax = axes[0, k]
    ax.set_facecolor("#131416")
    plot_particles(coll_micro, t_index=ti, color_dim=2, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                   box=box_np, s=1.5, edgecolors="none", ax=ax)
    ax.set_title(f"t = {t_burn + ti * dt_frame:.0f}")
    axes[1, k].imshow(flock_rgb(fields[ti]), origin="lower", extent=[0, L, 0, L],
                      interpolation="bilinear")
    for a in axes[:, k]:
        a.set_xlim(0, L); a.set_ylim(0, L); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("particles (heading)")
axes[1, 0].set_ylabel(r"fields $(\rho, \mathbf{m})$")
fig.suptitle("From particles to fields")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Candidate hydrodynamic terms
# ----------------------------
#
# The library contains the terms allowed by symmetry at low order —
# those of the Toner–Tu equations — built with
# :class:`~SFI.statefunc.layout.GridLayout`:
#
# - **ρ-equation**: only *divergences*, because particle number is
#   exactly conserved — :math:`\nabla\cdot\mathbf m`,
#   :math:`\nabla\cdot(\rho\mathbf m)`, :math:`\nabla^2\rho`,
#   :math:`\nabla^2\rho^2`, :math:`\nabla^2|\mathbf m|^2`;
# - **m-equation**: Landau ordering :math:`(a_0 + a_1\rho - a_2|\mathbf m|^2)\,\mathbf m`,
#   elasticity :math:`\nabla^2\mathbf m`, :math:`\nabla(\nabla\cdot\mathbf m)`,
#   pressures :math:`\nabla\rho`, :math:`\nabla\rho^2`, :math:`\nabla|\mathbf m|^2`,
#   and the advective terms :math:`(\mathbf m\cdot\nabla)\mathbf m`,
#   :math:`\mathbf m(\nabla\cdot\mathbf m)`.

from SFI.statefunc.layout import GridLayout, ScalarSector, VectorSector
from SFI.statefunc.structexpr import StructuredExpr

layout = GridLayout(
    rho=ScalarSector([0]),
    m=VectorSector([1, 2], sdim=2, spatial=True),
    dim=3, ndim=2, bc="pbc",
)
rho, m = layout.rho, layout.m
m_sq = StructuredExpr.einsum("i,i->", m, m).with_label("|m|²")

rho_terms = [
    layout.div(m).with_label("∇·m"),
    layout.lap(rho).with_label("∇²ρ"),
    layout.div(rho * m).with_label("∇·(ρm)"),
    layout.lap(rho * rho).with_label("∇²ρ²"),
    layout.lap(m_sq).with_label("∇²|m|²"),
]
m_terms = [
    m.with_label("m"),
    (rho * m).with_label("ρm"),
    (m_sq * m).with_label("|m|²m"),
    layout.lap(m),
    layout.grad(layout.div(m)).with_label("∇(∇·m)"),
    layout.grad(rho).with_label("∇ρ"),
    layout.grad(rho * rho).with_label("∇ρ²"),
    layout.grad(m_sq).with_label("∇|m|²"),
    layout.advection_by(m, m),
    (m * layout.div(m)).with_label("m(∇·m)"),
]
rho_basis, m_basis = rho_terms[0], m_terms[0]
for t in rho_terms[1:]:
    rho_basis = rho_basis & t
for t in m_terms[1:]:
    m_basis = m_basis & t
BASIS = layout.embed(rank=1, rho=rho_basis, m=m_basis)
labels = list(BASIS.labels)
print(f"{BASIS.n_features} candidate terms ({len(rho_terms)} for ρ, {len(m_terms)} for m)")

# %%
# Inference: every allowed term is real
# -------------------------------------
#
# We fit all fifteen terms at once.  The fields are exact functions of the
# particle positions — there is no measurement noise — so we use
# ``preset="clean"``; ``WeakNoise`` estimates the noise amplitude, since
# over one frame the deterministic change of the fields is not small
# compared with their random change.

from SFI import OverdampedLangevinInference


def fit_drift(basis):
    """Linear SFI fit of the field drift on a candidate library."""
    inf_ = OverdampedLangevinInference(coll_fields)
    inf_.compute_diffusion_constant(method="WeakNoise")
    inf_.infer_force_linear(basis, preset="clean")
    inf_.compute_force_error()
    return inf_


inf = fit_drift(BASIS)
c_all = np.asarray(inf.force_coefficients)
s_all = np.asarray(inf.force_coefficients_stderr)
result = inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.print_report()

# %%
# PASTIS keeps (almost) all the terms, and the reason is instructive.
# Removing term :math:`j` from the full model costs
# :math:`z_j^2/2` nats of log-likelihood (:math:`z_j` = coefficient /
# standard error), while PASTIS charges :math:`\ln(n_0/p)` per term —
# under 6 nats here.  With millions of cell increments the information
# budget runs into millions of nats, so even terms that barely change the
# dynamics are statistically resolved.  A coarse-grained equation is
# **not sparse** within its symmetry class: coarse-graining generates
# every term the symmetries allow, and with enough data all of them are
# detected.  Sparsity selection answers *"which terms are real?"* — here,
# essentially all.  The physicist's question is a different one:
# *"which terms matter?"*

z2_half = 0.5 * (c_all / s_all) ** 2
penalty = np.log(len(labels) / result.selection["p_param"])
print(f"{(coll_fields.T - 1) * coll_fields.N:.3g} cell increments; total information "
      f"{result.total_info:.3g} nats; PASTIS penalty per term ln(n0/p) = {penalty:.1f} nats; "
      f"PASTIS keeps {len(inf.force_support)}/{len(labels)} terms")
print("information carried by each term (z²/2, nats):")
print("  " + ",  ".join(f"{l} {v:.3g}" for l, v in zip(labels, z2_half)))

# %%
# Which terms matter: 98% of the information, equation by equation
# ----------------------------------------------------------------
#
# The information of a model — its log-likelihood gain — measures how
# much of the observed field dynamics it predicts, in units that do not
# depend on how the fields or coefficients are normalised.  The natural
# summary of an equation is therefore the smallest model on its Pareto
# front (best information at each model size) that captures 98% of the
# information.  We apply this criterion to each equation separately:
# the density is conserved and nearly deterministic at the scale of a
# cell, so the ρ-equation carries the lion's share of the total
# information, and a single cut on the total would delete the
# polarisation dynamics wholesale.  Each equation's front is computed by
# fitting it alone; the selected terms are then refitted together.  (This
# is why a highly significant density term such as
# :math:`\nabla^2|\mathbf m|^2` can fall outside the 98% model while
# weaker polarisation terms stay in: it adds little to an equation
# dominated by continuity.)

fronts = {}
selected = []
for name, sector_basis in [("ρ", layout.embed(rank=1, rho=rho_basis)),
                           ("m", layout.embed(rank=1, m=m_basis))]:
    inf_eq = fit_drift(sector_basis)
    front = inf_eq.sparsify_force(criterion="PASTIS")
    frac = np.asarray(front.best_info_by_k) / front.total_info
    k98 = int(np.argmax(frac >= 0.98))
    terms = [list(sector_basis.labels)[j] for j in front.best_support_by_k[k98]]
    fronts[name] = dict(frac=frac, k98=k98, terms=terms, info=front.total_info)
    selected += terms
    print(f"{name}-equation: {front.total_info:.3g} nats; 98% model = {k98}/{sector_basis.n_features} "
          f"terms ({frac[k98]:.1%}): {', '.join(terms)}")
share = fronts["ρ"]["info"] / (fronts["ρ"]["info"] + fronts["m"]["info"])
print(f"share of the information in the ρ-equation: {share:.1%}")

BASIS_98 = BASIS[sorted(labels.index(t) for t in selected)]
inf98 = fit_drift(BASIS_98)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
for ax, (name, fr), col in zip(axes[:2], fronts.items(), [SFI_COLORS["data"], SFI_COLORS["inferred"]]):
    ks = np.arange(1, len(fr["frac"]))
    ax.plot(ks, fr["frac"][1:], "o-", color=col, ms=5, lw=1.5)
    ax.axhline(0.98, color="#B0B0B0", ls="--", lw=1)
    ax.text(ks[-1], 0.975, "98%", color="#B0B0B0", ha="right", va="top", fontsize=9)
    ax.plot(fr["k98"], fr["frac"][fr["k98"]], "o", ms=13, mfc="none", mec=SFI_COLORS["error"], mew=2)
    ax.text(0.97, 0.05, "98% model:\n" + "\n".join(fr["terms"]), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#D0D0D0")
    ax.set_xticks(ks)
    ax.set_xlabel("number of terms")
    ax.set_ylabel("fraction of the information")
    ax.set_title(f"{name}-equation ({fr['info']:.2g} nats)")
ax = axes[2]
order = np.argsort(z2_half)
in98 = np.array([l in selected for l in labels])
is_rho = np.arange(len(labels)) < len(rho_terms)
colors = [SFI_COLORS["data"] if is_rho[j] else SFI_COLORS["inferred"] for j in order]
ax.barh(np.arange(len(labels)), z2_half[order], color=colors,
        alpha=1.0, edgecolor="none")
for y, j in enumerate(order):
    if not in98[j]:
        ax.barh(y, z2_half[j], color="#131416", alpha=0.55, edgecolor="none")
ax.axvline(penalty, color=SFI_COLORS["error"], ls="--", lw=1.5, label=r"PASTIS penalty $\ln(n_0/p)$")
ax.legend(loc="lower right", fontsize=8)
ax.set_xscale("log")
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels([labels[j] for j in order], fontsize=8)
ax.set_xlabel(r"information of each term, $z^2/2$ (nats)")
ax.set_title("Information of each term (faded: not in the 98% model)", fontsize=11)
fig.suptitle("Model selection: statistical significance vs physical importance")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The 98% hydrodynamic model
# --------------------------
#
# Written out with its fitted coefficients (±1σ sampling error):


def print_equation(model, field, groups):
    names = list(model.force_basis.labels)
    c, s = np.asarray(model.force_coefficients), np.asarray(model.force_coefficients_stderr)
    lines = []
    for group in groups:
        terms = [f"({c[names.index(t)]:+.3f} ± {s[names.index(t)]:.3f}) {t}"
                 for t in group if t in names]
        if terms:
            lines.append("  ".join(terms))
    print(f"∂t{field} = " + ("\n" + " " * (4 + len(field))).join(lines))


rho_groups = [["∇·m", "∇·(ρm)"], ["∇²ρ", "∇²ρ²", "∇²|m|²"]]
m_groups = [["m", "ρm", "|m|²m"], ["∇ρ", "∇ρ²", "∇|m|²"], ["∇²m", "∇(∇·m)"],
            ["(m·∇)m", "m(∇·m)"]]
print_equation(inf98, "ρ", rho_groups)
print_equation(inf98, "m", m_groups)

# %%
# Term by term (each line of the printed equations is one physical
# mechanism):
#
# .. math::
#    \partial_t\rho &= -v\,\nabla\cdot\mathbf m + \kappa\,\nabla^2\rho^2,\\
#    \partial_t\mathbf m &= (a_0 + a_1\rho - a_2|\mathbf m|^2)\,\mathbf m
#      - \zeta\,\nabla\rho + \lambda_3\nabla|\mathbf m|^2 + K\,\nabla^2\mathbf m\\
#      &\quad + \lambda_1(\mathbf m\cdot\nabla)\mathbf m + \lambda_2\,\mathbf m(\nabla\cdot\mathbf m).
#
# - :math:`-v\nabla\cdot\mathbf m` — **continuity**: particles carry
#   density at their swim speed, the flux being :math:`c_0\mathbf m`.
#   This single term holds most of the density information.
# - :math:`\kappa\nabla^2\rho^2` — **collisional spreading**: soft
#   repulsion pushes particles out of crowded regions, a diffusion whose
#   coefficient :math:`2\kappa\rho` grows with density.
# - :math:`(a_0 + a_1\rho - a_2|\mathbf m|^2)\mathbf m` — **Landau
#   ordering**, the heart of flocking: polarisation decays in the dilute
#   gas (:math:`a_0<0`), grows where alignment wins (:math:`a_1\rho >
#   |a_0|`) and saturates (:math:`a_2>0`).  The density where the bracket
#   changes sign is the local flocking threshold.
# - :math:`-\zeta\nabla\rho` — **pressure**: more particles swim out of a
#   dense region than in, so polarisation points down density gradients.
# - :math:`K\nabla^2\mathbf m` — **elasticity**: polarisation spreads to
#   neighbouring cells, smoothing splay and bend.
# - :math:`\lambda_1(\mathbf m\cdot\nabla)\mathbf m`,
#   :math:`\lambda_2\mathbf m(\nabla\cdot\mathbf m)`,
#   :math:`\lambda_3\nabla|\mathbf m|^2` — **self-advection**: a polar
#   flock carries its own orientation along.  These come from the same
#   flux of headings as the pressure: in a polarised cell the anisotropic
#   part of :math:`c_0\rho\langle\hat e\hat e\rangle` is about
#   :math:`c_0(\mathbf m\mathbf m - \tfrac12|\mathbf m|^2\mathbf I)/\rho`, whose
#   divergence gives :math:`\lambda_{1,2}\approx -c_0/\rho` and
#   :math:`\lambda_3 \approx c_0/2\rho` — the signs found here, with
#   magnitudes of order :math:`c_0` at flock densities.  Unlike in a
#   Galilean-invariant fluid, nothing ties these coefficients together: a
#   hallmark of Toner–Tu hydrodynamics.
#
# This is the **Toner–Tu structure**: continuity, Landau ordering,
# pressure, elasticity and the convective nonlinearities.  What falls
# below the 2% line are corrections — the density dependence of the swim
# speed :math:`\nabla\cdot(\rho\mathbf m)` (crowding slows particles
# down), the :math:`\nabla^2|\mathbf m|^2` and :math:`\nabla^2\rho` fluxes,
# the nonlinear pressure :math:`\nabla\rho^2` and the splay–bend
# anisotropy :math:`\nabla(\nabla\cdot\mathbf m)`.  They are real (see the
# bar chart) but refine rather than define the dynamics.  Note that the
# coefficients move when terms are dropped (compare with the full table
# above) by much more than their error bars: the coefficients of a
# truncated model are *effective* values that absorb the correlated parts
# of the missing terms.

# %%
# The noise: shot noise of the particles
# --------------------------------------
#
# A cell's fields fluctuate because its particles turn and move at
# random, so the noise variance should grow with the number of particles
# in the cell — a multiplicative noise, as in Dean's equation for the
# density of Brownian particles.  We fit a noise amplitude proportional
# to :math:`\rho` for each channel; for the polarisation, rotational
# diffusion of the headings predicts
# :math:`D_m = \tfrac{2}{9}\,D_\theta\,\rho/\Delta x^2`, where the factor
# :math:`\tfrac49` is the mean squared weight of the bilinear deposition.

from SFI.bases import symmetric_matrix_basis, x_coordinate

rho_scalar = x_coordinate(0, dim=3, label="ρ")
noise_basis = rho_scalar * symmetric_matrix_basis(3)[[0, 3, 5]]   # ρ × diag(ρ, m_x, m_y)
for model in (inf, inf98):
    model.infer_diffusion_linear(noise_basis)
    model.compute_diffusion_error()
d_c = np.asarray(inf.diffusion_coefficients)
d_s = np.asarray(inf.diffusion_coefficients_stderr)
print(f"D_ρ = ({d_c[0]:.2e} ± {d_s[0]:.0e}) ρ")
print(f"D_m = ({d_c[1]:.2e} ± {d_s[1]:.0e}) ρ  [x],  ({d_c[2]:.2e} ± {d_s[2]:.0e}) ρ  [y];  "
      f"mean {(d_c[1] + d_c[2]) / 2:.2e};  prediction (2/9) D_θ/Δx² = {2 / 9 * D_theta / DX**2:.2e}")

# %%
# The polarisation noise matches the prediction from rotational
# diffusion: the mean of the two components agrees with it to about 1%
# (they differ by ±7% because headings fluctuate perpendicular to the
# flocks' direction of motion, which over this run favours one axis).
# The density noise is some forty times weaker — the density changes only
# when particles cross cell boundaries, mostly carried by the
# deterministic flux :math:`c_0\mathbf m` — which is why the ρ-equation
# dominates the information budget.

# %%
# Checking the coefficients against the microscopic model
# -------------------------------------------------------
#
# Several coefficients have a direct microscopic counterpart:
#
# - continuity: :math:`v = c_0`;
# - pressure from the isotropic part of the heading flux,
#   :math:`\partial_t\mathbf m \ni -c_0\nabla\cdot(\rho\langle\hat e\hat e\rangle)
#   = -\tfrac{c_0}{2}\nabla\rho`: :math:`\zeta = c_0/2`;
# - collisional spreading in mean field:
#   :math:`\kappa = \tfrac12\int V\,d^2r = \pi\varepsilon R_0^3`
#   (:math:`V` the repulsive potential);
# - elasticity from persistent swimming, :math:`c_0^2/16D_\theta +
#   D_{xy}`, plus alignment with neighbours up to :math:`r_c` away, which
#   in mean field adds :math:`(\rho/8)\int K r^2\,d^2r` — a
#   density-dependent elasticity, compared through its best constant
#   approximation on the observed fields (as for the Landau terms below);
# - Landau terms from mean-field alignment, which for a cell of density
#   :math:`\rho` and polarisation :math:`\mathbf m` gives
#   :math:`\partial_t\mathbf m = [-D_\theta + a_1(\rho - |\mathbf m|^2/\rho)]\,\mathbf m`.
#   This is *not* a polynomial: alignment cannot polarise a cell beyond
#   :math:`|\mathbf m| = \rho`.  The Landau polynomial is its expansion
#   near onset; the dense flocks here are strongly polarised, so the
#   right comparison is the best polynomial approximation of the
#   mean-field law over the observed fields — a weighted least-squares
#   projection, computed below.

r_obs = fields[::5, :, 0].ravel()
m2_obs = (fields[::5, :, 1] ** 2 + fields[::5, :, 2] ** 2).ravel()
g_mf = -D_theta + a1_mf * (r_obs - m2_obs / np.maximum(r_obs, 1e-12))
design = np.stack([np.ones_like(r_obs), r_obs, -m2_obs], axis=1)
w = np.sqrt(m2_obs)                       # residual weight |m| in the m-equation
a0_p, a1_p, a2_p = np.linalg.lstsq(design * w[:, None], g_mf * w, rcond=None)[0]

K_swim = c0_true**2 / (16 * D_theta) + D_xy
xc = r_cut / La_true
K_align = np.pi * A_true * La_true**4 / 4 * (6 - np.exp(-xc) * (xc**3 + 3 * xc**2 + 6 * xc + 6))
grid_m = fields[::5, :, 1:].reshape(-1, *GRID, 2)
lap_m = sum(np.roll(grid_m, sh, axis=ax) for sh in (1, -1) for ax in (1, 2)) - 4 * grid_m
w_lap = (lap_m**2).sum(-1).ravel()                # residual weight |∇²m|² of the K term
K_p = K_swim + K_align * (fields[::5, :, 0].ravel() * w_lap).sum() / w_lap.sum()

predictions = [
    ("∇·m", "-c0", -c0_true),
    ("∇ρ", "-c0/2", -c0_true / 2),
    ("∇²ρ²", "π ε R0³ (mean field)", np.pi * eps_true * R0_true**3),
    ("∇²m", "swimming;  + projected alignment", K_swim, K_p),
    ("m", f"-D_θ;  projected mean field", -D_theta, a0_p),
    ("ρm", f"a1;  projected mean field", a1_mf, a1_p),
    ("|m|²m", "projected mean field", None, -a2_p),
]
full = dict(zip(labels, zip(c_all, s_all)))
trunc = dict(zip(BASIS_98.labels, zip(np.asarray(inf98.force_coefficients),
                                      np.asarray(inf98.force_coefficients_stderr))))
print(f"{'term':7s} {'full model':>16s} {'98% model':>16s}   microscopic estimate")
for lab, desc, *vals in predictions:
    est = "  /  ".join(f"{v:+.2f}" for v in vals if v is not None)
    tr = f"{trunc[lab][0]:+.3f} ± {trunc[lab][1]:.3f}" if lab in trunc else "—"
    print(f"{lab:7s} {full[lab][0]:+.3f} ± {full[lab][1]:.3f} {tr:>16s}   {est}  ({desc})")

# %%
# Reading the comparison:
#
# - The **pressure** :math:`\zeta \approx c_0/2` is recovered within
#   about 7% by both models.
# - The **advection speed** :math:`v` exceeds :math:`c_0` by about 20–30%:
#   on a grid whose cells are as small as the flock structure, centred
#   finite differences underestimate the divergence of :math:`\mathbf m`,
#   and the fit compensates.
# - The **collisional spreading** :math:`\kappa` is below its mean-field
#   value: repulsion keeps particles apart, so they overlap less than
#   mean field assumes.
# - The **Landau coefficients** are far below the naive mean-field
#   alignment rate :math:`a_1`, but within about 20% of the *projected*
#   mean-field values for the growth :math:`a_1` and the saturation
#   :math:`a_2`: the polynomial is an effective description of a saturating
#   law.  The decay rate :math:`a_0` is about half of :math:`D_\theta`; at
#   this resolution part of the local relaxation is carried by the grid
#   Laplacian, whose on-site part :math:`-4K/\Delta x^2` also damps
#   :math:`\mathbf m`.
# - The **elasticity** :math:`K` is about eight times the swimming
#   estimate :math:`c_0^2/16D_\theta`.  Alignment across cell boundaries
#   supplies most of the difference: with it the estimate reaches the
#   right magnitude.  The remainder reflects cells (≈3) comparable to the
#   persistence length :math:`c_0/D_\theta = 2` and to the alignment range,
#   where a gradient expansion is only approximate.
#
# Error bars cover sampling noise only: the gap between the full and
# truncated models, and between both and the microscopic estimates,
# reflects coarse-graining at a finite scale — the inferred coefficients
# are the *effective* parameters of the hydrodynamics at the 3-unit scale
# we chose.

# %%
# Bootstrap: do the inferred equations make flocks?
# -------------------------------------------------
#
# We integrate both the PASTIS-selected equations (all resolved terms)
# and the 98% equations, with their inferred noise, from the first
# coarse-grained frame and for as long as the data,
# and compare statistics the fit never targeted: the global polar order,
# the distribution of local densities (whose tail measures the dense
# flocks), and the local polar order as a function of density — the
# flocking transition as seen cell by cell.

coll_boot, _ = inf.simulate_bootstrapped_trajectory(random.PRNGKey(seed + 1), oversampling=4)
coll_boot98, _ = inf98.simulate_bootstrapped_trajectory(random.PRNGKey(seed + 2), oversampling=4)
runs = {"agent-based": fields,
        f"PASTIS SPDE ({len(inf.force_support)} terms)": np.asarray(coll_boot.to_array()),
        f"98% SPDE ({len(selected)} terms)": np.asarray(coll_boot98.to_array())}


def polar_order(F):
    return np.hypot(F[:, :, 1].sum(1), F[:, :, 2].sum(1)) / F[:, :, 0].sum(1)


rho_bins = np.linspace(0.0, 2.0, 11)


def order_vs_density(F):
    r, mm = F[..., 0].ravel(), np.hypot(F[..., 1], F[..., 2]).ravel()
    idx = np.digitize(r, rho_bins) - 1
    ok = (idx >= 0) & (idx < len(rho_bins) - 1)
    num = np.bincount(idx[ok], weights=mm[ok], minlength=len(rho_bins) - 1)
    den = np.bincount(idx[ok], weights=r[ok], minlength=len(rho_bins) - 1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


q = [0.5, 0.9, 0.99]
for name, F in runs.items():
    phi = polar_order(F)
    print(f"{name:24s} polar order φ = {phi.mean():.3f} ± {phi.std():.3f};  density quantiles {q}: "
          f"{np.round(np.quantile(F[..., 0], q), 2)}")

# sphinx_gallery_start_ignore
run_colors = [SFI_COLORS["data"], SFI_COLORS["bootstrap"], SFI_COLORS["tertiary"]]
t_cg = t_burn + np.arange(len(fields)) * dt_frame
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
bins = np.linspace(-0.2, float(np.percentile(fields[..., 0], 99.95)), 60)
centres = 0.5 * (rho_bins[1:] + rho_bins[:-1])
for (name, F), col in zip(runs.items(), run_colors):
    first = name == "agent-based"
    axes[0].plot(t_cg, polar_order(F)[:len(t_cg)], color=col, lw=1.6 if first else 1.2, label=name)
    axes[1].hist(F[..., 0].ravel(), bins=bins, density=True,
                 histtype="stepfilled" if first else "step", alpha=0.45 if first else 1.0,
                 lw=1.6, color=col, label=name)
    axes[2].plot(centres, order_vs_density(F), "o-", color=col, ms=4, lw=1.4, label=name)
axes[0].set_xlabel("time"); axes[0].set_ylabel(r"global polar order $\varphi$")
axes[0].set_ylim(0, None); axes[0].legend(fontsize=8)
axes[1].set_yscale("log"); axes[1].set_xlabel(r"local density $\rho$"); axes[1].set_ylabel("pdf")
axes[2].set_xlabel(r"local density $\rho$")
axes[2].set_ylabel(r"local polar order $\langle|\mathbf{m}|\rangle / \langle\rho\rangle$")
axes[2].set_ylim(0, 1)
fig.suptitle("Emergent statistics: agent-based flock vs inferred SPDEs")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Both SPDEs reproduce the distribution of local densities, including
# the dense-flock tail, and the cell-by-cell ordering curve — slightly
# less polarised inside the densest flocks.  The PASTIS and 98% models
# behave alike: the 2% of information we dropped does not change the
# emergent behaviour.  Their *global* polar order, however, creeps upward
# over the run and ends near twice the agent-based value: the inferred
# equations let flocks merge into co-moving groups more readily than the
# particles do.  Large-scale coordination is a weak, slow signal in a
# 250-time-unit movie of a 143-wide box — the place where an inferred
# model is least constrained by the data, and where the bootstrap is most
# telling.

# %%
# Side by side: the coarse-grained particles (left) and the two inferred
# SPDEs running freely (centre, right), in the colour code of the field
# snapshots above.  All three start from the same frame and then evolve
# independently; what should match is not the trajectory but the
# phenomenology — flocks that form, travel, collide and dissolve.

# sphinx_gallery_start_ignore
fig_spde, axes_spde = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=100)
fig_spde.patch.set_facecolor("#131416")
images = []
for ax, (name, F) in zip(axes_spde, runs.items()):
    images.append(ax.imshow(flock_rgb(F[0]), origin="lower", extent=[0, L, 0, L],
                            interpolation="bilinear"))
    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
stamp = fig_spde.suptitle("")
spde_frames = list(range(0, min(len(F) for F in runs.values()), movie_skip))


def _update_spde(fr):
    ti = spde_frames[fr]
    for im, F in zip(images, runs.values()):
        im.set_data(flock_rgb(F[ti]))
    stamp.set_text(f"t = {t_burn + ti * dt_frame:.0f}")
    return images + [stamp]


anim_spde = FuncAnimation(fig_spde, _update_spde, frames=len(spde_frames), interval=90)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
ax_t.set_facecolor("#131416")
t_thumb = int(np.argmax(contrast_t[i0:]))      # the most heterogeneous frame of the run
plot_particles(coll_micro, t_index=t_thumb, color_dim=2, cmap="hsv",
               vmin=-np.pi, vmax=np.pi, box=box_np, s=3, edgecolors="none", ax=ax_t)
ax_t.set_xlim(0, L); ax_t.set_ylim(0, L)
ax_t.set_xticks([]); ax_t.set_yticks([])
for spine in ax_t.spines.values():
    spine.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore
