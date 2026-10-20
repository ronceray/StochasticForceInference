r"""
Lotka–Volterra ecosystem — a food web from a few noisy censuses
=================================================================

Ecological time series are short, sparse and noisy.  How little data
does it take to map who inhibits whom in a community?  Here, eight
replicate cultures of a 6-species Lotka–Volterra community are
inoculated at random low densities and counted every two time
units, with a 10% counting error: **20
censuses per culture, 160 noisy snapshots in total**, taken at an
interval as long as the species' growth time.  From these, the
parametric force estimator and PASTIS model selection recover the
sparse interaction network.

In log-population coordinates :math:`x_i = \log n_i` the dynamics read

.. math::

   \dot x_i = r_i + \sum_j A_{ij}\, e^{x_j} + \sqrt{2D}\,\xi_i(t),

with growth rates :math:`r_i`, a sparse interaction matrix :math:`A`
(:math:`A_{ij}` is the effect of species :math:`j` on species
:math:`i`) and demographic noise :math:`D`.  The candidate library has
42 terms — a constant and the six populations :math:`e^{x_j}`, for each
of the six species — and the true model uses 20 of them: 6 growth rates
and 14 interactions (6 self-limitations and 8 cross-species couplings).

.. rubric:: Tags

synthetic · overdamped · nonlinear · sparsity · ecology
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "sparsity", "ecology"]
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import NullLocator

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import (
    SFI_COLORS,
    apply_style,
    stamp_output,
)
from SFI.utils.plotting import (
    plot_pareto_front,
    plot_recovery_bar,
    timeseries,
)

apply_style()
# ``Basis.vectorize`` is built on a feature-axis product and emits a generic
# "Cartesian product" notice that does not apply to it.
warnings.filterwarnings("ignore", message=r"\[StateExpr\] Binary op")

SPECIES_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"][:6]
INHIBIT, FACILITATE = "#C8C8C8", "#7CDB8A"


def draw_network(ax, A_hat, A_ref=None, title=""):
    """Interaction graph: an arrow j -> i for every non-zero A_ij (i != j).

    Arrow width grows with |A_ij|; the node ring width with the
    self-limitation |A_ii|.  With a reference matrix, true couplings that
    are missing are dotted grey and spurious couplings dashed.
    """
    n = A_hat.shape[0]
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    xy = np.c_[np.cos(ang), np.sin(ang)]
    for i in range(n):
        for j in range(n):
            a = A_hat[i, j]
            a_ref = a if A_ref is None else A_ref[i, j]
            if i == j or (a == 0 and a_ref == 0):
                continue
            if a == 0:
                kw = dict(color="#808080", ls=":", lw=1.4)
            else:
                kw = dict(color=FACILITATE if a > 0 else INHIBIT, lw=1.0 + 3.0 * abs(a),
                          ls="--" if a_ref == 0 else "-")
            ax.add_patch(FancyArrowPatch(
                xy[j], xy[i], arrowstyle="-|>,head_length=6,head_width=4",
                connectionstyle="arc3,rad=0.18", shrinkA=19, shrinkB=19,
                mutation_scale=1.6, zorder=2, **kw))
    for i in range(n):
        ax.add_patch(Circle(xy[i], 0.17, facecolor=SPECIES_COLORS[i], edgecolor="#D0D0D0",
                            lw=1.0 + 4.0 * abs(A_hat[i, i]), zorder=3))
        ax.text(*xy[i], f"{i + 1}", ha="center", va="center", fontsize=11,
                color="#131416", fontweight="bold", zorder=4)
    ax.text(0.0, 1.42, title, ha="center", va="bottom", fontsize=12, color="#D0D0D0")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.62 if title else 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


NETWORK_LEGEND = [
    Line2D([], [], color=INHIBIT, lw=2.5, label="inhibition ($A_{ij}<0$)"),
    Line2D([], [], color=FACILITATE, lw=2.5, label="facilitation ($A_{ij}>0$)"),
    Line2D([], [], color="#808080", lw=1.4, ls=":", label="missed coupling"),
    Line2D([], [], color=INHIBIT, lw=1.5, ls="--", label="spurious coupling"),
]
# sphinx_gallery_end_ignore
stamp_output()

# %%
# The community
# -------------
#
# Every species limits itself (negative diagonal of :math:`A`); a few
# off-diagonal couplings encode competition (both negative) or
# exploitation (one positive, one negative).  The positive growth rates
# and the interactions admit a coexistence equilibrium
# :math:`n^\star = -A^{-1} r` with all six species present.

from SFI import make_sf
from SFI.langevin import OverdampedProcess

dim = 6
r = jnp.array([0.50, 0.65, 0.55, 0.65, 0.58, 0.62])      # growth rates
A = jnp.array([                                            # A[i, j]: effect of j on i
    [-0.90, -0.60,  0.00,  0.00,  0.00,  0.00],
    [ 0.00, -0.70, -0.60,  0.00,  0.00,  0.00],
    [+0.60,  0.00, -0.90,  0.00,  0.00, -0.40],
    [ 0.00,  0.00,  0.00, -0.90, -0.50,  0.00],
    [ 0.00,  0.00,  0.00,  0.00, -0.40, -0.60],
    [ 0.00,  0.00, -0.40, +0.40,  0.00, -1.00],
])
D0 = 0.002                                                 # demographic noise


def lv_force(x):
    """Lotka–Volterra force in log-population coordinates."""
    return r + A @ jnp.exp(x)


proc = OverdampedProcess(make_sf(lv_force, dim=dim, rank=1), D=jnp.eye(dim) * D0)
n_star = jnp.linalg.solve(A, -r)
print("coexistence equilibrium n* =", np.round(np.asarray(n_star), 3))

# %%
# The experiment: eight cultures, twenty censuses each
# -----------------------------------------------------
#
# Each culture is seeded with every species at a random 0.1–10% of its
# equilibrium abundance and followed for 40 time units.  The cultures are
# independent replicates of the same process; passing one initial
# condition per culture (shape ``(8, 6)``) simulates them together, and
# SFI stores them as eight non-interacting trajectories of one dataset.
# The simulation runs at a fine time step; the "experiment" then keeps
# one census every :math:`\Delta t = 2` and adds a 10% counting error
# (:math:`\sigma = 0.1` in log units) to every count.

n_cultures = 8
duration = 40.0
census_interval = 2.0
counting_noise = 0.10
dt_sim = 0.01


def run_cultures(seed):
    """Simulate `n_cultures` replicate cultures from random low inocula."""
    key_seed, key_noise = random.split(random.PRNGKey(seed))
    log_fraction = random.uniform(key_seed, (n_cultures, dim),
                                  minval=jnp.log(1e-3), maxval=jnp.log(1e-1))
    proc.initialize(jnp.log(n_star) + log_fraction)     # one row per culture
    return proc.simulate(dt=dt_sim, Nsteps=int(duration / dt_sim),
                         key=key_noise, oversampling=20)


def census(coll_true, interval, seed):
    """Keep one frame every `interval` and add the counting error."""
    return coll_true.degrade(downsample=int(round(interval / dt_sim)),
                             noise=counting_noise, seed=100 + seed)


coll_true = run_cultures(seed=0)
coll = census(coll_true, census_interval, seed=0)
n_census = coll.T
print(f"{n_cultures} cultures x {n_census} censuses = {n_cultures * n_census} noisy snapshots "
      f"(Δt = {census_interval}, σ = {counting_noise})")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    timeseries(coll_true, particles=[k], transform=np.exp, ax=ax, lw=0.9, alpha=0.45)
    ax.set_prop_cycle(None)                        # same species colours for the censuses
    timeseries(coll, particles=[k], transform=np.exp, ax=ax, ls="none", marker="o", ms=3.2)
    ax.set_yscale("log")
    ax.set_title(f"culture {k + 1}", fontsize=10)
    ax.set_xlabel("time" if k >= 4 else "")
    ax.set_ylabel("population $n_i$" if k % 4 == 0 else "")
handles = [Line2D([], [], color=c, marker="o", ms=4, lw=0.9, label=f"species {i + 1}")
           for i, c in enumerate(SPECIES_COLORS)]
fig.legend(handles=handles, loc="outside right center", fontsize=8)
fig.suptitle(f"The whole dataset: {n_cultures} cultures × {n_census} noisy censuses "
             "(dots; lines: unobserved true dynamics)", fontsize=12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each culture shows the same two phases: a growth phase, where every
# species multiplies at its own rate (straight lines on the log scale,
# about threefold between censuses), then saturation at the coexistence
# equilibrium, where the populations fluctuate by ~10–20% — comparable
# to the counting error.  The growth phase pins the growth rates; the
# way each species bends away from straight growth as the others become
# abundant pins the interactions.

# %%
# Candidate library: a constant plus the six populations
# ------------------------------------------------------
#
# In log coordinates the force is linear in the populations
# :math:`e^{x_j}`, so the library is
# :math:`\{1,\; e^{x_1},\; \dots,\; e^{x_6}\}` — built from the
# coordinate primitives with an elementwise ``exp`` — vectorised over the
# six species: :math:`7 \times 6 = 42` candidate terms.

from SFI.bases import ones_basis, x_components

xs = x_components(dim)
X_all = xs[0]
for xi in xs[1:]:
    X_all = X_all & xi                                   # (x0, ..., x5) as features
populations = X_all.elementwisemap(jnp.exp, label_fn=lambda s: f"exp({s})")
B = (ones_basis(dim) & populations).vectorize(dim)       # 42 force terms

labels = list(B.labels)
theta_true = np.asarray(jnp.vstack((r[None, :], A.T))).ravel()   # same order as B
support_true = np.flatnonzero(theta_true)
print(f"{len(labels)} candidate terms, {len(support_true)} in the true model")

# %%
# A quick linear fit, and what it says about the sampling
# --------------------------------------------------------
#
# The closed-form linear estimator is the fast first pass.  The
# finite-:math:`\Delta t` diagnostic
# :func:`~SFI.diagnostics.finite_dt_scale` then measures
# :math:`\Delta t\,\|\nabla \hat F\|` — how much the force changes
# within one census interval.  Above about 0.2 the linear estimator's
# finite-:math:`\Delta t` corrections are no longer guaranteed, and the
# parametric estimator, which integrates the flow across each interval,
# is the safe tool.

import SFI
from SFI.diagnostics import finite_dt_scale

inf_lin = SFI.OverdampedLangevinInference(coll)
inf_lin.compute_diffusion_constant()
inf_lin.infer_force_linear(B)
scale = finite_dt_scale(inf_lin)
print(f"linear preset chosen by 'auto': {inf_lin.metadata['force_preset']} "
      f"(noise estimate Tr Λ = {float(inf_lin.Lambda_trace):.2f}; true {dim * counting_noise**2:.2f})")
print(f"Δt·|∇F| = {scale['grad_phi_dt']:.2f} (99th percentile {scale['grad_p99_dt']:.2f}) "
      f"-> verdict: {scale['verdict']}")

inf_lin.compute_force_error()
inf_lin.sparsify_force(criterion="PASTIS")
inf_lin.compute_force_error()

# %%
# Note which preset ``"auto"`` picked: ``"clean"``.  Its noise detector
# reads measurement noise from the anticorrelation of successive
# increments, and on these data the steady growth trends correlate
# successive increments positively and hide it.  The linear fit below is
# therefore the clean-data estimator (Itô moments with a trapezoid-rule
# Gram); we keep the default, as a user would, and let the comparison
# show what it gives.

# %%
# The parametric estimator
# ------------------------
#
# :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` predicts
# each census from the previous one with a Runge–Kutta flow step, models
# the counting error explicitly (its level is estimated together with
# the force and the demographic noise), and removes the bias the noise
# would otherwise cause inside the exponential features.  The one
# non-default argument is the number of Runge–Kutta substeps per census
# interval.  The diagnostic above puts :math:`\Delta t\,\|\nabla \hat F\|`
# well above 1 — the dynamics change substantially within one interval —
# so we split each interval into :math:`\lceil \Delta t\,\|\nabla \hat
# F\| \rceil` substeps (99th percentile over the data), each shorter than
# the fastest local time scale :math:`1/\|\nabla \hat F\|`.  This is a
# numerical resolution choice, not a tuning knob: beyond it, more
# substeps leave the fit unchanged.

n_substeps = int(np.ceil(scale["grad_p99_dt"]))
print(f"n_substeps = {n_substeps}")

inf = SFI.OverdampedLangevinInference(coll)
inf.infer_force(B, n_substeps=n_substeps)
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()
inf.compare_to_exact(force_exact=proc.force_sf, maxpoints=2000)
inf.print_report()

# %%
# The force, which is what we are after, comes with a predicted error;
# the realised error against the true model (both printed in the report)
# is of the same order.  The 6×6 demographic-noise and counting-noise
# matrices printed above are only loosely pinned: over one census
# interval the counting error (:math:`2\sigma^2 = 0.02`) outweighs the
# demographic noise (:math:`2D\Delta t = 0.008`), and 160 snapshots
# split the two only roughly.  Their diagonals against the truth:

sigma_hat = np.sqrt(np.diag(np.asarray(inf.Lambda)))
D_hat = np.diag(np.asarray(inf.diffusion_average))
print(f"counting noise σ per species: {np.round(sigma_hat, 3)}  (true {counting_noise})")
print(f"demographic noise D per species: {np.round(D_hat, 4)}  (true {D0})")

# %%
# Which terms were found?
# -----------------------
#
# Both estimators' selections against the true support, and the relative
# error of the 42 coefficients (zeros included) after selection.  Basis
# labels count species from 0: ``exp(x0)·e2`` is the effect of species 1
# on species 3, :math:`A_{31}`.

from SFI.inference.sparse import overlap_metrics


def recovery(fit):
    """Support recovery (F1, misses, false positives) and coefficient error."""
    m = overlap_metrics(list(support_true), list(np.asarray(fit.force_support)))
    f1 = 2 * m["TP"] / (2 * m["TP"] + m["FP"] + m["FN"])
    coeffs = np.asarray(fit.force_coefficients_full)
    err = np.linalg.norm(coeffs - theta_true) / np.linalg.norm(theta_true)
    return f1, m["FN"], m["FP"], err


for name, fit in [("linear", inf_lin), ("parametric", inf)]:
    f1, fn, fp, err = recovery(fit)
    sel = set(np.asarray(fit.force_support).tolist())
    print(f"{name:>10}: keeps {len(sel)}/{len(labels)} terms, F1 = {f1:.2f}, "
          f"missed {[labels[i] for i in sorted(set(support_true) - sel)]}, "
          f"spurious {[labels[i] for i in sorted(sel - set(support_true))]}, "
          f"coefficient error {100 * err:.1f}%")

# %%
# Pareto front and coefficients with error bars
# ---------------------------------------------
#
# Left: the information gained by the best model of each size along the
# parametric fit's Pareto front; PASTIS (gold) stops where an extra term
# no longer pays for itself, while AIC (orange) would keep a few more.
# Right: the selected coefficients (gold bars, 1-σ error bars) against
# the truth (orange outlines), with the linear estimator's selection for
# comparison (teal diamonds).
# :math:`r_i` are growth rates and :math:`A_{ij}` the effect of species
# :math:`j` on species :math:`i`.

coeffs, cov = inf.coeff_block(slice(0, len(labels)))
stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
coeffs_lin, cov_lin = inf_lin.coeff_block(slice(0, len(labels)))
stderr_lin = np.sqrt(np.clip(np.diag(cov_lin), 0.0, None))

# sphinx_gallery_start_ignore
math_labels = [f"$r_{{{k % dim + 1}}}$" if k < dim else f"$A_{{{k % dim + 1}{k // dim}}}$"
               for k in range(len(labels))]
shown = np.union1d(np.union1d(support_true, np.asarray(inf.force_support)),
                   np.asarray(inf_lin.force_support))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 2.6]})
plot_pareto_front(inf.force_sparsity_result, criteria=("PASTIS", "AIC"), ax=axes[0])
axes[0].set_title("Pareto front (parametric)")
plot_recovery_bar(coeffs[shown], shown, coeffs_true=theta_true[shown], support_true=shown,
                  stderr=stderr[shown], labels=math_labels, ax=axes[1])
x_lin = np.arange(len(shown)) - min(0.6, 6.0 / len(shown)) / 2
sel_lin = coeffs_lin[shown] != 0
axes[1].errorbar(x_lin[sel_lin], coeffs_lin[shown][sel_lin], yerr=stderr_lin[shown][sel_lin],
                 fmt="D", ms=4, color=SFI_COLORS["highlight"], ecolor=SFI_COLORS["highlight"],
                 elinewidth=1, capsize=2, label="linear", zorder=5)
axes[1].axhline(0, color="#808080", lw=0.6)
axes[1].set_xticklabels([math_labels[s] for s in shown], rotation=0, ha="center", fontsize=8)
h, l = axes[1].get_legend_handles_labels()
names = {"inferred": "parametric", "exact": "true", "linear": "linear"}
axes[1].legend(h, [names[x] for x in l], loc="upper right", ncol=3, fontsize=8)
axes[1].set_title(f"Selected coefficients ({len(np.asarray(inf.force_support))} of {len(labels)} terms)")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The interaction network
# -----------------------
#
# The off-diagonal couplings drawn as a network: an arrow from
# :math:`j` to :math:`i` for every non-zero :math:`A_{ij}`, its width
# growing with :math:`|A_{ij}|`; the ring around each species grows with
# its self-limitation :math:`|A_{ii}|`.  At this census interval both
# estimators recover every coupling with no spurious one (see the
# printed selections above); the parametric coefficients are about twice
# as accurate.

A_true = np.asarray(A)
A_lin = coeffs_lin.reshape(dim + 1, dim)[1:].T            # rows: affected species
A_par = coeffs.reshape(dim + 1, dim)[1:].T

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
draw_network(axes[0], A_true, title="True network")
draw_network(axes[1], A_lin, A_true, title="Linear estimator + PASTIS")
draw_network(axes[2], A_par, A_true, title="Parametric estimator + PASTIS")
fig.legend(handles=NETWORK_LEGEND, loc="outside lower center", ncol=4, fontsize=9)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Bootstrap: replaying the experiment with the inferred model
# -----------------------------------------------------------
#
# Does the selected model reproduce the cultures, not just the
# coefficients?  We restart each culture 20 times from its first census
# with the inferred force and noise and compare the spread of these
# replays (median and 10–90% band) with the censuses.  The replays are
# recorded at a fine time step to show the full trajectories between
# censuses.

boot_proc = inf.simulate_bootstrapped_trajectory(random.PRNGKey(1), simulate=False)
n_replays = 20
first_census = jnp.asarray(coll.to_array()[0])            # (cultures, species)
boot_proc.initialize(jnp.tile(first_census, (n_replays, 1)))
coll_boot = boot_proc.simulate(dt=0.1, Nsteps=int(duration / 0.1), key=random.PRNGKey(2),
                               oversampling=20)

# sphinx_gallery_start_ignore
t_b, X_b, _ = coll_boot.to_arrays()
pop_b = np.exp(np.asarray(X_b).reshape(len(t_b), n_replays, n_cultures, dim))
lo, med, hi = np.quantile(pop_b, [0.1, 0.5, 0.9], axis=1)
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    for i in range(dim):
        ax.fill_between(t_b, lo[:, k, i], hi[:, k, i], color=SPECIES_COLORS[i], alpha=0.25, lw=0)
        ax.plot(t_b, med[:, k, i], color=SPECIES_COLORS[i], lw=0.9)
    timeseries(coll, particles=[k], transform=np.exp, ax=ax, ls="none", marker="o", ms=3.2)
    ax.set_yscale("log")
    ax.set_title(f"culture {k + 1}", fontsize=10)
    ax.set_xlabel("time" if k >= 4 else "")
    ax.set_ylabel("population $n_i$" if k % 4 == 0 else "")
fig.legend(handles=handles, loc="outside right center", fontsize=8)
fig.suptitle("Censuses (dots) vs replays of the inferred model (median, 10–90% band)",
             fontsize=12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The replays track the censuses through both phases: the timing and
# order of the growth spurts, the overshoots, and the equilibrium
# levels.  All replays of a culture start from the same noisy census, so
# their spread comes from the demographic noise alone; it is widest for
# species 1, whose slow return to equilibrium lets fluctuations
# accumulate (and somewhat wider than the truth, since :math:`\hat D` is
# over-estimated).

# %%
# How coarse, how few? Linear vs parametric
# -----------------------------------------
#
# Keep the experiment fixed — eight cultures followed for 40 time units,
# 10% counting error — and vary only how often they are counted: from
# :math:`\Delta t = 0.5` (80 censuses per culture) to
# :math:`\Delta t = 8` (5 censuses).  Each setting is repeated on three
# independent experiments (new inocula, demographic and counting noise),
# and both estimators run exactly as above, followed by PASTIS.

census_intervals = [0.5, 1.0, 2.0, 4.0, 8.0]
seeds = [0, 1, 2]
cultures = {seed: run_cultures(seed) for seed in seeds}

sweep = {"linear": np.zeros((len(census_intervals), len(seeds), 2)),
         "parametric": np.zeros((len(census_intervals), len(seeds), 2))}
D_ratio = np.zeros((len(census_intervals), len(seeds)))      # parametric D̂ / D
presets = set()                                               # linear presets picked by "auto"
for a, interval in enumerate(census_intervals):
    for b, seed in enumerate(seeds):
        data = census(cultures[seed], interval, seed)
        lin = SFI.OverdampedLangevinInference(data)
        lin.compute_diffusion_constant()
        lin.infer_force_linear(B)
        presets.add(lin.metadata["force_preset"])
        n_sub = int(np.ceil(finite_dt_scale(lin)["grad_p99_dt"]))
        lin.sparsify_force(criterion="PASTIS")

        par = SFI.OverdampedLangevinInference(data)
        par.infer_force(B, n_substeps=n_sub)
        par.sparsify_force(criterion="PASTIS")

        for name, fit in [("linear", lin), ("parametric", par)]:
            f1, fn, fp, err = recovery(fit)
            sweep[name][a, b] = f1, err
        D_ratio[a, b] = np.mean(np.diag(np.asarray(par.diffusion_average))) / D0
    print(f"Δt = {interval:>3} ({data.T:>2} censuses/culture): "
          + "; ".join(f"{name} F1 = {np.round(sweep[name][a, :, 0], 2)}, "
                      f"error = {np.round(100 * sweep[name][a, :, 1]).astype(int)}%"
                      for name in sweep)
          + f"; parametric D/D_true = {np.round(D_ratio[a], 1)}")
print(f"linear preset(s) picked by 'auto' across the sweep: {sorted(presets)}")

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
styles = {"linear": dict(color=SFI_COLORS["highlight"], marker="D"),
          "parametric": dict(color=SFI_COLORS["inferred"], marker="o")}
dts = np.asarray(census_intervals)
for name, st in styles.items():
    for q, ax in enumerate(axes):
        vals = sweep[name][:, :, q]
        jitter = 1.0 + 0.04 * (np.arange(len(seeds)) - 1) * (1 if name == "linear" else -1)
        for b in range(len(seeds)):
            ax.plot(dts * jitter[b], vals[:, b], ls="none", marker=st["marker"], ms=4,
                    color=st["color"], alpha=0.55)
        ax.plot(dts, vals.mean(axis=1), color=st["color"], lw=2, label=f"{name} (mean of 3)")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"census interval $\Delta t$")
    ax.axvline(census_interval, color="#808080", lw=0.8, ls="--")
    top = ax.secondary_xaxis("top")
    top.set_xticks(dts)
    top.set_xticklabels([f"{int(round(duration / d))}" for d in dts])
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("censuses per culture", fontsize=9)
axes[0].set_ylabel("support recovery F1")
axes[0].set_ylim(-0.03, 1.05)
axes[0].legend(loc="lower left", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_ylabel(r"coefficient error $\|\hat\theta-\theta\|/\|\theta\|$")
fig.suptitle("Network recovery vs census interval (8 cultures, 40 time units, 10% counting error)",
             fontsize=12)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Reading the sweep (the numbers are printed above):
#
# * **Dense censuses** (:math:`\Delta t \le 1`, 40–80 per culture):
#   consecutive counts differ mostly by counting error.  The parametric
#   estimator models that error and finds the exact network in every run.
#   The linear estimator runs in its clean-data mode throughout (its
#   automatic noise detection is fooled by the growth trends, see the
#   printed presets); the noise costs it missed or spurious couplings and
#   coefficient errors several times larger than the parametric ones.
# * **Coarse censuses** (:math:`\Delta t = 2`–4, 20–10 per culture): both
#   estimators find the network.  The counting error is now small
#   compared with the change between censuses, and the linear
#   estimator's trapezoid-rule Gram integrates the force across each
#   interval accurately on these smooth growth-then-saturation curves —
#   well beyond the regime the finite-:math:`\Delta t` diagnostic
#   certifies.  The parametric estimator's demographic-noise estimate,
#   about twice too large at finer sampling, grows to about six times the
#   truth at :math:`\Delta t = 4` (the printed :math:`\hat D/D`); this
#   widens its error bars, and PASTIS then drops a weak coupling in one
#   run out of three.
# * **Five censuses per culture** (:math:`\Delta t = 8`) are too few:
#   32 census intervals for 42 candidate terms and two noise matrices.
#   The parametric estimator's error bars become so wide that PASTIS
#   keeps at most a handful of terms, while the linear estimator returns
#   a partly wrong network.
#
# Bottom line: with the parametric estimator, eight cultures counted 20
# to 80 times each recovered the full network in every run, and counted
# 10 times in two runs out of three.  The linear estimator needs the
# populations to change by more than the counting error between
# censuses; when they do, it is a fast and accurate first pass.

# %%
# Thumbnail
# ---------
#
# The inferred network, for the gallery grid.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
draw_network(ax_t, A_par, A_true)
plt.show()
# sphinx_gallery_end_ignore
