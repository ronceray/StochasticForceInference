"""
Lotka–Volterra ecosystem — sparse network recovery
=====================================================

Recover the **sparse interaction network** of a 6-species
Lotka–Volterra ecosystem from a heavily downsampled stochastic
trajectory, using a custom polynomial-of-exponential basis and
**PASTIS** model selection.

The model lives in **log-population** coordinates
:math:`x_i = \\log n_i`, where the force takes the simple form

.. math::

   F_i(x) = r_i + \\sum_j A_{ij}\\,e^{x_j}

with growth rates :math:`r` and a *sparse* interaction matrix
:math:`A`.  Only 18 of the 42 candidate parameters are nonzero —
PASTIS recovers this structure from downsampled, noisy data.

.. note::

   This example uses SFI's parametric force estimator (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`).
   It is needed here: the measurement noise acts through the exponential
   basis (an errors-in-variables effect) and, together with the stiff
   nonlinear dynamics, defeats the linear Itô estimator (which does not
   recover the network).  The parametric estimator profiles the noise
   level and uses the noise-clean skip instrument, and so stays unbiased.

.. rubric:: Tags

synthetic · overdamped · nonlinear · sparsity · ecology · custom-basis
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "sparsity", "ecology", "custom-basis"]
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import (
    SFI_COLORS,
    apply_style,
    stamp_output,
)
from SFI.utils.plotting import (
    plot_pareto_front,
    plot_recovery_bar,
    plot_recovery_matrix,
    timeseries,
)

apply_style()
# sphinx_gallery_end_ignore
# %%
# System definition
# ------------------
#
# A 6-species Lotka–Volterra ecosystem in log-population coordinates.
# The interaction matrix :math:`A` is deliberately sparse: most pairs
# of species do not interact.  Self-interactions (diagonal) are all
# negative (density-dependent mortality), while a handful of off-diagonal
# entries encode predation (+/−) or competition (−/−).

from SFI.langevin import OverdampedProcess
from SFI import make_sf

dim = 6

# Intrinsic growth rates
r = jnp.array([0.50, 0.65, 0.55, 0.65, 0.58, 0.62])

# Interaction matrix (sparse: 18 nonzero entries out of 42 parameters)
A = jnp.array([
    [-0.90, -0.60,  0.00,  0.00,  0.00,  0.00],
    [ 0.00, -0.70, -0.60,  0.00,  0.00,  0.00],
    [+0.60,  0.00, -0.90,  0.00,  0.00, -0.40],
    [ 0.00,  0.00,  0.00, -0.90, -0.50,  0.00],
    [ 0.00,  0.00,  0.00,  0.00, -0.40, -0.60],
    [ 0.00,  0.00, -0.40, +0.40,  0.00, -1.00],
])

D0 = 0.002  # small isotropic noise

dt = 0.01
Nsteps = 40_000
oversampling = 20
seed = 0


def lv_force(x):
    """Lotka–Volterra force in log-population coordinates."""
    return A @ jnp.exp(x) + r


key = random.PRNGKey(seed)
F_sf = make_sf(lv_force, dim=dim, rank=1)

proc = OverdampedProcess(F_sf, D=jnp.eye(dim) * D0)
proc.initialize(jnp.full(dim, -6.0))

key, sub = random.split(key)
coll_clean = proc.simulate(
    dt=dt, Nsteps=Nsteps, key=sub, prerun=0, oversampling=oversampling,
)
print(f"Clean trajectory: {coll_clean.T} frames, dim={coll_clean.d}, dt={dt}")

# %%
# Degrade the data
# ------------------
#
# Downsample the trajectory and add measurement noise to simulate a
# realistic low-data regime.
# The inference will use only ~2 000 frames instead of 40 000.

downsample_factor = 50
noise_level = 0.1  # measurement noise in log-population coordinates
coll_degraded = coll_clean.degrade(downsample=downsample_factor, noise=noise_level, seed=42)
print(f"Degraded trajectory: {coll_degraded.T} frames "
      f"(downsampled {downsample_factor}×, effective dt={dt * downsample_factor}, "
      f"noise={noise_level})")

# %%
# Population dynamics
# ---------------------
#
# Left: the full simulated trajectory (40 000 frames) showing rich
# oscillatory dynamics of six coexisting species.
# Right: the downsampled, noisy data actually used for inference.

# Population-space views (n_i = e^{x_i}) for plotting, via the
# ``transform`` hook on ``timeseries`` (exponentiates the log-population
# states on the fly).

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

timeseries(coll_clean, transform=np.exp, ax=axes[0], lw=0.4, alpha=0.8)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Population $n_i = e^{x_i}$")
axes[0].set_title(f"Full trajectory ({coll_clean.T} frames)")

timeseries(coll_degraded, transform=np.exp, ax=axes[1], marker=".", linestyle="-",
           markersize=1.5, lw=0.6, alpha=0.8)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Population $n_i = e^{x_i}$")
axes[1].set_title(f"Degraded data ({coll_degraded.T} frames, {downsample_factor}× downsampled)")

fig.suptitle("Lotka–Volterra ecosystem — 6 species", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Custom polynomial-of-exponential basis
# -----------------------------------------
#
# In log-population coordinates the force is linear in the
# populations :math:`e^{x_j}`, so the natural basis dictionary is
#
# .. math::
#
#    \{1,\; e^{x_1},\; \dots,\; e^{x_6}\}
#
# giving 7 scalar features.  Vectorising to 6 species yields
# :math:`7 \times 6 = 42` force parameters.  The true model uses
# only 18 of them (6 growth rates + 12 nonzero interactions).

from SFI.statefunc import make_basis

import SFI


def polyexp_basis_func(x):
    """Scalar basis: {1, exp(x_1), ..., exp(x_d)}."""
    return jnp.concatenate([jnp.ones(1), jnp.exp(x)])  # shape (dim+1,)


# Labels: constant + each exponential
basis_labels = ["1"] + [f"exp(x{i+1})" for i in range(dim)]

B_scalar = make_basis(
    polyexp_basis_func, dim=dim, rank=0,
    n_features=dim + 1, labels=basis_labels,
)
B = B_scalar.vectorize(dim)

# %%
# Inference and sparsification
# ------------------------------
#
# We fit the force with the parametric estimator (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`), which
# profiles the measurement-noise level and so stays unbiased on the
# noisy exponential basis.  PASTIS then selects the sparse interaction
# network from the candidate basis.

# This system sits deep in the noise-dominated regime
# (β = σ²/(2DΔt) ≈ 12): the residual correlations decay slowly, so we
# widen the precision window beyond its default (``extra_radius=3``) —
# the documented escape hatch for β ≫ 1.  The skip-trick instrument
# (on by default) is what removes the errors-in-variables bias here.
inf = SFI.OverdampedLangevinInference(coll_degraded)
inf.infer_force(B, extra_radius=3)
inf.compute_force_error()

inf.sparsify_force(criterion="PASTIS", p=0.1)
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, maxpoints=2000)

k_sel, support_sel, _, coeffs_sel = inf.force_sparsity_result.select_by_ic("PASTIS")

inf.print_report()

# %%
# Pareto front and sparse model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_pareto_front(inf.force_sparsity_result, ax=axes[0])

labels_all = B.labels if hasattr(B, "labels") and B.labels else [str(i) for i in range(B.n_features)]
plot_recovery_bar(np.array(coeffs_sel), support_sel, labels=labels_all, ax=axes[1])
axes[1].set_title(f"Sparse model: {k_sel} / {B.n_features} features")

fig.suptitle("PASTIS model selection — Lotka–Volterra", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Interaction matrix recovery
# -----------------------------
#
# The standout result: the inferred sparse coefficients reshaped as
# a :math:`(d{+}1) \\times d` parameter matrix (growth rates on top,
# interaction columns below) compared to the ground truth.
# Zeros in the true matrix should appear as zeros in the inferred one.

# True parameter matrix: first row = r, rows 1..d = A^T
true_params = np.array(jnp.vstack((r[None, :], A.T)))

# Reconstruct inferred parameter matrix from sparse coefficients
inferred_flat = np.zeros(B.n_features)
inferred_flat[np.array(support_sel)] = np.array(coeffs_sel)
inferred_params = inferred_flat.reshape((dim + 1, dim))

vmax = max(np.abs(true_params).max(), np.abs(inferred_params).max()) * 1.05

row_labels = ["$r_i$"] + [f"$A_{{i{j+1}}}$" for j in range(dim)]
col_labels = [f"sp. {i+1}" for i in range(dim)]

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_recovery_matrix(
    true_params, inferred_params,
    row_labels=row_labels, col_labels=col_labels, vmax=vmax, axes=axes,
)
axes[0].set_title("True parameters")
axes[1].set_title("Inferred (PASTIS)")
# Emphasise the separation between growth rates (top row) and interactions.
for ax in axes:
    ax.axhline(0.5, color="#B0B0B0", lw=1.2)

fig.suptitle("Interaction network recovery", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Bootstrap trajectory
# ---------------------
#
# Simulate a new trajectory from the inferred sparse model and compare
# population dynamics to the original data.

key_boot = random.PRNGKey(seed + 99)
coll_boot, _ = inf.simulate_bootstrapped_trajectory(key_boot, oversampling=oversampling)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

timeseries(coll_degraded, transform=np.exp, ax=axes[0], lw=0.6, alpha=0.8)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Population $n_i$")
axes[0].set_title("Data (degraded)")

timeseries(coll_boot, transform=np.exp, ax=axes[1], lw=0.6, alpha=0.8)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Population $n_i$")
axes[1].set_title("Bootstrap (inferred sparse model)")

fig.suptitle("Data vs bootstrapped population dynamics", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Thumbnail
# ---------
#
# Compact population time series for the gallery grid.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 3))
ax_t = fig_thumb.add_subplot(111)
timeseries(coll_clean, transform=np.exp, ax=ax_t, lw=0.3, alpha=0.7)
ax_t.set_xticks([])
ax_t.set_yticks([])
ax_t.set_xlabel("")
ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
