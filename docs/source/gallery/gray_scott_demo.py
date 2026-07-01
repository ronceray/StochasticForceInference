r"""
Gray-Scott reaction-diffusion: SPDE inference
================================================

.. note::

   Uses the **experimental** SPDE toolbox — see :doc:`/spde/index`.

Infer the reaction-diffusion dynamics of a stochastic Gray-Scott
system on a 2D grid.  This demonstrates SFI's SPDE capabilities:

- **GridLayout** composing differential operators on named field
  sectors: ``layout.lap(U)`` applies a 5-point Laplacian stencil
  to the U sector, ``layout.grad(U).dot(layout.grad(V))`` builds
  cross-gradient features, and ``layout.embed(rank=1, ...)``
  compiles everything into a single force model
- Auto-generated **feature labels** from the expression tree,
  using :meth:`~SFI.statefunc.structexpr.StructuredExpr.with_label`
  and operator auto-labelling
- **PASTIS sparsification** recovering the exact 7-term model from
  a 34-feature over-specified basis (all order-3 monomials,
  Laplacians, gradient products, and biharmonics)
- **Conserved noise** — :func:`~SFI.bases.spde.conserved_noise_pbc`
  provides :math:`\nabla\!\cdot\!\boldsymbol\eta` noise whose
  spatial average is exactly zero at every time step
- Overdamped inference on high-dimensional state spaces
- Degradation of spatial data and bootstrap validation
- **Multi-regime robustness** — the same basis recovers the same
  7-term structure across distinct :math:`(F, K)` regimes

The Gray-Scott model defines two interacting fields :math:`U, V`
with conserved (divergence-form) stochastic transport:

.. math::

   \dot{U} = D_U \nabla^2 U - U V^2 + F (1-U)
             + \sigma\,\nabla\!\cdot\!\boldsymbol\eta_U

   \dot{V} = D_V \nabla^2 V + U V^2 - (F+K) V
             + \sigma\,\nabla\!\cdot\!\boldsymbol\eta_V

where :math:`\boldsymbol\eta_{U,V}` are independent spatiotemporal
white vector noises.  The conserved noise ensures that any spatial
average :math:`\langle U \rangle, \langle V \rangle` changes only
through the deterministic dynamics, not through noise.

.. rubric:: Tags

synthetic · overdamped · SPDE · reaction-diffusion · 2D · sparsification · multi-experiment
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "SPDE", "experimental", "reaction-diffusion", "2D", "sparsification", "multi-experiment"]
# sphinx_gallery_thumbnail_number = 5

# sphinx_gallery_start_ignore
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Avoid GPU pre-allocation OOM when running after other gallery examples
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gallery_utils.helpers import (
    SFI_COLORS,
    apply_style,
    stamp_output,
)
from SFI.utils.formatting import model_summary
from SFI.utils.plotting import (
    animate_spde_comparison,
    plot_pareto_front,
    plot_recovery_bar,
    plot_recovery_bar_multi,
    plot_spde_snapshot,
)

apply_style()
# sphinx_gallery_end_ignore
# %%
# System setup
# ---------------
#
# We use a **64×64 grid** with ``dx = 1`` (physical domain
# :math:`L = 64`).  The Turing wavelength
# :math:`\lambda \approx 2\pi\sqrt{D_U/F} \approx 12` fits about five
# times across the domain, giving well-developed patterns.  Inference
# is *local* — it fits the finite-difference stencil at every cell — so
# the 4096 cells supply ample constraints for the operators.
#
# The force model is built with :class:`~SFI.statefunc.layout.GridLayout`:
# two named ``ScalarSector`` fields ``U`` and ``V`` share a single
# layout.  Differential operators (``layout.lap``) and pointwise
# algebra (``U * V * V``) compose freely; ``layout.embed`` compiles
# the whole expression tree into an inference-ready :class:`~SFI.statefunc.Basis`.

from SFI.bases.spde import conserved_noise_pbc, square_grid_extras
from SFI.langevin import OverdampedProcess
from SFI.statefunc.layout import GridLayout, ScalarSector

DIM = 2  # field dimension (U, V)
GRID = (64, 64)
DT = 0.5
DX = 1.0
STEPS = 500
OVERS = 4
PRERUN = 200   # burn-in frames: let patterns develop before recording
SEED = 0

# Gray-Scott parameters — "coral" regime
DU, DV = 0.16, 0.08
F_gs, K_gs = 0.042, 0.063

# Conserved noise amplitude (same for both fields)
SIGMA = 0.01

# --- GridLayout: two scalar sectors on a 2D periodic grid ---
layout = GridLayout(
    U=ScalarSector([0]),
    V=ScalarSector([1]),
    dim=DIM, ndim=2, bc="pbc",
)
U = layout.U   # StructuredExpr for U field
V = layout.V   # StructuredExpr for V field

# %%
# Over-specified basis
# ----------------------
#
# Rather than hand-coding the exact 7-term Gray-Scott structure, we
# build a **generic** basis: all monomials up to **order 3** in
# :math:`(U, V)`, differential operators
# :math:`\nabla^2, \nabla^4`, and **nonlinear gradient terms**
# :math:`|\nabla U|^2, |\nabla V|^2, \nabla U \cdot \nabla V`.
# This gives **17 candidate features per channel = 34 total**.
# PASTIS will prune this down to the correct 7.
#
# Labels are generated automatically from the expression tree —
# each single-feature term carries its own human-readable string
# via operator auto-labelling (e.g. ``U ** 2`` → ``"U²"``,
# ``layout.lap(U)`` → ``"∇²U"``, ``gU.dot(gV)`` → ``"∇U·∇V"``).

ONE = layout.const(1)                 # auto: "1"
gU = layout.grad(U)
gV = layout.grad(V)

terms = [
    # --- pointwise monomials ---
    ONE,                              # auto: "1"
    U, V,                             # auto: "U", "V"
    U**2, U * V, V**2,               # auto: "U²", "UV", "V²"
    U**3, U**2 * V, U * V**2, V**3,  # auto: "U³", "U²V", "UV²", "V³"
    # --- Laplacian ---
    layout.lap(U),                    # auto: "∇²U"
    layout.lap(V),                    # auto: "∇²V"
    # --- gradient product terms ---
    gU.dot(gU),                       # auto: "∇U·∇U"
    gV.dot(gV),                       # auto: "∇V·∇V"
    gU.dot(gV),                       # auto: "∇U·∇V"
    # --- biharmonic ---
    layout.biharmonic(U),             # auto: "∇⁴U"
    layout.biharmonic(V),             # auto: "∇⁴V"
]

# Concatenate features using &
generic = terms[0]
for t in terms[1:]:
    generic = generic & t

# Same candidate set for both channels
BASIS = layout.embed(rank=1, U=generic, V=generic)

# Labels are auto-derived from the expression tree
auto_labels = list(generic.labels)
n_per = len(auto_labels)   # 17
n_feat = 2 * n_per         # 34
labels = [f"{lbl}→U̇" for lbl in auto_labels] + \
         [f"{lbl}→V̇" for lbl in auto_labels]

# Ground truth: 7 non-zero out of 34
#   U channel (indices 0–16):  F·1, −F·U, −1·UV², D_U·∇²U
#   V channel (indices 17–33): +1·UV², −(F+K)·V, D_V·∇²V
theta_dense = np.zeros(n_feat)
theta_dense[0] = F_gs             # 1→U̇
theta_dense[1] = -F_gs            # U→U̇
theta_dense[8] = -1.0             # UV²→U̇
theta_dense[10] = DU              # ∇²U→U̇
theta_dense[n_per + 8] = +1.0     # UV²→V̇
theta_dense[n_per + 2] = -(F_gs + K_gs)  # V→V̇
theta_dense[n_per + 11] = DV      # ∇²V→V̇

support_true = list(np.nonzero(theta_dense)[0])
coeffs_true = theta_dense[support_true]

theta_sim = jnp.array(theta_dense, dtype=jnp.float32)

# %%
# Simulate with burn-in
# -----------------------
#
# A ``prerun`` of 200 frames (100 time units) lets the Turing
# pattern develop from the initial seed before we start recording.
# This improves inference quality because the data covers the
# developed attractor rather than a featureless transient.

noise = conserved_noise_pbc(sigma=SIGMA, grid_shape=GRID, dx=DX, n_fields=DIM)
box_extras = square_grid_extras(grid_shape=GRID, dx=DX)

# Initial condition (symmetry-broken seed in the centre)
Nx, Ny = GRID
U0 = jnp.ones((Nx, Ny), dtype=jnp.float32)
V0 = jnp.zeros((Nx, Ny), dtype=jnp.float32)
r = min(Nx, Ny) // 8
cx, cy = Nx // 2, Ny // 2
U0 = U0.at[cx - r:cx + r, cy - r:cy + r].set(0.50)
V0 = V0.at[cx - r:cx + r, cy - r:cy + r].set(0.25)

key = random.PRNGKey(SEED)
key, sub = random.split(key)
U0 = U0 + 0.02 * random.normal(sub, U0.shape, dtype=U0.dtype)
key, sub = random.split(key)
V0 = V0 + 0.02 * random.normal(sub, V0.shape, dtype=V0.dtype)

X0 = jnp.stack([U0, V0], axis=-1).reshape((Nx * Ny, DIM))

proc = OverdampedProcess(BASIS, D=noise)
proc.set_params(theta_F=theta_sim)
proc.set_extras(extras_global=box_extras)
proc.initialize(X0)

key, sub = random.split(key)
t0 = time.perf_counter()
coll = proc.simulate(
    dt=DT, Nsteps=STEPS, key=sub, prerun=PRERUN, oversampling=OVERS,
)
elapsed = time.perf_counter() - t0
print(f"Simulation: {coll.T} frames, prerun={PRERUN}  "
      f"({elapsed:.1f}s)")

# %%
# Simulated fields
# -------------------
#
# Snapshots of the U and V fields after burn-in.  By frame 0 the
# pattern is already developing; by the final frame the
# characteristic spot/labyrinth structure has emerged.

T_total = coll.T
snapshots = [0, T_total // 3, T_total - 1]

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
plot_spde_snapshot(coll, snapshots, scalar_channel=0, grid_shape=GRID,
                   cmap="viridis", vmin=0, vmax=1, axes=axes[0])
plot_spde_snapshot(coll, snapshots, scalar_channel=1, grid_shape=GRID,
                   cmap="magma", vmin=0, vmax=1, axes=axes[1])
for j, ti in enumerate(snapshots):
    axes[0, j].set_title(f"t = {(PRERUN + ti) * DT:.0f}")
axes[0, 0].set_ylabel("U field")
axes[1, 0].set_ylabel("V field")
fig.colorbar(axes[0, -1].images[0], ax=axes[0, :], shrink=0.8,
             label="Concentration")
fig.suptitle("Gray-Scott simulation: pattern formation", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Degrade and infer
# --------------------
#
# We introduce a small random pixel-loss fraction, mimicking
# missing data in experimental recordings.  The 34-feature dense
# model is inferred first, before sparsification.
#
# .. note::
#
#    Spatial coarsening (``downscale > 1``) is *not* applied here.
#    For stencil-based operators like the Laplacian, down-sampling
#    changes which neighbours the stencil sees; the coarse-grid
#    Laplacian is a different finite-difference approximation from
#    the fine-grid one that generated the data.

from SFI import OverdampedLangevinInference
from SFI.statefunc.nodes.interactions.prepare import purge_cache_extras
from SFI.trajectory.degrade import degrade_spatial_data

coll_deg = degrade_spatial_data(
    coll, downscale=1,
    data_loss_fraction=0.001, noise=0.0, bc="pbc",
)
coll_deg.extras_global = purge_cache_extras(coll_deg.extras_global)

inf = OverdampedLangevinInference(coll_deg)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(BASIS, M_mode="Ito")
inf.compute_force_error()

# %%
# PASTIS sparsification
# -----------------------
#
# From the 34-feature dense solution, **PASTIS** selects the minimal
# model that passes the significance criterion.  The 7 reaction-diffusion
# terms are always recovered; additionally, tiny biharmonic corrections
# (:math:`\nabla^4 U, \nabla^4 V` with coefficients :math:`\sim 10^{-4}`)
# may appear because the finite-difference Laplacian stencil introduces
# a small systematic :math:`O(dx^2)` numerical artefact.

inf.sparsify_force(criterion="PASTIS", p=0.001)
inf.compute_force_error()
inf.compare_to_exact(model_exact=proc, maxpoints=30)

k_sel, support_sel, _, coeffs_sel = \
    inf.force_sparsity_result.select_by_ic("PASTIS")
print(f"True support recovered: {set(support_true).issubset(set(support_sel))}")

inf.print_report()

# %%
# Pareto front and sparse recovery
# -----------------------------------
#
# Left: information gain vs model size, with information-criterion
# thresholds.  Right: inferred sparse coefficients overlaid on
# ground truth.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

plot_pareto_front(inf.force_sparsity_result, ax=axes[0])
axes[0].set_title(f"PASTIS: {k_sel} / {n_feat} features selected")

plot_recovery_bar(
    np.array(coeffs_sel), support_sel,
    coeffs_true=coeffs_true, support_true=support_true,
    labels=labels, ax=axes[1],
)
axes[1].set_title("Sparse coefficient recovery")

fig.suptitle("PASTIS model selection — Gray-Scott", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Coefficient comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~

print(model_summary(
    labels, np.array(coeffs_sel), support=support_sel,
    coeffs_true=coeffs_true, support_true=support_true,
    title="Gray-Scott sparse model: true vs inferred",
))

# %%
# Bootstrap validation
# -----------------------
#
# Re-simulate from the inferred sparse model, starting from the
# first post-burn-in frame.

X0_boot = coll.X[0]  # post-burn-in initial condition

key, sub = random.split(key)
proc_boot = inf.simulate_bootstrapped_trajectory(sub, simulate=False)
proc_boot.set_extras(extras_global=box_extras)
proc_boot.initialize(X0_boot)

key, sub = random.split(key)
_boot_ok = False
for _boot_try in range(5):
    try:
        key, sub = random.split(key)
        coll_boot = proc_boot.simulate(
            dt=DT, Nsteps=STEPS, key=sub, prerun=0, oversampling=OVERS,
        )
        _boot_ok = True
        break
    except ValueError:
        print(f"  Bootstrap attempt {_boot_try + 1} diverged, retrying...")
if not _boot_ok:
    raise RuntimeError("Bootstrap diverged after 5 attempts")

ti_final = min(coll.T, coll_boot.T) - 1

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
plot_spde_snapshot(coll, ti_final, scalar_channel=0, grid_shape=GRID,
                   cmap="viridis", vmin=0, vmax=1, axes=[axes[0, 0]])
plot_spde_snapshot(coll_boot, ti_final, scalar_channel=0, grid_shape=GRID,
                   cmap="viridis", vmin=0, vmax=1, axes=[axes[0, 1]])
plot_spde_snapshot(coll, ti_final, scalar_channel=1, grid_shape=GRID,
                   cmap="magma", vmin=0, vmax=1, axes=[axes[1, 0]])
plot_spde_snapshot(coll_boot, ti_final, scalar_channel=1, grid_shape=GRID,
                   cmap="magma", vmin=0, vmax=1, axes=[axes[1, 1]])
axes[0, 0].set_title("U — simulated")
axes[0, 0].set_ylabel("U field")
axes[0, 1].set_title("U — bootstrapped")
axes[1, 0].set_title("V — simulated")
axes[1, 0].set_ylabel("V field")
axes[1, 1].set_title("V — bootstrapped")
t_phys = (PRERUN + ti_final) * DT
fig.suptitle(f"Bootstrap comparison (t = {t_phys:.0f})", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore
# %%
# Side-by-side bootstrap movie
# --------------------------------
#
# Animated comparison of original simulation (left) and bootstrap
# resimulation from the inferred sparse model (right).  Both start
# from the same initial condition but evolve under independent noise
# realisations — the morphological statistics (spot density, ring
# thickness …) should match.

T_frames = min(coll.T, coll_boot.T)
skip_gs = max(1, T_frames // 150)

# sphinx_gallery_start_ignore
anim_gs = animate_spde_comparison(
    coll, coll_boot, field_component=0, grid_shape=GRID,
    skip=skip_gs, vmin=0, vmax=1, cmap="viridis",
    titles=("Original U", "Bootstrap U"),
)
plt.gcf().suptitle("Gray-Scott: original vs bootstrap", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Multi-regime robustness
# -------------------------
#
# The 7-term reaction-diffusion structure should be recoverable across
# different :math:`(F, K)` regimes that produce visually distinct
# patterns.  We re-run the simulate → degrade → infer → PASTIS pipeline
# on three regimes using the **same** over-specified basis, and check
# that PASTIS selects the same structural terms each time.  The
# regime-independent coefficients (:math:`D_U`, :math:`D_V`, the
# :math:`\pm 1` couplings) should stay fixed, while :math:`F` and
# :math:`(F{+}K)` track the parameter change.  The "Coral" entry reuses
# the fit from above.


def _true_theta(F, K):
    """The 7 non-zero coefficients for a given (F, K)."""
    th = np.zeros(n_feat)
    th[0] = F                       # 1   → U̇
    th[1] = -F                      # U   → U̇
    th[8] = -1.0                    # UV² → U̇
    th[10] = DU                     # ∇²U → U̇
    th[n_per + 8] = +1.0            # UV² → V̇
    th[n_per + 2] = -(F + K)        # V   → V̇
    th[n_per + 11] = DV             # ∇²V → V̇
    return th


def _seed_ic(key):
    """Symmetry-broken central seed, identical recipe to the main run."""
    U0r = jnp.ones((Nx, Ny), dtype=jnp.float32)
    V0r = jnp.zeros((Nx, Ny), dtype=jnp.float32)
    rr = min(Nx, Ny) // 8
    cxr, cyr = Nx // 2, Ny // 2
    U0r = U0r.at[cxr - rr:cxr + rr, cyr - rr:cyr + rr].set(0.50)
    V0r = V0r.at[cxr - rr:cxr + rr, cyr - rr:cyr + rr].set(0.25)
    k1, k2 = random.split(key)
    U0r = U0r + 0.02 * random.normal(k1, U0r.shape, dtype=U0r.dtype)
    V0r = V0r + 0.02 * random.normal(k2, V0r.shape, dtype=V0r.dtype)
    return jnp.stack([U0r, V0r], axis=-1).reshape((Nx * Ny, DIM))


# "Coral" is the regime fit above; add two more.
regimes = [("Spots", 0.030, 0.057), ("Coral", F_gs, K_gs), ("Holes", 0.050, 0.065)]
regime_results = {
    "Coral": dict(F=F_gs, K=K_gs, support=list(support_sel),
                  coeffs=np.array(coeffs_sel)),
}

for ri, (rname, Fr, Kr) in enumerate(regimes):
    if rname == "Coral":
        print(f"  {rname:>6} (F={Fr}, K={Kr}): reusing fit from above "
              f"({len(support_sel)} terms)")
        continue
    proc_r = OverdampedProcess(BASIS, D=noise)
    proc_r.set_params(theta_F=jnp.array(_true_theta(Fr, Kr), dtype=jnp.float32))
    proc_r.set_extras(extras_global=box_extras)
    key, sub_ic = random.split(key)
    proc_r.initialize(_seed_ic(sub_ic))
    key, sub = random.split(key)
    coll_r = proc_r.simulate(dt=DT, Nsteps=STEPS, key=sub,
                             prerun=PRERUN, oversampling=OVERS)
    coll_rd = degrade_spatial_data(
        coll_r, downscale=1, data_loss_fraction=0.001, noise=0.0, bc="pbc",
    )
    coll_rd.extras_global = purge_cache_extras(coll_rd.extras_global)
    inf_r = OverdampedLangevinInference(coll_rd)
    inf_r.compute_diffusion_constant(method="WeakNoise")
    inf_r.infer_force_linear(BASIS, M_mode="Ito")
    inf_r.compute_force_error()
    inf_r.sparsify_force(criterion="PASTIS", p=0.001)
    inf_r.compute_force_error()
    ks, sup_s, _, co_s = inf_r.force_sparsity_result.select_by_ic("PASTIS")
    regime_results[rname] = dict(F=Fr, K=Kr, support=list(sup_s),
                                 coeffs=np.array(co_s))
    print(f"  {rname:>6} (F={Fr}, K={Kr}): "
          f"true terms recovered = {set(support_true).issubset(set(sup_s))}")


def _coeff(res, idx):
    """Inferred coefficient at full-basis index idx (0 if pruned)."""
    return float(res["coeffs"][res["support"].index(idx)]) \
        if idx in res["support"] else 0.0


# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: regime-independent coefficients (should match the truth step)
const_labels = ["−1 (UV²→U̇)", "+1 (UV²→V̇)", "Dᵤ (∇²U→U̇)", "Dᵥ (∇²V→V̇)"]
const_idx = [8, n_per + 8, 10, n_per + 11]
const_true = [-1.0, +1.0, DU, DV]
plot_recovery_bar_multi(
    [[_coeff(regime_results[rname], idx) for idx in const_idx]
     for rname, _, _ in regimes],
    const_labels, coeffs_true=const_true,
    group_names=[rname for rname, _, _ in regimes], ax=axes[0],
)
axes[0].set_ylabel("Coefficient")
axes[0].set_title("Regime-independent coefficients")

# Right: F and -(F+K) track the regime
names = [r[0] for r in regimes]
xr = np.arange(len(names))
w2 = 0.35
F_true_v = [regime_results[n]["F"] for n in names]
FK_true_v = [-(regime_results[n]["F"] + regime_results[n]["K"]) for n in names]
F_inf_v = [_coeff(regime_results[n], 0) for n in names]
FK_inf_v = [_coeff(regime_results[n], n_per + 2) for n in names]
axes[1].bar(xr - w2 / 2, F_true_v, w2, color=SFI_COLORS["exact"],
            alpha=0.7, label="True F")
axes[1].bar(xr + w2 / 2, F_inf_v, w2, color=SFI_COLORS["inferred"],
            alpha=0.7, label="Inferred F")
axes[1].bar(xr - w2 / 2, FK_true_v, w2, color=SFI_COLORS["exact"],
            alpha=0.35, hatch="//")
axes[1].bar(xr + w2 / 2, FK_inf_v, w2, color=SFI_COLORS["inferred"],
            alpha=0.35, hatch="//", label="Inferred −(F+K)")
axes[1].axhline(0, color="#808080", lw=0.5, ls="--")
axes[1].set_xticks(xr)
axes[1].set_xticklabels(names, fontsize=9)
axes[1].set_ylabel("Coefficient")
axes[1].set_title("Regime-dependent coefficients")
axes[1].legend(fontsize=8)

fig.suptitle("Multi-regime recovery — Gray-Scott", fontsize=13)
plt.show()
# sphinx_gallery_end_ignore

stamp_output()
