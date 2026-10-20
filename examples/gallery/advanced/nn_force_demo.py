r"""
Neural-network energy landscape — Müller–Brown potential
=========================================================

Can a small neural network learn an **energy landscape** from a single
trajectory?  A particle hops between the three wells of the Müller–Brown
surface; we parametrize its potential by a network built from SFI's
compositional basis operations and obtain the force as minus its
gradient, by automatic differentiation — every fitted force is
conservative **by construction**.

In symbols, the network is a potential :math:`U_\theta(\mathbf{x})` with
weights :math:`\theta`, and the force
:math:`\mathbf{F} = -\nabla U_\theta` comes from its ``.d_x()``
derivative.  The benchmark is the standard linear route — a polynomial
potential — on the same data:

1. **Polynomial potential** (closed-form linear fit): fast and with error
   bars, but its smooth monomials cannot follow the narrow wells.  Its
   *predicted* error (sampling only) is far below the error it actually
   makes: the gap is basis bias.
2. **Neural-network potential** (parametric fit, nonlinear in the
   weights), finished by a **linear refit of its last layer**, which
   restores the closed-form error bars of the linear route.

Finally, a trajectory simulated from the learned landscape hops
between the same wells as the data.

.. note::

   This is an **advanced** example: the network is fitted with the
   parametric estimator
   (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`).
   Start with the main gallery if you are new to SFI.

.. rubric:: Tags

synthetic · overdamped · nonlinear · neural-network · thermodynamics · 2D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "nonlinear", "neural-network", "thermodynamics", "2D"]
# sphinx_gallery_thumbnail_number = 5

# sphinx_gallery_start_ignore
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from jax import random
from matplotlib.colors import ListedColormap

if "__file__" in dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

from SFI.utils.formatting import print_model_comparison
from SFI.utils.plotting import animate_particles_comparison, dark_fig, phase2d, plot_field_error

apply_style()
# Movies are encoded on the figure's own (black) background, so their light
# titles stay readable; static figures keep the transparent gallery style.
plt.rcParams["savefig.facecolor"] = "auto"
# sphinx_gallery_end_ignore
stamp_output()

# %%
# A particle on the Müller–Brown surface
# --------------------------------------
#
# The Müller–Brown potential is a classic benchmark from computational
# chemistry: three wells joined by two saddles,
#
# .. math::
#
#    V(x,y) = \alpha \sum_{k=1}^{4} A_k
#      \exp\!\bigl[a_k(x-\bar x_k)^2 + b_k(x-\bar x_k)(y-\bar y_k)
#      + c_k(y-\bar y_k)^2\bigr],
#
# rescaled by :math:`\alpha = 0.01` so that forces are
# :math:`\mathcal{O}(1)`.  With :math:`D = 0.2` the barriers are a few
# :math:`k_BT`: the particle dwells in each well and repeatedly hops
# between them over the 150 time units of the recording.

from SFI.langevin import OverdampedProcess
from SFI.statefunc import make_sf

# Standard Müller–Brown parameters
_A = jnp.array([-200.0, -100.0, -170.0, 15.0])
_a = jnp.array([-1.0, -1.0, -6.5, 0.7])
_b = jnp.array([0.0, 0.0, 11.0, 0.6])
_c = jnp.array([-10.0, -10.0, -6.5, 0.7])
_xbar = jnp.array([1.0, 0.0, -0.5, -1.0])
_ybar = jnp.array([0.0, 0.5, 1.5, 1.0])
ALPHA = 0.01


def muller_brown_potential(xy):
    """Müller–Brown potential (rescaled)."""
    x, y = xy[0], xy[1]
    exponents = (
        _a * (x - _xbar) ** 2
        + _b * (x - _xbar) * (y - _ybar)
        + _c * (y - _ybar) ** 2
    )
    return ALPHA * jnp.sum(_A * jnp.exp(exponents))


F_exact = make_sf(jax.grad(lambda xy: -muller_brown_potential(xy)), dim=2, rank=1)

D0 = 0.2
dt = 0.01

proc = OverdampedProcess(F_exact, D=D0 * jnp.eye(2))
proc.initialize(jnp.array([-0.5, 1.5]))
coll = proc.simulate(dt=dt, Nsteps=15_000, key=random.PRNGKey(42), prerun=500, oversampling=10)

# %%
# Bright = low energy.  The trajectory concentrates in the three wells,
# with thin threads across the saddles.

# sphinx_gallery_start_ignore
X_traj = np.asarray(coll.to_array()[:, 0, :])   # (T, 2) single-particle positions
pad = 0.15
xlo, xhi = float(X_traj[:, 0].min()) - pad, float(X_traj[:, 0].max()) + pad
ylo, yhi = float(X_traj[:, 1].min()) - pad, float(X_traj[:, 1].max()) + pad
xg = np.linspace(xlo, xhi, 120)
yg = np.linspace(ylo, yhi, 120)
XG, YG = np.meshgrid(xg, yg)
pts_grid = jnp.stack([XG.ravel(), YG.ravel()], axis=-1)
V_grid = np.asarray(jax.vmap(muller_brown_potential)(pts_grid)).reshape(XG.shape)

# Colour scale set from the visited region
_X_sub = jnp.array(X_traj[::10])
V_data = np.asarray(jax.vmap(muller_brown_potential)(_X_sub))
V_lo, V_hi = float(V_data.min()), float(V_data.max())
levels = np.linspace(V_lo - 0.3 * (V_hi - V_lo), V_hi + 0.3 * (V_hi - V_lo), 40)

fig, ax = plt.subplots(figsize=(6.5, 5.2))
cs = ax.contourf(XG, YG, V_grid, levels=levels, cmap="viridis_r", extend="both")
phase2d(coll, dims=(0, 1), color="#14213d", alpha=0.15, linewidth=0.3, ax=ax)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Müller–Brown potential and the recorded trajectory")
cb = plt.colorbar(cs, ax=ax, label=r"$V(x, y)$", shrink=0.85)
cb.set_ticks([t for t in mticker.MaxNLocator(6).tick_values(levels[0], levels[-1])
              if levels[0] <= t <= levels[-1]])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Polynomial potential — the linear baseline
# ------------------------------------------
#
# Write the potential on scalar monomials, :math:`U = \sum_i c_i\, u_i`,
# and differentiate the *library* once: :math:`\mathbf{b}_i = -\nabla u_i`
# (``.d_x()``).  The force is still linear in the coefficients, so the
# closed-form estimator applies and returns the energy coefficients
# directly.  The data are clean, so we use the ``"clean"`` preset and the
# ``WeakNoise`` diffusion estimator (the drift is strong compared with the
# noise here).

from SFI import OverdampedLangevinInference
from SFI.bases import monomials_up_to

poly_order = 5
U_poly_lib = monomials_up_to(order=poly_order, dim=2, include_constant=False, rank="scalar")
B_poly = -(U_poly_lib.d_x())   # b_i = -grad u_i  (rank-0 -> rank-1)

inf_poly = OverdampedLangevinInference(coll)
inf_poly.compute_diffusion_constant(method="WeakNoise")
inf_poly.infer_force_linear(B_poly, preset="clean")
inf_poly.compute_force_error()
inf_poly.compare_to_exact(model_exact=proc)
theta_poly = np.asarray(inf_poly.force_coefficients_full)


def U_poly(pts):
    """Fitted polynomial potential U = sum_i c_i u_i."""
    return np.asarray(U_poly_lib(pts)) @ theta_poly


print(f"polynomial ({theta_poly.size} terms): predicted NMSE = {inf_poly.force_predicted_MSE:.4f},"
      f"  NMSE vs truth = {float(inf_poly.NMSE_force):.3f}")

# %%
# The error against the truth is far above the predicted (sampling-only)
# error: with this much data the limitation is the *basis*, not the
# sampling.
#
# Neural-network potential
# ------------------------
#
# A two-hidden-layer network built in SFI's expression tree,
# ``2 → 16 → 16 → 1``:
#
# - ``X(dim=2).rank_to_features()`` puts :math:`(x, y)` on the feature axis;
# - ``.dense(n, weight=…, bias=…)`` is a learnable affine layer and
#   ``.elementwisemap(jnp.tanh)`` the activation;
# - the scalar head (no bias — an energy offset is unobservable) is the
#   potential :math:`U_\theta`, and ``-U.d_x()`` its conservative force.

from SFI.bases import X

H = 16  # hidden width

mlp_U = (
    X(dim=2).rank_to_features()
    .dense(H, weight="W1", bias="b1").elementwisemap(jnp.tanh)
    .dense(H, weight="W2", bias="b2").elementwisemap(jnp.tanh)
    .dense(1, weight="W3", bias=None)      # U(x): rank-0, 1 feature
)
mlp = -(mlp_U.d_x())                        # F = -grad U

# Glorot initialisation of the weights, zero biases.
theta0, init_key = {}, random.PRNGKey(123)
for name, shape in [("W1", (2, H)), ("b1", (H,)), ("W2", (H, H)), ("b2", (H,)), ("W3", (H, 1))]:
    init_key, sub = random.split(init_key)
    theta0[name] = (jnp.sqrt(2.0 / sum(shape)) * random.normal(sub, shape)
                    if name.startswith("W") else jnp.zeros(shape))
print(f"MLP potential: {mlp.template.size} parameters")

# %%
# The weights enter nonlinearly, so the fit uses the parametric
# estimator, here with its L-BFGS solver (``inner="lbfgs"``): on a
# saturating network the default Gauss–Newton path diverges and would
# fall back to L-BFGS anyway.  A short optimisation budget
# (``max_outer``, ``inner_maxiter``) is enough because the last-layer
# refit below finishes the job.

inf_nn = OverdampedLangevinInference(coll)
inf_nn.infer_force(mlp, theta0, inner="lbfgs", max_outer=2, inner_maxiter=60)
inf_nn.compare_to_exact(model_exact=proc)
print(f"neural network (optimiser output): NMSE vs truth = {float(inf_nn.NMSE_force):.3f}")

# %%
# Last-layer refit: error bars for the network
# --------------------------------------------
#
# Freeze the trained hidden layers and treat their activations
# :math:`z_h(\mathbf{x})` (plus the linear tilts :math:`x, y`) as a
# scalar potential library — exactly like the monomials above.  One
# ``.d_x()`` turns it into gradient force features, and the closed-form
# linear estimator refits the last layer in seconds, with error bars and
# a predicted error.  This is the network result we report.

from SFI.statefunc import make_basis

theta_nn = mlp.unflatten_params(inf_nn.force_coefficients_full)


def hidden_features(x, *, mask=None, extras=None):
    z = jnp.tanh(theta_nn["W1"].T @ x + theta_nn["b1"])
    z = jnp.tanh(theta_nn["W2"].T @ z + theta_nn["b2"])
    return jnp.concatenate([z, x])          # (H + 2,) potential features


Z_last = make_basis(hidden_features, dim=2, rank=0, n_features=H + 2)
B_last = -(Z_last.d_x())

inf_last = OverdampedLangevinInference(coll)
inf_last.compute_diffusion_constant(method="WeakNoise")
inf_last.infer_force_linear(B_last, preset="clean")
inf_last.compute_force_error()
inf_last.compare_to_exact(model_exact=proc)
theta_last = np.asarray(inf_last.force_coefficients_full)


def U_nn(pts):
    """Network potential U = sum_h c_h z_h + c_x x + c_y y."""
    return np.asarray(Z_last(pts)) @ theta_last


print(print_model_comparison(
    [inf_poly, inf_nn, inf_last],
    [f"Polynomial (deg {poly_order})", "NN (optimiser)", "NN + last-layer refit"],
    metrics=["n_params", "force_predicted_MSE", "NMSE_force"],
))

# %%
# The network cuts the error about tenfold, with 18 refitted
# coefficients on top of its frozen hidden layers (the refit also
# improves on the optimiser's own output).  Its realised error is still
# a few times the predicted one, which counts sampling noise only: a
# 16-unit network remains an approximation of the sharp wells.
#
# Inferred energy landscapes
# --------------------------
#
# Both models parametrize :math:`U` itself, so the landscape is read off
# directly.  Each surface is shifted by its mean over the visited region
# (an additive constant is unobservable).  The polynomial cannot carve
# the shallow middle well; the network recovers the three-well
# topography.  Far from the data, both extrapolate freely.

# sphinx_gallery_start_ignore
_gauge_pts = jnp.array(X_traj[::10])
U_true_g = np.asarray(jax.vmap(muller_brown_potential)(_gauge_pts)).mean()
U_true_grid = V_grid - U_true_g
U_nn_grid = U_nn(pts_grid).reshape(XG.shape) - U_nn(_gauge_pts).mean()
U_panels = [
    ("True potential", U_true_grid),
    (f"Polynomial (degree {poly_order})",
     U_poly(pts_grid).reshape(XG.shape) - U_poly(_gauge_pts).mean()),
    ("Neural network", U_nn_grid),
]
levels_g = levels - U_true_g

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
for ax, (title, U_grid) in zip(axes, U_panels):
    csU = ax.contourf(XG, YG, U_grid, levels=levels_g, cmap="viridis_r", extend="both")
    ax.contour(XG, YG, U_grid, levels=levels_g[::4], colors="white", linewidths=0.4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_aspect("equal")
axes[0].set_ylabel("$y$")
cb = fig.colorbar(csU, ax=axes, label=r"$U - \langle U \rangle_{\rm data}$", fraction=0.02)
cb.set_ticks([t for t in mticker.MaxNLocator(6).tick_values(levels_g[0], levels_g[-1])
              if levels_g[0] <= t <= levels_g[-1]])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Point-wise force error
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The polynomial's error concentrates on the steep well walls and
# saddles; the network's is lower and more uniform.  Both colour scales
# are capped at the 99th percentile of the errors along the trajectory,
# so cells far outside the data (where both models extrapolate)
# saturate.

force_poly = inf_poly.force_inferred
force_nn = inf_last.force_inferred

# sphinx_gallery_start_ignore
_epts = jnp.array(X_traj[::5])
_Fe = np.asarray(F_exact(_epts))
err_vmax = float(max(
    np.percentile(np.linalg.norm(np.asarray(force_poly(_epts)) - _Fe, axis=-1), 99),
    np.percentile(np.linalg.norm(np.asarray(force_nn(_epts)) - _Fe, axis=-1), 99),
))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, fld, ttl in [(axes[0], force_poly, f"Polynomial (degree {poly_order})"),
                     (axes[1], force_nn, "Neural network")]:
    plot_field_error(coll, fld, F_exact, ax=ax, cmap="magma_r", vmax=err_vmax)
    ax.set_title(ttl)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
for cax in fig.axes[2:]:
    cax.set_ylabel(r"$|\hat{\mathbf{F}} - \mathbf{F}_{\rm true}|$")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Bootstrap: does the learned landscape hop like the data?
# --------------------------------------------------------
#
# A trajectory simulated from the network's force and the inferred
# diffusion should dwell in the same wells and hop across the same
# saddles.  Left: the data on the true landscape; right: the bootstrap on
# the learned one.

coll_boot, _ = inf_last.simulate_bootstrapped_trajectory(random.PRNGKey(1), oversampling=10)

# sphinx_gallery_start_ignore
fig, axes = dark_fig(1, 2, figsize=(6.6, 3.4), sharex=True, sharey=True)
for ax, U_grid, ttl in [(axes[0], U_true_grid, "data, true landscape"),
                        (axes[1], U_nn_grid, "bootstrap, learned landscape")]:
    ax.contourf(XG, YG, U_grid, levels=levels_g, cmap="viridis_r", extend="both", alpha=0.85)
    ax.set_title(ttl, fontsize=9)
anim_boot = animate_particles_comparison(
    [coll, coll_boot], axes=axes, trail=1200, skip=75, interval=70, s=30,
    cmap=ListedColormap([SFI_COLORS["error"]]),
)
for ax in axes:
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xticks([]); ax.set_yticks([])
    for ln in ax.lines:                       # darker trails read better on the bright wells
        ln.set_color("#14213d"); ln.set_alpha(0.6); ln.set_linewidth(0.9)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Thumbnail
# ---------

# sphinx_gallery_start_ignore
fig_thumb, axes_t = plt.subplots(1, 2, figsize=(6, 3))
plot_field_error(coll, force_poly, F_exact, ax=axes_t[0], cmap="magma_r", vmax=err_vmax)
axes_t[0].set_title(f"Poly (deg {poly_order})", fontsize=9)
plot_field_error(coll, force_nn, F_exact, ax=axes_t[1], cmap="magma_r", vmax=err_vmax)
axes_t[1].set_title("Neural network", fontsize=9)
for _axt in axes_t:
    _axt.set_xticks([]); _axt.set_yticks([])
    _axt.set_xlabel(""); _axt.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
