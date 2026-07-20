"""
Neural-network energy landscape — Müller-Brown potential
==========================================================

Infer the 2D **energy landscape** of the Müller-Brown potential surface
with a neural network (multi-layer perceptron), built entirely from
SFI's compositional basis operations, and compare with polynomial-basis
inference.  Both models fit the *potential* :math:`U(\\mathbf{x})` —
a scalar (rank-0) expression — and obtain the force by automatic
differentiation, :math:`\\mathbf{F} = -\\nabla U` via ``.d_x()``, so the
fitted force fields are conservative **by construction**.

.. note::

   This is an **advanced** example: it fits a nonlinear-in-θ force
   family with the parametric estimator (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force`), which runs
   frozen-precision L-BFGS for :class:`~SFI.statefunc.PSF` models.  Start with the main
   gallery if you are new to SFI.

This example demonstrates:

1. Building an MLP *potential* by chaining ``.rank_to_features()``,
   ``.dense()``, and ``.elementwisemap()`` on a
   :class:`~SFI.statefunc.Basis` object — a natural **dim → H → H → 1**
   architecture — and differentiating it with ``.d_x()``.
2. Running ``infer_force`` on the resulting parametric state function
   (``PSF``) — the nonlinear-in-θ L-BFGS path.
3. The same trick at zero extra cost in the **linear** route: a scalar
   monomial library differentiated with ``.d_x()`` is a gradient basis,
   so :meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear` fits :math:`U = \\sum_i c_i u_i` directly.

.. note::

   The key compositional operations are:

   - ``.rank_to_features()`` — folds spatial (rank) axes into the
     feature axis, turning a vector into a flat feature vector.
   - ``.dense(n, weight=…, bias=…)`` — learnable affine layer on
     the feature axis.
   - ``.elementwisemap(jnp.tanh)`` — activation function.
   - ``.d_x()`` — automatic spatial differentiation: a scalar (rank-0)
     potential becomes its (rank-1) gradient field, so ``-U.d_x()`` is
     the conservative force of the landscape ``U``.

   Fitting the potential rather than a generic vector field *builds the
   physics in*: what the data cannot express as a gradient shows up as
   residual, not as spurious non-conservative force terms.  See
   :doc:`/inference/stochastic_thermodynamics` for the concepts.

.. rubric:: Tags

synthetic · overdamped · nonlinear · neural-network · thermodynamics · 2D · Müller-Brown
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
import numpy as np
from jax import random

if "__file__" in dir():  # not set when run by sphinx-gallery
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gallery_utils.helpers import SFI_COLORS, apply_style, stamp_output

from SFI.utils.formatting import print_model_comparison
from SFI.utils.plotting import phase2d, plot_field, plot_field_error

apply_style()
# sphinx_gallery_end_ignore
# %%
# System: Müller-Brown potential energy surface
# -----------------------------------------------
#
# The Müller-Brown potential is a classic 2D benchmark from
# computational chemistry with three local minima connected by two
# saddle points:
#
# .. math::
#
#    V(x,y) = \alpha \sum_{k=1}^{4} A_k
#      \exp\!\bigl[a_k(x-\bar x_k)^2 + b_k(x-\bar x_k)(y-\bar y_k)
#      + c_k(y-\bar y_k)^2\bigr]
#
# The force is :math:`\mathbf{F} = -\nabla V`.  We rescale by
# :math:`\alpha = 0.01` so that forces are :math:`\mathcal{O}(1)`.

from SFI.langevin import OverdampedProcess
from SFI.statefunc import make_sf

# Standard Müller-Brown parameters
_A = jnp.array([-200.0, -100.0, -170.0, 15.0])
_a = jnp.array([-1.0, -1.0, -6.5, 0.7])
_b = jnp.array([0.0, 0.0, 11.0, 0.6])
_c = jnp.array([-10.0, -10.0, -6.5, 0.7])
_xbar = jnp.array([1.0, 0.0, -0.5, -1.0])
_ybar = jnp.array([0.0, 0.5, 1.5, 1.0])

ALPHA = 0.01  # rescaling factor


def muller_brown_potential(xy):
    """Müller-Brown potential (rescaled)."""
    x, y = xy[0], xy[1]
    exponents = (
        _a * (x - _xbar) ** 2
        + _b * (x - _xbar) * (y - _ybar)
        + _c * (y - _ybar) ** 2
    )
    return ALPHA * jnp.sum(_A * jnp.exp(exponents))


_neg_grad_V = jax.grad(lambda xy: -muller_brown_potential(xy))


def mb_force(x):
    """Force F = −∇V for the rescaled Müller-Brown potential."""
    return _neg_grad_V(x)


# Simulation parameters
D0 = 0.5
dt = 0.01
Nsteps = 15_000
seed = 42

F_exact = make_sf(mb_force, dim=2, rank=1)
proc = OverdampedProcess(F_exact, D=D0 * jnp.eye(2))
proc.initialize(jnp.array([-0.5, 1.5]))

key = random.PRNGKey(seed)
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=key, prerun=500, oversampling=10)

# %%
# Potential landscape and trajectory
# ------------------------------------
#
# The contour plot shows the three-well structure (reversed-viridis:
# bright wells, dark barriers).  Thermal noise (:math:`D = 0.5`) allows
# the particle to explore all basins — the trajectory ink concentrates
# where the particle dwells, i.e. in the wells.

_, X_full, _ = coll.to_arrays(dataset=0)   # (T,), (T, N, d), (T, N)
X_traj = np.asarray(X_full[:, 0, :])       # single particle -> (T, 2)

# Bounding box from trajectory (with margin) — used throughout
pad = 0.15
xlo, xhi = float(X_traj[:, 0].min()) - pad, float(X_traj[:, 0].max()) + pad
ylo, yhi = float(X_traj[:, 1].min()) - pad, float(X_traj[:, 1].max()) + pad

# Evaluation grid for potential
xg = np.linspace(xlo, xhi, 120)
yg = np.linspace(ylo, yhi, 120)
XG, YG = np.meshgrid(xg, yg)
pts_grid = jnp.stack([XG.ravel(), YG.ravel()], axis=-1)
V_grid = np.asarray(jax.vmap(muller_brown_potential)(pts_grid)).reshape(XG.shape)

# Data-consistent scales: evaluate along trajectory to set contour/quiver ranges
_X_sub = jnp.array(X_traj[::10])
V_data = np.asarray(jax.vmap(muller_brown_potential)(_X_sub))
F_data_mag = np.linalg.norm(np.asarray(F_exact(_X_sub)), axis=-1)
F_clip = float(np.percentile(F_data_mag, 99)) * 2  # ceiling for quiver arrows
V_lo, V_hi = float(V_data.min()), float(V_data.max())
V_margin = 0.3 * (V_hi - V_lo)
levels = np.linspace(V_lo - V_margin, V_hi + V_margin, 40)

fig, ax = plt.subplots(figsize=(7, 6))
cs = ax.contourf(XG, YG, V_grid, levels=levels, cmap="viridis_r", extend="both")
phase2d(coll, dims=(0, 1), color="#14213d", alpha=0.15, linewidth=0.3, ax=ax)
ax.set_title("Müller-Brown potential & trajectory")
plt.colorbar(cs, ax=ax, label=r"$V(x, y)$")
plt.show()

# %%
# Polynomial potential (linear) inference — baseline
# ----------------------------------------------------
#
# Write the potential on a scalar monomial library,
# :math:`U = \sum_i c_i\, u_i` — the constant is dropped, its gradient
# is the zero feature — and differentiate the *library* once:
# :math:`\mathbf{b}_i = -\nabla u_i` via ``.d_x()``.  The force model
# :math:`\mathbf{F} = \sum_i c_i \mathbf{b}_i` is still **linear in the
# coefficients**, so the closed-form estimator applies unchanged and
# returns the energy coefficients directly.  Monomials up to degree 6
# give 27 potential features (a degree-5 force — the conservative
# constraint costs almost nothing relative to a free degree-5 vector
# fit); the polynomial captures smooth, low-order trends but cannot
# represent the sharp Gaussian channels of the Müller-Brown surface.

from SFI import OverdampedLangevinInference
from SFI.bases import monomials_up_to

poly_order = 6
U_poly_lib = monomials_up_to(
    order=poly_order, dim=2, include_constant=False, rank="scalar"
)
B_poly = -(U_poly_lib.d_x())   # b_i = -grad u_i  (rank-0 -> rank-1)

inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()

inf.infer_force_linear(B_poly, M_mode="Ito")
inf.compare_to_exact(model_exact=proc, maxpoints=5000)
nmse_poly = float(inf.NMSE_force)
force_poly = inf.force_inferred
theta_poly = jnp.asarray(inf.force_coefficients_full)


def U_poly(pts):
    """Fitted polynomial potential U = sum_i c_i u_i."""
    return np.asarray(U_poly_lib(pts)) @ np.asarray(theta_poly)


inf.print_report()

# %%
# Neural-network architecture (MLP potential)
# ----------------------------------------------
#
# We build a two-hidden-layer MLP *energy* entirely within SFI's
# expression tree:
#
# 1. **Start from position** — ``X(dim=2)`` is a rank-1 basis with 1
#    feature, representing the position vector :math:`\mathbf{x} \in \mathbb{R}^2`.
# 2. **Flatten to features** — ``.rank_to_features()`` folds the spatial
#    axis into features, giving a rank-0 expression with ``dim`` features.
#    Now :math:`(x, y)` lives on the feature axis where dense layers operate.
# 3. **Hidden layers** — ``dense(32) → tanh → dense(32) → tanh``.
# 4. **Scalar head** — ``dense(1, bias=None)`` produces the potential
#    :math:`U_\theta(\mathbf{x})`: rank-0, one feature.  No output bias:
#    a constant shift of the energy is pure gauge (the force cannot see
#    it), so we do not fit one.
# 5. **Differentiate** — ``-U.d_x()`` is the conservative force
#    :math:`\mathbf{F}_\theta = -\nabla U_\theta` — exactly the PSF
#    shape :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` expects.
#
# This gives a natural **2 → 64 → 64 → 1** MLP for the energy landscape,
# and the physics guarantees the fitted force is a gradient field.

from SFI.bases import X

dim = 2
H = 32  # hidden layer width

mlp_U = (
    X(dim=dim)                                            # rank-1, 1 feature
    .rank_to_features()                                   # rank-0, dim features
    .dense(H, weight="W1", bias="b1")                    # rank-0, H features
    .elementwisemap(jnp.tanh)                             # activation
    .dense(H, weight="W2", bias="b2")                    # rank-0, H features
    .elementwisemap(jnp.tanh)                             # activation
    .dense(1, weight="W3", bias=None)                    # rank-0, 1 feature: U(x)
)
mlp = -(mlp_U.d_x())                                      # rank-1: F = -grad U

n_params = mlp.template.size
print(f"MLP architecture: {dim} → {H} → {H} → 1 (scalar U)   ({n_params} parameters)")

# %%
# Parameter initialisation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Xavier/Glorot initialisation breaks weight symmetry and prevents
# dead neurons at start-up.  Biases are set to zero.

theta0 = {}
init_key = random.PRNGKey(123)

for name, shape in [
    ("W1", (dim, H)),  ("b1", (H,)),
    ("W2", (H, H)),   ("b2", (H,)),
    ("W3", (H, 1)),
]:
    init_key, subkey = random.split(init_key)
    if name.startswith("W"):
        fan_in, fan_out = shape
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        theta0[name] = std * random.normal(subkey, shape)
    else:
        theta0[name] = jnp.zeros(shape)

# %%
# NN potential inference (nonlinear optimisation)
# -------------------------------------------------
#
# For a (nonlinear-in-θ) :class:`~SFI.statefunc.PSF` the parametric :meth:`~SFI.inference.OverdampedLangevinInference.infer_force` minimises
# the exact banded NLL of the single-step flow residuals with
# frozen-precision L-BFGS, re-profiling ``(D, Λ)`` once at the fitted
# parameters.  We raise the inner L-BFGS budget for the NN landscape.
# A fresh inference object keeps the NN fit cleanly separated from the
# linear baseline.

# ``inner="lbfgs"`` selects the frozen-precision route explicitly; for a
# nonlinear-in-θ PSF ``inner="auto"`` already resolves here (Gauss–Newton
# is reserved for linear-in-θ bases, where its errors-in-variables
# instrument is safe).  The inner L-BFGS budget is deliberately *shallow*:
# deep inner solves against the provisional frozen precision overfit the
# wrong metric before the (D, Λ) reprofile can correct it (the classic
# IRLS trap; quantified in the companion NN study).
inf_nn = OverdampedLangevinInference(coll)
inf_nn.infer_force(
    mlp, theta0,
    inner="lbfgs",
    inner_maxiter=60,
    max_outer=2,
)

inf_nn.compare_to_exact(model_exact=proc, maxpoints=5000)
nmse_nn = float(inf_nn.NMSE_force)

nn_info = inf_nn.metadata["force_parametric_info"]
inf_nn.print_report()
print(f"L-BFGS IRLS: {nn_info['outer_iterations']} outer steps, "
      f"best loss = {nn_info['loss']:.6g}")

# %%
# Last-layer Gauss–Newton polish
# --------------------------------
#
# The recommended finishing move: freeze the warm-started network
# *body* and refit the final layer as a **linear basis** through the
# fast Gauss–Newton path.  The hidden activations become scalar
# *potential* features :math:`z_h(\mathbf{x})` (we add the two linear
# tilts :math:`x, y`), and one ``.d_x()`` turns them into gradient
# force features — the last layer's weights are ordinary linear
# coefficients, solved in seconds with proper error bars, and the
# polished force is still exactly conservative.

from SFI.statefunc import make_basis

theta_nn = mlp.unflatten_params(inf_nn.force_coefficients_full)


def body_scalars(x, *, mask=None, extras=None):
    z = jnp.tanh(theta_nn["W1"].T @ x + theta_nn["b1"])
    z = jnp.tanh(theta_nn["W2"].T @ z + theta_nn["b2"])
    return jnp.concatenate([z, x])        # (H + dim,) potential features


Z_last = make_basis(body_scalars, dim=dim, rank=0, n_features=H + dim)
B_last = -(Z_last.d_x())                  # gradient force features

inf_polish = OverdampedLangevinInference(coll)
inf_polish.infer_force(B_last, eiv=False)   # clean data: symmetric GN
inf_polish.compare_to_exact(model_exact=proc, maxpoints=5000)
nmse_polish = float(inf_polish.NMSE_force)
force_nn = inf_polish.force_inferred        # use the polished field below
theta_last = np.asarray(inf_polish.force_coefficients_full)


def U_nn(pts):
    """Polished NN potential U = sum_h c_h z_h + c_x x + c_y y."""
    return np.asarray(Z_last(pts)) @ theta_last


inf_polish.print_report()
nmse_nn = min(nmse_nn, nmse_polish)

# %%
# Force field comparison
# -------------------------
#
# Quiver plots of the true, polynomial, and NN force fields on a
# regular 2D grid clipped to the region explored by the trajectory.
# The neural network closely tracks the true force in the narrow
# saddle regions where the polynomial deteriorates.

# Quiver grid: ``plot_field`` drops arrows in unvisited cells
# (``mask_unvisited``) and caps each arrow at ``F_clip`` (``clip_magnitude``),
# so all three panels share one arrow scale.
Nq = 20
_rad = 0.5 * float((X_traj.max(axis=0) - X_traj.min(axis=0)).max())
arrow_scale = 1.6 * _rad / (Nq - 1)   # longest arrow ≈ one grid cell

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

titles = [
    "True force",
    f"Polynomial potential (deg {poly_order})\nNMSE = {nmse_poly:.3f}",
    f"Neural-network potential\nNMSE = {nmse_nn:.3f}",
]
fields = [F_exact, force_poly, force_nn]
colors = [SFI_COLORS["exact"], SFI_COLORS["inferred"], SFI_COLORS["highlight"]]

for ax, title, field, color in zip(axes, titles, fields, colors):
    ax.contourf(XG, YG, V_grid, levels=levels, cmap="viridis_r", alpha=0.35,
                extend="both")
    plt.sca(ax)
    plot_field(
        coll, field, N=Nq, color=color,
        mask_unvisited=True, clip_magnitude=F_clip,
        autoscale=True, scale=arrow_scale,
    )
    ax.set_title(title)
    ax.set_xlabel("x")

axes[0].set_ylabel("y")
fig.suptitle("Force field comparison — Müller-Brown potential", fontsize=14)
plt.show()

# %%
# Inferred energy landscapes
# ----------------------------
#
# Because both models parametrize :math:`U` itself, the landscape is a
# *direct read-out* of the fit — no line integration needed.  Energies
# are gauge-fixed by subtracting each surface's mean over the visited
# region before comparing (an additive constant is unobservable).  The
# polynomial extrapolates wildly outside the data support (contours
# saturate); the NN stays close to the true three-well topography.

_gauge_pts = jnp.array(X_traj[::10])
U_true_g = np.asarray(jax.vmap(muller_brown_potential)(_gauge_pts)).mean()

U_panels = [
    ("True potential", V_grid - U_true_g),
    (f"Polynomial potential (deg {poly_order})",
     U_poly(pts_grid).reshape(XG.shape) - U_poly(_gauge_pts).mean()),
    ("Neural-network potential",
     U_nn(pts_grid).reshape(XG.shape) - U_nn(_gauge_pts).mean()),
]
levels_g = levels - U_true_g

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for ax, (title, U_grid) in zip(axes, U_panels):
    csU = ax.contourf(XG, YG, U_grid, levels=levels_g, cmap="viridis_r",
                      extend="both")
    ax.contour(XG, YG, U_grid, levels=levels_g[::4], colors="white",
               linewidths=0.4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(csU, ax=axes, label=r"$U(x,y) - \langle U \rangle_{\rm data}$",
             fraction=0.02)
fig.suptitle("Inferred energy landscapes", fontsize=14)
plt.show()

# %%
# Point-wise force error
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The error map highlights where each model fails.  The polynomial
# concentrates error near saddle points and channel walls; the NN
# distributes residual error more uniformly and at a lower level.
# Cells outside the sampled region saturate at the colour ceiling —
# both models extrapolate freely there, and the true force itself
# diverges toward the box corners.

# Shared colour ceiling so the two error maps are directly comparable.
# Set it from the *visited* region (99th percentile along the
# trajectory): the true Müller-Brown force blows up at the unvisited
# box corners, and a max over the full bounding box would flatten all
# of the interesting structure.
_epts = jnp.array(X_traj[::5])
_Fe = np.asarray(F_exact(_epts))
err_vmax = float(max(
    np.percentile(np.linalg.norm(np.asarray(force_poly(_epts)) - _Fe, axis=-1), 99),
    np.percentile(np.linalg.norm(np.asarray(force_nn(_epts)) - _Fe, axis=-1), 99),
))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
plot_field_error(coll, force_poly, F_exact, ax=axes[0], cmap="magma_r", vmax=err_vmax)
axes[0].set_title(f"|F_inferred − F_true|:  Polynomial (deg {poly_order})")
plot_field_error(coll, force_nn, F_exact, ax=axes[1], cmap="magma_r", vmax=err_vmax)
axes[1].set_title("|F_inferred − F_true|:  Neural network")
plt.show()

# %%
# Summary
# --------
#
# The MLP energy landscape captures the non-polynomial Gaussian
# structure of the Müller-Brown surface more faithfully than a degree-6
# monomial potential.  Parametric SFI refines the polynomial estimate
# using RK4 splitting and Gauss–Newton, improving accuracy without
# switching to a neural-network architecture — and every model in the
# table is conservative by construction.

import time as _time

F_psf_poly = B_poly.to_psf()
theta0_parametric = {"coeff": theta_poly}

inf_parametric = OverdampedLangevinInference(coll)
t0_parametric = _time.perf_counter()
inf_parametric.infer_force(F_psf_poly, theta0_parametric)
t_parametric = _time.perf_counter() - t0_parametric
inf_parametric.compare_to_exact(model_exact=proc, maxpoints=5000)

print()
print(print_model_comparison(
    [inf, inf_parametric, inf_polish],
    [f"Poly (deg {poly_order})", "Poly + Parametric SFI", "NN (MLP)"],
    metrics=["n_params", "NMSE_force"],
    extra_cols={"Time (s)": {"Poly + Parametric SFI": round(t_parametric, 1)}},
))
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

stamp_output()
