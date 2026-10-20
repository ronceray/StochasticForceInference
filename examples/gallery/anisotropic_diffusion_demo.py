r"""
Anisotropic diffusion tensor field
====================================

Recover a **position-dependent, anisotropic diffusion tensor** — and
the force — from a single 2D trajectory.  A
tracer in a ring trap experiences fluctuations whose radial component
grows with distance from the center (think of a probe in a radially
stretched gel, or near a topological defect in an active film): the
local noise ellipse rotates with position.

We fit both fields on generic polynomial dictionaries, let PASTIS decide
which terms the data support — 6 of 20 force terms, 5 of 18 tensor
terms — and read the selected fields with their error bands, including
the principal axes of the noise.

.. rubric:: Tags

synthetic · overdamped · multiplicative-noise · anisotropic · sparsity · 2D
"""

# sphinx_gallery_tags = ["synthetic", "overdamped", "multiplicative-noise", "anisotropic", "sparsity", "2D"]
# sphinx_gallery_thumbnail_number = 6

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
from SFI.utils.plotting import phase2d, plot_pareto_front, plot_tensor_field

apply_style()


def selection_figure(result, labels, support, coeffs, stderr, truth, title, legend_kw):
    """Pareto front (left) and selected coefficients / truth with ±2σ bars (right).

    ``legend_kw`` places the Pareto legend in a free corner, clear of the
    vertical selection lines.
    """
    support = [int(i) for i in support]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    plot_pareto_front(result, ax=axes[0])
    axes[0].legend(fontsize=8, **legend_kw)
    axes[0].set_title(f"Pareto front — PASTIS keeps {len(support)} / {len(labels)}")
    true_idx = [labels.index(k) for k in truth]
    shown = sorted(set(true_idx) | set(support))
    val = {i: c for i, c in zip(support, coeffs)}
    err = {i: e for i, e in zip(support, stderr)}
    ax = axes[1]
    ax.axhline(1.0, color="#808080", lw=0.8, ls="--")
    for j, i in enumerate(shown):
        if i in true_idx:
            t = truth[labels[i]]
            ax.errorbar(j, val.get(i, 0.0) / t, yerr=2 * err.get(i, 0.0) / abs(t), fmt="o",
                        ms=6, capsize=4, color=SFI_COLORS["inferred"], ecolor="#B0B0B0")
        else:  # selected but absent from the truth
            ax.plot(j, 0.0, "x", ms=8, color=SFI_COLORS["error"])
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([labels[i] for i in shown], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("inferred / true")
    ax.set_title("Selected coefficients (±2σ)")
    fig.suptitle(title, fontsize=13)
    return fig
# sphinx_gallery_end_ignore
stamp_output()

# %%
# Model and simulation
# ---------------------
#
# A particle in a ring potential :math:`U = \tfrac{k}{4}(r^2 - R^2)^2`
# with a radially anisotropic diffusion tensor (Itô convention):
#
# .. math::
#
#    D(\mathbf{x}) = D_0\,\mathbb{I}
#      + a\, r^2\, \hat{\mathbf{r}}\hat{\mathbf{r}}^{\!\top}
#    = D_0\,\mathbb{I} + a \begin{pmatrix} x^2 & xy \\ xy & y^2 \end{pmatrix}.
#
# Radial kicks grow with :math:`r`; tangential noise stays at
# :math:`D_0`.  Both fields are polynomial, so we can build them
# compositionally from coordinate and matrix-template primitives.

import SFI
from SFI.bases import identity_matrix_basis, symmetric_matrix_basis, unit_axes, x_components
from SFI.langevin import OverdampedProcess

k = 4.0       # ring trap stiffness
R = 1.5       # ring radius
D0 = 0.2      # isotropic (tangential) diffusion
a = 0.15      # radial anisotropy growth
dt = 0.005
Nsteps = 50_000

x, y = x_components(2)
ex, ey = unit_axes(2)
I = identity_matrix_basis(2)
S = symmetric_matrix_basis(2)            # templates Sxx, Sxy, Syy
Sxx, Sxy, Syy = S[0], S[1], S[2]

r2 = x**2 + y**2
F_model = (-k * (r2 - R**2)) * (x * ex + y * ey)
D_model = D0 * I + a * (x**2 * Sxx + x * y * Sxy + y**2 * Syy)

proc = OverdampedProcess(F=F_model, D=D_model,
                         theta_F=jnp.ones(F_model.n_features),
                         theta_D=jnp.ones(D_model.n_features))
proc.initialize(jnp.array([R, 0.0]))
coll = proc.simulate(dt=dt, Nsteps=Nsteps, key=random.PRNGKey(11),
                     prerun=500, oversampling=10)

# %%
# Trajectory
# -----------
#
# The tracer diffuses around the annulus, sampling all orientations of
# the local anisotropy.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(4.5, 4.5))
phase2d(coll, dims=(0, 1), ax=ax, color=SFI_COLORS["data"], linewidth=0.3, alpha=0.6)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(R * np.cos(th), R * np.sin(th), ls=":", lw=1, color="#808080")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_title("Ring-trap trajectory")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Force: fit and select
# ----------------------
#
# We do not assume the form of the trap.  The force is fitted on all
# vector monomials up to degree 3 (20 candidate terms); the truth,
# :math:`F = -k(r^2 - R^2)\,\mathbf{x}`, uses six of them.  The data are
# clean, and the stiff ring confinement can be mistaken for measurement
# noise by the automatic choice, so we select the clean-data estimators
# explicitly (``WeakNoise`` diffusion constant, ``clean`` force preset).
# :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_force` then
# runs PASTIS on the Pareto front of sparse models.

from SFI.bases import monomials_up_to
from SFI.utils.formatting import model_summary

B_force = monomials_up_to(order=3, dim=2, rank="vector")
F_labels = list(B_force.labels)
F_true = {"x0·e0": k * R**2, "x0^3·e0": -k, "(x0·x1^2)·e0": -k,
          "x1·e1": k * R**2, "(x0^2·x1)·e1": -k, "x1^3·e1": -k}

inf = SFI.OverdampedLangevinInference(coll)
inf.compute_diffusion_constant(method="WeakNoise")
inf.infer_force_linear(B_force, preset="clean")
inf.sparsify_force(criterion="PASTIS")
inf.compute_force_error()            # standard errors of the selected terms

F_sup = [int(i) for i in np.asarray(inf.force_support)]
F_true_idx = [F_labels.index(lbl) for lbl in F_true]
print(f"PASTIS keeps {len(F_sup)}/{len(F_labels)} force terms; "
      f"exactly the true ones: {sorted(F_sup) == sorted(F_true_idx)}")
print(model_summary(F_labels, np.asarray(inf.force_coefficients),
                    stderr=np.asarray(inf.force_coefficients_stderr), support=F_sup,
                    coeffs_true=np.array(list(F_true.values())), support_true=F_true_idx,
                    title="Force: selected vs true"))

# %%
# Left: the Pareto front — the best information gain at each model size;
# it climbs until the six true terms are in and is flat beyond, and
# PASTIS stops at the knee.  Right: each selected coefficient divided by
# its true value, with ±2σ error bars from
# :meth:`~SFI.inference.OverdampedLangevinInference.compute_force_error`.

# sphinx_gallery_start_ignore
selection_figure(inf.force_sparsity_result, F_labels, F_sup, np.asarray(inf.force_coefficients),
                 np.asarray(inf.force_coefficients_stderr), F_true,
                 "Force model selection — ring trap", legend_kw={"loc": "upper left"})
plt.show()
# sphinx_gallery_end_ignore

# %%
# Diffusion tensor: fit and select
# ---------------------------------
#
# The diffusion field gets the same treatment, with a rank-2 tensor
# basis: degree-2 matrix monomials spanning all symmetric-tensor fields
# with quadratic entries (18 candidates; the truth uses 5).
# :meth:`~SFI.inference.OverdampedLangevinInference.sparsify_diffusion`
# runs PASTIS on the diffusion fit, scoring each subset by its likelihood
# in the metric of the fit's own sandwich covariance, and
# :meth:`~SFI.inference.OverdampedLangevinInference.compute_diffusion_error`
# returns the covariance of the selected model.  ``M_mode="WeakNoise"``
# is the clean-data local estimator, for the same reason as above.

B_diff = monomials_up_to(order=2, dim=2, rank="symmetric_matrix")
D_labels = list(B_diff.labels)
D_true = {"1·S00": D0, "1·S11": D0, "x0^2·S00": a, "(x0·x1)·S01": a, "x1^2·S11": a}

inf.infer_diffusion_linear(B_diff, M_mode="WeakNoise")
inf.sparsify_diffusion(criterion="PASTIS")
inf.compute_diffusion_error()        # standard errors of the selected terms

D_sup = [int(i) for i in np.asarray(inf.diffusion_support)]
D_true_idx = [D_labels.index(lbl) for lbl in D_true]
print(f"PASTIS keeps {len(D_sup)}/{len(D_labels)} diffusion terms; "
      f"exactly the true ones: {sorted(D_sup) == sorted(D_true_idx)}")
print(model_summary(D_labels, np.asarray(inf.diffusion_coefficients),
                    stderr=np.asarray(inf.diffusion_coefficients_stderr), support=D_sup,
                    coeffs_true=np.array(list(D_true.values())), support_true=D_true_idx,
                    title="Diffusion: selected vs true"))

c_D, cov_D = inf.coeff_block(slice(None), field="diffusion")   # full-basis order
dev = [f"{lbl} {(c_D[i] - tv) / np.sqrt(cov_D[i, i]):+.1f}σ"
       for lbl, tv in D_true.items() for i in [D_labels.index(lbl)]]
print("deviation from truth:", ", ".join(dev))

inf.compare_to_exact(model_exact=proc)
print(f"NMSE vs truth: force {float(inf.NMSE_force):.4f}, "
      f"diffusion {float(inf.NMSE_diffusion):.4f}")

# %%
# Same reading as for the force: the Pareto front flattens once five
# terms are in (AIC, a less strict criterion, would admit a sixth).  The
# selected coefficients scatter around the truth by about their error
# bars; the largest deviation, on the :math:`x^2` part of
# :math:`D_{xx}`, is offset by the constant part of :math:`D_{xx}`, and
# its symmetric partner on :math:`D_{yy}` sits on the truth.

# sphinx_gallery_start_ignore
selection_figure(inf.diffusion_sparsity_result, D_labels, D_sup,
                 np.asarray(inf.diffusion_coefficients),
                 np.asarray(inf.diffusion_coefficients_stderr), D_true,
                 "Diffusion-tensor model selection",
                 legend_kw={"loc": "lower left", "bbox_to_anchor": (0.16, 0.02)})
plt.show()
# sphinx_gallery_end_ignore

# %%
# Tensor glyphs: exact vs inferred
# ----------------------------------
#
# Each glyph is the ellipse :math:`\mathbf{u}^{\!\top} D^{-1}\mathbf{u}
# = \text{const}` of the local tensor — long axis along the fast
# direction.  Exact field on the left, the selected model on the right:
# the radial orientation and the :math:`r^2` growth are both recovered.

# Glyph positions on a polar grid covering the sampled annulus, and a
# callable for the exact tensor field D(x) = D0 I + a r^2 r-hat r-hat^T.
r_glyph = np.linspace(0.9, 2.0, 4)
th_glyph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
pts = np.array([[r * np.cos(t), r * np.sin(t)] for r in r_glyph for t in th_glyph])


def D_exact_field(p):
    p = np.asarray(p).reshape(-1, 2)
    return np.array([D0 * np.eye(2) + a * np.outer(q, q) for q in p])


def D_inferred_field(p):
    return inf.diffusion_inferred(jnp.asarray(p))


# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
plt.sca(axes[0])
plot_tensor_field(coll, D_exact_field, mode="ellipse", positions=pts,
                  center=(0.0, 0.0), radius=2.4, scale=0.4,
                  color=SFI_COLORS["exact"])
axes[0].set_title("Exact $D(\\mathbf{x})$")
plt.sca(axes[1])
plot_tensor_field(coll, D_inferred_field, mode="ellipse", positions=pts,
                  center=(0.0, 0.0), radius=2.4, scale=0.4,
                  color=SFI_COLORS["inferred"])
axes[1].set_title("Inferred $D(\\mathbf{x})$ (selected model)")
for ax in axes:
    ax.plot(R * np.cos(th), R * np.sin(th), ls=":", lw=0.8, color="#808080")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Radial and tangential eigenvalues
# -----------------------------------
#
# Projecting the inferred tensor onto the local radial and tangential
# directions separates the two physical components: the radial
# diffusivity grows as :math:`D_0 + a r^2` while the tangential one
# stays flat at :math:`D_0`.  Curves are averaged over the polar angle;
# bands are ±2σ, propagated from the covariance of the selected
# coefficients.  The grey histogram shows where the tracer actually went:
# the bands are narrowest on the ring, where the data are.

r_prof = np.linspace(0.8, 2.1, 30)
th_s = np.linspace(0, 2 * np.pi, 24, endpoint=False)


def radial_tangential_rows(basis):
    """Linear maps coefficients -> angle-averaged D_rr(r), D_θθ(r)."""
    L_rr = np.zeros((len(r_prof), basis.n_features))
    L_tt = np.zeros_like(L_rr)
    for t_ in th_s:
        pts_ray = np.column_stack([r_prof * np.cos(t_), r_prof * np.sin(t_)])
        Bf = np.asarray(basis(jnp.array(pts_ray)))          # (n_r, 2, 2, F)
        rhat = np.array([np.cos(t_), np.sin(t_)])
        that = np.array([-np.sin(t_), np.cos(t_)])
        L_rr += np.einsum("i,nijf,j->nf", rhat, Bf, rhat) / len(th_s)
        L_tt += np.einsum("i,nijf,j->nf", that, Bf, that) / len(th_s)
    return L_rr, L_tt


L_rr, L_tt = radial_tangential_rows(B_diff)
C_D, Cov_D = inf.coeff_block(slice(None), field="diffusion")   # full-basis order
D_rr, D_tt = L_rr @ C_D, L_tt @ C_D
D_rr_err = np.sqrt(np.einsum("nf,fg,ng->n", L_rr, Cov_D, L_rr))
D_tt_err = np.sqrt(np.einsum("nf,fg,ng->n", L_tt, Cov_D, L_tt))

# sphinx_gallery_start_ignore
_, X_all, _ = coll.to_arrays()
r_data = np.linalg.norm(np.asarray(X_all)[:, 0, :], axis=-1)
fig, ax = plt.subplots(figsize=(6.5, 4))
axh = ax.twinx()
axh.hist(r_data, bins=60, range=(r_prof[0], r_prof[-1]), color="#808080", alpha=0.25)
axh.set_yticks([])
axh.spines["right"].set_visible(False)
ax.set_zorder(axh.get_zorder() + 1)
ax.patch.set_visible(False)
ax.plot(r_prof, D0 + a * r_prof**2, lw=2, color=SFI_COLORS["exact"],
        label=r"Exact $D_{rr} = D_0 + a r^2$")
ax.plot(r_prof, np.full_like(r_prof, D0), lw=2, ls=":",
        color=SFI_COLORS["exact"], label=r"Exact $D_{\theta\theta} = D_0$")
ax.plot(r_prof, D_rr, "--", lw=2, color=SFI_COLORS["inferred"],
        label="Inferred $D_{rr}$")
ax.fill_between(r_prof, D_rr - 2 * D_rr_err, D_rr + 2 * D_rr_err,
                color=SFI_COLORS["inferred"], alpha=0.25)
ax.plot(r_prof, D_tt, "--", lw=2, color=SFI_COLORS["highlight"],
        label=r"Inferred $D_{\theta\theta}$")
ax.fill_between(r_prof, D_tt - 2 * D_tt_err, D_tt + 2 * D_tt_err,
                color=SFI_COLORS["highlight"], alpha=0.25)
ax.set_xlabel("r")
ax.set_ylabel("eigenvalue of $D$")
ax.set_title("Radial vs tangential diffusivity (±2σ)")
ax.legend(fontsize=8, loc="upper left")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The same workflow applies unchanged to experimental 2D tracking data;
# see the :doc:`multiplicative-noise demo <multiplicative_diffusion_demo>`
# for localisation noise and the Itô-convention caveats that come with
# state-dependent noise.

# %%
# Thumbnail
# ---------
#
# Dedicated single-panel figure for the gallery thumbnail.

# sphinx_gallery_start_ignore
fig_thumb = plt.figure(figsize=(4, 4))
ax_t = fig_thumb.add_subplot(111)
phase2d(coll, dims=(0, 1), ax=ax_t, color=SFI_COLORS["data"], linewidth=0.2, alpha=0.35)
plt.sca(ax_t)
plot_tensor_field(coll, D_inferred_field, mode="ellipse", positions=pts,
                  center=(0.0, 0.0), radius=2.4, scale=0.4,
                  color=SFI_COLORS["inferred"])
ax_t.set_xticks([]); ax_t.set_yticks([])
ax_t.set_xlabel(""); ax_t.set_ylabel("")
plt.show()
# sphinx_gallery_end_ignore
