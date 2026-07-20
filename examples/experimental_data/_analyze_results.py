# TODO: review this file
import numpy as np, matplotlib.pyplot as plt

labels  = list(BASIS_FULL.labels)
K       = len(labels)
support = np.asarray(inf.force_support)   # indices of PASTIS-selected terms

# Expand sparse → full-length coefficient vectors (zeros for inactive terms)
c_full  = np.zeros(K)
se_full = np.zeros(K)
c_full[support]  = np.asarray(inf.force_coefficients)
se_full[support] = np.asarray(inf.force_coefficients_stderr)
c_hat, stderr = c_full, se_full

# D_avg is the full (D×D) diffusion tensor; take sector diagonal means
D_avg    = np.asarray(inf.diffusion_average)
D_u      = float((D_avg[CH_UX, CH_UX] + D_avg[CH_UY, CH_UY]) / 2)
D_Q      = float((D_avg[CH_Q11, CH_Q11] + D_avg[CH_Q12, CH_Q12]) / 2)

# Paper reference values — Q sector only (Golden et al. 2023)
paper = {
    "[Ω,Q]":  +1.0,
    "E[u]":   -1.0,
    "(u·∇)Q": -1.0,
}

print(f"Diffusion constants:  D(u) = {D_u:.5f}   D(Q) = {D_Q:.5f}")
print(f"PASTIS selected {len(support)}/{K} terms: {[labels[i] for i in support]}")
print()
print(f"  {'Term':20s}  {'Inferred':>12s}  {'±stderr':>10s}  {'Paper':>8s}  {'Active?':>7s}")
print("  " + "─" * 72)
for name, c, se in zip(labels, c_hat, stderr):
    pv      = paper.get(name)
    pv_str  = f"{pv:+.2f}" if pv is not None else "   N/A"
    active  = "✓" if c != 0.0 else ""
    print(f"  {name:20s}  {c:+12.4f}  {se:10.4f}  {pv_str}  {active:>7s}")

# ── Bar chart ──────────────────────────────────────────────────────────────────
paper_vals = np.array([paper.get(n, np.nan) for n in labels], dtype=float)
x     = np.arange(K)
width = 0.35

# u-sector = first 11 terms, Q-sector = last 10 terms
n_u_terms = 11
sector_colors = ["C1"] * n_u_terms + ["C0"] * (K - n_u_terms)

fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(x - width/2, c_hat, width, yerr=stderr, capsize=4,
       color=sector_colors, alpha=0.9, label="Inferred (SFI)")
ax.bar(x + width/2, paper_vals, width,
       label="Paper (Golden 2023)", color="salmon",
       alpha=0.8, hatch="//", edgecolor="black")
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Coefficient value")
ax.set_title(f"Coupled (u, Q) SPDE — 21-term basis, PASTIS sparse ({len(support)}/{K} active)   D(u)={D_u:.4f}  D(Q)={D_Q:.4f}")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="C1", alpha=0.9, label="u-sector (inferred)"),
    Patch(facecolor="C0", alpha=0.9, label="Q-sector (inferred)"),
    Patch(facecolor="salmon", alpha=0.8, hatch="//", edgecolor="black", label="Paper (Golden 2023)"),
])
plt.tight_layout()
out = DATA_DIR / "golden2023_coefficients_full.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
