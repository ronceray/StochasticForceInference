"""Side-by-side RK4-decision figure: minimal-Euler vs RK4·n1 vs RK4·n4, with the
linear and legacy methods as reference, for the OD Lorenz and UD Van der Pol x10
benchmarks.

NMSE(force) vs dt_eff (log-log), one column per noise level, one row per system.
Linear/legacy come from the full-budget sweeps (dense, dashed); the core variants
from the budget-matched sweep_core.jsonl (markers).  Blow-ups (NMSE>1) are drawn
at the top clip line so divergence is visible rather than off-scale.

    python -m scripts.render_rk4_decision
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CLIP = 3.0  # NMSE display ceiling; values above are drawn here (divergence marker)


def _load(path):
    p = REPO / path
    if not p.exists():
        return []
    out = []
    for line in p.open():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _curve(rows, method, noise, dt_base):
    """median NMSE vs dt_eff for one method at one noise level."""
    by_dtf = defaultdict(list)
    for r in rows:
        if r.get("method") != method or r.get("error") is not None:
            continue
        if abs(float(r.get("noise", -9)) - noise) > 1e-9:
            continue
        try:
            v = float(r["NMSE_force"])
        except (TypeError, ValueError, KeyError):
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_dtf[int(r["dt_factor"])].append(v)
    xs, ys = [], []
    for dtf in sorted(by_dtf):
        xs.append(dtf * dt_base)
        ys.append(float(np.median(by_dtf[dtf])))
    return np.array(xs), np.array(ys)


SYSTEMS = [
    dict(name="Lorenz (OD)", dt_base=1e-3,
         linear=("benchmark_results/lorenz_method_evolution/sweep.jsonl", "sfi_v1_5", "best-linear"),
         legacy=("benchmark_results/lorenz_method_evolution/sweep.jsonl", "sfi_v2_window", "legacy"),
         core="benchmark_results/lorenz_method_evolution/sweep_core.jsonl"),
    dict(name="Van der Pol (UD)", dt_base=1e-3,
         linear=("benchmark_results/vdp_uli_10x/sweep.jsonl", "rect_antic", "linear (rect·antic)"),
         legacy=("benchmark_results/vdp_uli_10x/sweep_parametric_v2.jsonl", "infer_force", "legacy"),
         core="benchmark_results/vdp_uli_10x/sweep_core.jsonl"),
]
CORE = [("core_euler", "Euler·n₁ (no skip)", "#f1a340", "o"),
        ("core_rk4_n1", "minimal RK4·n₁", "#01665e", "s"),
        ("core_rk4_n2", "RK4·n₂", "#35978f", "D")]
# Per-system noise facets are picked from the available core data (the two
# standard grids use different noise ladders), nearest to these targets:
NOISE_TARGETS = [0.0, 0.01, 1.0]


def _pick_facets(core_rows):
    avail = sorted({float(r["noise"]) for r in core_rows
                    if r.get("noise") is not None and r.get("error") is None})
    if not avail:
        return NOISE_TARGETS
    out = []
    for t in NOISE_TARGETS:
        out.append(min(avail, key=lambda v: abs(v - t)))
    # dedupe preserving order
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


def _plot(ax, x, y, **kw):
    if len(x) == 0:
        return
    yc = np.clip(y, None, CLIP)
    ax.plot(x, yc, **kw)
    blow = y > 1.0
    if blow.any():
        ax.scatter(x[blow], np.full(blow.sum(), CLIP), marker="x", color=kw.get("color"),
                   s=60, zorder=5)


def main():
    ncol = len(NOISE_TARGETS)
    fig, axes = plt.subplots(len(SYSTEMS), ncol,
                             figsize=(4.2 * ncol, 3.6 * len(SYSTEMS)),
                             squeeze=False)
    for i, sysd in enumerate(SYSTEMS):
        lin_rows = _load(sysd["linear"][0])
        leg_rows = _load(sysd["legacy"][0])
        core_rows = [r for r in _load(sysd["core"])
                     if r.get("dt_base") is None
                     or abs(float(r["dt_base"]) - sysd["dt_base"]) < 1e-12]
        facets = _pick_facets(core_rows)
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(facets):
                ax.axis("off"); continue
            nz = facets[j]
            x, y = _curve(lin_rows, sysd["linear"][1], nz, sysd["dt_base"])
            _plot(ax, x, y, color="#999999", ls="--", lw=1.5, label=sysd["linear"][2])
            x, y = _curve(leg_rows, sysd["legacy"][1], nz, sysd["dt_base"])
            _plot(ax, x, y, color="#d94801", ls="-.", lw=1.5, label=sysd["legacy"][2])
            for m, lbl, col, mk in CORE:
                x, y = _curve(core_rows, m, nz, sysd["dt_base"])
                _plot(ax, x, y, color=col, marker=mk, lw=2.0, ms=7, label=lbl)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
            ax.set_title(f"{sysd['name']}  σ={nz:g}", fontsize=10)
            if i == len(SYSTEMS) - 1:
                ax.set_xlabel(r"$\Delta t_{\rm eff}$")
            if j == 0:
                ax.set_ylabel("NMSE(force)")
            ax.grid(True, which="both", alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("RK4 decision — minimal Euler·n₁ vs RK4·n₁ vs RK4·n₄  "
                 "(× = NMSE>1 blow-up; linear/legacy = full-budget reference)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO / "benchmark_results/rk4_decision/rk4_decision.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
