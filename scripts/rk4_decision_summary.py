"""Summarise the RK4 decision sweep: minimal-Euler vs minimal-RK4 vs legacy vs
best-linear, on the OD Lorenz and UD Van der Pol x10 grids.

Reads the same JSONL the renders read.  Partial-data safe.  For each system and
noise level it prints median NMSE(force) per method across dt_eff, plus a
blow-up count (NMSE>1 or non-finite) — the headline robustness signal.

    python -m scripts.rk4_decision_summary
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _val(r):
    v = r.get("NMSE_force")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v


SYSTEMS = {
    "Lorenz (OD)": dict(
        # Budget-matched comparison: all methods run at the same point budget /
        # dt grid into sweep_core.jsonl (the full-budget sweep.jsonl is a separate,
        # different-T_eff reference shown by the renders).
        files=["benchmark_results/lorenz_method_evolution/sweep_core.jsonl"],
        dt_base=1e-3,
        compare=["sfi_v1_5", "sfi_v2_window", "core_euler", "core_rk4_n1", "core_rk4"],
        labels={"sfi_v1_5": "best-linear", "sfi_v2_window": "legacy",
                "core_euler": "minimal·Euler", "core_rk4_n1": "RK4·n1",
                "core_rk4": "minimal·RK4"},
    ),
    "Van der Pol (UD)": dict(
        files=["benchmark_results/vdp_uli_10x/sweep_core.jsonl"],
        dt_base=1e-3,
        compare=["__best_linear__", "infer_force", "kalman", "core_euler", "core_rk4_n1", "core_rk4"],
        labels={"infer_force": "legacy", "kalman": "kalman",
                "core_euler": "minimal·Euler", "core_rk4_n1": "RK4·n1",
                "core_rk4": "minimal·RK4"},
    ),
}


def _agg(rows, method, dtf, noise):
    vals = [_val(r) for r in rows
            if r.get("method") == method and int(r.get("dt_factor", -1)) == dtf
            and abs(float(r.get("noise", -1)) - noise) < 1e-12 and r.get("error") is None]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def _blowups(rows, method):
    n = bad = 0
    for r in rows:
        if r.get("method") != method or r.get("error") is not None:
            continue
        v = _val(r)
        if v is None:
            continue
        n += 1
        if (not math.isfinite(v)) or v > 1.0:
            bad += 1
    return bad, n


def _best_linear(rows, linear_methods, dtfs, noises):
    """Method with the lowest overall median NMSE among the linear set."""
    best, best_med = None, float("inf")
    for m in linear_methods:
        vals = [_val(r) for r in rows if r.get("method") == m and r.get("error") is None]
        vals = [v for v in vals if v is not None and math.isfinite(v) and v <= 1.0]
        if not vals:
            continue
        med = float(np.median(vals))
        if med < best_med:
            best, best_med = m, med
    return best


def _dtfs_for(rows, method):
    return sorted({int(r["dt_factor"]) for r in rows
                   if r.get("method") == method and "dt_factor" in r})


def _agg_nearest(rows, method, dtf, noise):
    """Exact dt_factor median if present, else the nearest dt_factor (~)."""
    v = _agg(rows, method, dtf, noise)
    if not math.isnan(v):
        return v, False
    avail = _dtfs_for(rows, method)
    if not avail:
        return float("nan"), False
    near = min(avail, key=lambda d: abs(d - dtf))
    return _agg(rows, method, near, noise), True


def main():
    for name, cfg in SYSTEMS.items():
        rows = []
        for f in cfg["files"]:
            rows += _load(REPO / f)
        methods_present = {r.get("method") for r in rows}
        compare = list(cfg["compare"])
        labels = dict(cfg["labels"])
        if "__best_linear__" in compare:
            linear = [m for m in methods_present
                      if m and not m.startswith(("infer_force", "core", "kalman"))]
            bl = _best_linear(rows, linear, None, None)
            i = compare.index("__best_linear__")
            compare[i] = bl or "__none__"
            labels[bl] = f"best-linear ({bl})"
        compare = [m for m in compare if m in methods_present]
        core_methods = [m for m in compare if m.startswith("core_")]

        # Anchor the table on the CORE grid (the fair, matched-budget comparison);
        # non-core methods (full-budget linear/legacy) are shown at the nearest
        # dt_eff for context only (~), since their T_eff differs.
        anchor = sorted({d for m in core_methods for d in _dtfs_for(rows, m)}) \
            or sorted({int(r["dt_factor"]) for r in rows if "dt_factor" in r})
        noises = sorted({float(r["noise"]) for r in rows
                         if r.get("method") in core_methods and "noise" in r})
        print(f"\n{'='*84}\n{name}   (core grid; ~ = nearest-dt context, different T_eff)\n{'='*84}")

        for nz in noises:
            print(f"\n  noise σ={nz:g}   — median NMSE(force) vs dt_eff")
            print("    dt_eff   " + "".join(f"{labels.get(m,m)[:14]:>16}" for m in compare))
            for dtf in anchor:
                line = f"    {dtf*cfg['dt_base']:<8.4g} "
                for m in compare:
                    if m in core_methods:
                        v = _agg(rows, m, dtf, nz); approx = False
                    else:
                        v, approx = _agg_nearest(rows, m, dtf, nz)
                    if math.isnan(v):
                        cell = "—"
                    elif v > 1:
                        cell = f"{v:.1e}!"
                    else:
                        cell = f"{'~' if approx else ''}{v:.4g}"
                    line += f"{cell:>16}"
                print(line)

        print("\n  blow-ups (NMSE>1 or non-finite) / valid cells [core grid]:")
        for m in compare:
            bad, tot = _blowups(rows, m)
            flag = "  <== FRAGILE" if tot and bad / tot > 0.15 else ""
            print(f"    {labels.get(m,m):24s} {bad:3d}/{tot:<3d}{flag}")


if __name__ == "__main__":
    main()
