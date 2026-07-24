# TODO: review this file
"""
Scenario picker — select representative (dt, noise) cells from a reference grid.
==================================================================================

Given a cached reference grid of fast methods, classifies cells by
difficulty and picks a diverse subset for targeted nonlinear-estimator
evaluation.

Difficulty levels (best-reference-method NMSE):
- **easy**   — NMSE < 0.01
- **medium** — 0.01 ≤ NMSE < 0.1
- **hard**   — NMSE ≥ 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────

EASY_THRESHOLD = 0.01
HARD_THRESHOLD = 0.10


@dataclass
class Scenario:
    """A single benchmark scenario extracted from the reference grid."""
    dt_factor: int
    noise: float
    difficulty: str          # "easy", "medium", "hard"
    ref_NMSE: float          # best NMSE achieved by reference methods
    best_method: str         # name of the method that achieved ref_NMSE
    seed: int = 0            # repeat / seed to use (first repeat)


def _best_nmse(row: dict, method_keys: Sequence[str]):
    """Return (best_nmse, best_method) across the given keys."""
    best_val = float("inf")
    best_name = ""
    for k in method_keys:
        v = row.get(k, float("nan"))
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_name = k
    if not np.isfinite(best_val):
        return float("nan"), ""
    return best_val, best_name


def classify_difficulty(nmse: float) -> str:
    if np.isnan(nmse):
        return "failed"
    if nmse < EASY_THRESHOLD:
        return "easy"
    if nmse < HARD_THRESHOLD:
        return "medium"
    return "hard"


def pick_scenarios(
    grid_results: List[dict],
    method_keys: Optional[Sequence[str]] = None,
    n_easy: int = 1,
    n_medium: int = 2,
    n_hard: int = 1,
) -> List[Scenario]:
    """Select representative scenarios from a reference-grid cache.

    Parameters
    ----------
    grid_results : list of dict
        Flat list of per-cell results (as stored by BenchmarkCache).
        Must contain ``dt_factor``, ``noise``, and per-method NMSE keys.
    method_keys : sequence of str, optional
        Keys in each row that contain NMSE values (e.g.
        ``["NMSE_Itô-rect", "NMSE_Itô-trap", ...]``).
        Auto-detected if omitted (all keys starting with ``NMSE_``).
    n_easy, n_medium, n_hard : int
        How many scenarios to pick per difficulty level.

    Returns
    -------
    list of Scenario, sorted by difficulty (easy → medium → hard).
    """
    if not grid_results:
        return []

    # Auto-detect NMSE keys
    if method_keys is None:
        sample = grid_results[0]
        method_keys = sorted(k for k in sample if k.startswith("NMSE_"))
    if not method_keys:
        raise ValueError("No NMSE columns found in grid_results")

    # Aggregate repeats: average NMSE per (dt_factor, noise) cell
    from collections import defaultdict

    cells: Dict[tuple, list] = defaultdict(list)
    for row in grid_results:
        key = (int(row["dt_factor"]), float(row["noise"]))
        cells[key].append(row)

    # Build per-cell summary
    summaries = []
    for (dt_f, nv), rows in cells.items():
        # Average each method's NMSE across repeats
        avg_row: dict = {}
        for mk in method_keys:
            vals = [r.get(mk, float("nan")) for r in rows]
            finite = [v for v in vals if np.isfinite(v)]
            avg_row[mk] = float(np.mean(finite)) if finite else float("nan")
        best_val, best_name = _best_nmse(avg_row, method_keys)
        summaries.append((dt_f, nv, best_val, best_name))

    # Bucket by difficulty
    buckets: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for dt_f, nv, bv, bn in summaries:
        d = classify_difficulty(bv)
        if d in buckets:
            buckets[d].append((dt_f, nv, bv, bn))

    # Pick from each bucket with diversity in (dt, noise) axes
    def _pick(bucket, n):
        if not bucket or n == 0:
            return []
        # Sort by NMSE so we pick "most representative" of each level
        bucket.sort(key=lambda x: x[2])
        if len(bucket) <= n:
            return bucket
        # Spread evenly across the sorted list
        indices = np.linspace(0, len(bucket) - 1, n, dtype=int)
        return [bucket[i] for i in indices]

    picks = (
        _pick(buckets["easy"], n_easy)
        + _pick(buckets["medium"], n_medium)
        + _pick(buckets["hard"], n_hard)
    )

    return [
        Scenario(
            dt_factor=dt_f,
            noise=nv,
            difficulty=classify_difficulty(bv),
            ref_NMSE=bv,
            best_method=bn.replace("NMSE_", ""),
        )
        for dt_f, nv, bv, bn in picks
    ]


def print_scenarios(scenarios: List[Scenario]) -> None:
    """Pretty-print a list of scenarios."""
    if not scenarios:
        print("  (no scenarios)")
        return
    print(f"  {'#':>2} {'Difficulty':>10} {'dt_f':>5} {'noise':>6} "
          f"{'ref_NMSE':>9} {'best_method'}")
    print(f"  {'─'*2} {'─'*10} {'─'*5} {'─'*6} {'─'*9} {'─'*15}")
    for i, s in enumerate(scenarios):
        print(f"  {i+1:>2} {s.difficulty:>10} {s.dt_factor:>5d} "
              f"{s.noise:>6.3f} {s.ref_NMSE:>9.5f} {s.best_method}")
