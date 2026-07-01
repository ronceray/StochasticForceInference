# TODO: review this file
"""Diagnose which cells have NaN NMSE and why."""
import json
import numpy as np
from collections import defaultdict

rows = [json.loads(l) for l in open("benchmark_results/lorenz_10x/sweep.jsonl")]

all_dt    = sorted({r["dt_factor"] for r in rows})
all_noise = sorted({r["noise"]     for r in rows})
METHODS   = ["baseline", "sfi_v1", "sfi_v1_5", "sfi_v2_window"]

buckets_n = defaultdict(list)
buckets_err = defaultdict(list)
for r in rows:
    k = (r["method"], r["dt_factor"], r["noise"])
    if r.get("error") is not None:
        buckets_err[k].append(r["error"])
    else:
        n = r.get("NMSE_force")
        if n is not None and not (isinstance(n, float) and n != n):
            buckets_n[k].append(float(n))

# Find NaN cells per method
print("=== NaN cells (no valid NMSE) per method ===\n")
for m in METHODS:
    nan_cells = []
    for dt in all_dt:
        for no in all_noise:
            k = (m, dt, no)
            n_valid = len(buckets_n.get(k, []))
            n_err   = len(buckets_err.get(k, []))
            if n_valid == 0:
                errs = buckets_err.get(k, [])
                # Categorize error types
                types = set()
                for e in errs:
                    if "timeout" in str(e).lower() or "timed" in str(e).lower():
                        types.add("timeout")
                    elif e:
                        types.add("error")
                if not errs:
                    types.add("missing")
                nan_cells.append((dt, no, n_err, sorted(types)))
    print(f"Method {m}: {len(nan_cells)} NaN cells")
    if nan_cells:
        # group by error type
        by_type = defaultdict(list)
        for dt, no, ne, types in nan_cells:
            for t in types:
                by_type[t].append((dt, no))
        for t, cells in by_type.items():
            print(f"  {t}: {len(cells)} cells")
            # show dt breakdown
            dt_counts = defaultdict(int)
            for dt, no in cells:
                dt_counts[dt] += 1
            for dt, cnt in sorted(dt_counts.items()):
                print(f"    dt_factor={dt}: {cnt} noise levels")
        # Print a few example NaN cells
        print(f"  First few NaN cells: {nan_cells[:5]}")
    print()

# For sfi_v2_window: show NMSE distribution for non-NaN cells
m = "sfi_v2_window"
valid_nmse = []
for k, v in buckets_n.items():
    if k[0] == m:
        valid_nmse.extend(v)
if valid_nmse:
    arr = np.array(valid_nmse)
    print(f"sfi_v2_window valid NMSE: n={len(arr)}, median={np.median(arr):.4f}, "
          f"max={np.max(arr):.4f}, >0.99: {(arr>0.99).sum()}, >10: {(arr>10).sum()}")

# Show timeout error example
for r in rows:
    if r.get("error") and r["method"] == "sfi_v2_window":
        print(f"\nExample sfi_v2_window error: dt_factor={r['dt_factor']}, noise={r['noise']}, error={str(r['error'])[:120]}")
        break
