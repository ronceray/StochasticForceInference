#!/usr/bin/env python3
"""Regenerate the parametric golden reference values.

Runs every scenario in ``tests/inference/_parametric_golden_scenarios.py``
on this machine and stores inputs + outputs in
``tests/inference/_golden/parametric_golden.npz``.  The consumer is
``tests/inference/test_parametric_golden.py``.

Regenerate ONLY when a deliberate, documented numerical change lands
(record the old-vs-new deltas in the commit message)::

    .venv/bin/python scripts/gen_parametric_goldens.py            # all
    .venv/bin/python scripts/gen_parametric_goldens.py ud_gn_eiv  # subset

Float64 and CPU are forced; existing entries for scenarios not being
regenerated are preserved.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"  # must precede any jax import

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))                      # this tree's SFI
sys.path.insert(0, str(REPO_ROOT / "tests" / "inference"))

import numpy as np  # noqa: E402

from _parametric_golden_scenarios import GOLDEN_PATH, SCENARIOS  # noqa: E402


def main(argv):
    names = argv[1:] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s): {unknown}; have {list(SCENARIOS)}")

    out_path = REPO_ROOT / GOLDEN_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = {}
    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as old:
            store = {k: old[k] for k in old.files}

    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()

    for name in names:
        sc = SCENARIOS[name]
        t0 = time.perf_counter()
        data = sc.make_data()
        outputs = sc.run(data)
        dt_wall = time.perf_counter() - t0
        old_keys = [k for k in store if k.startswith(f"{name}/out/")]
        deltas = []
        for key, val in outputs.items():
            full = f"{name}/out/{key}"
            if full in store and store[full].shape == np.shape(val):
                prev = store[full]
                scale = max(np.max(np.abs(prev)), 1e-30)
                deltas.append((key, float(np.max(np.abs(prev - val)) / scale)))
        for k in old_keys:
            del store[k]
        for key, val in data.items():
            store[f"{name}/in/{key}"] = np.asarray(val)
        for key, val in outputs.items():
            store[f"{name}/out/{key}"] = np.asarray(val)
        store[f"{name}/meta/commit"] = np.array(commit)
        msg = f"[golden] {name}: {dt_wall:6.1f}s"
        if deltas:
            worst = max(deltas, key=lambda kv: kv[1])
            msg += f"  (max rel delta vs previous: {worst[1]:.3e} on {worst[0]!r})"
        print(msg, flush=True)

    np.savez_compressed(out_path, **store)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} kB, "
          f"{len(store)} arrays)")


if __name__ == "__main__":
    main(sys.argv)
