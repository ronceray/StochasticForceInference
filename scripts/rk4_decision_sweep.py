"""RK4 decision sweep for the v2.0 minimal-parametric path.

Runs the two new ``infer_force_core`` variants — ``core_euler`` (minimal:
single Euler step, n_substeps=1) and ``core_rk4`` (RK4, fixed n_substeps=4) —
across the existing x10 stress grids for **both** the OD Lorenz
(``lorenz_method_evolution``) and the UD Van der Pol (``vdp_uli_10x``)
benchmarks, reusing each package's ``run_cell``.  Streams to the same JSONL
files the renders read (``sweep_core.jsonl``), so the linear + legacy methods
already computed there/in ``sweep.jsonl`` render side-by-side.

Point-budgeted (cost ∝ window count) + resumable + joblib-parallel, mirroring
``run_sweep_core.py``.

Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 RK4GATE_NJOBS=6 \\
        python -m scripts.rk4_decision_sweep
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from _gallery_utils.bench_utils import _clear_jax_caches  # noqa: E402,F401

METHODS = tuple(os.environ.get("RK4GATE_METHODS", "core_euler,core_rk4").split(","))
DT_FACTORS = tuple(int(x) for x in os.environ.get("RK4GATE_DTF", "1,10,50,137,276").split(","))
NOISE_LEVELS = tuple(float(x) for x in os.environ.get("RK4GATE_NOISE", "0.0,0.1,1.0").split(","))
N_REPEATS = int(os.environ.get("RK4GATE_REPEATS", "2"))
N_JOBS = int(os.environ.get("RK4GATE_NJOBS", "6"))
TIMEOUT_S = int(os.environ.get("RK4GATE_TIMEOUT_S", "900"))
POINTS = int(os.environ.get("RK4GATE_POINTS", "3500"))
NSTEPS_CAP = int(os.environ.get("RK4GATE_NSTEPS_CAP", "400000"))
NSTEPS_FLOOR = 3500
# Matched-N (consolidated) mode: when >0, every cell uses this fixed Nsteps so the
# core runs on the SAME trajectory length as the linear/legacy standard sweep.
FIXED_NSTEPS = int(os.environ.get("RK4GATE_FIXED_NSTEPS", "0"))
# Output basename override (e.g. "sweep_core_b_gate.jsonl") — lets validation runs
# write next to, instead of into, the consolidated sweep_core.jsonl.
OUT_NAME = os.environ.get("RK4GATE_OUT", "sweep_core.jsonl")

SYSTEMS = {
    "lorenz": dict(
        out=REPO / "benchmark_results/lorenz_method_evolution" / OUT_NAME,
        dt_base=1e-3,   # standard Lorenz dt_base (problems.py)
    ),
    "vdp": dict(
        out=REPO / "benchmark_results/vdp_uli_10x" / OUT_NAME,
        dt_base=1e-4,   # standard VdP dt_base (run_sweep.py uses DT_BASE_FINE)
    ),
}


def _nsteps_for(dt_factor: int) -> int:
    if FIXED_NSTEPS > 0:
        return FIXED_NSTEPS
    return int(min(max(POINTS * dt_factor, NSTEPS_FLOOR), NSTEPS_CAP))


def _run_cell(system, method, dt_factor, noise, seed):
    if system == "lorenz":
        from scripts.lorenz_method_evolution.runner import run_cell
        kw = {}
    else:
        from scripts.vdp_uli_10x.runner import run_cell
        kw = dict(dt_base=SYSTEMS[system]["dt_base"])
    try:
        r = run_cell(method=method, dt_factor=dt_factor, noise=noise, seed=seed,
                     Nsteps=_nsteps_for(dt_factor), timeout_s=TIMEOUT_S, **kw)
        r["system"] = system
        return r
    except Exception as exc:  # noqa: BLE001
        return dict(system=system, method=method, dt_factor=int(dt_factor),
                    noise=float(noise), seed=int(seed),
                    NMSE_force=float("nan"), error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            _clear_jax_caches()
        except Exception:
            pass


def _done_keys(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("method") in METHODS and r.get("error") is None:
                done.add((r["method"], int(r["dt_factor"]), float(r["noise"]), int(r["seed"])))
    return done


def run_system(system: str):
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    out = SYSTEMS[system]["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    tasks = [(m, df, float(nz), s)
             for m in METHODS for df in DT_FACTORS for nz in NOISE_LEVELS for s in range(N_REPEATS)
             if (m, df, float(nz), s) not in done]
    # Run methods in the *passed* order (priority), and within a method do the
    # cheap large-dt_factor (small T_eff) cells first so previews fill fast.
    tasks.sort(key=lambda t: (METHODS.index(t[0]), -t[1], t[2], t[3]))
    print(f"\n[{system}] {len(done)} done, {len(tasks)} remaining -> {out.name}", flush=True)
    if not tasks:
        return
    f_out = open(out, "a", buffering=1)
    t0 = time.perf_counter()
    BATCH = N_JOBS
    for bi in range((len(tasks) + BATCH - 1) // BATCH):
        batch = tasks[bi * BATCH:(bi + 1) * BATCH]
        results = Parallel(n_jobs=N_JOBS)(
            delayed(_run_cell)(system, m, df, nz, s) for m, df, nz, s in batch)
        for r in results:
            f_out.write(json.dumps(r, default=str) + "\n")
        f_out.flush(); os.fsync(f_out.fileno())
        ok = sum(1 for r in results if r.get("error") is None)
        blow = sum(1 for r in results
                   if r.get("error") is None and r.get("NMSE_force", 0) > 1.0)
        print(f"  [{system} {bi+1}] ok={ok}/{len(batch)} blowups(NMSE>1)={blow} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m", flush=True)
        try:
            get_reusable_executor().shutdown(kill_workers=True)
        except Exception:
            pass
        _clear_jax_caches()
    f_out.close()


def main():
    print(f"RK4 decision sweep  methods={METHODS}")
    print(f"  dt_factors={DT_FACTORS}  noise={NOISE_LEVELS}  seeds={N_REPEATS}  workers={N_JOBS}")
    for system in os.environ.get("RK4GATE_SYSTEMS", "lorenz,vdp").split(","):
        run_system(system)
    print("\nRK4 decision sweep complete.")


if __name__ == "__main__":
    main()
