# TODO: review this file
"""Generate a synthetic optical-tweezer CSV for the experimental workflow demo.

Run once to create examples/experimental_data/optical_tweezer.csv.
The file is committed to the repo so the demo can load it directly.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

# Optical tweezer: 2D harmonic trap  F = -k*x,  D = 0.3
k = 1.0
D = 0.3
dt = 0.01
T = 10_000

X = np.zeros((T, 2))
for i in range(1, T):
    drift = -k * X[i - 1] * dt
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(2)
    X[i] = X[i - 1] + drift + noise

t = np.arange(T) * dt

out = Path(__file__).parent / "optical_tweezer.csv"
with open(out, "w") as f:
    f.write("t,x,y\n")
    for i in range(T):
        f.write(f"{t[i]:.4f},{X[i, 0]:.6f},{X[i, 1]:.6f}\n")

print(f"Wrote {out} ({T} rows, {out.stat().st_size / 1024:.0f} KB)")
