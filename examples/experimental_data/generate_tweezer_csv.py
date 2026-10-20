"""Generate the synthetic optical-tweezer CSV used by the experimental workflow demo.

Run once to (re)create ``examples/experimental_data/optical_tweezer.csv``;
the file is committed so the demo loads it directly, like a real recording.

Model (overdamped, 2D, natural units):

    F(x, y) = -k (x, y) - alpha (x^3, 0) + omega (-y, x)

a harmonic trap with a weak Duffing anharmonicity along x and a slow
rotational (non-conservative) drive, isotropic diffusion D, recorded every
dt with Gaussian localisation noise of standard deviation sigma per axis.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random

from SFI.bases import unit_axes, x_components
from SFI.langevin import OverdampedProcess

k, alpha, omega = 1.0, 1.0, 0.5
D = 0.5
dt = 0.01
T = 100_000
sigma = 0.1           # localisation noise (per axis)

x, y = x_components(2)
ex, ey = unit_axes(2)
F = (-k * x - alpha * x * x * x - omega * y) * ex + (-k * y + omega * x) * ey

proc = OverdampedProcess(F, D=D)
proc.initialize(jnp.zeros(2))
coll = proc.simulate(dt=dt, Nsteps=T, key=random.PRNGKey(42), prerun=500, oversampling=10)
coll = coll.degrade(noise=sigma, seed=7)

_, X, _ = coll.to_arrays(dataset=0)
X = np.asarray(X)[:, 0, :]

out = Path(__file__).parent / "optical_tweezer.csv"
with open(out, "w") as f:
    f.write(f"# dt: {dt}\n")
    f.write("# description: 2D optical tweezer (synthetic stand-in) with anharmonic "
            "correction, rotation and localisation noise\n")
    f.write("particle_id,frame,x,y\n")
    for i in range(X.shape[0]):
        f.write(f"0,{i},{X[i, 0]:.5f},{X[i, 1]:.5f}\n")

print(f"Wrote {out} ({X.shape[0]} rows, {out.stat().st_size / 1024:.0f} KB)")
