# Stochastic Force Inference (SFI)

[![Documentation](https://readthedocs.org/projects/stochasticforceinference/badge/?version=latest)](https://stochasticforceinference.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/StochasticForceInference.svg)](https://pypi.org/project/StochasticForceInference/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ronceray/StochasticForceInference/blob/main/LICENSE)

**Infer force and diffusion fields from stochastic trajectory data.**

SFI is a JAX-based Python package for learning the drift (force) and diffusion
of Langevin stochastic differential equations from time-series observations.
It handles both overdamped and underdamped dynamics, supports interacting
particles and spatially-extended (SPDE) systems, and provides built-in
diagnostics, sparse model selection, and bootstrapped validation.

> **Designed for experimental data.**  SFI is built for real experimental
> trajectories (tracked particles, cells, organisms, …) where *no dynamical
> model pre-exists*.  Two first-class estimator families share one API:
> fast closed-form **linear estimators** that require no initial guess, and
> **parametric estimators** that model measurement noise and finite sampling
> explicitly — robust where real data is hard.  The PASTIS information
> criterion rigorously identifies which terms the data actually supports.
> Synthetic examples in the gallery serve as runnable demonstrations.

## Key features

- **Two estimator families** — closed-form linear estimators
  (`infer_force_linear`, seconds even on large datasets) and parametric
  estimators (`infer_force` / `infer_diffusion`: noise-aware likelihood
  fit, robust to localization error and coarse sampling, accepts
  nonlinear models such as neural-network drifts).
- **Overdamped & underdamped inference** — works with position-only data;
  velocities are reconstructed automatically for underdamped systems.
- **Composable state functions** — build force/diffusion models from
  monomials, custom basis functions, pair interactions, or arbitrary
  parametric families, all with automatic differentiation via JAX.
- **Sparse model selection** — Pareto-front beam search with AIC / BIC /
  PASTIS information criteria.
- **Simulation** — simulate Langevin SDEs (Euler–Maruyama) from the same
  model objects used for inference.
- **Trajectory toolkit** — mask-aware increments, synthetic degradation
  (noise, downsampling, data loss, motion blur), streaming for large
  datasets, I/O (CSV / Parquet / HDF5).
- **Diagnostics** — compare inferred fields to exact models, compute
  normalized errors, generate bootstrapped trajectories.

## Installation

```bash
pip install StochasticForceInference
```

For development (editable install with test/doc dependencies):

```bash
git clone https://github.com/ronceray/StochasticForceInference.git
cd StochasticForceInference
pip install -e ".[dev,io]"
```

> **Note:** SFI requires Python ≥ 3.11 and JAX ≥ 0.10.

## Quick start

For **experimental data**, load your trajectories and infer directly:

```python
import numpy as np
from SFI import OverdampedLangevinInference, TrajectoryCollection
from SFI.bases import monomials_up_to

# Load your tracked data (positions: T×d array, dt: time step)
positions = np.load("my_experiment.npz")["positions"]  # shape (T, d)
coll = TrajectoryCollection.from_arrays(X=positions, dt=0.01)

# Infer using a polynomial basis — no model needed
B = monomials_up_to(order=3, dim=2, rank="vector")
inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()
inf.infer_force_linear(B)
inf.sparsify_force(criterion="PASTIS")  # optional: find the minimal model
inf.print_report()

# Noisy or coarsely-sampled recordings? Use the parametric estimator
# instead: inf.infer_force(B) — it models the measurement noise natively.
```

For a **synthetic example** (useful for validation):

```python
import jax.numpy as jnp
from jax import random

from SFI import OverdampedLangevinInference
from SFI.bases import X, monomials_up_to
from SFI.langevin import OverdampedProcess

# 1. Build a force model (Ornstein–Uhlenbeck: F = -k x) and simulate a trajectory.
#    X(dim) is the identity basis x ↦ x; theta_F holds its coefficient, so F(x) = -x.
proc = OverdampedProcess(F=X(dim=2), D=jnp.eye(2) * 0.5, theta_F=jnp.array([-1.0]))
proc.initialize(jnp.zeros(2))
coll = proc.simulate(dt=0.01, Nsteps=10_000, key=random.PRNGKey(0))

# 2. Infer from the trajectory using a generic polynomial basis
B = monomials_up_to(order=2, dim=2, rank="vector")
inf = OverdampedLangevinInference(coll)
inf.compute_diffusion_constant()
inf.infer_force_linear(B)
inf.compute_force_error()

# 3. Inspect results and validate against the known model
inf.print_report()
inf.compare_to_exact(model_exact=proc)
```

## Documentation

Full documentation is available at
[stochasticforceinference.readthedocs.io](https://stochasticforceinference.readthedocs.io), including:

- **[Tutorial](https://stochasticforceinference.readthedocs.io/en/latest/gallery/ou_demo.html)** — end-to-end walkthrough
- **[Examples gallery](https://stochasticforceinference.readthedocs.io/en/latest/gallery/index.html)** — worked examples covering sparsity, multi-particle, and SPDE inference
- **[Model building](https://stochasticforceinference.readthedocs.io/en/latest/statefunc/user_guide.html)** — composable state functions
- **[Trajectory handling](https://stochasticforceinference.readthedocs.io/en/latest/trajectory/user_guide.html)** — data ingestion, masking, degradation
- **[Running inference](https://stochasticforceinference.readthedocs.io/en/latest/inference/user_guide.html)** — estimator choice, overdamped/underdamped engines
- **[API reference](https://stochasticforceinference.readthedocs.io/en/latest/api_frontend.html)** — full autodoc

## Package structure

| Subpackage       | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `SFI.statefunc`  | Composable state functions: `Basis`, `PSF`, `SF`     |
| `SFI.inference`  | Overdamped & underdamped inference engines            |
| `SFI.trajectory` | `TrajectoryCollection` / `TrajectoryDataset`          |
| `SFI.langevin`   | Langevin simulators (`OverdampedProcess`, …)          |
| `SFI.bases`      | Ready-made basis builders (monomials, constants, …)  |
| `SFI.integrate`  | Time-averaging integration engine                    |
| `SFI.utils`      | Math helpers, formatting, plotting                   |

For contributors and AI coding agents: see [`AGENTS.md`](https://github.com/ronceray/StochasticForceInference/blob/main/AGENTS.md) for
the canonical imports table, task playbooks, and "do not re-implement"
guidance.

## Citation

If you use SFI, please cite the software (see [`CITATION.cff`](CITATION.cff))
together with the method paper(s) relevant to the features you use. See
**[How to cite](https://stochasticforceinference.readthedocs.io/en/latest/#how-to-cite)** in the
documentation for the full, feature-specific list (overdamped, underdamped/ULI,
trapeze, and PASTIS references).

## License

MIT — see [LICENSE](https://github.com/ronceray/StochasticForceInference/blob/main/LICENSE) for details.
