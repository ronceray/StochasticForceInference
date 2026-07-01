# AGENTS.md — guide for AI coding agents working on SFI

> **Read this first.** Canonical entry point for automated coding
> agents (GitHub Copilot, Claude Code, Cursor, …) on the Stochastic
> Force Inference (SFI) repository. Human contributors should read
> [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 1. What this project is

SFI is a JAX-based Python package for **inferring drift (force) and
diffusion fields from stochastic trajectory data** (overdamped and
underdamped Langevin SDEs). Targets real experimental data — tracked
particles, cells, organisms — where no dynamical model exists yet.
See [`README.md`](README.md) and
[`docs/source/index.rst`](docs/source/index.rst).

## 2. Before you write code

1. Read this file end-to-end.
2. Open the playbook matching your task:
   - Apply inference to a dataset →
     [`docs/source/agent_playbooks/apply_inference.rst`](docs/source/agent_playbooks/apply_inference.rst)
   - Add a feature to SFI →
     [`docs/source/agent_playbooks/add_feature.rst`](docs/source/agent_playbooks/add_feature.rst)
3. Skim §4 below — the canonical-imports table.
4. Plotting? read [`GALLERY_STYLE_GUIDE.md`](GALLERY_STYLE_GUIDE.md).
5. Need the full public surface?
   [`_project_index/api_map.md`](_project_index/api_map.md) (human)
   or [`_project_index/api_map.json`](_project_index/api_map.json)
   (machine).

## 3. Core principle — reuse, don't re-implement

SFI has mature, well-tested abstractions for every common operation.
**Do not re-implement them.**

| If you need to…                          | Use this (not custom code)                              |
| ---------------------------------------- | ------------------------------------------------------- |
| Load trajectories / compute increments   | `SFI.trajectory.TrajectoryCollection`                   |
| Simulate an SDE                          | `SFI.langevin.OverdampedProcess` / `UnderdampedProcess` |
| Build a polynomial / pair / SPDE basis   | `SFI.bases.*` helpers                                   |
| Write a force as a symbolic expression   | Compositional primitives: `x_components`, `v_components`, `unit_axes`, `frame`, `named_scalar(s)` — see [`docs/source/bases/user_guide.rst`](docs/source/bases/user_guide.rst) |
| Wrap a user function as a state function | `SFI.statefunc.make_basis` / `make_sf` / `make_psf`     |
| Run force / diffusion inference          | `SFI.OverdampedLangevinInference` / `UnderdampedLangevinInference` |
| Select a sparse model                    | `inf.sparsify_force(criterion="PASTIS", p=0.1)`         |
| Compare to ground truth                  | `inf.compare_to_exact(model_exact=...)`                 |
| Plot in the SFI style                    | `SFI.utils.plotting.SFI_COLORS`, `dark_fig`, `phase2d`  |
| Save / load a fitted model               | `SFI.inference.save_model` / `load_model`               |

If you find yourself writing an Euler step, a finite-difference
velocity reconstructor, a mask-aware increment, a polynomial feature
builder, a Gram-matrix assembler, or an information-criterion scorer —
**stop and search the codebase first**.

## 4. Canonical imports

Top-level symbols are re-exported from `SFI/__init__.py`; prefer
`from SFI import X` whenever available. For the exhaustive surface
(every public function and class), see
[`_project_index/api_map.md`](_project_index/api_map.md).

### 4.1 Top-level

```python
from SFI import (
    OverdampedLangevinInference,   # 1st-order Langevin inference
    UnderdampedLangevinInference,  # 2nd-order (velocity-unobserved)
    InferenceResultSF,             # fitted SF with covariance + metadata
    TrajectoryCollection,          # multi-trajectory container
    TrajectoryDataset,             # single-trajectory container
    Basis, PSF, SF, make_sf,       # state-function core
    classify_dynamics,             # OD-vs-UD classifier -> DynamicsOrderReport
)
```

The dynamics-order classifier (`classify_dynamics`, returning a
`DynamicsOrderReport`) decides overdamped vs underdamped from raw
positions; see `SFI.diagnostics` and `examples/gallery/dynamics_order_demo.py`.

### 4.2 `SFI.statefunc` — composable state functions

| Symbol                           | Purpose                                   |
| -------------------------------- | ----------------------------------------- |
| `Basis`                          | Dictionary of functions (no params)       |
| `PSF`                            | Parametric family `F(x; θ)`               |
| `SF`                             | State function with frozen parameters     |
| `make_sf(func, dim, rank, ...)`  | Wrap a Python callable as an `SF`         |
| `make_psf(func, dim, rank, ...)` | Wrap a callable with parameters as `PSF`  |
| `make_basis(func, dim, ...)`     | Wrap a vector-valued callable as `Basis`  |
| `Interactor`, `make_interactor`  | Multi-scale interaction graphs            |

### 4.3 `SFI.inference` — inference engines

| Symbol                                              | Purpose                                    |
| --------------------------------------------------- | ------------------------------------------ |
| `OverdampedLangevinInference(collection)`           | Overdamped inference engine                |
| `UnderdampedLangevinInference(collection)`          | Underdamped inference engine               |
| `InferenceResultSF`                                 | Fitted model + parameter covariance        |
| `save_model` / `load_model`                         | Model (SF/PSF) serialisation               |
| `save_results` / `load_results`                     | Full inference-object serialisation        |

**Inference paths — two first-class estimator families.** They trade
compute for robustness; route by data regime, not by habit:

| Data regime | Use | Why |
| --- | --- | --- |
| Clean data, fine sampling | linear | exact in this limit; fastest |
| Quick exploration / huge dataset | linear | closed form, no initial guess |
| Measurement (localization) noise | parametric | profiles Λ; EIV instrument removes the noise bias |
| Coarse sampling (large Δt) | parametric | RK4 flow step replaces the Euler secant |
| Model nonlinear in θ (NN drift, gated PSF) | parametric | the L-BFGS path is the only option |
| Multi-particle / interacting | either | linear for speed; parametric when noise/coarse Δt too |
| SPDE / grid fields | linear | the (experimental) SPDE toolbox is linear-only |

When in doubt, run the linear estimator first to fix scales and
candidate terms, then confirm with the parametric estimator —
agreement is itself a diagnostic.

* **Linear estimators (fast, closed-form).** Methods
  `compute_diffusion_constant`, `infer_force_linear`,
  `infer_diffusion_linear`. Closed-form projection onto a `Basis`,
  exact in the fine-sampling, low-noise limit; with `"auto"` settings
  they apply the published SFI methodology's best refinements. Canonical
  read-out after a fit (each accessor has a distinct job):

  | Accessor | What it gives |
  | --- | --- |
  | `inf.force_inferred(x)` | Evaluate the inferred force at points (callable; carries parameter covariance) |
  | `inf.force_predicted_MSE` | Scalar predicted NMSE (accuracy estimate) |
  | `inf.force_coefficients` | Fitted coefficients |
  | `inf.print_report()` / `inf.report_dict()` | Human / machine summary |
  | `inf.compare_to_exact(model_exact=...)` | Validate against a known model |
  | `inf.holdout_score(test)` | Held-out NMSE on `coll.split_time(...)` test data (data-abundant scenarios only) |

* **Parametric estimators (robust, flexible; compute-intensive).**
  Robust to measurement noise
  and finite Δt.  Methods `infer_force` and `infer_diffusion`
  (implemented in `SFI.inference.parametric_core`).  Takes a `Basis`
  (linear-in-θ; PASTIS sparsification wired) or a `PSF` (general
  parametric family): a single RK4 flow step per observation interval
  defines the residual, the banded residual covariance gives a
  windowed-precision NLL, and `(D, Λ)` are profiled natively
  (moment-estimator init + one conditional-NLL refinement).  Two inner
  solvers, selected by `inner="auto"`:

  * `inner="gn"` — direct Gauss–Newton on the windowed Gram (the
    linear-in-θ fast path) with the skip-trick errors-in-variables
    instrument (`eiv=True`, consistent under measurement noise).
  * `inner="lbfgs"` — frozen-precision L-BFGS for non-linear-in-θ
    families (neural-net drift, gated PSFs, …); raise `inner_maxiter`
    for large models.  Worked example:
    [`examples/gallery/advanced/nn_force_demo.py`](examples/gallery/advanced/nn_force_demo.py).

  Multi-particle / interacting systems are supported in both overdamped
  and underdamped regimes (the multi-particle path uses the per-edge
  `d_x(same_particle=True)` / `d_v(same_particle=True)` Jacobian
  protocol; O(N) per window).

  See
  [`docs/source/inference/user_guide.rst`](docs/source/inference/user_guide.rst)
  for the regime table and workflow.

**Linear-estimator sequence:**

```python
inf = OverdampedLangevinInference(collection)
inf.compute_diffusion_constant(method="auto")        # 1. constant D ("noisy"/"WeakNoise"/"MSD")
inf.infer_force_linear(basis, preset="auto")         # 2. force ("auto"/"robust"/"clean"/"KM"/legacy)
inf.infer_diffusion_linear()                         # 3. optional: constant sym D (default basis)
inf.compute_force_error()                            # 4. error analysis
inf.sparsify_force(criterion="PASTIS", p=0.1)        # 5. optional: sparsity
inf.print_report()                                   # 6. summary
inf.compare_to_exact(model_exact=process)            # 7. optional: validate
```

**Parametric-estimator sequence:**

```python
inf = OverdampedLangevinInference(collection)
inf.infer_force(F)                                   # Basis or PSF; profiles (D, Λ)
inf.infer_diffusion()                                # no-arg: defaults to symmetric_matrix_basis
inf.compute_force_error()
```

**Sparsity.** PASTIS is the canonical criterion in SFI. Alternative
criteria (`AIC`, `BIC`, `SIC`) and strategies (greedy stepwise,
STLSQ, LASSO) are available — see
[`_project_index/api_map.md`](_project_index/api_map.md#sfiinferencesparse)
for the full inventory. Precision/recall and held-out NMSE helpers
live in `SFI.inference.sparse.overlap_metrics` and `predictive_nmse`.

### 4.4 `SFI.trajectory` — data containers

Construction:

| Symbol                                                 | Purpose                                                |
| ------------------------------------------------------ | ------------------------------------------------------ |
| `TrajectoryDataset.from_arrays(X=, dt=, ...)`          | Single dataset from dense tensors `(T, N, d)`          |
| `TrajectoryCollection.from_arrays(X=, dt=, ...)`       | Collection wrapping one dense dataset                  |
| `TrajectoryCollection.from_dataframe(df, particle=, time=, coords=)` | Pandas tracking table → collection: columns by **name** (auto-detected when omitted), junk columns dropped, extras prefixes parsed |
| `TrajectoryCollection.from_columns(particle_idx, time_idx, state_vectors, ...)` | **Canonical multi-particle entry**: flat columns (one row per observation) → collection. Accepts tabular data where particles enter/leave at different times |
| `TrajectoryCollection.from_dataset(ds, weights=...)`   | Promote a single dataset into a collection             |
| `TrajectoryCollection.concat(items, weights="equal")`  | Combine several collections (multi-experiment)         |
| `TrajectoryCollection.load(path)`                      | Load from CSV/Parquet/HDF5 (multi-file via list)       |
| `TimeSeriesExtra` / `time_series_extra`                | Time-dependent extras (leading time axis; sliced per frame in inference, per-frame schedules in simulation) |
| `FunctionExtra`, `function_extra`                      | Register on-demand computed extras                     |

Notes:
- For tracked-particle data where the number of particles changes
  over time, `from_columns` is the canonical path — **do not pre-pad
  with NaNs and call `from_arrays`.**
- Synthetic degradation (noise / downsampling / data-loss /
  motion blur) lives in `SFI.trajectory.degrade`.
- Low-level I/O lives in `SFI.trajectory.io` (used internally by
  `TrajectoryCollection.load`).

### 4.5 `SFI.langevin` — simulators

| Symbol                                           | Purpose                          |
| ------------------------------------------------ | -------------------------------- |
| `OverdampedProcess(F, D)`                        | Euler–Maruyama / Heun integrator |
| `UnderdampedProcess(F, D, M)`                    | Velocity-Verlet / BAOAB          |
| `WhiteNoise`, `ConservedNoise`, `CompositeNoise` | Noise models                     |

Pass a `Basis` (or a `PSF`, for a parametric family) directly as `F`,
with coefficients via `theta_F` — e.g.
`OverdampedProcess(F=X(dim=1), D=0.5, theta_F=jnp.array([-1.0]))`. Avoid
`make_basis(...).to_psf()` unless a parametric family is genuinely
required.

### 4.6 `SFI.bases` — ready-made basis builders

| Symbol                                              | Purpose                              |
| --------------------------------------------------- | ------------------------------------ |
| `monomials_up_to(order, dim, rank="scalar", ...)`   | Polynomials up to total order        |
| `monomials_degree(degree, dim, ...)`                | Polynomials of exact total degree    |
| `ones_basis(dim)`                                   | Constant 1                           |
| `unit_vector_basis(dim)`                            | Cartesian unit vectors               |
| `unit_axes(dim)`                                    | Per-axis unit-vector primitives `ex, ey, …`   |
| `x_components(dim)` / `v_components(dim)`           | Per-axis coordinate / velocity primitives     |
| `frame(dim, velocity=False)`                        | `(x, axes)` or `(x, v, axes)` convenience     |
| `named_scalar(name, default=...)`                   | Parametric scalar PSF with optional default   |
| `extra_scalar(name)`                                | Basis symbol reading `extras[name]` (conditions, drive protocols, per-particle properties) |
| `per_dataset_scalar(name, n)` / `dataset_indicator(n)` | Experiment-specific parameters in pooled multi-experiment fits (parametric / linear route), via the auto-injected `dataset_index` extra |
| `named_scalars(*names)` / `named_scalars(**kw)`     | Batch of named scalars (mutually exclusive)   |
| `identity_matrix_basis(dim)`                        | Isotropic I                          |
| `symmetric_matrix_basis(dim)`                       | Symmetric-matrix templates           |
| `linear_basis(dim)`                                 | Coordinate-extraction identity       |
| `X`, `V`, `x_coordinate`, `v_coordinate`, …         | Coordinate accessors                 |
| `SFI.bases.pairs.*`                                 | Pair-interaction builders (lazy)     |
| `SFI.bases.spde.*`                                  | PDE differential operators (lazy)    |

**Canonical idiom for vector force bases.** For a polynomial library,
use the built-in shortcut `monomials_up_to(order, dim, rank='vector')`.
To lift a scalar basis you built compositionally, call
`scalar_basis.vectorize(dim)` (not
`scalar_basis * unit_vector_basis(dim)`). When the force is literally a
coordinate, use `X(dim)` / `V(dim)`.

See [`docs/source/bases/user_guide.rst`](docs/source/bases/user_guide.rst)
for the decision tree (compositional algebra vs. factories) and
canonical patterns (Lorenz, harmonic UD, limit cycle, double-well).

ABP / active-matter forces are **not** prepackaged as a basis — they are
too example-specific. Build them directly from `SFI.bases.pairs`
primitives (``heading_vector``, ``pbc_displacement``, ``wrap_angle``); see
`examples/_gallery_utils/abp.py` for a worked composition.

### 4.7 `SFI.integrate` — time-averaging engine (backend)

**Backend module — do not call from user code.** Used internally by
the inference engines to build Gram matrices and time-averaged
operators. Central to the design: agents *extending* SFI (new
estimators, new observables) will interact with it; agents *applying*
SFI to data should never need to.

Entry points (for feature work only):

| Symbol                                      | Purpose                                 |
| ------------------------------------------- | --------------------------------------- |
| `integrate(collection, program, reduce)`    | One-shot time-average                   |
| `make_parametric_integrator(program)`       | Build a reusable JIT-compiled integrator|
| `Integrand`, `Term`                         | Compose expressions with Einstein sums  |
| `stream`, `timeop`, `velocity`              | Operand builders                        |

### 4.8 `SFI.utils`

```python
# Plotting (gallery/notebook helpers — not in __all__).  Always take a
# TrajectoryCollection; never reach into coll.datasets[0].X by hand.
from SFI.utils.plotting import SFI_COLORS, dark_fig, dark_ax, wrap_positions
from SFI.utils.plotting import timeseries, timeseries_colored      # x_d(t) lines / colored
from SFI.utils.plotting import phase2d, phase2d_scalar, phase3d    # phase-space (2d/3d/scalar-colored)
from SFI.utils.plotting import trajectory_scatter                  # all-frames density cloud
from SFI.utils.plotting import plot_field, plot_tensor_field, stream_field
from SFI.utils.plotting import plot_profile_1d, plot_field_error   # 1d F/D overlay; 2d ‖ΔF‖ heatmap
from SFI.utils.plotting import plot_particles, plot_nematic_director, plot_rods
from SFI.utils.plotting import plot_spde_snapshot, spatial_acorr2d # gridded/SPDE fields
from SFI.utils.plotting import animate_particles, animate_spde_comparison
from SFI.utils.plotting import comparison_scatter, plot_pareto_front
from SFI.utils.plotting import plot_recovery_bar, plot_recovery_bar_multi, plot_recovery_matrix
from SFI.utils.plotting import plot_time_profile_comparison

# Formatting
from SFI.utils.formatting import model_summary, print_model_comparison

# Numerics (also re-exported via SFI.utils)
from SFI.utils.maths import stable_pinv, sqrtm_psd, solve_or_pinv, fd_velocity
from SFI.utils.maths import default_float_dtype, as_default_float

# Neighbor lists (host-side, call between JIT chunks)
from SFI.utils.neighbors import build_neighbor_csr, make_neighbor_extras, pad_neighbor_csr
```

Inferred-vs-exact read-outs live on the inference object:
`inf.comparison_scatter(model_exact=, field=)`,
`inf.force_comparison_arrays(model_exact=)`,
`inf.coeff_block(block, field=)`, `inf.predict_time_profile(basis, t)`,
`inf.compare_params_to_exact(theta_true, psf=)`.  Collection helpers:
`coll.to_arrays()` / `coll.to_array()` (numpy materialization),
`coll.velocity_array(scheme=)` (finite-diff velocity), `coll.merge([...])`.
Use `coll.T` (frames) and `coll.d` (dimension) instead of
`coll.datasets[0].X.shape`.

`SFI_COLORS` is the semantic palette (`data`, `inferred`, `exact`,
…) — see
[`GALLERY_STYLE_GUIDE.md`](GALLERY_STYLE_GUIDE.md#color-palette).

## 5. What to treat as private / not authoritative

- **Anything prefixed `_`** (modules, functions, attributes) is a private
  internal: it may change without notice, so don't depend on it from user
  code or new features.
- `*.bak` files, if you ever see one, are stale snapshots — ignore them.

Deprecated / soon-to-be-removed symbols (do not use in new code):
- (none currently)

## 6. Testing, building, environment

Full details: [`docs/source/dev_notes.rst`](docs/source/dev_notes.rst).

```bash
source .venv/bin/activate
pytest tests/ -v                             # full suite (CPU-only via conftest)
pytest tests/ -x --ff                        # quick smoke
cd docs && make clean && make html           # fast docs build
cd docs && SFI_DOCS_RUN_GALLERY=1 make html  # full build (slow)
```

JAX persistent cache is opt-in:
`export SFI_JAX_CACHE_DIR=~/.cache/sfi/jax_cache`.

## 7. Conventions

- Import each symbol from the **highest level at which it is exposed**:
  `from SFI import X` when re-exported at top level, otherwise from the
  subpackage (`from SFI.bases import monomials_up_to`) — never from a
  leaf module (`SFI.bases.monomials`, `SFI.bases.linear`).
- New public symbols → add to the submodule's `__init__.py` and, if
  top-level, also to `SFI/__init__.py`.
- Plots: use `SFI_COLORS`, `dark_fig` for dark-theme figures; never pure black.
- Tests: unit tests in `tests/<subpackage>/test_*.py`; files named
  `validate_*.py`, `audit_*.py` are **not**
  collected by pytest.
- Docstrings: numpy-style (Sphinx autodoc).
- JIT: toggle globally with `SFI.statefunc.set_jit(False)` to debug.

## 8. Asking vs guessing

If this file, the playbooks, and the source code together do not
answer a design question, **ask the user** rather than guessing.
Silent assumptions in a feature PR are the main cause of rework here.

## 9. Keeping these agent docs fresh

This file and the machine-readable API map drift out of sync fast.
**When you change public API, you must:**

1. Update §4 of this file (add / remove / rename symbols).
2. Update the matching row in `_project_index/api_map.md` **or**
   regenerate both files:
   ```bash
   python scripts/gen_api_map.py
   ```
3. If you deprecated a symbol, add a row to §5.

Automated guards (run in CI; see
[`.github/workflows/agent_docs.yml`](.github/workflows/agent_docs.yml)):

- `scripts/gen_api_map.py --check` fails if the committed API map is
  stale relative to the current `SFI/__init__.py`.
- A pre-commit hook is available —
  `pre-commit install` once, and the map is regenerated automatically
  on any commit that touches `SFI/**/*.py` (config:
  [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

The add-feature playbook
([`docs/source/agent_playbooks/add_feature.rst`](docs/source/agent_playbooks/add_feature.rst))
has a matching checklist step — don't skip it.
