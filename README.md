# This branch is retired

`master` held the original SFI scripts (`StochasticForceInference.py`,
2019–2025), which accompanied the 2020 PRX paper. SFI 2, a rewrite with a
new API, lives on [`main`](https://github.com/ronceray/StochasticForceInference/tree/main):

```bash
pip install StochasticForceInference      # then: import SFI
```

Documentation: https://stochasticforceinference.readthedocs.io

## Old calls in SFI 2

| Original scripts | SFI 2 |
| --- | --- |
| `StochasticTrajectoryData(X, t)` | `TrajectoryCollection.from_arrays(X=X, t=t)` |
| `StochasticForceInference(data)` | `OverdampedLangevinInference(coll)` |
| `UnderdampedLangevinInference(data)` (separate repository) | `SFI.UnderdampedLangevinInference(coll)` |
| `S.compute_drift(basis={"type": "polynomial", "order": n})` | `inf.infer_force_linear(monomials_up_to(order=n, dim=d, rank="vector"))` |
| `S.compute_diffusion(method=..., basis=...)` | `inf.infer_diffusion_linear(basis)` |
| `S.compute_drift_error()` | `inf.compute_force_error()` |
| `S.compute_entropy()` | `inf.compute_entropy_production()` |
| `print_report`, `compare_to_exact`, `simulate_bootstrapped_trajectory` | same names, new signatures |

The theory manual, `SFI_TheoryManual.pdf`, stays on this branch. The original
scripts are preserved at the tag
[`legacy-scripts`](https://github.com/ronceray/StochasticForceInference/tree/legacy-scripts)
and in this branch's history.
