"""
SFI.inference — Stochastic Force Inference engines.

Public classes
--------------
OverdampedLangevinInference
    Inference for overdamped Langevin dynamics (dx/dt = F(x) + sqrt(2D) dW).
UnderdampedLangevinInference
    Inference for underdamped Langevin dynamics (velocities unobserved).
InferenceResultSF
    Callable fitted state function carrying parameter covariance and metadata.
SparseScorer
    Scores candidate supports via the normal equations (M, G).
SparsityResult
    Immutable container returned by every sparsity strategy.  Provides
    information-criterion selection (AIC, BIC, PASTIS, SIC).
BeamSearchStrategy / GreedyStepwiseStrategy / HillClimbStrategy / STLSQStrategy / LassoStrategy
    Pluggable search strategies for sparse model selection.
overlap_metrics / predictive_nmse
    Benchmark helpers for comparing to ground truth.
"""

from .overdamped import OverdampedLangevinInference
from .result import InferenceResultSF, kernel_predict_ci
from .serialization import load_model, load_results, save_model, save_results
from .sparse import (
    BeamSearchStrategy,
    GreedyStepwiseStrategy,
    HillClimbStrategy,
    LassoStrategy,
    SparseScorer,
    SparsityResult,
    SparsityStrategy,
    STLSQStrategy,
    overlap_metrics,
    predictive_nmse,
)
from .underdamped import UnderdampedLangevinInference

__all__ = [
    "OverdampedLangevinInference",
    "UnderdampedLangevinInference",
    "InferenceResultSF",
    "SparseScorer",
    "SparsityResult",
    "SparsityStrategy",
    "BeamSearchStrategy",
    "GreedyStepwiseStrategy",
    "HillClimbStrategy",
    "STLSQStrategy",
    "LassoStrategy",
    "overlap_metrics",
    "predictive_nmse",
    "save_results",
    "load_results",
    "save_model",
    "load_model",
]
