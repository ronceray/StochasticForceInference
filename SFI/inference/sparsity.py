"""
===============================================================================
SFI.inference.sparsity — Sparse model selection
===============================================================================

Public façade that re-exports the key symbols from
:mod:`SFI.inference.sparse`.  User code should import from here or from
the ``SFI.inference`` namespace::

    from SFI.inference import SparseScorer, SparsityResult
    from SFI.inference.sparse import BeamSearchStrategy, LassoStrategy

The heavy lifting lives in ``SFI.inference.sparse``; this module keeps
the import path short and backward-compatible.
"""

# Re-export everything downstream code may need
from SFI.inference.sparse import (  # noqa: F401
    BeamSearchStrategy,
    GreedyStepwiseStrategy,
    LassoStrategy,
    SparseScorer,
    SparsityResult,
    SparsityStrategy,
    STLSQStrategy,
    overlap_metrics,
    predictive_nmse,
)

__all__ = [
    "SparseScorer",
    "SparsityResult",
    "SparsityStrategy",
    "BeamSearchStrategy",
    "GreedyStepwiseStrategy",
    "STLSQStrategy",
    "LassoStrategy",
    "overlap_metrics",
    "predictive_nmse",
]
