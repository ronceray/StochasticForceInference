"""
SFI.inference.sparse — Pluggable sparse model selection.
=========================================================

This sub-package separates the *scoring* of candidate supports from the
*search strategy* used to navigate the combinatorial lattice.

Public API
----------
SparseScorer
    Owns the ``(M, G)`` normal-equations data.  Scores any support *B* by
    solving the restricted linear system and computing the information gain.

SparsityResult
    Container returned by every strategy.  Holds the Pareto front
    (best info / support / coefficients per cardinality *k*) and provides
    information-criterion selection (AIC, BIC, EBIC, PASTIS, SIC).

SparsityStrategy
    Abstract base class for search algorithms.  Subclasses must implement
    ``run(scorer, *, max_k, **kwargs) -> SparsityResult``.

Concrete strategies
~~~~~~~~~~~~~~~~~~~
BeamSearchStrategy
    Bidirectional beam search (the original PASTIS algorithm).
GreedyStepwiseStrategy
    Forward, backward, or bidirectional stepwise selection.
HillClimbStrategy
    Stochastic hill-climbing with random add/remove moves
    (Gerardos & Ronceray, 2025).
STLSQStrategy
    Sequential Thresholded Least Squares (à la SINDy).
LassoStrategy
    Coordinate-descent :math:`\\ell_1`-penalised regression on the normal
    equations.

Benchmark helpers
~~~~~~~~~~~~~~~~~
overlap_metrics
    Compare predicted vs true support (TP / FP / FN / precision / recall).
predictive_nmse
    Normalised mean-squared error on a held-out design matrix.
"""

from .base import SparsityStrategy
from .beam import BeamSearchStrategy
from .greedy import GreedyStepwiseStrategy
from .hillclimb import HillClimbStrategy
from .lasso import LassoStrategy
from .metrics import overlap_metrics, predictive_nmse
from .result import SparsityResult
from .scorer import SparseScorer
from .stlsq import STLSQStrategy

__all__ = [
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
]
