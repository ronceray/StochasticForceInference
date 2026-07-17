"""
SFI.inference.sparse.base — Strategy protocol
==============================================

Every sparse-selection algorithm subclasses :class:`SparsityStrategy`
and implements :meth:`run`, which receives a :class:`SparseScorer` and
returns a :class:`SparsityResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .result import SparsityResult
from .scorer import SparseScorer


class SparsityStrategy(ABC):
    """Abstract base class for sparsity search strategies.

    Subclasses must implement :meth:`run`.
    """

    #: Short identifier used in :attr:`SparsityResult.method`.
    name: str = "base"

    @abstractmethod
    def run(self, scorer: SparseScorer, *, max_k: int, **kwargs) -> SparsityResult:
        """Execute the search and return a :class:`SparsityResult`.

        Parameters
        ----------
        scorer : SparseScorer
            Provides ``info_and_coeffs`` / ``vmap_info`` for evaluating
            candidate supports.
        max_k : int
            Maximum model size to explore.

        Returns
        -------
        SparsityResult
        """
        ...
