# SFI/inference/parametric_core/gram.py
"""Packing convention for the Gauss–Newton Gram bundle.

The exact core (:class:`runner.ExactRuns`) packs its per-solve
Gauss–Newton objects into one flat vector ``concat[G, f, H, nll]`` so a
single jitted reduction produces all of them; :func:`unpack_gram`
recovers the pieces.  ``H = ψ_leftᵀ P ψ_left`` is the estimating-function
variance — the sandwich meat on the errors-in-variables path, equal to
``G`` on the symmetric path.
"""

from __future__ import annotations


def unpack_gram(packed, n_params):
    """Flat ``concat[G.ravel(), f, H.ravel(), nll]`` → ``(G, f, H, nll)``."""
    n2 = n_params * n_params
    G = packed[:n2].reshape(n_params, n_params)
    f = packed[n2:n2 + n_params]
    H = packed[n2 + n_params:2 * n2 + n_params].reshape(n_params, n_params)
    return G, f, H, packed[-1]
