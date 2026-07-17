# SFI/bases/__init__.py
"""
SFI.bases
=========

Library of ready-made basis builders on top of :mod:`SFI.statefunc`.

Submodules
----------
- monomials  : scalar (or lifted) monomial families in x and/or v.
- constants  : structural bases: ones, unit vectors, identity/symmetric matrices.
- linear     : coordinate-extraction helpers (X, V, x_coordinate, ...).
- pairs      : pair-interaction toolkit: radial kernels, PBC, heading vectors.
- spde       : composable spatial operators (Laplacian, Gradient, Divergence, Curl, ...) on regular grids.
"""

from .constants import (
    constant_array,
    dataset_indicator,
    extra_scalar,
    identity_matrix_basis,
    named_scalar,
    named_scalars,
    ones_basis,
    per_dataset_scalar,
    symmetric_matrix_basis,
    time_fourier,
    unit_vector_basis,
)
from .linear import (
    V,
    X,
    field_component,
    frame,
    linear_basis,
    unit_axes,
    v_components,
    v_coordinate,
    v_coordinates,
    x_components,
    x_coordinate,
    x_coordinates,
)
from .monomials import monomials_degree, monomials_up_to

__all__ = [
    # monomials
    "monomials_degree",
    "monomials_up_to",
    # constants / structural
    "ones_basis",
    "unit_vector_basis",
    "identity_matrix_basis",
    "symmetric_matrix_basis",
    "constant_array",
    "named_scalar",
    "named_scalars",
    "extra_scalar",
    "time_fourier",
    "per_dataset_scalar",
    "dataset_indicator",
    # linear / coordinates
    "linear_basis",
    "X",
    "V",
    "x_coordinate",
    "x_coordinates",
    "field_component",
    "v_coordinate",
    "v_coordinates",
    # component / axis unpackers + frame bundle
    "x_components",
    "v_components",
    "unit_axes",
    "frame",
]


# Lazy imports for submodules that have heavier dependencies
def __getattr__(name):
    if name in ("pairs", "spde"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
