# SFI/statefunc/__init__.py
"""
High-level API for state functions.

Public classes:
    - Basis: deterministic dictionary façade
    - PSF: parametric family F(x; θ)
    - SF: state-function with fixed θ
    - Interactor: local interaction expression (pre-dispatch)
    - StateExpr: immutable expression tree base class

Factory helpers:
    - make_basis, make_psf, make_sf, make_interactor

Control:
    - set_jit: enable/disable JIT on __call__

Power-user primitives:
    - Rank, ParamSpec, ParamSuite
"""

from .basis import Basis
from .core.runtime import set_jit
from .factory import make_basis, make_interactor, make_psf, make_sf
from .interactor import Interactor
from .nodes import Rank
from .params import ParamSpec, ParamSuite
from .psf import PSF
from .sf import SF
from .stateexpr import StateExpr

__all__ = [
    # façades
    "Basis",
    "PSF",
    "SF",
    "Interactor",
    "StateExpr",
    "set_jit",
    # factory
    "make_basis",
    "make_psf",
    "make_sf",
    "make_interactor",
    # power-user primitives
    "Rank",
    "ParamSpec",
    "ParamSuite",
]
