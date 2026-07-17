"""
SFI.integrate — Time-averaging integration engine.

Public API
----------
integrate
    Run an Integrand over a TrajectoryCollection and reduce.
make_parametric_integrator
    Build a reusable, jittable integrator for a parametric Integrand.
make_minibatch_parametric_integrator
    Like make_parametric_integrator but also returns a stochastic mini-batch runner.
Integrand
    Compose state expressions and time operands via einsum per time slice.
Term, ExprOperand, TimeOperand, ConstOperand
    Building blocks for Integrand programs.
stream, timeop, velocity, scale, add
    TimeOp constructors for stream access and linear combinations.
"""

from .api import integrate, make_minibatch_parametric_integrator, make_parametric_integrator
from .integrand import ConstOperand, ExprOperand, Integrand, Term, TimeOperand
from .timeops import add, scale, stream, timeop, velocity

__all__ = [
    "integrate",
    "make_parametric_integrator",
    "make_minibatch_parametric_integrator",
    "Integrand",
    "Term",
    "ExprOperand",
    "TimeOperand",
    "ConstOperand",
    "stream",
    "timeop",
    "velocity",
    "scale",
    "add",
]
