"""
SFI.langevin
============

Langevin simulators (overdamped, underdamped, …).

This submodule provides simulation classes built on top of the statefunc API.
They accept `PSF`/`SF` force and diffusion models, enforce shape contracts,
and integrate trajectories using stochastic schemes.

Public classes
--------------
- OverdampedProcess : overdamped Langevin simulator (Euler–Maruyama or
                      stochastic Heun) with entropy/information observables.
- UnderdampedProcess : underdamped Langevin simulator (velocity-Verlet).
"""

from .noise import CompositeNoise, ConservedNoise, NoiseModel, WhiteNoise
from .overdamped import OverdampedProcess
from .underdamped import UnderdampedProcess

__all__ = [
    "OverdampedProcess",
    "UnderdampedProcess",
    "NoiseModel",
    "WhiteNoise",
    "ConservedNoise",
    "CompositeNoise",
]
