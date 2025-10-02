"""
SFI – Stochastic Force Inference - main entry point
"""

from importlib.metadata import version as _v

# Public API --------------------------------------------------------------
from .SFI_data import StochasticTrajectoryData
from .OLI_inference import OverdampedLangevinInference
from .ULI_inference import UnderdampedLangevinInference
from . import  OLI_bases, OLI_inference, SFI_data, SFI_Langevin, SFI_plotting_toolkit, SFI_sparsity, SFI_utils, ULI_bases, ULI_inference


__all__ = [
    "StochasticTrajectoryData",
    "OverdampedLangevinInference",
    "UnderdampedLangevinInference",
    "OLI_bases", "OLI_inference", "SFI_data", "SFI_Langevin", "SFI_plotting_toolkit", "SFI_sparsity", "SFI_utils", "ULI_bases", "ULI_inference"
]

def _resolve_version():
    for dist in ("stochasticforceinference", "SFI"):
        try:
            return _v(dist)
        except PackageNotFoundError:
            continue
    return "0+unknown"

__version__ = _resolve_version()
del _v, PackageNotFoundError, _resolve_version



