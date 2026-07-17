"""
SFI.utils — Mathematical, formatting, and plotting utilities.
"""

from .formatting import model_summary, print_model_comparison
from .maths import as_default_float, default_float_dtype, fd_velocity, solve_or_pinv, sqrtm_psd, stable_pinv

__all__ = [
    "stable_pinv",
    "sqrtm_psd",
    "solve_or_pinv",
    "fd_velocity",
    "model_summary",
    "print_model_comparison",
    "default_float_dtype",
    "as_default_float",
]
