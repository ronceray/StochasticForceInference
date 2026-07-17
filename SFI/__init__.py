"""
SFI – Stochastic Force Inference - main entry point
"""

import logging as _logging
import os as _os
from importlib.metadata import version as _v

# ---------------------------------------------------------------------------
# JAX persistent compilation cache  (opt-in)
# ---------------------------------------------------------------------------
# Loading cached XLA executables triggers a per-hit C++ warning from the
# PjRt-IFRT layer ("Assume version compatibility …"), so the cache is OFF
# by default.  Enable it by setting SFI_JAX_CACHE_DIR to a directory path,
# e.g.  export SFI_JAX_CACHE_DIR=~/.cache/sfi/jax_cache
# XLA compilations dominate small-data runtime (~5 s for lorenz_demo T=100);
# caching cuts repeat runs to ~1 s.
_cache_dir = _os.environ.get("SFI_JAX_CACHE_DIR", "")
if _cache_dir:
    import jax

    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    # Many SFI compilations are 50–500 ms each; the default 1 s threshold
    # would skip them, defeating the purpose of the cache.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    del jax
del _cache_dir

# Library-level null handler (silences output unless the user configures logging)
_logging.getLogger(__name__).addHandler(_logging.NullHandler())


def enable_logging(level: str = "INFO") -> None:
    """Quick helper to turn on SFI log output.

    >>> import SFI
    >>> SFI.enable_logging()          # INFO-level messages
    >>> SFI.enable_logging("DEBUG")   # everything
    """
    _logger = _logging.getLogger(__name__)
    _logger.setLevel(getattr(_logging, level.upper(), _logging.INFO))
    if not any(
        isinstance(h, _logging.StreamHandler) for h in _logger.handlers if not isinstance(h, _logging.NullHandler)
    ):
        _handler = _logging.StreamHandler()
        _handler.setFormatter(_logging.Formatter("[SFI %(levelname)s] %(message)s"))
        _logger.addHandler(_handler)


# Public API --------------------------------------------------------------
from . import bases, diagnostics, inference, integrate, langevin, trajectory, utils

# Convenience re-exports so users can write ``from SFI import ...``
from .diagnostics import DiagnosticsReport, DynamicsOrderReport, assess, classify_dynamics
from .inference import (
    InferenceResultSF,
    OverdampedLangevinInference,
    UnderdampedLangevinInference,
)
from .statefunc import PSF, SF, Basis, make_sf
from .trajectory import TrajectoryCollection, TrajectoryDataset

__version__ = _v("StochasticForceInference")
del _v
