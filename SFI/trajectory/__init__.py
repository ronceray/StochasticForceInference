# SFI/trajectory/__init__.py
"""
Trajectory submodule public API.

Exports:
- TrajectoryDataset / TrajectoryCollection — main user-facing containers
- FunctionExtra / function_extra — pass JAX-traceable callables as extras
- TimeSeriesExtra / time_series_extra — wrap time-varying array extras

I/O (save_trajectory, load_trajectory, columns_and_extras_to_dataset) is
available via ``SFI.trajectory.io`` but is not re-exported here; most users
should use ``TrajectoryCollection.save`` / ``TrajectoryCollection.load``.
"""

from .collection import TrajectoryCollection
from .dataset import (
    FunctionExtra,
    TimeSeriesExtra,
    TrajectoryDataset,
    function_extra,
    time_series_extra,
)

__all__ = [
    "FunctionExtra",
    "function_extra",
    "TimeSeriesExtra",
    "time_series_extra",
    "TrajectoryCollection",
    "TrajectoryDataset",
]
