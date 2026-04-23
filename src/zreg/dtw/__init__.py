"""Dynamic Time Warping for temporal alignment of 3D point cloud trajectories.

This package provides functionality to compute optimal temporal alignment between
two sequences of point clouds using Dynamic Time Warping (DTW).

Public API
----------
Classes:
    DynamicTimeWarping : Main DTW algorithm class

Dataclasses:
    DTWResult : Container for DTW computation results

Functions:
    compose_constraints : Combine multiple windowing constraint functions

Note: The DTW algorithm is metric-agnostic by design. It delegates metric
computation to pairwise_distance_matrix.create_pairwise_distance_matrix().
For custom constraints, use set_cost_matrix() with a pre-masked matrix.
"""

# Result dataclass
from .result import DTWResult

# Constraint utilities
from .constraints import compose_constraints

# DynamicTimeWarping class will be added in Plan 02

__all__ = [
    # Dataclasses
    "DTWResult",
    # Functions
    "compose_constraints",
    # Classes (added in Plan 02)
    # "DynamicTimeWarping",
]
