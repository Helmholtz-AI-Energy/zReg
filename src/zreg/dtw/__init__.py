"""Dynamic Time Warping for temporal alignment of 3D point cloud trajectories.

This package provides functionality to compute optimal temporal alignment between
two sequences of point clouds using Dynamic Time Warping (DTW).

Public API
----------
Classes:
    DynamicTimeWarping : Main DTW algorithm class for temporal alignment

Dataclasses:
    DTWResult : Container for DTW computation results

Functions:
    compose_constraints : Combine multiple windowing constraint functions

Note
----
The DTW algorithm is metric-agnostic by design. It delegates metric
computation to pairwise_distance_matrix.create_pairwise_distance_matrix().

For custom constraints beyond the built-in Sakoe-Chiba band (window parameter):
1. Use compose_constraints() to combine constraint functions
2. Use set_cost_matrix() with a pre-masked cost matrix for arbitrary constraints
"""

# Result dataclass
from .result import DTWResult

# Constraint utilities
from .constraints import compose_constraints

# Main algorithm class
from .core import DynamicTimeWarping

__all__ = [
    # Classes
    "DynamicTimeWarping",
    # Dataclasses
    "DTWResult",
    # Functions
    "compose_constraints",
]
