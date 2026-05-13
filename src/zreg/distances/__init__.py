"""Distance metrics for point cloud comparison.

This module provides distance metrics for comparing point clouds, including:
- Standard metrics (Euclidean, Manhattan, Minkowski)
- Sliced Wasserstein Distance variants (SWD, ASWD, OSWD, GSWD, PSWD, MaxSWD)

Public API
----------
Protocols:
    DistanceMetric

Functions:
    euclidean_distance, manhattan_distance, minkowski_distance

Classes (nn.Module):
    SlicedWassersteinDistance, MaxSlicedWassersteinDistance,
    ProjectedWassersteinDistance, AdaptiveSlicedWassersteinDistance,
    OrthogonalSlicedWassersteinDistance, GeneralisedSlicedWassersteinDistance
"""

from ._protocol import DistanceMetric
from .general import euclidean_distance, manhattan_distance, minkowski_distance
from .sw_varients import (
    SlicedWassersteinDistance,
    MaxSlicedWassersteinDistance,
    ProjectedWassersteinDistance,
    AdaptiveSlicedWassersteinDistance,
    OrthogonalSlicedWassersteinDistance,
    GeneralisedSlicedWassersteinDistance,
)

__all__ = [
    # Protocol
    "DistanceMetric",
    # Functions
    "euclidean_distance",
    "manhattan_distance",
    "minkowski_distance",
    # SWD variant classes
    "SlicedWassersteinDistance",
    "MaxSlicedWassersteinDistance",
    "ProjectedWassersteinDistance",
    "AdaptiveSlicedWassersteinDistance",
    "OrthogonalSlicedWassersteinDistance",
    "GeneralisedSlicedWassersteinDistance",
]
