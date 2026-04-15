"""Coherent Point Drift (CPD) registration algorithms.

This package provides implementations of the CPD algorithm for point cloud
registration, supporting rigid, affine, and non-rigid transformations.
"""

# Types (always exported)
from ._types import EstepResult, MstepResult

# Kernel utilities
from .kernels import rbf_kernel_matrix

# Public API will be extended in subsequent plans
__all__ = [
    # Types
    "EstepResult",
    "MstepResult",
    # Utilities
    "rbf_kernel_matrix",
]
