"""Synthetic data generators for the zReg evaluation framework.

This package provides building blocks for generating and corrupting synthetic
3D point cloud trajectories. It is organised into four submodules:

- ``generators``: from-scratch trajectory factory (``generate_trajectory``)
- ``transforms``: immutable rigid and affine transform wrappers
  (``apply_rigid``, ``apply_affine``)
- ``corruption``: noise and outlier injection wrappers
  (``add_gaussian_noise``, ``add_outliers``) — added in Plan 02
- ``labels``: integer label generation and removal utilities
  (``generate_labels``, ``remove_labels``) — added in Plan 02

All generators and wrappers accept ``seed: int | None = 42`` for
reproducible stochastic behaviour. Corruption and transform wrappers are
immutable: they accept a ``dict[int, zRegPointCloud]`` and return a new
dict without modifying the input.
"""

from .generators import generate_trajectory
from .transforms import apply_rigid, apply_affine

__all__ = ["apply_affine", "apply_rigid", "generate_trajectory"]
