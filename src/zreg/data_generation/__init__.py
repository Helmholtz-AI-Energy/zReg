"""Synthetic data generators for the zReg evaluation framework.

This package provides building blocks for generating and corrupting synthetic
3D point cloud trajectories. It is organised into four submodules:

- ``generators``: from-scratch trajectory factory (``generate_trajectory``) and
  single-frame bowl/ball geometry samplers (``sample_ball``, ``sample_bowl``)
- ``transforms``: immutable rigid and affine transform wrappers
  (``apply_rigid``, ``apply_affine``)
- ``corruption``: noise and outlier injection wrappers
  (``add_gaussian_noise``, ``add_outliers``)
- ``labels``: integer label generation and removal utilities
  (``generate_labels``, ``remove_labels``, ``assign_cap_labels``,
  ``assign_gaussian_labels``)

All generators and wrappers accept ``seed: int | None = 42`` for
reproducible stochastic behaviour. Corruption and transform wrappers are
immutable: they accept a ``dict[int, zRegPointCloud]`` and return a new
dict without modifying the input.
"""

from .generators import generate_trajectory, sample_ball, sample_bowl
from .transforms import apply_rigid, apply_affine
from .corruption import add_gaussian_noise, add_outliers
from .labels import generate_labels, remove_labels, assign_cap_labels, assign_gaussian_labels

__all__ = [
    "generate_trajectory",
    "sample_ball",
    "sample_bowl",
    "apply_rigid",
    "apply_affine",
    "add_gaussian_noise",
    "add_outliers",
    "generate_labels",
    "remove_labels",
    "assign_cap_labels",
    "assign_gaussian_labels",
]
