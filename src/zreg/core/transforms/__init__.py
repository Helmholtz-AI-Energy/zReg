"""Point cloud transformation classes and utilities.

This package provides transformation classes for manipulating 3D point clouds,
supporting rigid, affine, non-rigid (RBF), combined, and TPS transformations.

Public API
==========

Base Class
----------
TransformBase
    Abstract base for all transformations

Transformation Classes
----------------------
RigidTransformation
    Rotation, translation, and uniform scaling
AffineTransformation
    General affine matrix transformation
NonRigidTransformation
    RBF kernel-based deformation
CombinedTransformation
    Rigid + non-rigid deformation
TPSTransformation
    Thin Plate Spline transformation

Functions
---------
transform_points_homogeneous
    Apply 4x4 matrix transformations to point clouds

Examples
========

Rigid transformation:

    >>> import torch
    >>> from zreg.core.transforms import RigidTransformation
    >>> tf = RigidTransformation(t=torch.tensor([1., 0., 0.]))
    >>> points = torch.randn(100, 3)
    >>> transformed = tf.transform(points)

Composition:

    >>> tf1 = RigidTransformation(t=torch.tensor([1., 0., 0.]))
    >>> tf2 = RigidTransformation(t=torch.tensor([0., 1., 0.]))
    >>> tf_composed = tf1 * tf2

Non-rigid transformation:

    >>> from zreg.core.transforms import NonRigidTransformation
    >>> points = torch.randn(50, 3)
    >>> w = torch.randn(50, 3) * 0.01
    >>> tf = NonRigidTransformation(w=w, points=points, beta=2.0)
    >>> transformed = tf.transform(points)

Homogeneous coordinates:

    >>> from zreg.core.transforms import transform_points_homogeneous
    >>> from zreg.core.dataset import zRegPointCloud
    >>> pc = zRegPointCloud(pos=torch.randn(10, 3))
    >>> T = torch.eye(4)
    >>> result = transform_points_homogeneous(pc, T)
"""

from .base import TransformBase
from .rigid import RigidTransformation
from .affine import AffineTransformation
from .nonrigid import NonRigidTransformation
from .combined import CombinedTransformation
from .tps import TPSTransformation
from .homogeneous import transform_points_homogeneous

__all__ = [
    "TransformBase",
    "RigidTransformation",
    "AffineTransformation",
    "NonRigidTransformation",
    "CombinedTransformation",
    "TPSTransformation",
    "transform_points_homogeneous",
]
