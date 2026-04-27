"""Point cloud transformation classes.

This module provides transformation classes for manipulating 3D point clouds,
supporting rigid, affine, non-rigid (RBF), combined, and TPS transformations.

Public API
----------
Base Class:
    TransformBase - Abstract base for all transformations (user subclassing)

Transformation Classes:
    RigidTransformation - Rotation, translation, and uniform scaling
    AffineTransformation - General affine matrix transformation
    NonRigidTransformation - RBF kernel-based deformation
    CombinedTransformation - Rigid + non-rigid deformation
    TPSTransformation - Thin Plate Spline transformation

Functions:
    transform_points_homogeneous - Apply 4x4 matrix to point cloud

Kernel Patterns
---------------
NonRigidTransformation and TPSTransformation share a common kernel-based
deformation pattern:

1. Both use kernel functions from utils.py (rbf_kernel, tps_kernel)
2. Both store a kernel matrix G = kernel(control_pts, control_pts)
3. Both compute deformation as: points + G @ weights

The kernel functions differ in their radial basis:
- RBF: exp(-beta * ||x - y||^2)
- TPS: ||x - y||^2 * log(||x - y||)

See NonRigidTransformation and TPSTransformation docstrings for details.
"""

# This file takes insperation from https://github.com/neka-nat/probreg/
# Some algorithms are the same, but the implementations make use of
# pytorch as well as some efficiency changes

import torch
import open3d as o3d

from .dataset import open3d_to_zreg, zreg_to_open3d, zRegPointCloud
from . import utils

# try:
#     _imp_dq = True
# except:
#     _imp_dq = False


__all__ = [
    "TransformBase",
    "RigidTransformation",
    "AffineTransformation",
    "NonRigidTransformation",
    "CombinedTransformation",
    "TPSTransformation",
    "transform_points_homogeneous",
]


def transform_points_homogeneous(
    points: zRegPointCloud | o3d.t.geometry.PointCloud,
    transform_matrix: torch.Tensor,
    return_o3d: bool = False,
) -> zRegPointCloud | o3d.t.geometry.PointCloud:
    """Transform points using a 4x4 homogeneous transformation matrix.

    This function applies a 4x4 transformation matrix to 3D points using
    homogeneous coordinates. It handles both zRegPointCloud and Open3D
    point cloud formats.

    Homogeneous Coordinate Handling
    -------------------------------
    For zRegPointCloud inputs, the transformation process is:

    1. Extend points to homogeneous: [x, y, z] -> [x, y, z, 1]
    2. Apply transformation: [x', y', z', w'] = [x, y, z, 1] @ M.T
    3. Perspective divide: [x', y', z'] / w'

    W-Clamping: The w coordinate is clamped to avoid division by near-zero
    values. This prevents inf/NaN results when transformation matrices
    have degenerate bottom rows. The clamping uses torch.finfo(dtype).eps
    as the minimum value.

    For Open3D inputs, the native transform() method is used.

    Parameters
    ----------
    points : zRegPointCloud or o3d.t.geometry.PointCloud
        The 3D points to transform.
    transform_matrix : torch.Tensor
        4x4 transformation matrix.
    return_o3d : bool, optional
        If True, return Open3D PointCloud. Default False.

    Returns
    -------
    zRegPointCloud or o3d.t.geometry.PointCloud
        Transformed points in requested format.

    Raises
    ------
    TypeError
        If points is a raw torch.Tensor (must be wrapped in zRegPointCloud).

    Notes
    -----
    This function modifies the input zRegPointCloud in-place (updates 'pos' key)
    and returns the same object. For Open3D inputs, a new object is returned.

    See Also
    --------
    RigidTransformation : For rigid transformations without homogeneous coords
    AffineTransformation : For affine transformations
    """
    if isinstance(points, zRegPointCloud):
        pos = points["pos"]

        # Add a homogeneous coordinate (w=1) to each point
        homogeneous_points = torch.hstack((pos, torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)))

        # Apply the transformation
        transformed_points = homogeneous_points @ torch.tensor(transform_matrix, dtype=pos.dtype, device=pos.device).T

        # Divide by the homogeneous coordinate to get back to 3D
        # Clamp w to avoid division by near-zero (per D-14, D-15)
        w = transformed_points[:, 3:]
        w = torch.clamp(w, min=torch.finfo(pos.dtype).eps)
        transformed_points = transformed_points[:, :3] / w
        points["pos"] = transformed_points

    elif isinstance(points, o3d.t.geometry.PointCloud):
        # Use Open3D's built-in transform method
        try:
            # convert to numpy from torch if needed...?
            transform_matrix = transform_matrix.numpy()
        except AttributeError:
            pass
        transformed_points = points.transform(transform_matrix)

    elif isinstance(points, torch.Tensor):
        raise TypeError("points given are torch tensor, pass whole Dict or the o3d PointCloud class!")

    else:
        raise TypeError("Unsupported point type. Expected dict or o3d.t.geometry.PointCloud.")

    if return_o3d:
        if isinstance(points, zRegPointCloud):
            return zreg_to_open3d(points)  # Assuming you have a function for this conversion
        return transformed_points
    else:
        if isinstance(points, o3d.t.geometry.PointCloud):
            return open3d_to_zreg(transformed_points)  # Assuming you have a function for this conversion
        return points


class TransformBase(object):
    """Abstract base class for point cloud transformations.

    All transformation classes inherit from this base, which provides:
    - A standard `transform()` method that handles both 3D and extended point data
    - A `_transform()` template method for subclasses to implement

    Subclassing
    -----------
    To create a custom transformation, subclass TransformBase and implement
    `_transform(self, points)`:

        class MyTransform(TransformBase):
            def _transform(self, points):
                # points: torch.Tensor of shape (n, 3)
                return transformed_points

    The `transform()` method handles points with extra columns (e.g., colors)
    by only transforming the first 3 columns (xyz) and preserving the rest.

    See Also
    --------
    RigidTransformation : Rotation + translation + scale
    AffineTransformation : General affine transformation
    NonRigidTransformation : RBF kernel deformation
    TPSTransformation : Thin Plate Spline deformation
    """

    def __init__(self) -> None:
        pass

    def _transform(points): ...

    def transform(self, points):
        if points.shape[1] <= 3:
            return self._transform(points)
        # else
        ret = points.clone()
        ret[:, :3] = self._transform(ret[:, :3])
        return ret


class RigidTransformation(TransformBase):
    """Rigid Transformation

    Args:
        rot (numpy.ndarray, optional): Rotation matrix.
        t (numpy.ndarray, optional): Translation vector.
        scale (Float, optional): Scale factor.
    """

    def __init__(self, rot=None, t=None, scale=1.0, device=None, dtype=None):
        super(RigidTransformation, self).__init__()
        if rot is None:
            rot = torch.eye(3, dtype=dtype, device=device)
        if t is None:
            t = torch.zeros(3, dtype=dtype, device=device)

        if dtype is not None:
            rot = rot.to(dtype=dtype)
            t = t.to(dtype=dtype)
        if device is not None:
            rot = rot.to(device=device)
            t = t.to(device=device)
        self.rot = rot
        self.t = t
        self.scale = scale

    def reset(self):
        self.rot = torch.eye(3, dtype=self.rot.dtype, device=self.rot.device)
        self.t = torch.zeros(3, dtype=self.t.dtype, device=self.t.device)

    def _transform(self, points):
        return self.scale * torch.matmul(points, self.rot.T) + self.t  # dot

    def inverse(self):
        # unclean if a squeeze is needed from rot.T @ t
        return RigidTransformation(self.rot.T, -torch.matmul(self.rot.T, self.t) / self.scale, 1.0 / self.scale)

    def __mul__(self, other):
        """Compose two rigid transformations.

        Computes the composition self * other, which applies other first,
        then self. The composed transformation satisfies:

            (self * other).transform(points) == self.transform(other.transform(points))

        Composition Validation
        ----------------------
        The composed rotation matrix is validated:
        - Determinant must be within 1e-6 of 1.0 (preserves orientation)
        - Condition number must be below 1e6 (numerical stability)

        Parameters
        ----------
        other : RigidTransformation
            The transformation to apply first.

        Returns
        -------
        RigidTransformation
            The composed transformation.

        Raises
        ------
        ValueError
            If composed rotation has invalid determinant or condition number.

        Examples
        --------
        >>> tf1 = RigidTransformation(t=torch.tensor([1., 0., 0.]))
        >>> tf2 = RigidTransformation(t=torch.tensor([0., 1., 0.]))
        >>> tf_composed = tf1 * tf2
        >>> # tf_composed translates by (1, 1, 0)
        """
        rot_composed = torch.matmul(self.rot, other.rot)
        # Validate composed rotation (per D-10, D-11, D-12, D-13)
        det = torch.det(rot_composed)
        if abs(det.item() - 1.0) > 1e-6:
            raise ValueError(
                f"RigidTransformation composition produced invalid rotation: "
                f"det={det.item():.6f}, expected 1.0"
            )
        cond = torch.linalg.cond(rot_composed)
        if cond.item() > 1e6:
            raise ValueError(
                f"RigidTransformation composition produced invalid rotation: "
                f"det={det.item():.6f}, cond={cond.item():.2e}"
            )
        return RigidTransformation(
            rot_composed,
            self.t + self.scale * torch.matmul(self.rot, other.t),
            self.scale * other.scale,
        )


class AffineTransformation(TransformBase):
    """Affine Transformation

    Args:
        b (numpy.ndarray, optional): Affine matrix.
        t (numpy.ndarray, optional): Translation vector.
        xp (module, optional): Numpy or Cupy.
    """

    def __init__(self, b=None, t=None, device=None, dtype=None):
        super(AffineTransformation, self).__init__()
        if b is None:
            b = torch.eye(3, dtype=dtype, device=device)
        if t is None:
            t = torch.ones(3, dtype=dtype, device=device)
        self.b = b
        self.t = t

    def _transform(self, points):
        return torch.matmul(points, self.b.T) + self.t


class NonRigidTransformation(TransformBase):
    """Non-rigid transformation using RBF (Radial Basis Function) kernel.

    This transformation computes point displacements using an RBF kernel
    matrix and learned weights. The deformation at each point is a weighted
    sum of Gaussian basis functions centered at the control points.

    Kernel Pattern
    --------------
    This class shares a common kernel-based deformation pattern with
    TPSTransformation:

    1. Kernel matrix: G = rbf_kernel(points, points, beta)
       Shape: (n_points, n_points)
    2. Deformation: transformed = points + G @ weights
       Where weights has shape (n_points, 3)

    The RBF kernel is: k(x, y) = exp(-beta * ||x - y||^2)

    See utils.rbf_kernel() for the kernel implementation.

    Parameters
    ----------
    w : torch.Tensor
        Deformation weights, shape (n_points, 3).
    points : torch.Tensor
        Control points for kernel computation, shape (n_points, 3).
    beta : float, optional
        Kernel bandwidth parameter. Default 2.0.
        Smaller values = smoother deformation, larger = more local.

    Attributes
    ----------
    g : torch.Tensor
        Precomputed kernel matrix, shape (n_points, n_points).
    w : torch.Tensor
        Deformation weights.

    See Also
    --------
    TPSTransformation : TPS kernel variant with different basis function
    utils.rbf_kernel : The kernel function used
    CombinedTransformation : Combines rigid with non-rigid
    """

    def __init__(self, w, points, beta=2.0):
        super(NonRigidTransformation, self).__init__()
        self.g = utils.rbf_kernel(points, points, beta)
        self.w = w

    def _transform(self, points):
        return points + torch.matmul(self.g, self.w)  # dot, maybe need squeeze


class CombinedTransformation(TransformBase):
    """Combined Transformation

    Args:
        rot (numpy.array, optional): Rotation matrix.
        t (numpy.array, optional): Translation vector.
        scale (float, optional): Scale factor.
        v (numpy.array, optional): Nonrigid term.
    """

    def __init__(self, rot=None, t=None, scale=1.0, v=0.0):
        super(CombinedTransformation, self).__init__()
        self.rigid_trans = RigidTransformation(rot, t, scale)
        self.v = v

    def _transform(self, points):
        return self.rigid_trans.transform(points + self.v)


class TPSTransformation(TransformBase):
    """Thin Plate Spline (TPS) transformation.

    TPS is a spline-based interpolation method that minimizes bending
    energy while passing through control points. It's commonly used for
    smooth non-rigid registration.

    Kernel Pattern
    --------------
    This class shares a common kernel-based deformation pattern with
    NonRigidTransformation:

    1. Kernel function: k(x, y) = ||x - y||^2 * log(||x - y||)
    2. Basis construction: combines affine and kernel terms
    3. Transformation: basis @ [affine_params; kernel_weights]

    The TPS kernel naturally produces smoother deformations than RBF
    for large displacements.

    See utils.tps_kernel() for the kernel implementation.

    Parameters
    ----------
    a : torch.Tensor
        Affine parameters, shape (d+1, d) where d is dimension.
    v : torch.Tensor
        Kernel weights, shape (n_control, d).
    control_pts : torch.Tensor
        Control points, shape (n_control, d).
    kernel : callable, optional
        Kernel function. Default is utils.tps_kernel.

    Methods
    -------
    prepare(landmarks)
        Compute basis and kernel matrices for given landmarks.
    transform_basis(basis)
        Apply transformation given precomputed basis.

    See Also
    --------
    NonRigidTransformation : RBF kernel variant
    utils.tps_kernel : The kernel function used
    """

    def __init__(self, a, v, control_pts, kernel=utils.tps_kernel):
        super(TPSTransformation, self).__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts
        self._kernel = kernel
        self.fact = {"dtype": self.a.dtype, "device": self.a.device}

    def prepare(self, landmarks):
        control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape
        # Concatenate tensors
        pm = torch.cat([torch.ones((m, 1), **self.fact), torch.tensor(landmarks, **self.fact)], dim=1)
        pn = torch.cat([torch.ones((n, 1), **self.fact), control_pts], dim=1)
        u, _, _ = torch.linalg.svd(pn, full_matrices=True)  # T or F?
        pp = u[:, d + 1 :]
        kk = self._kernel(control_pts, control_pts)
        # Convert landmarks to tensor - transpose kernel output to get shape (m, n)
        uu = self._kernel(torch.tensor(landmarks, **self.fact), control_pts).T
        basis = torch.cat([pm, torch.matmul(uu, pp)], dim=1)  # Use torch.matmul for matrix multiplication
        kernel = torch.matmul(pp.T, torch.matmul(kk, pp))
        return basis, kernel

    def transform_basis(self, basis):
        return torch.matmul(basis, torch.cat((self.a, self.v), dim=0))  # dot

    def _transform(self, points):
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)


# TODO: this is only used in one place, and would need to be adapted for torch...
# https://github.com/neka-nat/probreg/blob/master/probreg/filterreg.py
# class DeformableKinematicModel(object):
#     """Deformable Kinematic Transformation

#     Args:
#         dualquats (:obj:`list` of :obj:`dq3d.dualquat`): Transformations for each link.
#         weights (DeformableKinematicModel.SkinningWeight): Skinning weight.
#     """

#     class SkinningWeight(torch.Tensor):
#         """SkinningWeight
#                 Transformations and weights for each point.

#         .       tf = SkinningWeight['val'][0] * dualquats[SkinningWeight['pair'][0]] + SkinningWeight['val'][1] * dualquats[SkinningWeight['pair'][1]]
#         """

#         def __new__(cls, n_points):
#             return super(DeformableKinematicModel.SkinningWeight, cls).__new__(
#                 cls, n_points, dtype=[("pair", "i4", 2), ("val", "f4", 2)]
#             )

#         @property
#         def n_nodes(self):
#             return self["pair"].max() + 1

#         def pairs_set(self):
#             return itertools.permutations(range(self.n_nodes), 2)

#         def in_pair(self, pair):
#             """
#             Return indices of the pairs equal to the given pair.
#             """
#             return torch.argwhere((self["pair"] == pair).all(1)).flatten()

#     @classmethod
#     def make_weight(cls, pairs, vals):
#         weights = cls.SkinningWeight(pairs.shape[0])
#         weights["pair"] = pairs
#         weights["val"] = vals
#         return weights

#     def __init__(self, dualquats, weights):
#         if not _imp_dq:
#             raise RuntimeError("No dq3d python package, deformable kinematic model not available.")
#         super(DeformableKinematicModel, self).__init__()
#         self.weights = weights
#         self.dualquats = dualquats
#         self.trans = [op.dlb(w[1], [self.dualquats[i] for i in w[0]]) for w in self.weights]

#     def transform(self, points):
#         return np.array([t.transform_point(p) for t, p in zip(self.trans, points)])
