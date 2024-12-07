# This file takes insperation from https://github.com/neka-nat/probreg/
# Some algorithms are the same, but the implementations make use of
# pytorch as well as some efficiency changes

import torch
from typing import Union
import open3d as o3d

from .dataset import open3d_to_zreg, zreg_to_open3d, zRegPointCloud
from . import utils

# try:
#     _imp_dq = True
# except:
#     _imp_dq = False


__all__ = [
    "transform_points_homogeneous",
    "RigidTransformation",
    "AffineTransformation",
    "NonRigidTransformation",
    "CombinedTransformation",
    "TPSTransformation",
]


def transform_points_homogeneous(
    points: Union[zRegPointCloud, o3d.t.geometry.PointCloud],
    transform_matrix: torch.Tensor,
    return_o3d: bool = False,
):
    """Transforms a set of 3D points using a 4x4 transformation matrix.

    This function handles point data in two formats:
        - A dictionary with a 'pos' key containing a torch.Tensor of shape (n, 3).
        - An Open3D PointCloud object.

    The transformation is applied using homogeneous coordinates.

    Parameters
    ----------
    points : dict or o3d.t.geometry.PointCloud
        The 3D points to transform.
    transform_matrix : torch.Tensor
        The 4x4 transformation matrix.
    return_o3d : bool, optional
        If True, returns an Open3D PointCloud.
        If False (default), returns a dictionary with the transformed points.

    Returns
    -------
    dict or o3d.t.geometry.PointCloud
        The transformed 3D points in the specified format.

    Raises
    ------
    TypeError
        If `points` is a torch.Tensor and not a dictionary or Open3D PointCloud.

    Examples
    --------
    >>> points = {'pos': torch.tensor([[1, 2, 3], [4, 5, 6]])}
    >>> transform_matrix = np.eye(4)  # Identity matrix
    >>> transformed_points = transform_points(points, transform_matrix)
    >>> print(transformed_points)
    """
    if isinstance(points, zRegPointCloud):
        pos = points["pos"]

        # Add a homogeneous coordinate (w=1) to each point
        homogeneous_points = torch.hstack((pos, torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)))

        # Apply the transformation
        transformed_points = homogeneous_points @ torch.tensor(transform_matrix, dtype=pos.dtype, device=pos.device).T

        # Divide by the homogeneous coordinate to get back to 3D
        transformed_points = transformed_points[:, :3] / transformed_points[:, 3:]
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
            t = torch.ones(3, dtype=dtype, device=device)
        self.rot = rot
        self.t = t
        self.scale = scale

    def reset(self):
        self.rot = torch.eye(3, dtype=self.rot.dtype, device=self.rot.device)
        self.t = torch.ones(3, dtype=self.t.dtype, device=self.t.device)

    def _transform(self, points):
        return self.scale * torch.matmul(points, self.rot.T) + self.t  # dot

    def inverse(self):
        # unclean if a squeeze is needed from rot.T @ t
        return RigidTransformation(self.rot.T, -torch.matmul(self.rot.T, self.t) / self.scale, 1.0 / self.scale)

    def __mul__(self, other):
        return RigidTransformation(
            torch.matmul(self.rot, other.rot),
            self.t + self.scale * torch.matmul(self.rot, other.t),  # unclean if a squeeze is needed
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
    """Nonrigid Transformation

    Args:
        w (numpy.array): Weights for kernel.
        points (numpy.array): Source point cloud data.
        beta (float, optional): Parameter for gaussian kernel.
        xp (module): Numpy or Cupy.
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
    """Thin Plate Spline transformaion.

    Args:
        a (numpy.array): Affine matrix.
        v (numpy.array): Translation vector.
        control_pts (numpy.array): Control points.
        kernel (function, optional): Kernel function.
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
        # Convert landmarks to tensor
        uu = self._kernel(torch.tensor(landmarks, **self.fact), control_pts)
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
