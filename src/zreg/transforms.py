import numpy as np
import torch
from typing import Union
import open3d as o3d

from .dataset import open3d_to_torch, torch_to_open3d


__all__ = [
    "transform_points",
]


def transform_points(
    points: Union[dict, o3d.t.geometry.PointCloud],
    transform_matrix: Union[torch.Tensor, np.ndarray],
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
    transform_matrix : torch.Tensor or np.ndarray
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
    if isinstance(points, dict):
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
        transformed_points = points.transform(transform_matrix)

    elif isinstance(points, torch.Tensor):
        raise TypeError("points given are torch tensor, pass whole Dict or the o3d PointCloud class!")

    else:
        raise TypeError("Unsupported point type. Expected dict or o3d.t.geometry.PointCloud.")

    if return_o3d:
        if isinstance(points, dict):
            return torch_to_open3d(points)  # Assuming you have a function for this conversion
        return transformed_points
    else:
        if isinstance(points, o3d.t.geometry.PointCloud):
            return open3d_to_torch(transformed_points)  # Assuming you have a function for this conversion
        return points
