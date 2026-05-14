"""Homogeneous coordinate transformations using 4×4 matrices."""

import torch

from ..dataset import zRegPointCloud, open3d_to_zreg, zreg_to_open3d


def _get_open3d():
    try:
        import open3d as o3d
        return o3d, True
    except (ImportError, OSError):
        return None, False

__all__ = ["transform_points_homogeneous"]


def transform_points_homogeneous(
    points: "zRegPointCloud | o3d.t.geometry.PointCloud",
    transform_matrix: torch.Tensor,
    return_o3d: bool = False,
) -> "zRegPointCloud | o3d.t.geometry.PointCloud":
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
    as the minimum value (per D-14, D-15).

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

    Examples
    --------
    >>> from zreg.dataset import zRegPointCloud
    >>> import torch
    >>> pc = zRegPointCloud(pos=torch.randn(10, 3))
    >>> T = torch.eye(4)
    >>> T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation
    >>> result = transform_points_homogeneous(pc, T)

    See Also
    --------
    RigidTransformation : For rigid transformations without homogeneous coords
    AffineTransformation : For affine transformations
    """
    o3d, HAS_OPEN3D = _get_open3d()
    if isinstance(points, zRegPointCloud):
        pos = points["pos"]

        # Add a homogeneous coordinate (w=1) to each point
        homogeneous_points = torch.hstack(
            (pos, torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device))
        )

        # Apply the transformation
        transform_tensor = torch.as_tensor(transform_matrix, dtype=pos.dtype, device=pos.device)
        transformed_points = homogeneous_points @ transform_tensor.T

        # Divide by the homogeneous coordinate to get back to 3D
        # Clamp w to avoid division by near-zero (per D-14, D-15)
        w = transformed_points[:, 3:]
        w = torch.clamp(w, min=torch.finfo(pos.dtype).eps)
        transformed_points = transformed_points[:, :3] / w
        points["pos"] = transformed_points

    elif HAS_OPEN3D and isinstance(points, o3d.t.geometry.PointCloud):
        # Use Open3D's built-in transform method
        try:
            transform_matrix = transform_matrix.numpy()
        except AttributeError:
            pass
        transformed_points = points.transform(transform_matrix)

    elif isinstance(points, torch.Tensor):
        raise TypeError(
            "points given are torch tensor, pass whole Dict or the o3d PointCloud class!"
        )

    else:
        raise TypeError(
            "Unsupported point type. Expected dict or o3d.t.geometry.PointCloud."
        )

    if return_o3d:
        if isinstance(points, zRegPointCloud):
            return zreg_to_open3d(points)
        return transformed_points
    else:
        if HAS_OPEN3D and isinstance(points, o3d.t.geometry.PointCloud):
            return open3d_to_zreg(transformed_points)
        return points
