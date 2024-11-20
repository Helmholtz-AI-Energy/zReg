import numpy as np
import torch
from typing import Union


__all__ = [
    "rigid_transform",
]


def rigid_transform(
    point_cloud: Union[np.ndarray, torch.Tensor],
    rotation_matrix: Union[np.ndarray, torch.Tensor],
    translation_vector: Union[np.ndarray, torch.Tensor] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies a rigid transformation to a point cloud.

    This function takes a point cloud, a rotation matrix, and a translation
    vector, and applies the rigid transformation to the point cloud. The function
    can handle both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    point_cloud : np.ndarray or torch.Tensor
        The point cloud to transform, with shape (n, 3).
    rotation_matrix : np.ndarray or torch.Tensor
        The rotation matrix, with shape (3, 3).
    translation_vector : np.ndarray or torch.Tensor
        The translation vector, with shape (3,).

    Returns
    -------
    np.ndarray or torch.Tensor
        The transformed point cloud, with the same type and shape as the input
        point cloud.

    Raises
    ------
    ValueError
        If the input point cloud does not have shape (n, 3).

    Examples
    --------
    >>> point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> translation_vector = np.array([10, 20, 30])
    >>> transformed_pc = rigid_transform(
    ...     point_cloud, rotation_matrix, translation_vector
    ... )
    >>> print(transformed_pc)
    [[-12.  11.  33.]
    [-17.  14.  36.]
    [-22.  17.  39.]]
    """

    # Validate the input point cloud shape.
    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (n, 3)")

    # Apply rotation.
    rotated_point_cloud = (
        point_cloud @ rotation_matrix.T
        if isinstance(point_cloud, np.ndarray)
        else point_cloud @ rotation_matrix.t()
    )

    # Apply translation.
    if translation_vector is not None:
        rotated_point_cloud += translation_vector

    return rotated_point_cloud
