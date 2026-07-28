"""Kernel functions for CPD registration.

This module provides kernel computation specifically for the Coherent Point Drift
algorithm. The primary function is rbf_kernel_matrix which computes the Gaussian
RBF kernel used in non-rigid registration.
"""

import torch

__all__ = ["rbf_kernel_matrix"]


def rbf_kernel_matrix(points: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute the RBF kernel matrix for non-rigid CPD.

    This computes G_ij = exp(-||y_i - y_j||^2 / (2 * beta^2)) for all pairs
    of points. Used in NonRigidCPD to model smooth deformations.

    .. note::
        Input points must be pre-normalized (e.g., via normalize_point_cloud)
        to prevent numerical instability.

    Parameters
    ----------
    points : torch.Tensor
        Point cloud data with shape (N, D). Must be pre-normalized.
    beta : float
        Bandwidth parameter controlling kernel width. Larger values
        produce smoother deformations.

    Returns
    -------
    torch.Tensor
        Kernel matrix with shape (N, N).

    Examples
    --------
    >>> points = torch.randn(100, 3)
    >>> G = rbf_kernel_matrix(points, beta=2.0)
    >>> G.shape
    torch.Size([100, 100])
    """
    # Squared pairwise distances
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, D)
    sq_dist = (diff**2).sum(dim=2)  # (N, N)
    return torch.exp(-sq_dist / (2.0 * beta**2))
