"""Kernel functions for CPD registration.

This module provides kernel computation specifically for the Coherent Point Drift
algorithm. The primary function is rbf_kernel_matrix which computes the Gaussian
RBF kernel used in non-rigid registration.
"""

import torch

from ...utils import rbf_kernel

__all__ = ["rbf_kernel_matrix"]


def rbf_kernel_matrix(points: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute the RBF kernel matrix for non-rigid CPD.

    This computes ``G_ij = exp(-||y_i - y_j||^2 / (2 beta))`` for all pairs
    of points, by delegating to :func:`zreg.utils.rbf_kernel`. It is the same
    kernel matrix that NonRigidCPD uses.

    zreg's ``beta`` is the variance parameter of the Gaussian, i.e. the
    quantity written ``beta^2`` in Myronenko & Song (2010).

    .. note::
        Input points must be pre-normalized (e.g., via normalize_point_cloud)
        to prevent numerical instability.

    Parameters
    ----------
    points : torch.Tensor
        Point cloud data with shape (N, D). Must be pre-normalized.
    beta : float
        Variance parameter controlling the kernel width (Myronenko & Song's
        beta^2). Larger values produce smoother deformations.

    Returns
    -------
    torch.Tensor
        Symmetric kernel matrix with shape (N, N).

    Examples
    --------
    >>> points = torch.randn(100, 3)
    >>> G = rbf_kernel_matrix(points, beta=2.0)
    >>> G.shape
    torch.Size([100, 100])
    """
    return rbf_kernel(points, points, beta)
