"""Non-rigid transformation using RBF (Radial Basis Function) kernels."""

import torch

from .base import TransformBase
from ...utils import rbf_kernel

__all__ = ["NonRigidTransformation"]


class NonRigidTransformation(TransformBase):
    """Non-rigid transformation using RBF (Radial Basis Function) kernel.

    This transformation computes point displacements using an RBF kernel
    matrix and learned weights. The deformation at each point is a weighted
    sum of Gaussian basis functions centered at the control points.

    Kernel Pattern
    --------------
    This class uses a common kernel-based deformation pattern:

    1. Kernel matrix: G = rbf_kernel(points, points, beta)
       Shape: (n_points, n_points)
    2. Deformation: transformed = points + G @ weights
       Where weights has shape (n_points, 3)

    The RBF kernel is: k(x, y) = exp(-||x - y||^2 / (2 beta)), where beta is
    the variance parameter (Myronenko & Song's beta^2). This is the kernel
    NonRigidCPD uses.

    Parameters
    ----------
    w : torch.Tensor
        Deformation weights, shape (n_points, 3).
    points : torch.Tensor
        Control points for kernel computation, shape (n_points, 3).
    beta : float, optional
        Kernel variance parameter. Default 2.0.
        Larger values = wider kernel and smoother deformation, smaller = more local.

    Attributes
    ----------
    g : torch.Tensor
        Precomputed kernel matrix, shape (n_points, n_points).
    w : torch.Tensor
        Deformation weights.

    Examples
    --------
    >>> import torch
    >>> from zreg.core.transforms import NonRigidTransformation
    >>> points = torch.randn(50, 3)
    >>> w = torch.zeros(50, 3)
    >>> tf = NonRigidTransformation(w=w, points=points, beta=2.0)
    >>> transformed = tf.transform(points)

    See Also
    --------
    CombinedTransformation : Combines rigid with non-rigid
    TPSTransformation : TPS kernel variant with different basis function
    utils.rbf_kernel : The kernel function used
    """

    def __init__(
        self,
        w: torch.Tensor,
        points: torch.Tensor,
        beta: float = 2.0,
    ) -> None:
        """Initialize non-rigid transformation.

        Parameters
        ----------
        w : torch.Tensor
            Deformation weights, shape (n_points, 3).
        points : torch.Tensor
            Control points, shape (n_points, 3).
        beta : float, optional
            Kernel bandwidth. Default: 2.0.
        """
        super().__init__()
        self.g = rbf_kernel(points, points, beta)
        self.w = w

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply non-rigid transformation.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        return points + torch.matmul(self.g, self.w)
