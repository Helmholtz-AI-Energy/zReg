"""Thin Plate Spline (TPS) transformation for smooth non-rigid registration."""

import torch

from .base import TransformBase
from .. import utils

__all__ = ["TPSTransformation"]


class TPSTransformation(TransformBase):
    """Thin Plate Spline (TPS) transformation.

    TPS is a spline-based interpolation method that minimizes bending
    energy while passing through control points. It's commonly used for
    smooth non-rigid registration.

    Kernel Pattern
    --------------
    This class uses a common kernel-based deformation pattern:

    1. Kernel function: k(x, y) = ||x - y||^2 * log(||x - y||)
    2. Basis construction: combines affine and kernel terms
    3. Transformation: basis @ [affine_params; kernel_weights]

    The TPS kernel naturally produces smoother deformations than RBF
    for large displacements.

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

    Examples
    --------
    >>> a = torch.randn(4, 3)  # Affine parameters
    >>> v = torch.randn(10, 3)  # Kernel weights
    >>> control_pts = torch.randn(10, 3)  # Control points
    >>> tf = TPSTransformation(a=a, v=v, control_pts=control_pts)
    >>> points = torch.randn(50, 3)
    >>> transformed = tf.transform(points)

    See Also
    --------
    NonRigidTransformation : RBF kernel variant
    utils.tps_kernel : The kernel function used
    """

    def __init__(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        control_pts: torch.Tensor,
        kernel: callable = utils.tps_kernel,
    ) -> None:
        """Initialize TPS transformation.

        Parameters
        ----------
        a : torch.Tensor
            Affine parameters, shape (d+1, d).
        v : torch.Tensor
            Kernel weights, shape (n_control, d).
        control_pts : torch.Tensor
            Control points, shape (n_control, d).
        kernel : callable, optional
            Kernel function. Default: utils.tps_kernel.
        """
        super().__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts
        self._kernel = kernel
        self.fact = {"dtype": self.a.dtype, "device": self.a.device}

    def prepare(self, landmarks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute basis and kernel matrices for given landmarks.

        Parameters
        ----------
        landmarks : torch.Tensor
            Landmark points, shape (n_landmarks, d).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (basis, kernel) matrices.
        """
        control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape

        # Concatenate tensors with homogeneous coordinate
        pm = torch.cat([torch.ones((m, 1), **self.fact), torch.tensor(landmarks, **self.fact)], dim=1)
        pn = torch.cat([torch.ones((n, 1), **self.fact), control_pts], dim=1)

        # SVD decomposition to get null space of control points
        u, _, _ = torch.linalg.svd(pn, full_matrices=True)
        pp = u[:, d + 1 :]

        # Compute kernel matrices
        kk = self._kernel(control_pts, control_pts)
        uu = self._kernel(torch.tensor(landmarks, **self.fact), control_pts).T

        # Construct basis: [affine part | kernel part in null space]
        basis = torch.cat([pm, torch.matmul(uu, pp)], dim=1)

        # Kernel in null space
        kernel = torch.matmul(pp.T, torch.matmul(kk, pp))

        return basis, kernel

    def transform_basis(self, basis: torch.Tensor) -> torch.Tensor:
        """Apply transformation given precomputed basis.

        Parameters
        ----------
        basis : torch.Tensor
            Precomputed basis matrix.

        Returns
        -------
        torch.Tensor
            Transformed points.
        """
        return torch.matmul(basis, torch.cat((self.a, self.v), dim=0))

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply TPS transformation.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)
