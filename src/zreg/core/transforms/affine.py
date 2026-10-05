"""Affine transformation: general linear transformation with translation."""

import torch

from .base import TransformBase

__all__ = ["AffineTransformation"]


class AffineTransformation(TransformBase):
    """Affine transformation combining linear matrix and translation.

    An affine transformation applies a 3×3 linear matrix B and translation t:
    T(p) = p @ B.T + t

    Unlike rigid transformations, the matrix B does not need to be orthogonal,
    allowing for scaling, shearing, and other deformations.

    Parameters
    ----------
    b : torch.Tensor, optional
        3×3 affine matrix. Default: identity matrix.
    t : torch.Tensor, optional
        3D translation vector. Default: zero vector (identity).
    device : torch.device, optional
        Device for tensors (cpu/cuda). Default: None.
    dtype : torch.dtype, optional
        Data type for tensors. Default: None.

    Attributes
    ----------
    b : torch.Tensor
        3×3 affine matrix.
    t : torch.Tensor
        3D translation vector.

    Examples
    --------
    >>> import torch
    >>> from zreg.core.transforms import AffineTransformation
    >>> # Create identity affine transformation
    >>> tf = AffineTransformation()
    >>> points = torch.randn(100, 3)
    >>> transformed = tf.transform(points)

    >>> # Create scaling transformation
    >>> b = torch.diag(torch.tensor([2.0, 2.0, 2.0]))
    >>> tf = AffineTransformation(b=b)
    """

    def __init__(
        self,
        b: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize affine transformation."""
        super().__init__()

        if b is None:
            b = torch.eye(3, dtype=dtype, device=device)
        if t is None:
            t = torch.zeros(3, dtype=dtype, device=device)

        if dtype is not None:
            b = b.to(dtype=dtype)
            t = t.to(dtype=dtype)
        if device is not None:
            b = b.to(device=device)
            t = t.to(device=device)

        self.b = b
        self.t = t

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to 3D points.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        return torch.matmul(points, self.b.T) + self.t
