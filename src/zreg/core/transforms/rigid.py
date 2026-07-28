"""Rigid transformation: rotation, translation, and uniform scaling."""

import torch

from .base import TransformBase

__all__ = ["RigidTransformation"]


class RigidTransformation(TransformBase):
    """Rigid transformation with rotation, translation, and uniform scaling.

    A rigid transformation preserves distances and angles, consisting of:
    - Rotation matrix R (3×3, orthogonal with det(R) = 1)
    - Translation vector t (3D)
    - Uniform scale s (positive scalar)

    The transformation is: T(p) = s * p @ R.T + t

    Composition of rigid transformations is supported via the `*` operator,
    with automatic validation of the composed rotation matrix.

    Parameters
    ----------
    rot : torch.Tensor, optional
        3×3 rotation matrix. Default: identity matrix.
    t : torch.Tensor, optional
        3D translation vector. Default: zero vector.
    scale : float, optional
        Uniform scale factor. Default: 1.0.
    device : torch.device, optional
        Device for tensors (cpu/cuda). Default: None.
    dtype : torch.dtype, optional
        Data type for tensors. Default: None.

    Attributes
    ----------
    rot : torch.Tensor
        3×3 rotation matrix.
    t : torch.Tensor
        3D translation vector.
    scale : float
        Uniform scale factor.

    Examples
    --------
    >>> # Create identity transformation
    >>> tf = RigidTransformation()
    >>> points = torch.randn(100, 3)
    >>> transformed = tf.transform(points)

    >>> # Create translation
    >>> t = torch.tensor([1.0, 2.0, 3.0])
    >>> tf = RigidTransformation(t=t)

    >>> # Compose transformations
    >>> tf1 = RigidTransformation(t=torch.tensor([1., 0., 0.]))
    >>> tf2 = RigidTransformation(t=torch.tensor([0., 1., 0.]))
    >>> tf_composed = tf1 * tf2  # Apply tf2 first, then tf1
    """

    def __init__(
        self,
        rot: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize rigid transformation."""
        super().__init__()

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

    def reset(self) -> None:
        """Reset to identity transformation."""
        self.rot = torch.eye(3, dtype=self.rot.dtype, device=self.rot.device)
        self.t = torch.zeros(3, dtype=self.t.dtype, device=self.t.device)
        self.scale = 1.0

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply transformation to 3D points.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        return self.scale * torch.matmul(points, self.rot.T) + self.t

    def inverse(self) -> "RigidTransformation":
        """Compute the inverse transformation.

        Returns
        -------
        RigidTransformation
            The inverse transformation.
        """
        return RigidTransformation(
            rot=self.rot.T,
            t=-torch.matmul(self.rot.T, self.t) / self.scale,
            scale=1.0 / self.scale,
        )

    def __mul__(self, other: "RigidTransformation") -> "RigidTransformation":
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
        # Compute both metrics before raising so the error message always includes
        # full context regardless of which constraint is violated.
        det = torch.det(rot_composed)
        cond = torch.linalg.cond(rot_composed)
        if abs(det.item() - 1.0) > 1e-6 or cond.item() > 1e6:
            raise ValueError(
                f"RigidTransformation composition produced invalid rotation: "
                f"det={det.item():.6f}, cond={cond.item():.2e} (expected det=1.0, cond<1e6)"
            )

        return RigidTransformation(
            rot=rot_composed,
            t=self.t + self.scale * torch.matmul(self.rot, other.t),
            scale=self.scale * other.scale,
        )
