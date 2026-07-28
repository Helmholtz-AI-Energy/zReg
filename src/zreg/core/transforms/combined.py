"""Combined transformation: rigid transformation plus non-rigid deformation."""

import torch

from .base import TransformBase
from .rigid import RigidTransformation

__all__ = ["CombinedTransformation"]


class CombinedTransformation(TransformBase):
    """Combine rigid and non-rigid transformations.

    Applies a rigid transformation composed with a non-rigid deformation term.
    The transformation is: T(p) = rigid.transform(p + nonrigid_term)

    This is useful for modeling transformations that have both global rigid
    motion and local non-rigid deformations.

    Parameters
    ----------
    rot : torch.Tensor, optional
        3×3 rotation matrix. Default: identity.
    t : torch.Tensor, optional
        3D translation vector. Default: zero.
    scale : float, optional
        Uniform scale factor. Default: 1.0.
    v : torch.Tensor | float, optional
        Non-rigid deformation term added before rigid transform.
        If float, broadcast to all points. Default: 0.0.

    Attributes
    ----------
    rigid_trans : RigidTransformation
        The rigid transformation component.
    v : torch.Tensor | float
        The non-rigid deformation term.

    Examples
    --------
    >>> # Create rigid + deformation transformation
    >>> t = torch.tensor([1.0, 0.0, 0.0])
    >>> v = torch.randn(100, 3) * 0.01  # Small deformations
    >>> tf = CombinedTransformation(t=t, v=v)
    >>> points = torch.randn(100, 3)
    >>> transformed = tf.transform(points)

    See Also
    --------
    RigidTransformation : Rigid component
    NonRigidTransformation : Non-rigid variants
    """

    def __init__(
        self,
        rot: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        scale: float = 1.0,
        v: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize combined transformation.

        Parameters
        ----------
        rot : torch.Tensor, optional
            3×3 rotation matrix. Default: identity.
        t : torch.Tensor, optional
            3D translation vector. Default: zero.
        scale : float, optional
            Uniform scale. Default: 1.0.
        v : torch.Tensor | float, optional
            Non-rigid deformation. Default: 0.0.
        """
        super().__init__()
        self.rigid_trans = RigidTransformation(rot=rot, t=t, scale=scale)
        self.v = v

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply combined transformation.

        Parameters
        ----------
        points : torch.Tensor
            Points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        return self.rigid_trans.transform(points + self.v)
