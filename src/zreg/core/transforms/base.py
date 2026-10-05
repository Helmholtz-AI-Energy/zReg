"""Abstract base class for point cloud transformations."""

import torch

__all__ = ["TransformBase"]


class TransformBase:
    """Abstract base class for point cloud transformations.

    All transformation classes inherit from this base, which provides:
    - A standard `transform()` method that handles both 3D and extended point data
    - A `_transform()` template method for subclasses to implement

    Subclassing
    -----------
    To create a custom transformation, subclass TransformBase and implement
    `_transform(self, points)`:

        class MyTransform(TransformBase):
            def _transform(self, points):
                # points: torch.Tensor of shape (n, 3)
                return transformed_points

    The `transform()` method handles points with extra columns (e.g., colors)
    by only transforming the first 3 columns (xyz) and preserving the rest.

    See Also
    --------
    RigidTransformation : Rotation + translation + scale
    AffineTransformation : General affine transformation
    NonRigidTransformation : RBF kernel deformation
    TPSTransformation : Thin Plate Spline deformation
    """

    def __init__(self) -> None:
        pass

    def _transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply transformation to 3D points (N, 3).

        Subclasses must implement this method.

        Parameters
        ----------
        points : torch.Tensor
            Input points of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed points of shape (N, 3).
        """
        raise NotImplementedError("Subclasses must implement _transform()")

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply transformation to points, preserving extra dimensions.

        If points have more than 3 columns (e.g., with colors), only the first
        3 columns (xyz) are transformed. Remaining columns are preserved unchanged.

        Parameters
        ----------
        points : torch.Tensor
            Input points of shape (N, D) where D >= 3.

        Returns
        -------
        torch.Tensor
            Transformed points of same shape as input.
        """
        if points.shape[1] <= 3:
            return self._transform(points)

        # Transform only xyz, preserve the rest
        ret = points.clone()
        ret[:, :3] = self._transform(ret[:, :3])
        return ret
