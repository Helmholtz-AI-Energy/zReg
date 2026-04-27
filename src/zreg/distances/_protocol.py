"""Internal protocol definition for distance metrics."""

from typing import Protocol

import torch


class DistanceMetric(Protocol):
    """Protocol for distance metric callables.

    This protocol defines the interface contract for distance metrics used
    in pairwise distance matrix computation. Both functional metrics
    (euclidean_distance) and callable classes (SlicedWassersteinDistance)
    satisfy this protocol.

    Note: This is an internal typing protocol and is NOT part of the public API.
    """

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute distance between two tensors.

        Parameters
        ----------
        x : torch.Tensor
            First tensor (point cloud or feature tensor).
        y : torch.Tensor
            Second tensor (point cloud or feature tensor).
        **kwargs
            Additional metric-specific parameters.

        Returns
        -------
        torch.Tensor
            Distance value(s). Shape depends on metric:
            - Pairwise metrics (euclidean, etc.): (n, m) matrix
            - SWD variants: scalar or (batch,) tensor
        """
        ...
