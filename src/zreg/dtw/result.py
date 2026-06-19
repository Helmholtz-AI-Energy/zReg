"""DTW result container."""

from dataclasses import dataclass, field

import torch

from ..types import StoredTransform

__all__ = ["DTWResult"]


@dataclass
class DTWResult:
    """Container for DTW computation results.

    Attributes
    ----------
    cost_matrix : torch.Tensor
        Pairwise distance matrix between all time points.
        Shape: (n_metrics, len(x), len(y)) or (len(x), len(y)) if single metric.
    accumulated_cost : torch.Tensor
        Accumulated cost matrix from DTW dynamic programming.
        Shape: same as cost_matrix.
    warping_path : list[tuple[int, int]]
        Optimal alignment path as list of (x_idx, y_idx) pairs.
        Path goes from (0, 0) to (len(x)-1, len(y)-1).
    distance : float
        Total DTW distance (accumulated cost at the end of the path).
    rotations : torch.Tensor | None
        CPD rotations if cpd_type was specified during computation.
    stored_transforms : dict[tuple[int, int], StoredTransform]
        CPD transforms and normalisation parameters captured during pairwise distance
        computation, keyed by (i, j) index pairs. Empty when cpd_type is None.
    """

    cost_matrix: torch.Tensor
    accumulated_cost: torch.Tensor
    warping_path: list[tuple[int, int]]
    distance: float
    rotations: torch.Tensor | None = None
    stored_transforms: dict[tuple[int, int], StoredTransform] = field(default_factory=dict)
