"""Shared type dataclasses for the zreg package.

This module is a thin layer containing only dataclasses and torch imports
to avoid circular dependencies. Both pairwise_distance_matrix.py and
dtw/result.py import from this module.
"""

from dataclasses import dataclass, field

import torch

__all__ = ["StoredTransform", "PairwiseResult", "DTWResult"]


@dataclass
class StoredTransform:
    """CPD transform and normalisation parameters captured during pairwise distance computation.

    Stores the CPD transform object and the per-frame min/max normalisation parameters
    from ``utils.normalize_point_cloud`` so that the same normalised-space transform can
    be reused in ``_build_aligned_cloud`` (Step 3) without re-running CPD from identity.

    Attributes
    ----------
    transform : object
        The CPD transformation object — one of RigidCPDTransformation,
        AffineCPDTransformation, or NonRigidCPDTransformation. Typed as ``object``
        to avoid circular imports with the cpd subpackage.
    src_min : torch.Tensor
        0-dimensional scalar tensor: global minimum returned by
        ``utils.normalize_point_cloud`` for the source frame (min_vals, byaxis=False).
    src_max : torch.Tensor
        0-dimensional scalar tensor: global maximum returned by
        ``utils.normalize_point_cloud`` for the source frame (max_vals, byaxis=False).
    tgt_min : torch.Tensor
        0-dimensional scalar tensor: global minimum for the target frame.
    tgt_max : torch.Tensor
        0-dimensional scalar tensor: global maximum for the target frame.
    """

    transform: object
    src_min: torch.Tensor
    src_max: torch.Tensor
    tgt_min: torch.Tensor
    tgt_max: torch.Tensor


@dataclass
class PairwiseResult:
    """Return type for ``create_pairwise_distance_matrix``.

    Replaces the former plain 2-tuple ``(distance_matrix, rotations)`` so that
    ``stored_transforms`` can be threaded through ``DynamicTimeWarping`` to
    ``_build_aligned_cloud`` without positional juggling.

    Attributes
    ----------
    cost_matrix : torch.Tensor
        Pairwise distance matrix of shape ``(n_metrics, len_x, len_y)``.
    rotations : torch.Tensor | None
        Stacked rotation tensors from rigid CPD registration, or ``None`` when
        ``cpd_type`` is not ``"rigid"`` or no rotations were computed.
    stored_transforms : dict[tuple[int, int], StoredTransform]
        Mapping from ``(src_sub_idx, tgt_sub_idx)`` to the corresponding
        ``StoredTransform``. Populated when ``cpd_type is not None``; empty
        dict ``{}`` when ``cpd_type is None``.
    """

    cost_matrix: torch.Tensor
    rotations: torch.Tensor | None
    stored_transforms: dict[tuple[int, int], StoredTransform] = field(default_factory=dict)


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
    config : dict | None
        DTW configuration as saved by ``DynamicTimeWarping.save()``; ``distance_metric``
        callables are restored by importing their module. None for results not loaded
        from disk or files written before Phase 60.
    """

    cost_matrix: torch.Tensor
    accumulated_cost: torch.Tensor
    warping_path: list[tuple[int, int]]
    distance: float
    rotations: torch.Tensor | None = None
    stored_transforms: dict[tuple[int, int], StoredTransform] = field(default_factory=dict)
    config: dict | None = None
