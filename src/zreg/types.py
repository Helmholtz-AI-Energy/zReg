"""Shared type dataclasses for the zreg package.

This module is a thin layer containing only dataclasses and torch imports
to avoid circular dependencies. Both pairwise_distance_matrix.py and
dtw/result.py import from this module.
"""

from dataclasses import dataclass, field

import torch

__all__ = ["StoredTransform", "PairwiseResult"]


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
