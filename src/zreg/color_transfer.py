"""Color transfer functionality for assigning cell type information between aligned point clouds.

This module provides methods to transfer color/cell type information from a source
point cloud to a target point cloud after spatial alignment.
"""

from typing import Optional, Union
import torch
import logging

from .dataset import zRegPointCloud

log = logging.getLogger(__name__)

__all__ = ["transfer_colors", "ColorTransferMethod"]


class ColorTransferMethod:
    """Enumeration of available color transfer methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CPD_WEIGHTED = "cpd_weighted"


def transfer_colors(
    source: Union[zRegPointCloud, torch.Tensor],
    target: Union[zRegPointCloud, torch.Tensor],
    method: str = ColorTransferMethod.NEAREST_NEIGHBOR,
    source_colors: Optional[torch.Tensor] = None,
    target_colors: Optional[torch.Tensor] = None,
    estep_result: Optional["EstepResult"] = None,
    **kwargs
) -> torch.Tensor:
    """Transfer colors from source to target point cloud.

    Parameters
    ----------
    source : zRegPointCloud or torch.Tensor
        Source point cloud data. If zRegPointCloud, uses 'pos' and 'color' fields.
        If torch.Tensor, should be positions of shape (n_points, n_dims).
    target : zRegPointCloud or torch.Tensor
        Target point cloud data. If zRegPointCloud, uses 'pos' field.
        If torch.Tensor, should be positions of shape (m_points, n_dims).
    method : str, optional
        Color transfer method. Options: 'nearest_neighbor', 'cpd_weighted'.
        Default: 'nearest_neighbor'.
    source_colors : torch.Tensor, optional
        Source colors of shape (n_points, n_color_channels). Required if source
        is torch.Tensor. If source is zRegPointCloud, uses source['color'].
    target_colors : torch.Tensor, optional
        Target colors of shape (m_points, n_color_channels). Only used for
        validation in some methods. If target is zRegPointCloud, uses target['color'].
    estep_result : EstepResult, optional
        Result from CPD E-step containing posterior probabilities. Required for
        'cpd_weighted' method.
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    torch.Tensor
        Transferred colors for target point cloud of shape (m_points, n_color_channels).

    Raises
    ------
    ValueError
        If method is not supported or required parameters are missing.
    """
    # Extract positions and colors
    if isinstance(source, zRegPointCloud):
        source_pos = source["pos"]
        source_colors = source["color"]
    else:
        source_pos = source
        if source_colors is None:
            raise ValueError("source_colors must be provided when source is torch.Tensor")

    if isinstance(target, zRegPointCloud):
        target_pos = target["pos"]
        target_colors = target.get("color")
    else:
        target_pos = target

    # Validate inputs
    if source_pos.shape[1] != target_pos.shape[1]:
        raise ValueError(f"Source and target must have same dimensionality: {source_pos.shape[1]} vs {target_pos.shape[1]}")

    if method == ColorTransferMethod.NEAREST_NEIGHBOR:
        return _transfer_colors_nearest_neighbor(source_pos, target_pos, source_colors)
    elif method == ColorTransferMethod.CPD_WEIGHTED:
        if estep_result is None:
            raise ValueError("estep_result is required for CPD-weighted method")
        return _transfer_colors_cpd_weighted(source_pos, target_pos, source_colors, estep_result)
    else:
        raise ValueError(f"Unknown color transfer method: {method}")


def _transfer_colors_nearest_neighbor(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor
) -> torch.Tensor:
    """Transfer colors using nearest neighbor assignment.

    For each target point, finds the closest source point and copies its color.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source colors of shape (n_points, n_color_channels).

    Returns
    -------
    torch.Tensor
        Transferred colors of shape (m_points, n_color_channels).
    """
    log.debug(f"Transferring colors using nearest neighbor: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find nearest neighbors
    _, nearest_indices = torch.min(distances, dim=1)  # (m_points,)

    # Assign colors
    transferred_colors = source_colors[nearest_indices]  # (m_points, n_color_channels)

    return transferred_colors


def _transfer_colors_cpd_weighted(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    estep_result: "EstepResult"
) -> torch.Tensor:
    """Transfer colors using CPD posterior probabilities.

    For each target point, computes weighted average of source colors
    based on correspondence probabilities from CPD.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source colors of shape (n_points, n_color_channels).
    estep_result : EstepResult
        Result from CPD E-step containing posterior probabilities.

    Returns
    -------
    torch.Tensor
        Transferred colors of shape (m_points, n_color_channels).
    """
    log.debug(f"Transferring colors using CPD weights: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Extract probability matrix from E-step result
    # pmat should be (n_target, n_source) - probability of each target point belonging to each source
    pmat = estep_result.pmat

    # For now, assume pmat is (n_target, n_source)
    # If it's the wrong shape, we might need to transpose
    if pmat.shape[0] == target_pos.shape[0] and pmat.shape[1] == source_pos.shape[0]:
        prob_matrix = pmat
    elif pmat.shape[0] == source_pos.shape[0] and pmat.shape[1] == target_pos.shape[0]:
        prob_matrix = pmat.T
    else:
        raise ValueError(f"Probability matrix shape {pmat.shape} doesn't match points: source {source_pos.shape[0]}, target {target_pos.shape[0]}")

    # Normalize probabilities (should already be normalized, but ensure)
    prob_matrix = prob_matrix / prob_matrix.sum(dim=1, keepdim=True)

    # Compute weighted color average for each target point
    # prob_matrix: (m_points, n_points), source_colors: (n_points, n_channels)
    # Result: (m_points, n_channels)
    transferred_colors = torch.matmul(prob_matrix, source_colors.float())

    return transferred_colors
