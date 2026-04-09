"""Color transfer functionality for assigning cell type information between aligned point clouds.

This module provides methods to transfer color/cell type information from a source
point cloud to a target point cloud after spatial alignment.
"""

from typing import Optional, Union
import torch
import logging

from zreg.cpd import EstepResult

from .dataset import zRegPointCloud

log = logging.getLogger(__name__)

__all__ = ["transfer_colors", "ColorTransferMethod"]


class ColorTransferMethod:
    """Enumeration of available color transfer methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CPD_WEIGHTED = "cpd_weighted"
    KNN_VOTING = "knn_voting"
    GAUSSIAN_KERNEL = "gaussian_kernel"


def transfer_colors(
    source: Union[zRegPointCloud, torch.Tensor],
    target: Union[zRegPointCloud, torch.Tensor],
    method: str = ColorTransferMethod.NEAREST_NEIGHBOR,
    source_colors: Optional[torch.Tensor] = None,
    target_colors: Optional[torch.Tensor] = None,
    estep_result: Optional[EstepResult] = None,
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
        Color transfer method. Options: 'nearest_neighbor', 'cpd_weighted', 'knn_voting', 'gaussian_kernel'.
        Default: 'nearest_neighbor'.
    source_colors : torch.Tensor, optional
        Source colors of shape (n_points, n_color_channels). Required if source
        is torch.Tensor. If source is zRegPointCloud, uses source['color'].
    target_colors : torch.Tensor, optional
        Target colors of shape (m_points, n_color_channels). Not used in current methods.
    estep_result : EstepResult, optional
        Result from CPD E-step containing posterior probabilities. Required for
        'cpd_weighted' method.
    **kwargs
        Additional method-specific parameters. For 'knn_voting': 'k' (int, default 5).
        For 'gaussian_kernel': 'sigma' (float, default 1.0).

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
    else:
        target_pos = target

    # Guard: empty source point cloud
    if source_pos.shape[0] == 0:
        raise ValueError(
            f"source has no points (shape: {tuple(source_pos.shape)})"
        )

    # Validate inputs
    if source_pos.shape[1] != target_pos.shape[1]:
        raise ValueError(f"Source and target must have same dimensionality: {source_pos.shape[1]} vs {target_pos.shape[1]}")

    if method == ColorTransferMethod.NEAREST_NEIGHBOR:
        return _transfer_colors_nearest_neighbor(source_pos, target_pos, source_colors)
    elif method == ColorTransferMethod.CPD_WEIGHTED:
        if estep_result is None:
            raise ValueError("estep_result is required for CPD-weighted method")
        return _transfer_colors_cpd_weighted(source_pos, target_pos, source_colors, estep_result)
    elif method == ColorTransferMethod.KNN_VOTING:
        k = kwargs.get('k', 5)
        return _transfer_colors_knn_voting(source_pos, target_pos, source_colors, k)
    elif method == ColorTransferMethod.GAUSSIAN_KERNEL:
        sigma = kwargs.get('sigma', 1.0)
        return _transfer_colors_gaussian_kernel(source_pos, target_pos, source_colors, sigma)
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


def _transfer_colors_knn_voting(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    k: int
) -> torch.Tensor:
    """Transfer colors using K-nearest neighbors with majority voting.

    For each target point, finds K nearest source points and assigns the most
    common color among them.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source colors of shape (n_points, n_color_channels).
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    torch.Tensor
        Transferred colors of shape (m_points, n_color_channels).

    Raises
    ------
    ValueError
        If n_color_channels != 1 (colors must be class indices).
    """
    log.debug(f"Transferring colors using KNN voting (k={k}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    if source_colors.shape[1] != 1:
        raise ValueError("KNN voting assumes single-channel colors (class indices)")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find K nearest neighbors
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)  # (m_points, k)

    # Get colors for nearest neighbors
    selected_colors = source_colors[indices].squeeze(-1)  # (m_points, k)

    # Compute mode for each target point
    transferred_colors = torch.mode(selected_colors, dim=1).values.unsqueeze(-1)  # (m_points, 1)

    return transferred_colors


def _transfer_colors_gaussian_kernel(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """Transfer colors using Gaussian kernel interpolation.

    Treats colors as a continuous field and interpolates using Gaussian kernels
    centered at source points.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source colors of shape (n_points, n_color_channels).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        Transferred colors of shape (m_points, n_color_channels).
    """
    log.debug(f"Transferring colors using Gaussian kernel (sigma={sigma}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Compute Gaussian weights
    weights = torch.exp(-distances**2 / (2 * sigma**2))  # (m_points, n_points)

    # Normalize weights to sum to 1 for each target point
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Compute weighted average of colors
    transferred_colors = torch.matmul(weights, source_colors.float())  # (m_points, n_color_channels)

    return transferred_colors
