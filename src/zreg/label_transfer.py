"""Label transfer functionality for assigning cell type information between aligned point clouds.

This module provides methods to transfer label/cell type information from a source
point cloud to a target point cloud after spatial alignment.
"""

from enum import Enum
import torch
import logging

from .algorithms.cpd import EstepResult

from .core.dataset import zRegPointCloud

log = logging.getLogger(__name__)

__all__ = ["transfer_labels", "LabelTransferMethod"]


class LabelTransferMethod(Enum):
    """Enumeration of available label transfer methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CPD_WEIGHTED = "cpd_weighted"
    KNN_VOTING = "knn_voting"
    GAUSSIAN_KERNEL = "gaussian_kernel"


def transfer_labels(
    source: zRegPointCloud | torch.Tensor,
    target: zRegPointCloud | torch.Tensor,
    method: str | LabelTransferMethod = LabelTransferMethod.NEAREST_NEIGHBOR,
    source_colors: torch.Tensor | None = None,
    target_colors: torch.Tensor | None = None,
    estep_result: EstepResult | None = None,
    **kwargs
) -> torch.Tensor:
    """Transfer labels from source to target point cloud.

    Parameters
    ----------
    source : zRegPointCloud or torch.Tensor
        Source point cloud data. If zRegPointCloud, uses 'pos' and 'label' fields.
        If torch.Tensor, should be positions of shape (n_points, n_dims).
    target : zRegPointCloud or torch.Tensor
        Target point cloud data. If zRegPointCloud, uses 'pos' field.
        If torch.Tensor, should be positions of shape (m_points, n_dims).
    method : str, optional
        Label transfer method. Options: 'nearest_neighbor', 'cpd_weighted', 'knn_voting', 'gaussian_kernel'.
        Default: 'nearest_neighbor'.
    source_colors : torch.Tensor, optional
        Source labels of shape (n_points, n_label_channels). Required if source
        is torch.Tensor. If source is zRegPointCloud, uses source['label'].
    target_colors : torch.Tensor, optional
        Target labels of shape (m_points, n_label_channels). Not used in current methods.
    estep_result : EstepResult, optional
        Result from CPD E-step containing posterior probabilities. Required for
        'cpd_weighted' method.
    **kwargs
        Additional method-specific parameters. For 'knn_voting': 'k' (int, default 5).
        For 'gaussian_kernel': 'sigma' (float, default 1.0).

    Returns
    -------
    torch.Tensor
        Transferred labels for target point cloud of shape (m_points, n_label_channels).

    Raises
    ------
    ValueError
        If method is not supported or required parameters are missing.
    """
    # Extract positions and labels
    if isinstance(source, zRegPointCloud):
        source_pos = source["pos"]
        source_colors = source["label"]
    else:
        source_pos = source
        if source_colors is None:
            raise ValueError("source_colors must be provided when source is torch.Tensor")

    if isinstance(target, zRegPointCloud):
        target_pos = target["pos"]
    else:
        target_pos = target

    # Validate inputs
    if source_pos.shape[0] == 0:
        raise ValueError("source has 0 points — cannot transfer labels from an empty point cloud")

    if source_pos.shape[1] != target_pos.shape[1]:
        raise ValueError(f"Source and target must have same dimensionality: {source_pos.shape[1]} vs {target_pos.shape[1]}")

    # Normalize method to enum (accepts both string and enum)
    if isinstance(method, str):
        try:
            method = LabelTransferMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown label transfer method: '{method}'. "
                f"Valid options: {[m.value for m in LabelTransferMethod]}"
            )

    if method == LabelTransferMethod.NEAREST_NEIGHBOR:
        return _transfer_labels_nearest_neighbor(source_pos, target_pos, source_colors)
    elif method == LabelTransferMethod.CPD_WEIGHTED:
        if estep_result is None:
            raise ValueError("estep_result is required for CPD-weighted method")
        return _transfer_labels_cpd_weighted(source_pos, target_pos, source_colors, estep_result)
    elif method == LabelTransferMethod.KNN_VOTING:
        k = kwargs.get('k', 5)
        return _transfer_labels_knn_voting(source_pos, target_pos, source_colors, k)
    elif method == LabelTransferMethod.GAUSSIAN_KERNEL:
        sigma = kwargs.get('sigma', 1.0)
        return _transfer_labels_gaussian_kernel(source_pos, target_pos, source_colors, sigma)
    else:  # pragma: no cover
        # Should not reach here if normalization above is correct, but guard anyway
        raise ValueError(
            f"Unknown label transfer method: '{method}'. "
            f"Valid options: {[m.value for m in LabelTransferMethod]}"
        )


def _transfer_labels_nearest_neighbor(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor
) -> torch.Tensor:
    """Transfer labels using nearest neighbor assignment.

    For each target point, finds the closest source point and copies its label.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).
    """
    log.debug(f"Transferring labels using nearest neighbor: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find nearest neighbors
    _, nearest_indices = torch.min(distances, dim=1)  # (m_points,)

    # Assign labels
    transferred_colors = source_colors[nearest_indices]  # (m_points, n_label_channels)

    return transferred_colors


def _transfer_labels_cpd_weighted(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    estep_result: "EstepResult"
) -> torch.Tensor:
    """Transfer labels using CPD posterior probabilities.

    For each target point, computes weighted average of source labels
    based on correspondence probabilities from CPD.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    estep_result : EstepResult
        Result from CPD E-step containing posterior probabilities.

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).
    """
    log.debug(f"Transferring labels using CPD weights: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Extract probability matrix from E-step result
    # pmat should be (n_target, n_source) - probability of each target point belonging to each source
    pmat = estep_result.pmat

    n_target = target_pos.shape[0]
    n_source = source_pos.shape[0]

    if pmat.shape == (n_target, n_source):
        prob_matrix = pmat
    elif pmat.shape == (n_source, n_target):
        raise ValueError(
            f"pmat has shape {pmat.shape} which looks transposed. "
            f"Expected (n_target={n_target}, n_source={n_source})"
        )
    else:
        raise ValueError(
            f"pmat has shape {pmat.shape} but expected "
            f"(n_target={n_target}, n_source={n_source})"
        )

    # Normalize probabilities (should already be normalized, but ensure)
    prob_matrix = prob_matrix / prob_matrix.sum(dim=1, keepdim=True)

    # Compute weighted label average for each target point
    # prob_matrix: (m_points, n_points), source_colors: (n_points, n_channels)
    # Result: (m_points, n_channels)
    transferred_colors = torch.matmul(prob_matrix, source_colors.float())

    return transferred_colors


def _transfer_labels_knn_voting(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    k: int
) -> torch.Tensor:
    """Transfer labels using K-nearest neighbors with majority voting.

    For each target point, finds K nearest source points and assigns the most
    common label among them.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).

    Raises
    ------
    ValueError
        If n_label_channels != 1 (labels must be class indices).
    """
    log.debug(f"Transferring labels using KNN voting (k={k}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    if source_colors.shape[1] != 1:
        raise ValueError("KNN voting assumes single-channel labels (class indices)")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find K nearest neighbors
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)  # (m_points, k)

    # Get labels for nearest neighbors
    selected_colors = source_colors[indices].squeeze(-1)  # (m_points, k)

    # Compute mode for each target point
    transferred_colors = torch.mode(selected_colors, dim=1).values.unsqueeze(-1)  # (m_points, 1)

    return transferred_colors


def _transfer_labels_gaussian_kernel(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """Transfer labels using Gaussian kernel interpolation.

    Treats labels as a continuous field and interpolates using Gaussian kernels
    centered at source points.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).
    """
    log.debug(f"Transferring labels using Gaussian kernel (sigma={sigma}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Compute Gaussian weights
    weights = torch.exp(-distances**2 / (2 * sigma**2))  # (m_points, n_points)

    # Normalize weights to sum to 1 for each target point
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Compute weighted average of labels
    transferred_colors = torch.matmul(weights, source_colors.float())  # (m_points, n_label_channels)

    return transferred_colors
