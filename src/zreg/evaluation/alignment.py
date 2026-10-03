"""Alignment metric functions for zReg.

Provides four pure alignment metrics:
- chamfer: bidirectional Chamfer distance between point clouds
- hausdorff: percentile-based Hausdorff distance
- chamfer_hausdorff: both metrics from one distance matrix
- path_smoothness: variance of the cross products of consecutive DTW path steps
"""

import torch

from ..utils.validation import _validate_tensors

__all__ = ["chamfer", "hausdorff", "chamfer_hausdorff", "path_smoothness"]


def _check_pair(source: torch.Tensor, target: torch.Tensor) -> None:
    """Shared validation of a (source, target) point-cloud pair."""
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source must be shape (N, 3), got {tuple(source.shape)}"
        )
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(
            f"target must be shape (M, 3), got {tuple(target.shape)}"
        )
    _validate_tensors(source, target, names=["source", "target"])

    if source.shape[0] == 0:
        raise ValueError("source must be non-empty (N > 0)")
    if target.shape[0] == 0:
        raise ValueError("target must be non-empty (M > 0)")


def _check_percentile(percentile: float) -> None:
    """Reject a Hausdorff percentile outside [0, 100]."""
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(
            f"percentile must be in [0, 100], got {percentile}"
        )


def _nearest_distances(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point nearest-neighbour distances in both directions (one cdist)."""
    dist = torch.cdist(source, target, p=2)
    return dist.min(dim=1).values, dist.min(dim=0).values


def _chamfer_from_minima(
    min_src: torch.Tensor, min_tgt: torch.Tensor, squared: bool
) -> torch.Tensor:
    """Chamfer reduction; squaring creates new tensors (inputs untouched)."""
    if squared:
        min_src = min_src ** 2
        min_tgt = min_tgt ** 2
    return (min_src.mean() + min_tgt.mean()) / 2.0


def _hausdorff_from_minima(
    min_src: torch.Tensor,
    min_tgt: torch.Tensor,
    percentile: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Percentile-Hausdorff reduction of the nearest-neighbour minima."""
    q = torch.tensor(percentile / 100.0, dtype=dtype, device=device)
    return torch.max(torch.quantile(min_src, q), torch.quantile(min_tgt, q))


def chamfer(
    source: torch.Tensor,
    target: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """Compute bidirectional Chamfer distance between two point clouds.

    For each point in source, finds the nearest point in target; and vice
    versa. Returns the mean of these minimum distances (averaged over both
    directions), optionally squaring the per-point distances before averaging.

    Parameters
    ----------
    source : torch.Tensor
        Point cloud of shape (N, 3).
    target : torch.Tensor
        Point cloud of shape (M, 3).
    squared : bool, optional
        If True, square each per-point minimum distance before averaging.
        Default: False.

    Returns
    -------
    torch.Tensor
        Scalar tensor on the same device as the inputs.

    Raises
    ------
    ValueError
        If source or target is not shape (N, 3).
        If source or target contains NaN or inf values.
        If source and target are on different devices.

    Notes
    -----
    Uses torch.cdist for GPU-safe pairwise distance computation. No NumPy
    on GPU path.
    """
    _check_pair(source, target)
    min_src, min_tgt = _nearest_distances(source, target)
    return _chamfer_from_minima(min_src, min_tgt, squared)


def hausdorff(
    source: torch.Tensor,
    target: torch.Tensor,
    percentile: float = 95.0,
) -> torch.Tensor:
    """Compute the percentile-based Hausdorff distance between two point clouds.

    For each point in source, finds the nearest point in target; and vice
    versa. Returns the maximum of the given percentile over both directions.

    Parameters
    ----------
    source : torch.Tensor
        Point cloud of shape (N, 3).
    target : torch.Tensor
        Point cloud of shape (M, 3).
    percentile : float, optional
        Percentile in [0, 100] used to compute the robust Hausdorff distance.
        Default: 95.0.

    Returns
    -------
    torch.Tensor
        Scalar tensor on the same device as the inputs.

    Raises
    ------
    ValueError
        If source or target is not shape (N, 3).
        If source or target contains NaN or inf values.
        If source and target are on different devices.

    Notes
    -----
    Uses torch.quantile (not numpy.percentile) for GPU-safe computation.
    """
    _check_pair(source, target)
    _check_percentile(percentile)
    min_src, min_tgt = _nearest_distances(source, target)
    return _hausdorff_from_minima(
        min_src, min_tgt, percentile, source.dtype, source.device
    )


def chamfer_hausdorff(
    source: torch.Tensor,
    target: torch.Tensor,
    squared: bool = False,
    percentile: float = 95.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Chamfer and percentile Hausdorff distance from one distance matrix.

    Equivalent to ``(chamfer(source, target, squared),
    hausdorff(source, target, percentile))`` -- the same operations on the
    same tensors, so both results are bit-identical to the separate calls --
    but the pairwise distance matrix is computed only once.

    Parameters
    ----------
    source : torch.Tensor
        Point cloud of shape (N, 3).
    target : torch.Tensor
        Point cloud of shape (M, 3).
    squared : bool, optional
        If True, square each per-point minimum distance before averaging for
        the Chamfer distance (the Hausdorff distance always uses the
        unsquared minima). Default: False.
    percentile : float, optional
        Percentile in [0, 100] for the robust Hausdorff distance.
        Default: 95.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(chamfer, hausdorff)`` as scalar tensors on the inputs' device.

    Raises
    ------
    ValueError
        If source or target is not shape (N, 3) or is empty.
        If source or target contains NaN or inf values.
        If source and target are on different devices.
        If percentile is outside [0, 100].
    """
    _check_pair(source, target)
    _check_percentile(percentile)
    min_src, min_tgt = _nearest_distances(source, target)
    return (
        _chamfer_from_minima(min_src, min_tgt, squared),
        _hausdorff_from_minima(
            min_src, min_tgt, percentile, source.dtype, source.device
        ),
    )


def path_smoothness(path: list[tuple[int, int]]) -> float:
    """Compute smoothness of a DTW alignment path as variance of cross-product curvature.

    Measures how much the alignment path "bends" between consecutive steps.
    Zero variance means a perfectly straight (constant-direction) path.

    Parameters
    ----------
    path : list[tuple[int, int]]
        Sequence of (i, j) index pairs representing the DTW alignment path.

    Returns
    -------
    float
        Variance of consecutive cross-product values along the path. Returns 0.0
        if fewer than 3 points are provided (need at least 2 step vectors for a
        cross-product to exist).

    Notes
    -----
    Uses the 2-D cross-product of consecutive step vectors to measure curvature,
    which avoids the numerical instability of slope-based approaches when DTW
    paths contain axis-aligned (horizontal or vertical) steps.
    Cross-product: (di1, dj1) x (di2, dj2) = di1*dj2 - di2*dj1.
    Uses unbiased=False so a single cross-product value still yields 0.0 (not NaN).
    """
    if len(path) < 3:
        return 0.0

    # Step vectors between consecutive path points
    steps = [
        (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        for k in range(1, len(path))
    ]
    # Signed curvature: cross-product of consecutive 2-D step vectors
    # (di1, dj1) x (di2, dj2) = di1*dj2 - di2*dj1
    cross = [
        steps[i][0] * steps[i + 1][1] - steps[i + 1][0] * steps[i][1]
        for i in range(len(steps) - 1)
    ]
    t = torch.tensor(cross, dtype=torch.float64)
    return float(t.var(unbiased=False).item())
