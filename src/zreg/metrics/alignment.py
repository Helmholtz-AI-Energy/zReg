"""Alignment metric functions for zReg.

Provides three pure alignment metrics:
- chamfer: bidirectional Chamfer distance between point clouds
- hausdorff: percentile-based Hausdorff distance
- path_smoothness: variance of slope changes along a DTW path
"""

import torch

from ..validation import _validate_tensors

__all__ = ["chamfer", "hausdorff", "path_smoothness"]


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
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source must be shape (N, 3), got {tuple(source.shape)}"
        )
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(
            f"target must be shape (M, 3), got {tuple(target.shape)}"
        )
    _validate_tensors(source, target, names=["source", "target"])

    dist = torch.cdist(source, target, p=2)
    min_src = dist.min(dim=1).values
    min_tgt = dist.min(dim=0).values

    if squared:
        min_src = min_src ** 2
        min_tgt = min_tgt ** 2

    return (min_src.mean() + min_tgt.mean()) / 2.0


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
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source must be shape (N, 3), got {tuple(source.shape)}"
        )
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(
            f"target must be shape (M, 3), got {tuple(target.shape)}"
        )
    _validate_tensors(source, target, names=["source", "target"])

    dist = torch.cdist(source, target, p=2)
    min_src = dist.min(dim=1).values
    min_tgt = dist.min(dim=0).values

    q = torch.tensor(percentile / 100.0, dtype=source.dtype, device=source.device)
    return torch.max(torch.quantile(min_src, q), torch.quantile(min_tgt, q))


def path_smoothness(path: list[tuple[int, int]]) -> float:
    """Compute smoothness of a DTW alignment path as variance of slope changes.

    Fixes the proto stub which used a plain-slope sum instead of variance of
    slope changes. Zero variance means perfectly linear (constant-slope) path.

    Parameters
    ----------
    path : list[tuple[int, int]]
        Sequence of (i, j) index pairs representing the DTW alignment path.

    Returns
    -------
    float
        Variance of consecutive slope changes along the path. Returns 0.0 if
        fewer than 3 points are provided (need at least 2 slopes for a
        slope-change to exist).

    Notes
    -----
    Slope between step k-1 and step k: (j_k - j_{k-1}) / (i_k - i_{k-1} + 1e-6).
    Uses unbiased=False so a single slope-change still yields 0.0 (not NaN).
    """
    if len(path) < 3:
        return 0.0

    slopes = []
    for k in range(1, len(path)):
        dt1 = path[k][0] - path[k - 1][0]
        dt2 = path[k][1] - path[k - 1][1]
        slopes.append(dt2 / (dt1 + 1e-6))

    deltas = [slopes[i] - slopes[i - 1] for i in range(1, len(slopes))]
    return float(torch.tensor(deltas).var(unbiased=False).item())
