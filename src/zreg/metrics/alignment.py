"""Alignment metric functions for zReg.

Provides five pure alignment metrics:
- chamfer: bidirectional Chamfer distance between point clouds
- hausdorff: percentile-based Hausdorff distance
- path_smoothness: variance of slope changes along a DTW path
- knn_consistency: fraction of k-nearest neighbours sharing the same label
- temporal_stability: mean Frobenius norm of consecutive transform differences
"""

import torch
from sklearn.neighbors import KDTree

from zreg.transforms import RigidTransformation, AffineTransformation
from ..validation import _validate_tensors

__all__ = ["chamfer", "hausdorff", "path_smoothness", "knn_consistency", "temporal_stability"]


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


def _rigid_to_matrix(tf: RigidTransformation) -> torch.Tensor:
    """Build a 4x4 homogeneous matrix from a RigidTransformation.

    Parameters
    ----------
    tf : RigidTransformation
        Rigid transformation with .rot (3x3), .t (3,), .scale (float).

    Returns
    -------
    torch.Tensor
        4x4 homogeneous transformation matrix on tf.rot.device.
    """
    device = tf.rot.device
    dtype = tf.rot.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.scale * tf.rot
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _affine_to_matrix(tf: AffineTransformation) -> torch.Tensor:
    """Build a 4x4 homogeneous matrix from an AffineTransformation.

    Parameters
    ----------
    tf : AffineTransformation
        Affine transformation with .b (3x3), .t (3,).

    Returns
    -------
    torch.Tensor
        4x4 homogeneous transformation matrix on tf.b.device.
    """
    device = tf.b.device
    dtype = tf.b.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.b
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _to_matrix(tf) -> torch.Tensor:
    """Dispatch to the correct matrix builder based on transform type.

    Parameters
    ----------
    tf : RigidTransformation or AffineTransformation
        The transformation to convert.

    Returns
    -------
    torch.Tensor
        4x4 homogeneous transformation matrix.

    Raises
    ------
    TypeError
        If tf is not a RigidTransformation or AffineTransformation.
    """
    if isinstance(tf, RigidTransformation):
        return _rigid_to_matrix(tf)
    elif isinstance(tf, AffineTransformation):
        return _affine_to_matrix(tf)
    else:
        raise TypeError(f"Unsupported transformation type: {type(tf).__name__}")


def knn_consistency(
    points: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> float:
    """Compute k-nearest-neighbour label consistency for a labelled point cloud.

    For each point, queries its k nearest neighbours and computes the fraction
    that share the same label. Returns the mean fraction across all points.

    Parameters
    ----------
    points : torch.Tensor
        Point cloud of shape (N, 3).
    labels : torch.Tensor
        Integer label tensor of shape (N,).
    k : int, optional
        Number of nearest neighbours to query. Default: 10.

    Returns
    -------
    float
        Mean label consistency score in [0.0, 1.0].

    Raises
    ------
    ValueError
        If points is not shape (N, 3).
        If labels is not shape (N,) or labels.shape[0] != points.shape[0].
        If k < 1 or k >= N.

    Notes
    -----
    Uses sklearn.neighbors.KDTree via .detach().cpu().numpy() CPU entry point
    (GPU-safe: all KDTree operations run on CPU numpy arrays).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must be shape (N, 3), got {tuple(points.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be shape (N,), got {tuple(labels.shape)}"
        )
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"points and labels must have the same length: "
            f"points.shape[0]={points.shape[0]}, labels.shape[0]={labels.shape[0]}"
        )
    n = points.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")
    if k >= n:
        raise ValueError(
            f"k must be < N (number of points), got k={k}, N={n}"
        )

    points_np = points.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    tree = KDTree(points_np)
    _, idx = tree.query(points_np, k=k + 1)

    scores = []
    for i in range(n):
        neighbour_labels = labels_np[idx[i, 1:]]
        match_count = (neighbour_labels == labels_np[i]).sum()
        scores.append(match_count / k)

    return float(sum(scores) / len(scores))


def temporal_stability(
    transforms: list[RigidTransformation | AffineTransformation],
) -> torch.Tensor:
    """Compute temporal stability as mean Frobenius norm of consecutive transform differences.

    Measures how smoothly a sequence of transformations evolves over time.
    A score of 0.0 indicates perfectly stable (identical) consecutive transforms.

    Parameters
    ----------
    transforms : list[RigidTransformation | AffineTransformation]
        Ordered sequence of transformations.

    Returns
    -------
    torch.Tensor
        Scalar tensor. Returns tensor(0.0) for lists of length 0 or 1.

    Raises
    ------
    TypeError
        If any element is not a RigidTransformation or AffineTransformation.

    Notes
    -----
    Builds 4x4 homogeneous matrices manually from .rot/.t/.scale (Rigid) or
    .b/.t (Affine). No .to_matrix method is called on any transform object.
    Uses torch.norm(..., p="fro") for Frobenius norm computation.
    """
    if len(transforms) < 2:
        return torch.tensor(0.0)

    matrices = [_to_matrix(tf) for tf in transforms]

    norms = []
    for i in range(1, len(matrices)):
        norms.append(torch.norm(matrices[i] - matrices[i - 1], p="fro"))

    return torch.stack(norms).mean()
