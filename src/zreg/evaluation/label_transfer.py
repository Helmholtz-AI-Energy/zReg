"""Label transfer metrics: F1 with sentinel masking, KNN consistency, temporal stability."""

import numpy as np
import torch
from sklearn.metrics import f1_score as _sklearn_f1
from sklearn.neighbors import KDTree

from zreg.core.transforms import RigidTransformation, AffineTransformation

__all__ = ["compute_f1", "knn_consistency", "temporal_stability"]


def compute_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "weighted",
    zero_division: int = 0,
) -> float:
    """Compute F1 score for label transfer evaluation with sentinel masking.

    Applies a sentinel mask to exclude entries where ``y_true == -1`` before
    computing the F1 score via scikit-learn. Both tensors are moved to CPU and
    converted to NumPy before delegation to sklearn.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth label tensor of shape ``(N,)``. Entries equal to ``-1``
        are treated as unlabelled and are excluded from the computation.
    y_pred : torch.Tensor
        Predicted label tensor of shape ``(N,)``. Must have the same length as
        ``y_true``.
    average : str, optional
        Averaging strategy passed to ``sklearn.metrics.f1_score``. Common
        choices are ``"weighted"`` (default, recommended for HPO) and
        ``"macro"`` (equal weight per class, used for reporting). Any value
        accepted by sklearn is forwarded without modification. Passing an
        unsupported value (e.g. ``"banana"``) will raise a ``ValueError``
        from sklearn.
    zero_division : int, optional
        Value to use when there is a zero division in the F1 computation
        (e.g. a class has no predicted or true positives). Forwarded directly
        to ``sklearn.metrics.f1_score``. Default ``0`` means undefined scores
        contribute zero to the average.

    Returns
    -------
    float
        F1 score as a Python ``float`` in the range ``[0.0, 1.0]``. Returns
        ``0.0`` when all entries are masked out (no labelled samples after
        applying the sentinel mask).

    Raises
    ------
    ValueError
        If ``y_true`` or ``y_pred`` is not 1-D, or if they have different
        lengths.

    Notes
    -----
    **Sentinel masking:** The label ``-1`` is the conventional sentinel code
    used throughout zReg to mark unlabelled or ignored points. Entries where
    ``y_true == -1`` are dropped from both ``y_true`` and ``y_pred`` before
    the F1 computation. This is important for datasets where only a subset of
    points carry ground-truth labels.

    **CPU entry:** scikit-learn does not accept CUDA tensors. Both masked
    arrays are transferred to CPU via ``.detach().cpu().numpy()`` before
    calling sklearn, so CUDA tensors are safe to pass as inputs.

    Examples
    --------
    >>> import torch
    >>> from zreg.metrics.label_transfer import compute_f1
    >>> y = torch.tensor([0, 1, 2, 0, 1])
    >>> compute_f1(y, y)  # perfect prediction
    1.0
    >>> compute_f1(torch.tensor([-1, -1, -1]), torch.tensor([0, 0, 0]))
    0.0
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D, got shapes "
            f"{tuple(y_true.shape)} and {tuple(y_pred.shape)}"
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true length {y_true.shape[0]} != y_pred length {y_pred.shape[0]}"
        )

    mask = y_true != -1
    if mask.sum().item() == 0:
        return 0.0

    y_true_np = y_true[mask].detach().cpu().numpy()
    y_pred_np = y_pred[mask].detach().cpu().numpy()

    score = _sklearn_f1(y_true_np, y_pred_np, average=average, zero_division=zero_division)
    return float(score)


def _rigid_to_matrix(tf: RigidTransformation) -> torch.Tensor:
    device = tf.rot.device
    dtype = tf.rot.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.scale * tf.rot
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _affine_to_matrix(tf: AffineTransformation) -> torch.Tensor:
    device = tf.b.device
    dtype = tf.b.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.b
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _to_matrix(tf) -> torch.Tensor:
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

    The query point is excluded by index, so coincident points with
    different labels are counted correctly.

    **Sentinel masking (62-REVIEW WR-06):** points labelled ``-1`` ("no
    label") are not scored, and ``-1`` neighbours are excluded from a point's
    fraction (its denominator is the number of labelled neighbours among the
    k nearest).  A point whose k neighbours are all ``-1`` is not scored.
    Returns ``0.0`` when no point can be scored.
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

    # Exclude the query point by index, not by position: with coincident
    # points the query point need not be in column 0. If more than k + 1
    # points coincide, the query point may be missing from the k + 1
    # returned neighbours; then the last (farthest) column is dropped so
    # every row keeps exactly k neighbours.
    is_self = idx == np.arange(n)[:, None]
    self_missing = ~is_self.any(axis=1)
    is_self[self_missing, -1] = True
    neigh = idx[~is_self].reshape(n, k)

    # 62-REVIEW WR-06: -1 is the "no label" sentinel (as in compute_f1):
    # unlabelled query points are not scored and unlabelled neighbours are
    # left out of each point's fraction.  Without -1 this is exactly the
    # plain mean over the k neighbours.
    neigh_labels = labels_np[neigh]
    valid_neigh = neigh_labels != -1
    n_valid = valid_neigh.sum(axis=1)
    scored = (labels_np != -1) & (n_valid > 0)
    if not scored.any():
        return 0.0
    agree = ((neigh_labels == labels_np[:, None]) & valid_neigh).sum(axis=1)
    scores = agree[scored] / n_valid[scored]
    return float(scores.mean())


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
        If any element is not a RigidTransformation/AffineTransformation, or
        if the transforms do not share one device and one dtype (mixed dtypes
        are rejected rather than promoted, per Phase 62 D-03).

    Notes
    -----
    Builds 4x4 homogeneous matrices manually from .rot/.t/.scale (Rigid) or
    .b/.t (Affine). No .to_matrix method is called on any transform object.
    Uses torch.norm(..., p="fro") for Frobenius norm computation.
    """
    for tf in transforms:
        if not isinstance(tf, (RigidTransformation, AffineTransformation)):
            raise TypeError(
                f"Unsupported transformation type: {type(tf).__name__}"
            )
    if len(transforms) < 2:
        if transforms:
            ref = _to_matrix(transforms[0])
            return torch.tensor(0.0, dtype=ref.dtype, device=ref.device)
        return torch.tensor(0.0)

    matrices = [_to_matrix(tf) for tf in transforms]

    ref_device = matrices[0].device
    ref_dtype = matrices[0].dtype
    if any(m.device != ref_device for m in matrices):
        raise TypeError(
            "temporal_stability: all transforms must be on the same device, "
            f"got {sorted({str(m.device) for m in matrices})}"
        )
    if any(m.dtype != ref_dtype for m in matrices):
        raise TypeError(
            "temporal_stability: all transforms must have the same dtype, "
            f"got {sorted({str(m.dtype) for m in matrices})}"
        )

    norms = []
    for i in range(1, len(matrices)):
        norms.append(torch.norm(matrices[i] - matrices[i - 1], p="fro"))

    return torch.stack(norms).mean()
