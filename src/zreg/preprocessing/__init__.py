"""Alignment preprocessing primitives (Phase 41).

This module provides two stateless functions that run *before* DTW inside
``AlignmentStage.run`` (wired in plan 41-02):

- ``compute_pca_rotation`` — principal-axes (PCA) rotation that aligns a source
  cloud's principal axes to a target cloud's, with a determinant-sign fix that
  guarantees a proper rotation (``det = +1``, no reflection).
- ``detect_velocity_landmarks`` — per-frame displacement thresholding that flags
  trajectory frames whose mean/max point velocity exceeds a threshold.

``src/zreg/`` is itself a zreg module, so the macOS libomp import-order
constraint (zreg before torch) does not apply *inside* the package; the
constraint applies to external consumers.
"""

from typing import Literal

import torch

from zreg.core.dataset import zRegPointCloud

__all__ = ["compute_pca_rotation", "detect_velocity_landmarks"]


def compute_pca_rotation(
    source_cloud: torch.Tensor, target_cloud: torch.Tensor
) -> torch.Tensor:
    """Compute the rotation aligning the source's principal axes to the target's.

    Both clouds are mean-centred, their principal axes are extracted via SVD
    (columns of ``V`` in descending-variance order), and the rotation
    ``R = P_tgt @ P_src.T`` maps source axes onto target axes.  If ``R`` is a
    reflection (``det < 0``) the least-significant source axis is flipped so the
    result is always a proper rotation (``det = +1``).  A single sign fix is
    sufficient — axis-by-axis flipping is an anti-pattern.

    Parameters
    ----------
    source_cloud : torch.Tensor
        Source points, shape ``[N, 3]``.
    target_cloud : torch.Tensor
        Target points, shape ``[M, 3]``.

    Returns
    -------
    torch.Tensor
        A ``(3, 3)`` proper rotation matrix (``det = +1``).
    """
    src_centered = source_cloud - source_cloud.mean(dim=0)
    tgt_centered = target_cloud - target_cloud.mean(dim=0)

    _, _, vt_src = torch.linalg.svd(src_centered, full_matrices=False)
    _, _, vt_tgt = torch.linalg.svd(tgt_centered, full_matrices=False)

    # Columns are principal axes in descending-variance order.
    p_src = vt_src.T
    p_tgt = vt_tgt.T

    rotation = p_tgt @ p_src.T
    if torch.linalg.det(rotation) < 0:
        p_src = p_src.clone()
        p_src[:, -1] = -p_src[:, -1]
        rotation = p_tgt @ p_src.T
    return rotation


def detect_velocity_landmarks(
    trajectory: dict[int, zRegPointCloud],
    threshold: float,
    metric: Literal["mean", "max"],
) -> list[int]:
    """Flag trajectory frames whose per-point velocity exceeds ``threshold``.

    Frames are processed in sorted-key order.  The velocity of a frame is the
    mean (``metric="mean"``) or max (``metric="max"``) L2 displacement of its
    points relative to the previous frame.  The first frame has velocity ``0.0``
    by definition and is never flagged.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Per-frame point clouds keyed by integer frame index.  Each cloud's
        ``"pos"`` tensor has shape ``[N, 3]``.
    threshold : float
        Velocity above which a frame is recorded as a landmark.
    metric : {"mean", "max"}
        Reduction applied to per-point displacement norms.

    Returns
    -------
    list[int]
        Integer frame keys whose velocity strictly exceeds ``threshold``.
    """
    sorted_keys = sorted(trajectory.keys())
    landmarks: list[int] = []
    for i, k in enumerate(sorted_keys):
        if i == 0:
            velocity = 0.0
        else:
            prev_k = sorted_keys[i - 1]
            diffs = trajectory[k]["pos"] - trajectory[prev_k]["pos"]
            norms = diffs.norm(dim=1)
            velocity = norms.mean().item() if metric == "mean" else norms.max().item()
        if velocity > threshold:
            landmarks.append(k)
    return landmarks
