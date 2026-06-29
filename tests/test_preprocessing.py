"""Tests for zreg.preprocessing (Phase 41 alignment preprocessing).

Covers ``compute_pca_rotation`` (principal-axes alignment with proper-rotation
determinant fix) and ``detect_velocity_landmarks`` (per-frame displacement
thresholding).  zreg.* is imported before torch per the macOS libomp lesson
(conftest.py enforces the order).
"""

from zreg.dataset import zRegPointCloud

import torch

from zreg import preprocessing
from zreg.preprocessing import compute_pca_rotation, detect_velocity_landmarks


def test_module_exports():
    """__all__ declares both public functions."""
    assert preprocessing.__all__ == [
        "compute_pca_rotation",
        "detect_velocity_landmarks",
    ]


def test_compute_pca_rotation_shape_and_proper_rotation():
    """R is (3, 3) with det ≈ +1 (proper rotation, no reflection)."""
    torch.manual_seed(0)
    R = compute_pca_rotation(torch.randn(200, 3), torch.randn(200, 3))
    assert R.shape == (3, 3)
    assert torch.linalg.det(R).item() > 0.99


def test_compute_pca_rotation_is_orthogonal():
    """R @ R.T ≈ I — the returned matrix is orthonormal."""
    torch.manual_seed(1)
    R = compute_pca_rotation(torch.randn(150, 3), torch.randn(150, 3))
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)


def test_compute_pca_rotation_identity_for_same_cloud():
    """compute_pca_rotation(X, X) returns approximately the identity."""
    torch.manual_seed(2)
    X = torch.randn(200, 3)
    R = compute_pca_rotation(X, X)
    assert torch.allclose(R, torch.eye(3), atol=0.1)


def test_compute_pca_rotation_recovers_z_rotation():
    """A 90° rotation about Z is recovered with small residual error."""
    torch.manual_seed(3)
    source = torch.randn(200, 3)
    # 90 degrees about Z
    R_true = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    target = source @ R_true.T
    R = compute_pca_rotation(source, target)
    src_c = source - source.mean(dim=0)
    tgt_c = target - target.mean(dim=0)
    aligned = src_c @ R.T
    mean_err = (aligned - tgt_c).norm(dim=1).mean().item()
    assert mean_err < 0.1, f"mean_err={mean_err}"


def _traj(*frames: torch.Tensor) -> dict[int, zRegPointCloud]:
    return {i: zRegPointCloud(pos=f) for i, f in enumerate(frames)}


def test_detect_velocity_landmarks_high_threshold_empty():
    """A normal trajectory under a huge threshold yields no landmarks."""
    traj = _traj(torch.zeros(10, 3), torch.zeros(10, 3), torch.ones(10, 3))
    assert detect_velocity_landmarks(traj, threshold=1000.0, metric="mean") == []


def test_detect_velocity_landmarks_frame_zero_never_included():
    """Frame 0 has velocity 0.0 by definition — excluded even at threshold 0."""
    traj = _traj(torch.zeros(10, 3), torch.ones(10, 3) * 5.0)
    landmarks = detect_velocity_landmarks(traj, threshold=0.0, metric="mean")
    assert 0 not in landmarks
    assert landmarks == [1]


def test_detect_velocity_landmarks_returns_int_keys():
    """Returned landmarks are the integer frame keys of the trajectory."""
    traj = _traj(torch.zeros(5, 3), torch.zeros(5, 3))
    result = detect_velocity_landmarks(traj, threshold=-1.0, metric="mean")
    assert all(isinstance(k, int) for k in result)


def test_detect_velocity_landmarks_mean_vs_max_differ():
    """metric='max' differs from 'mean' on a single-extreme-point frame."""
    prev = torch.zeros(10, 3)
    curr = torch.zeros(10, 3)
    curr[0] = torch.tensor([100.0, 0.0, 0.0])  # one extreme displacement
    traj = _traj(prev, curr)
    # Threshold between the mean displacement (10.0) and the max (100.0).
    mean_res = detect_velocity_landmarks(traj, threshold=50.0, metric="mean")
    max_res = detect_velocity_landmarks(traj, threshold=50.0, metric="max")
    assert mean_res == []
    assert max_res == [1]
