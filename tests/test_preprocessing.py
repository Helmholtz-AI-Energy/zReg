"""Tests for zreg.preprocessing (Phase 41 alignment preprocessing).

Covers ``compute_pca_rotation`` (principal-axes alignment with proper-rotation
determinant fix) and ``detect_velocity_landmarks`` (per-frame displacement
thresholding).  ``zreg.*`` is imported before ``torch`` per the macOS-ARM
libomp lesson (conftest.py enforces the order); this ordering applies in test
files even though it does not apply inside ``src/zreg/`` itself.
"""

import pytest

from zreg.core.dataset import zRegPointCloud
from zreg.preprocessing import compute_pca_rotation, detect_velocity_landmarks
from zreg import preprocessing
from zreg.data_generation import generate_trajectory

import torch


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_trajectory() -> dict[int, zRegPointCloud]:
    """8-frame Gaussian-blob source trajectory, seed=0."""
    return generate_trajectory(n_points=20, n_frames=8, seed=0)


@pytest.fixture
def target_trajectory() -> dict[int, zRegPointCloud]:
    """8-frame Gaussian-blob target trajectory, seed=1."""
    return generate_trajectory(n_points=20, n_frames=8, seed=1)


def _traj(*frames: torch.Tensor) -> dict[int, zRegPointCloud]:
    """Build a trajectory dict from positional pos tensors keyed 0..n-1."""
    return {i: zRegPointCloud(pos=f) for i, f in enumerate(frames)}


def test_module_exports():
    """__all__ declares both public functions."""
    assert preprocessing.__all__ == [
        "compute_pca_rotation",
        "detect_velocity_landmarks",
    ]


# ---------------------------------------------------------------------------
# TestComputePCARotation — ALIGN-06-01, ALIGN-06-02
# ---------------------------------------------------------------------------


class TestComputePCARotation:
    """PCA principal-axes rotation: known-rotation recovery, proper rotation, identity."""

    def test_known_rotation_recovered(self):
        """A 90° Z-axis rotation applied to the source is recovered with small residual."""
        # 90 degrees about Z
        r_true = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        source_cloud = torch.randn(200, 3, generator=torch.Generator().manual_seed(42))
        target_cloud = (r_true @ source_cloud.T).T
        rotation = compute_pca_rotation(source_cloud, target_cloud)
        # Centre both clouds before measuring residual (compute_pca_rotation is
        # defined on mean-centred principal axes).
        src_c = source_cloud - source_cloud.mean(dim=0)
        tgt_c = target_cloud - target_cloud.mean(dim=0)
        aligned = (rotation @ src_c.T).T
        mean_error = (aligned - tgt_c).norm(dim=1).mean().item()
        assert mean_error < 0.1, f"mean_error={mean_error}"

    def test_determinant_is_plus_one(self):
        """The returned matrix is a proper rotation (det ≈ +1, no reflection)."""
        source = torch.randn(200, 3, generator=torch.Generator().manual_seed(7))
        target = torch.randn(200, 3, generator=torch.Generator().manual_seed(8))
        rotation = compute_pca_rotation(source, target)
        assert torch.linalg.det(rotation).item() > 0.99

    def test_identity_clouds_return_near_identity(self):
        """compute_pca_rotation(X, X) returns approximately the identity matrix."""
        cloud = torch.randn(200, 3, generator=torch.Generator().manual_seed(9))
        rotation = compute_pca_rotation(cloud, cloud)
        assert torch.allclose(rotation, torch.eye(3), atol=0.1)


# ---------------------------------------------------------------------------
# TestDetectVelocityLandmarks — ALIGN-06-04
# ---------------------------------------------------------------------------


class TestDetectVelocityLandmarks:
    """Per-frame velocity thresholding: empty below threshold, detection, frame-0 rule, mean vs max."""

    def test_no_landmarks_below_threshold(self):
        """All-zero-displacement frames under a positive threshold yield no landmarks."""
        traj = _traj(*[torch.zeros(10, 3) for _ in range(5)])
        assert detect_velocity_landmarks(traj, threshold=0.001, metric="mean") == []

    def test_high_velocity_frame_detected(self):
        """A frame with a large displacement is flagged when its velocity exceeds the threshold."""
        frames = [torch.zeros(10, 3) for _ in range(5)]
        # Frame 3 jumps by 50; frame 4 mirrors frame 3 so only frame 3 has high velocity.
        frames[3] = torch.ones(10, 3) * 50.0
        frames[4] = torch.ones(10, 3) * 50.0
        traj = _traj(*frames)
        result = detect_velocity_landmarks(traj, threshold=1.0, metric="mean")
        assert 3 in result

    def test_frame_zero_always_zero_velocity(self):
        """Frame 0 has velocity 0.0 by definition — excluded even at threshold 0.0."""
        # Each frame differs from the previous by a constant offset -> all moving frames flagged.
        frames = [torch.ones(10, 3) * float(i) for i in range(5)]
        traj = _traj(*frames)
        result = detect_velocity_landmarks(traj, threshold=0.0, metric="mean")
        assert 0 not in result
        assert len(result) >= 1

    def test_metric_mean_vs_max(self):
        """metric='max' flags a single-extreme-point frame that metric='mean' misses."""
        n = 100
        base = torch.zeros(n, 3)
        moved = base.clone()
        moved[0] = torch.tensor([50.0, 0.0, 0.0])  # one extreme outlier point
        # Frame 2 introduces the outlier; later frames mirror it (zero displacement).
        traj = _traj(base.clone(), base.clone(), moved.clone(), moved.clone(), moved.clone())
        mean_result = detect_velocity_landmarks(traj, threshold=1.0, metric="mean")
        max_result = detect_velocity_landmarks(traj, threshold=1.0, metric="max")
        assert len(max_result) > len(mean_result)


# ---------------------------------------------------------------------------
# Reflection correction path (preprocessing.py:62-64)
# ---------------------------------------------------------------------------


class TestComputePCARotationReflectionCorrection:
    """preprocessing.py:62-64 — det<0 reflection correction is applied."""

    def test_reflection_branch_corrected_to_proper_rotation(self):
        """Force SVD to produce a reflection; verify lines 62-64 correct it to det=+1."""
        from unittest.mock import patch

        source = torch.randn(200, 3, generator=torch.Generator().manual_seed(10))
        target = torch.randn(200, 3, generator=torch.Generator().manual_seed(11))

        _real_svd = torch.linalg.svd
        call_count = [0]

        def _flipped_svd(m, **kwargs):
            call_count[0] += 1
            U, S, Vt = _real_svd(m, **kwargs)
            if call_count[0] == 2:  # second call = target — flip last row → det=-1
                Vt = Vt.clone()
                Vt[-1] = -Vt[-1]
            return U, S, Vt

        with patch.object(torch.linalg, "svd", side_effect=_flipped_svd):
            rotation = compute_pca_rotation(source, target)

        assert call_count[0] == 2, "SVD called exactly twice (source + target)"
        assert torch.linalg.det(rotation).item() > 0.99
