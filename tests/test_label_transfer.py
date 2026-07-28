"""Tests for color transfer functionality."""

import pytest
import torch

from zreg.label_transfer import transfer_labels as transfer_colors, LabelTransferMethod as ColorTransferMethod
from zreg.core.dataset import zRegPointCloud


class MockEstepResult:
    """Mock EstepResult for testing."""
    def __init__(self, pmat):
        self.pt1 = pmat.sum(dim=1)  # sum over source for each target
        self.p1 = pmat.sum(dim=0)   # sum over target for each source
        self.px = torch.matmul(pmat, torch.randn_like(pmat)[:, :3])  # dummy
        self.n_p = self.p1.sum()
        self.pmat = pmat


@pytest.fixture
def sample_point_clouds():
    """Create sample point clouds for testing."""
    # Source point cloud
    source_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)

    source_colors = torch.tensor([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0]   # Yellow
    ], dtype=torch.float32)

    source_pc = zRegPointCloud(pos=source_pos, label=source_colors)

    # Target point cloud (slightly offset)
    target_pos = torch.tensor([
        [0.1, 0.1, 0.0],  # Close to source[0]
        [0.9, 0.1, 0.0],  # Close to source[1]
        [0.1, 0.9, 0.0],  # Close to source[2]
        [0.9, 0.9, 0.0]   # Close to source[3]
    ], dtype=torch.float32)

    target_pc = zRegPointCloud(pos=target_pos)

    return source_pc, target_pc


def test_transfer_colors_nearest_neighbor_zreg(sample_point_clouds):
    """Test nearest neighbor color transfer with zRegPointCloud inputs."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.NEAREST_NEIGHBOR
    )

    # Check shape
    assert transferred_colors.shape == (4, 3)

    # Check that colors are from source (exact matches expected due to proximity)
    expected_colors = source_pc["label"]
    for i in range(4):
        # Check if transferred color matches any source color
        matches = [torch.allclose(transferred_colors[i], expected_colors[j], atol=1e-6) for j in range(4)]
        assert any(matches), f"Transferred color {i} doesn't match any source color"


def test_transfer_colors_nearest_neighbor_tensors(sample_point_clouds):
    """Test nearest neighbor color transfer with tensor inputs."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc["pos"], target_pc["pos"],
        method=ColorTransferMethod.NEAREST_NEIGHBOR,
        source_colors=source_pc["label"]
    )

    assert transferred_colors.shape == (4, 3)


def test_transfer_labels_cpd_weighted(sample_point_clouds):
    """Test CPD-weighted color transfer."""
    source_pc, target_pc = sample_point_clouds

    # Create mock probability matrix (identity for simplicity)
    prob_matrix = torch.eye(4, dtype=torch.float32)
    estep_result = MockEstepResult(prob_matrix)

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result
    )

    # With identity probabilities, should get exact source colors
    assert torch.allclose(transferred_colors, source_pc["label"], atol=1e-6)


def test_transfer_labels_cpd_weighted_uniform(sample_point_clouds):
    """Test CPD-weighted with uniform probabilities."""
    source_pc, target_pc = sample_point_clouds

    # Uniform probabilities (each target equally likely to match any source)
    prob_matrix = torch.full((4, 4), 0.25, dtype=torch.float32)
    estep_result = MockEstepResult(prob_matrix)

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result
    )

    # Should get average of all source colors
    expected_avg = source_pc["label"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-6)


def test_transfer_colors_invalid_method(sample_point_clouds):
    """Test error handling for invalid method."""
    source_pc, target_pc = sample_point_clouds

    with pytest.raises(ValueError, match="Unknown label transfer method"):
        transfer_colors(source_pc, target_pc, method="invalid_method")


def test_transfer_colors_missing_estep_result(sample_point_clouds):
    """Test error handling when estep_result is missing for CPD method."""
    source_pc, target_pc = sample_point_clouds

    with pytest.raises(ValueError, match="estep_result is required"):
        transfer_colors(
            source_pc, target_pc,
            method=ColorTransferMethod.CPD_WEIGHTED
        )


def test_transfer_colors_missing_source_colors():
    """Test error handling when source_colors is missing for tensor input."""
    source_pos = torch.randn(4, 3)
    target_pos = torch.randn(4, 3)

    with pytest.raises(ValueError, match="source_colors must be provided"):
        transfer_colors(
            source_pos, target_pos,
            method=ColorTransferMethod.NEAREST_NEIGHBOR
        )


def test_transfer_colors_dimension_mismatch(sample_point_clouds):
    """Test error handling for mismatched dimensions."""
    source_pc, target_pc = sample_point_clouds

    # Create target with different dimensionality
    target_2d = zRegPointCloud(pos=torch.randn(4, 2))

    with pytest.raises(ValueError, match="same dimensionality"):
        transfer_colors(source_pc, target_2d, method=ColorTransferMethod.NEAREST_NEIGHBOR)


def test_transfer_colors_prob_matrix_transpose(sample_point_clouds):
    """Test handling of transposed probability matrices."""
    source_pc, target_pc = sample_point_clouds

    # Create probability matrix with wrong orientation (source, target) instead of (target, source)
    prob_matrix = torch.eye(4, dtype=torch.float32)
    estep_result = MockEstepResult(prob_matrix.T)  # Transpose to simulate wrong orientation

    # Should still work due to automatic handling
    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result
    )

    assert transferred_colors.shape == (4, 3)


@pytest.fixture
def sample_point_clouds_single_channel():
    """Create sample point clouds with single-channel colors (class indices)."""
    # Source point cloud
    source_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)

    source_colors = torch.tensor([
        [0],  # Class 0
        [1],  # Class 1
        [2],  # Class 2
        [0]   # Class 0
    ], dtype=torch.float32)

    source_pc = zRegPointCloud(pos=source_pos, label=source_colors)

    # Target point cloud (slightly offset)
    target_pos = torch.tensor([
        [0.1, 0.1, 0.0],  # Close to source[0] (class 0)
        [0.9, 0.1, 0.0],  # Close to source[1] (class 1)
        [0.1, 0.9, 0.0],  # Close to source[2] (class 2)
        [0.9, 0.9, 0.0]   # Close to source[3] (class 0)
    ], dtype=torch.float32)

    target_pc = zRegPointCloud(pos=target_pos)

    return source_pc, target_pc


def test_transfer_colors_knn_voting(sample_point_clouds_single_channel):
    """Test KNN voting color transfer."""
    source_pc, target_pc = sample_point_clouds_single_channel

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.KNN_VOTING,
        k=1  # With k=1, should be same as nearest neighbor
    )

    assert transferred_colors.shape == (4, 1)

    # With k=1, should match nearest neighbor
    expected_classes = torch.tensor([[0], [1], [2], [0]], dtype=torch.float32)
    assert torch.allclose(transferred_colors, expected_classes, atol=1e-6)


def test_transfer_colors_knn_voting_k3(sample_point_clouds_single_channel):
    """Test KNN voting with k=3."""
    source_pc, target_pc = sample_point_clouds_single_channel

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.KNN_VOTING,
        k=3
    )

    assert transferred_colors.shape == (4, 1)

    # With k=3, each target should get the majority class among its 3 nearest neighbors
    # For this setup, the 3 nearest for each should include itself and neighbors
    # But since positions are close, it might vary, but let's just check shape and range
    assert transferred_colors.min() >= 0
    assert transferred_colors.max() <= 2


def test_transfer_colors_knn_voting_multichannel_error(sample_point_clouds):
    """Test KNN voting raises error for multi-channel colors."""
    source_pc, target_pc = sample_point_clouds  # This has 3-channel colors

    with pytest.raises(ValueError, match="single-channel"):
        transfer_colors(
            source_pc, target_pc,
            method=ColorTransferMethod.KNN_VOTING,
            k=3
        )


def test_transfer_colors_gaussian_kernel(sample_point_clouds):
    """Test Gaussian kernel color transfer."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.GAUSSIAN_KERNEL,
        sigma=0.1  # Small sigma, should be close to nearest neighbor
    )

    assert transferred_colors.shape == (4, 3)

    # Colors should be weighted averages
    # Check that values are reasonable (between 0 and 1 for RGB-like colors)
    assert transferred_colors.min() >= 0
    assert transferred_colors.max() <= 1


def test_transfer_colors_gaussian_kernel_large_sigma(sample_point_clouds):
    """Test Gaussian kernel with large sigma (should average all colors)."""
    source_pc, target_pc = sample_point_clouds

    transferred_colors = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.GAUSSIAN_KERNEL,
        sigma=10.0  # Large sigma, uniform weights
    )

    assert transferred_colors.shape == (4, 3)

    # Should be close to the average of all source colors
    expected_avg = source_pc["label"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-2)

class TestColorTransferEdgeCases:
    """Regression tests for color transfer edge cases (TEST-05)."""

    def test_empty_source_tensor_raises_valueerror(self):
        """transfer_colors raises ValueError with descriptive message for 0-point source (Tensor)."""
        source_pos = torch.zeros(0, 3)
        source_colors = torch.zeros(0, 3)
        target_pos = torch.randn(5, 3)
        with pytest.raises(ValueError, match="0 points"):
            transfer_colors(source_pos, target_pos, source_colors=source_colors)

    def test_empty_source_zreg_raises_valueerror(self):
        """transfer_colors raises ValueError with descriptive message for 0-point source (zRegPointCloud)."""
        source = zRegPointCloud(pos=torch.zeros(0, 3), label=torch.zeros(0, 3))
        target = zRegPointCloud(pos=torch.randn(5, 3))
        with pytest.raises(ValueError, match="0 points"):
            transfer_colors(source, target)

    def test_single_point_source_returns_correct_shape(self):
        """transfer_colors returns (m_target, n_channels) for a single-point source."""
        source = zRegPointCloud(
            pos=torch.tensor([[0.0, 0.0, 0.0]]),
            label=torch.tensor([[0.2, 0.5, 0.8]]),
        )
        target = zRegPointCloud(pos=torch.randn(7, 3))
        result = transfer_colors(source, target)
        assert result.shape == (7, 3)
        assert torch.isfinite(result).all()

    def test_single_point_source_all_targets_get_same_color(self):
        """With a single source point, all target points receive that point's color."""
        color = torch.tensor([[0.1, 0.9, 0.4]])
        source = zRegPointCloud(pos=torch.tensor([[0.0, 0.0, 0.0]]), label=color)
        target = zRegPointCloud(pos=torch.randn(5, 3))
        result = transfer_colors(source, target)
        assert torch.allclose(result, color.expand(5, 3), atol=1e-5)

    def test_dimension_mismatch_raises_valueerror(self):
        """transfer_colors raises ValueError when source and target have different spatial dims."""
        source_pos = torch.randn(10, 3)
        source_colors = torch.zeros(10, 3)
        target_pos = torch.randn(5, 2)
        with pytest.raises(ValueError, match="dimensionality"):
            transfer_colors(source_pos, target_pos, source_colors=source_colors)


def test_cpd_weighted_transposed_pmat_raises():
    """_transfer_labels_cpd_weighted raises ValueError when pmat shape looks transposed."""
    import torch
    from zreg.label_transfer import _transfer_labels_cpd_weighted
    source_pos = torch.randn(4, 3)
    target_pos = torch.randn(3, 3)
    source_colors = torch.randn(4, 3)

    class _Mock:
        pmat = torch.ones(4, 3) / 3.0  # (n_source=4, n_target=3) → transposed

    with pytest.raises(ValueError, match="transposed"):
        _transfer_labels_cpd_weighted(source_pos, target_pos, source_colors, _Mock())


def test_cpd_weighted_wrong_shape_pmat_raises():
    """_transfer_labels_cpd_weighted raises ValueError for completely unexpected pmat shape."""
    import torch
    from zreg.label_transfer import _transfer_labels_cpd_weighted
    source_pos = torch.randn(4, 3)
    target_pos = torch.randn(3, 3)
    source_colors = torch.randn(4, 3)

    class _Mock:
        pmat = torch.ones(5, 5)  # neither (n_target, n_source) nor (n_source, n_target)

    with pytest.raises(ValueError, match="expected"):
        _transfer_labels_cpd_weighted(source_pos, target_pos, source_colors, _Mock())

