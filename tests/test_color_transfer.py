"""Tests for color transfer functionality."""

import pytest
import torch

from zreg.color_transfer import transfer_colors, ColorTransferMethod
from zreg.dataset import zRegPointCloud


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

    source_pc = zRegPointCloud(pos=source_pos, color=source_colors)

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
    expected_colors = source_pc["color"]
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
        source_colors=source_pc["color"]
    )

    assert transferred_colors.shape == (4, 3)


def test_transfer_colors_cpd_weighted(sample_point_clouds):
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
    assert torch.allclose(transferred_colors, source_pc["color"], atol=1e-6)


def test_transfer_colors_cpd_weighted_uniform(sample_point_clouds):
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
    expected_avg = source_pc["color"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-6)


def test_transfer_colors_invalid_method(sample_point_clouds):
    """Test error handling for invalid method."""
    source_pc, target_pc = sample_point_clouds

    with pytest.raises(ValueError, match="Unknown color transfer method"):
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

    source_pc = zRegPointCloud(pos=source_pos, color=source_colors)

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

    with pytest.raises(ValueError, match="single-channel colors"):
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
    expected_avg = source_pc["color"].mean(dim=0)
    assert torch.allclose(transferred_colors, expected_avg, atol=1e-2)


class TestColorTransferEdgeCases:
    """Edge case tests for color transfer (TEST-05): empty source, single-point source,
    and dimension-mismatched inputs."""

    # --- Empty source tests (D-10, D-12) ---

    def test_empty_source_zreg_raises_valueerror(self):
        """Empty source zRegPointCloud raises ValueError."""
        source = zRegPointCloud(pos=torch.zeros(0, 3), color=torch.zeros(0, 3))
        target = zRegPointCloud(pos=torch.randn(5, 3))

        with pytest.raises(ValueError, match="source has no points"):
            transfer_colors(source, target, method=ColorTransferMethod.NEAREST_NEIGHBOR)

    def test_empty_source_tensor_raises_valueerror(self):
        """Empty source tensors raise ValueError."""
        source_pos = torch.zeros(0, 3)
        source_colors = torch.zeros(0, 3)
        target_pos = torch.randn(5, 3)

        with pytest.raises(ValueError, match="source has no points"):
            transfer_colors(
                source_pos, target_pos,
                method=ColorTransferMethod.NEAREST_NEIGHBOR,
                source_colors=source_colors,
            )

    def test_empty_source_error_contains_shape(self):
        """Empty source error message includes the shape tuple."""
        source = zRegPointCloud(pos=torch.zeros(0, 3), color=torch.zeros(0, 3))
        target = zRegPointCloud(pos=torch.randn(5, 3))

        with pytest.raises(ValueError) as exc_info:
            transfer_colors(source, target, method=ColorTransferMethod.NEAREST_NEIGHBOR)

        assert "(0, 3)" in str(exc_info.value)

    # --- Single-point source tests (D-11) ---

    def test_single_point_source_nearest_neighbor(self):
        """Single-point source returns valid output via nearest neighbor."""
        source = zRegPointCloud(
            pos=torch.tensor([[1.0, 2.0, 3.0]]),
            color=torch.tensor([[0.5, 0.8, 0.2]]),
        )
        target = zRegPointCloud(pos=torch.randn(10, 3))

        result = transfer_colors(source, target, method=ColorTransferMethod.NEAREST_NEIGHBOR)

        assert result.shape == (10, 3)
        assert torch.isfinite(result).all()
        # All target points must receive the single source color
        assert torch.allclose(result, torch.tensor([[0.5, 0.8, 0.2]]).expand(10, 3))

    def test_single_point_source_gaussian_kernel(self):
        """Single-point source returns valid output via Gaussian kernel."""
        source = zRegPointCloud(
            pos=torch.tensor([[1.0, 2.0, 3.0]]),
            color=torch.tensor([[0.5, 0.8, 0.2]]),
        )
        target = zRegPointCloud(pos=torch.randn(10, 3))

        result = transfer_colors(
            source, target, method=ColorTransferMethod.GAUSSIAN_KERNEL, sigma=1.0
        )

        assert result.shape == (10, 3)
        assert torch.isfinite(result).all()
        # With one source point all Gaussian weights go to it
        assert torch.allclose(result, torch.tensor([[0.5, 0.8, 0.2]]).expand(10, 3))

    def test_single_point_source_cpd_weighted(self):
        """Single-point source returns valid output via CPD weighted."""
        source = zRegPointCloud(
            pos=torch.tensor([[1.0, 2.0, 3.0]]),
            color=torch.tensor([[0.5, 0.8, 0.2]]),
        )
        target = zRegPointCloud(pos=torch.randn(10, 3))

        # pmat shape: (n_target=10, n_source=1)
        # Build mock manually to avoid MockEstepResult matrix size issue
        class _SimpleEstep:
            pass
        estep_result = _SimpleEstep()
        estep_result.pmat = torch.ones(10, 1)

        result = transfer_colors(
            source, target, method=ColorTransferMethod.CPD_WEIGHTED, estep_result=estep_result
        )

        assert result.shape == (10, 3)
        assert torch.isfinite(result).all()

    # --- Dimension mismatch tests ---

    def test_dimension_mismatch_3d_source_2d_target(self):
        """3D source with 2D target raises ValueError."""
        source = zRegPointCloud(pos=torch.randn(10, 3), color=torch.randn(10, 3))
        target = zRegPointCloud(pos=torch.randn(5, 2))

        with pytest.raises(ValueError, match="same dimensionality"):
            transfer_colors(source, target, method=ColorTransferMethod.NEAREST_NEIGHBOR)

    def test_dimension_mismatch_2d_source_3d_target(self):
        """2D source with 3D target raises ValueError."""
        source = zRegPointCloud(pos=torch.randn(10, 2), color=torch.randn(10, 2))
        target = zRegPointCloud(pos=torch.randn(5, 3))

        with pytest.raises(ValueError, match="same dimensionality"):
            transfer_colors(source, target, method=ColorTransferMethod.NEAREST_NEIGHBOR)