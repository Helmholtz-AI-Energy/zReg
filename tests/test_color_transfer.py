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