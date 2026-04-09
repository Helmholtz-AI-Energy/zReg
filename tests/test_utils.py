"""Tests for zreg.utils module."""

from unittest.mock import patch

import pytest
import torch

from zreg import utils


class TestSquaredKernel:
    """Tests for squared_kernel and squared_kernel_sum functions."""

    def test_squared_kernel_shape(self):
        """Test that squared_kernel returns correct shape."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.squared_kernel(x, y)
        # Result should be (m, n) = (15, 10)
        assert result.shape == (15, 10)

    def test_squared_kernel_same_points(self):
        """Test squared_kernel with same points returns zeros on diagonal."""
        x = torch.randn(5, 3)
        result = utils.squared_kernel(x, x)
        # Diagonal should be zeros (distance to self)
        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.zeros(5), atol=1e-6)

    def test_squared_kernel_positive(self):
        """Test that squared_kernel returns non-negative values."""
        x = torch.randn(10, 3)
        y = torch.randn(8, 3)
        result = utils.squared_kernel(x, y)
        assert (result >= 0).all()

    def test_squared_kernel_sum(self):
        """Test squared_kernel_sum returns scalar."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.squared_kernel_sum(x, y)
        assert result.ndim == 0  # scalar
        assert result >= 0


class TestRbfKernel:
    """Tests for RBF kernel function."""

    def test_rbf_kernel_shape(self):
        """Test that rbf_kernel returns correct shape."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.rbf_kernel(x, y, beta=1.0)
        assert result.shape == (15, 10)

    def test_rbf_kernel_range(self):
        """Test that rbf_kernel values are in (0, 1]."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.rbf_kernel(x, y, beta=1.0)
        assert (result > 0).all()
        assert (result <= 1).all()

    def test_rbf_kernel_same_points(self):
        """Test rbf_kernel with same points returns ones on diagonal."""
        x = torch.randn(5, 3)
        result = utils.rbf_kernel(x, x, beta=1.0)
        diagonal = torch.diag(result)
        assert torch.allclose(diagonal, torch.ones(5), atol=1e-5)

    def test_rbf_kernel_beta_effect(self):
        """Test that larger beta gives smoother kernel (values closer to 1)."""
        x = torch.randn(10, 3)
        y = torch.randn(10, 3)
        result_small_beta = utils.rbf_kernel(x, y, beta=0.1)
        result_large_beta = utils.rbf_kernel(x, y, beta=10.0)
        # Larger beta should give values closer to 1
        assert result_large_beta.mean() > result_small_beta.mean()


class TestRBFKernelNormalization:
    """Regression tests verifying rbf_kernel is a pure computation without internal normalization."""

    def _make_normalized(self, n, d):
        """Create a tensor normalized to [-1, 1]."""
        pts = torch.randn(n, d)
        pts, _ = utils.normalize_point_cloud(pts)
        return pts

    def test_rbf_kernel_shape(self):
        """rbf_kernel with pre-normalized inputs returns finite tensor of correct shape (m, n)."""
        x = self._make_normalized(10, 3)
        y = self._make_normalized(15, 3)
        result = utils.rbf_kernel(x, y, beta=2.0)
        assert result.shape == (15, 10)
        assert torch.isfinite(result).all()

    def test_rbf_kernel_no_internal_normalization(self):
        """rbf_kernel does NOT call normalize_point_cloud internally."""
        x = self._make_normalized(10, 3)
        y = self._make_normalized(15, 3)
        with patch("zreg.utils.normalize_point_cloud") as mock_norm:
            utils.rbf_kernel(x, y, 2.0)
            mock_norm.assert_not_called()

    def test_rbf_kernel_values_in_range(self):
        """rbf_kernel output values are in range (0, 1] for pre-normalized data."""
        x = self._make_normalized(10, 3)
        y = self._make_normalized(15, 3)
        result = utils.rbf_kernel(x, y, beta=2.0)
        assert (result > 0).all()
        assert (result <= 1).all()


class TestTpsKernel:
    """Tests for Thin Plate Spline kernel function."""

    def test_tps_kernel_2d(self):
        """Test TPS kernel with 2D points."""
        x = torch.randn(10, 2)
        y = torch.randn(15, 2)
        result = utils.tps_kernel(x, y)
        assert result.shape == (15, 10)

    def test_tps_kernel_3d(self):
        """Test TPS kernel with 3D points."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.tps_kernel(x, y)
        assert result.shape == (15, 10)

    def test_tps_kernel_invalid_dimension(self):
        """Test TPS kernel raises error for invalid dimension."""
        x = torch.randn(10, 4)
        y = torch.randn(15, 4)
        with pytest.raises(ValueError):
            utils.tps_kernel(x, y)


class TestInverseMultiquadricKernel:
    """Tests for inverse multiquadric kernel."""

    def test_inverse_multiquadric_shape(self):
        """Test shape of inverse multiquadric kernel."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.inverse_multiquadric_kernel(x, y, c=1.0)
        assert result.shape == (15, 10)

    def test_inverse_multiquadric_positive(self):
        """Test that inverse multiquadric kernel is positive."""
        x = torch.randn(10, 3)
        y = torch.randn(15, 3)
        result = utils.inverse_multiquadric_kernel(x, y, c=1.0)
        assert (result > 0).all()


class TestNormalizePointCloud:
    """Tests for normalize_point_cloud function."""

    def test_normalize_range(self):
        """Test that normalized points are in [-1, 1]."""
        points = torch.randn(100, 3) * 100 + 50
        normalized, _ = utils.normalize_point_cloud(points)
        assert normalized.min() >= -1
        assert normalized.max() <= 1

    def test_normalize_shape_preserved(self):
        """Test that normalization preserves shape."""
        points = torch.randn(50, 3)
        normalized, _ = utils.normalize_point_cloud(points)
        assert normalized.shape == points.shape

    def test_normalize_returns_minmax(self):
        """Test that normalization returns min/max values."""
        points = torch.randn(50, 3)
        normalized, (min_vals, max_vals) = utils.normalize_point_cloud(points)
        assert min_vals is not None
        assert max_vals is not None

    def test_normalize_with_provided_minmax(self):
        """Test normalization with provided min/max values."""
        points = torch.randn(50, 3)
        min_vals = torch.tensor([-10.0, -10.0, -10.0])
        max_vals = torch.tensor([10.0, 10.0, 10.0])
        normalized, _ = utils.normalize_point_cloud(points, max_vals=max_vals, min_vals=min_vals)
        assert normalized.shape == points.shape

    def test_normalize_byaxis(self):
        """Test per-axis normalization."""
        points = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        normalized, _ = utils.normalize_point_cloud(points, byaxis=True)
        # Each axis should span from -1 to 1
        assert torch.allclose(normalized.min(dim=0)[0], torch.tensor([-1.0, -1.0, -1.0]))
        assert torch.allclose(normalized.max(dim=0)[0], torch.tensor([1.0, 1.0, 1.0]))


class TestNormalizeToMostPoints:
    """Tests for normalize_to_pc_w_most_points function."""

    def test_normalize_uses_larger_cloud(self):
        """Test that normalization uses the cloud with more points."""
        pointx = torch.randn(100, 3) * 10  # More points
        pointy = torch.randn(50, 3) * 5
        xi, yi, (minv, maxv) = utils.normalize_to_pc_w_most_points(pointx, pointy)
        # xi should span [-1, 1] since pointx has more points
        assert torch.allclose(xi.min(), torch.tensor(-1.0), atol=0.1)
        assert torch.allclose(xi.max(), torch.tensor(1.0), atol=0.1)

    def test_normalize_shapes_preserved(self):
        """Test that shapes are preserved after normalization."""
        pointx = torch.randn(100, 3)
        pointy = torch.randn(50, 3)
        xi, yi, _ = utils.normalize_to_pc_w_most_points(pointx, pointy)
        assert xi.shape == pointx.shape
        assert yi.shape == pointy.shape


class TestUndoNormalize:
    """Tests for undo_normalize function."""

    def test_undo_normalize_roundtrip(self):
        """Test that undo_normalize reverses normalize_point_cloud."""
        original = torch.randn(50, 3) * 100 + 50
        normalized, (minv, maxv) = utils.normalize_point_cloud(original)
        restored = utils.undo_normalize(normalized, maxv, minv)
        assert torch.allclose(original, restored, atol=1e-4)

    def test_undo_normalize_shape(self):
        """Test that undo_normalize preserves shape."""
        points = torch.randn(50, 3)
        maxv = torch.tensor(10.0)
        minv = torch.tensor(-10.0)
        result = utils.undo_normalize(points, maxv, minv)
        assert result.shape == points.shape


class TestGenerateRandomRotationMatrix:
    """Tests for generate_random_rotation_matrix function."""

    def test_rotation_matrix_shape(self):
        """Test that rotation matrix is 3x3."""
        R = utils.generate_random_rotation_matrix()
        assert R.shape == (3, 3)

    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal (R @ R.T = I)."""
        R = utils.generate_random_rotation_matrix()
        identity = torch.eye(3)
        assert torch.allclose(R @ R.T, identity, atol=1e-5)

    def test_rotation_matrix_determinant(self):
        """Test that rotation matrix has determinant +1."""
        R = utils.generate_random_rotation_matrix()
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_rotation_matrix_with_angles(self):
        """Test rotation matrix with specified angles."""
        angles = torch.tensor([0.0, 0.0, 0.0])
        R = utils.generate_random_rotation_matrix(angles)
        identity = torch.eye(3)
        assert torch.allclose(R, identity, atol=1e-5)

    def test_rotation_matrix_preserves_norms(self):
        """Test that rotation preserves vector norms."""
        R = utils.generate_random_rotation_matrix()
        v = torch.randn(3)
        rotated = R @ v
        assert torch.allclose(v.norm(), rotated.norm(), atol=1e-5)
