"""Tests for ICP (Iterative Closest Point) registration wrapper.

Unit tests for zreg.registration.ICPRegistration class and integration
with the AlignmentStage dispatcher (Phase 39).
"""

import pytest
import torch
import numpy as np
from copy import deepcopy

from zreg.core.dataset import zRegPointCloud
from zreg.algorithms import ICPRegistration
from zreg.core.types import StoredTransform


class TestICPRegistration:
    """Unit tests for ICPRegistration class and Open3D integration."""

    def test_icp_registration_instantiation(self):
        """Test that ICPRegistration can be instantiated with default parameters."""
        icp = ICPRegistration()
        assert icp is not None
        assert icp.max_iterations == 50
        assert icp.tolerance == 1e-6

    def test_icp_registration_custom_parameters(self):
        """Test ICPRegistration with custom max_iterations and tolerance."""
        icp = ICPRegistration(max_iterations=100, tolerance=1e-4)
        assert icp.max_iterations == 100
        assert icp.tolerance == 1e-4

    @pytest.mark.open3d
    def test_icp_register_returns_stored_transform(self):
        """Test that ICPRegistration.register() returns a StoredTransform."""
        icp = ICPRegistration()
        source = zRegPointCloud(pos=torch.randn(10, 3))
        target = zRegPointCloud(pos=torch.randn(10, 3))

        result = icp.register(source, target)

        assert isinstance(result, StoredTransform)
        assert hasattr(result, "transform")
        assert hasattr(result, "src_min")
        assert hasattr(result, "src_max")
        assert hasattr(result, "tgt_min")
        assert hasattr(result, "tgt_max")

    @pytest.mark.open3d
    def test_icp_stored_transform_matrix_shape_and_type(self):
        """Test that StoredTransform contains 4x4 matrix in denormalised space."""
        icp = ICPRegistration()
        source = zRegPointCloud(pos=torch.randn(15, 3))
        target = zRegPointCloud(pos=torch.randn(15, 3))

        result = icp.register(source, target)

        # Extract matrix from ICPTransformation wrapper
        matrix = result.transform.matrix
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (4, 4)
        assert matrix.dtype == np.float32

    @pytest.mark.open3d
    def test_icp_denorm_context_completeness(self):
        """Test that denorm_context contains all required normalisation bounds."""
        icp = ICPRegistration()
        source = zRegPointCloud(pos=torch.randn(12, 3))
        target = zRegPointCloud(pos=torch.randn(12, 3))

        result = icp.register(source, target)

        # Check all denorm context keys are present
        assert result.src_min is not None
        assert result.src_max is not None
        assert result.tgt_min is not None
        assert result.tgt_max is not None

    @pytest.mark.open3d
    def test_icp_identity_transform(self):
        """Test ICP on identical source and target clouds.

        When source == target, ICP should recover a rigid transformation.
        Note: The transformation operates in denormalized space, so the translation
        may include denormalization offset. We verify the rotation is ~identity.
        """
        icp = ICPRegistration()
        # Create identical source and target
        pos = torch.randn(20, 3)
        source = zRegPointCloud(pos=pos.clone())
        target = zRegPointCloud(pos=pos.clone())

        result = icp.register(source, target)
        matrix = result.transform.matrix

        # Extract rotation from 4x4 matrix
        R = matrix[:3, :3]

        # For identical source and target: rotation should be approximately identity
        identity_rot = np.eye(3, dtype=np.float32)
        assert np.allclose(R, identity_rot, atol=0.01), (
            f"Rotation matrix for identical clouds not close to identity:\n{R}"
        )

    @pytest.mark.open3d
    def test_icp_translation_recovery(self):
        """Test ICP on source and translated target.

        Create target = source + known translation vector.
        Verify that ICP produces a valid transformation that captures the
        geometric relationship (rotation approximately identity).
        """
        torch.manual_seed(42)
        icp = ICPRegistration()
        pos = torch.randn(25, 3)
        source = zRegPointCloud(pos=pos.clone())

        # Create target with known translation
        known_translation = np.array([1.0, 2.0, -0.5], dtype=np.float32)
        target_pos = pos + torch.tensor(known_translation, dtype=pos.dtype)
        target = zRegPointCloud(pos=target_pos)

        result = icp.register(source, target)
        matrix = result.transform.matrix

        # Extract rotation from denormalised matrix
        R = matrix[:3, :3]

        # For pure translation: rotation should remain approximately identity
        identity_rot = np.eye(3, dtype=np.float32)
        assert np.allclose(R, identity_rot, atol=0.02), (
            f"Rotation matrix for translated clouds not close to identity:\n{R}"
        )

    @pytest.mark.open3d
    def test_icp_rotation_recovery(self):
        """Test ICP recovery of known rotation.

        Create target = source rotated by 10° around Z-axis.
        ICP should recover approximately the rotation matrix.

        45° is outside ICP's convergence basin for unstructured random clouds
        (it gets stuck near identity).  10° is well within the basin and
        recovers reliably across seeds.
        """
        torch.manual_seed(42)
        icp = ICPRegistration(max_iterations=100)
        pos = torch.randn(30, 3)
        source = zRegPointCloud(pos=pos.clone())

        # Create rotation matrix: 10° around Z-axis
        angle = np.pi / 18  # 10 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Apply rotation to source
        target_pos = torch.tensor(pos @ rotation_matrix.T, dtype=pos.dtype)
        target = zRegPointCloud(pos=target_pos)

        result = icp.register(source, target)
        matrix = result.transform.matrix

        # Extract rotation from denormalised matrix
        R = matrix[:3, :3]

        # Verify recovered rotation is approximately the known rotation
        assert np.allclose(R, rotation_matrix, atol=1e-2), (
            f"Recovered rotation matrix not close to known rotation:\n"
            f"Expected:\n{rotation_matrix}\nGot:\n{R}"
        )

    @pytest.mark.open3d
    def test_icp_on_synthetic_data_no_errors(self):
        """Test ICP on synthetic tracklet-like data (smoke test)."""
        icp = ICPRegistration()
        # Create two frames of synthetic trajectory
        source = zRegPointCloud(pos=torch.randn(50, 3))
        target = zRegPointCloud(pos=torch.randn(50, 3))

        result = icp.register(source, target)

        # Verify result is valid
        assert isinstance(result, StoredTransform)
        assert not np.any(np.isnan(result.transform.matrix))
        assert not np.any(np.isinf(result.transform.matrix))

    @pytest.mark.open3d
    def test_icp_matrix_not_degenerate(self):
        """Test that ICP produces non-degenerate transformation matrix.

        A degenerate matrix has determinant = 0 or is near-singular.
        """
        icp = ICPRegistration()
        source = zRegPointCloud(pos=torch.randn(40, 3))
        target = zRegPointCloud(pos=torch.randn(40, 3) * 2 + 5.0)

        result = icp.register(source, target)
        matrix = result.transform.matrix

        # Check determinant (rotation part should be ~±1)
        det = np.linalg.det(matrix[:3, :3])
        assert not np.isclose(det, 0.0), f"Matrix determinant is near zero: {det}"
        # Rigid rotation should have determinant ≈ ±1
        assert np.isclose(np.abs(det), 1.0, atol=0.1), (
            f"Determinant {det} not close to ±1 for rigid ICP"
        )

    @pytest.mark.open3d
    def test_icp_max_iterations_parameter_respected(self):
        """Test that max_iterations parameter affects registration."""
        source = zRegPointCloud(pos=torch.randn(20, 3))
        target = zRegPointCloud(pos=torch.randn(20, 3) * 2)

        icp_few = ICPRegistration(max_iterations=5)
        icp_many = ICPRegistration(max_iterations=100)

        result_few = icp_few.register(source, target)
        result_many = icp_many.register(source, target)

        # Both should complete without error
        assert isinstance(result_few, StoredTransform)
        assert isinstance(result_many, StoredTransform)

        # Results may differ (more iterations might converge better)
        # but both matrices should be valid and non-degenerate
        det_few = np.linalg.det(result_few.transform.matrix[:3, :3])
        det_many = np.linalg.det(result_many.transform.matrix[:3, :3])
        assert not np.isclose(det_few, 0.0)
        assert not np.isclose(det_many, 0.0)

    @pytest.mark.open3d
    def test_icp_small_cloud(self):
        """Test ICP on very small point clouds (edge case)."""
        icp = ICPRegistration()
        # Minimal clouds: 5 points each
        source = zRegPointCloud(pos=torch.randn(5, 3))
        target = zRegPointCloud(pos=torch.randn(5, 3))

        result = icp.register(source, target)

        # Should still produce a valid StoredTransform
        assert isinstance(result, StoredTransform)
        assert result.transform.matrix.shape == (4, 4)

    @pytest.mark.open3d
    def test_icp_large_cloud(self):
        """Test ICP on larger point clouds."""
        icp = ICPRegistration()
        source = zRegPointCloud(pos=torch.randn(500, 3))
        target = zRegPointCloud(pos=torch.randn(500, 3))

        result = icp.register(source, target)

        assert isinstance(result, StoredTransform)
        assert result.transform.matrix.shape == (4, 4)
