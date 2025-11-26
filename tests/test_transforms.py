"""Tests for zreg.transforms module."""

import pytest
import torch

from zreg import transforms


class TestRigidTransformation:
    """Tests for RigidTransformation class."""

    def test_default_initialization(self):
        """Test default initialization creates identity transform."""
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        assert tf.rot.shape == (3, 3)
        assert tf.t.shape == (3,)
        assert tf.scale == 1.0
        assert torch.allclose(tf.rot, torch.eye(3))
        assert torch.allclose(tf.t, torch.zeros(3))

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        rot = torch.eye(3) * 2
        t = torch.ones(3)
        scale = 0.5
        tf = transforms.RigidTransformation(rot=rot, t=t, scale=scale)
        assert torch.allclose(tf.rot, rot)
        assert torch.allclose(tf.t, t)
        assert tf.scale == scale

    def test_transform_identity(self):
        """Test that identity transform doesn't change points."""
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        points = torch.randn(10, 3)
        transformed = tf.transform(points)
        assert torch.allclose(points, transformed)

    def test_transform_translation(self):
        """Test translation transform."""
        t = torch.tensor([1.0, 2.0, 3.0])
        tf = transforms.RigidTransformation(t=t, device="cpu", dtype=torch.float32)
        points = torch.zeros(5, 3)
        transformed = tf.transform(points)
        expected = t.expand(5, 3)
        assert torch.allclose(transformed, expected)

    def test_transform_scale(self):
        """Test scaling transform."""
        tf = transforms.RigidTransformation(scale=2.0, device="cpu", dtype=torch.float32)
        points = torch.ones(5, 3)
        transformed = tf.transform(points)
        expected = torch.ones(5, 3) * 2.0
        assert torch.allclose(transformed, expected)

    def test_transform_with_extra_dims(self):
        """Test transform with more than 3 columns (e.g., with color)."""
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        tf.t = torch.tensor([1.0, 0.0, 0.0])
        points = torch.zeros(5, 6)  # 3 pos + 3 color
        transformed = tf.transform(points)
        assert transformed.shape == (5, 6)
        # Only first 3 columns should be transformed
        assert transformed[:, 0].sum() == 5.0  # translation applied
        assert transformed[:, 3:].sum() == 0.0  # color unchanged

    def test_inverse(self):
        """Test that inverse transformation reverses the original."""
        rot = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg rotation
        t = torch.tensor([1.0, 2.0, 3.0])
        scale = 2.0
        tf = transforms.RigidTransformation(rot=rot, t=t, scale=scale)
        tf_inv = tf.inverse()
        
        points = torch.randn(10, 3)
        transformed = tf.transform(points)
        restored = tf_inv.transform(transformed)
        assert torch.allclose(points, restored, atol=1e-5)

    def test_composition(self):
        """Test composition of two transformations."""
        tf1 = transforms.RigidTransformation(
            t=torch.tensor([1.0, 0.0, 0.0]), device="cpu", dtype=torch.float32
        )
        tf2 = transforms.RigidTransformation(
            t=torch.tensor([0.0, 1.0, 0.0]), device="cpu", dtype=torch.float32
        )
        tf_composed = tf1 * tf2
        
        points = torch.zeros(1, 3)
        result = tf_composed.transform(points)
        expected = torch.tensor([[1.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_reset(self):
        """Test reset method."""
        tf = transforms.RigidTransformation(
            rot=torch.randn(3, 3),
            t=torch.randn(3),
            scale=2.0,
            device="cpu",
            dtype=torch.float32,
        )
        tf.reset()
        assert torch.allclose(tf.rot, torch.eye(3))
        assert torch.allclose(tf.t, torch.zeros(3))


class TestAffineTransformation:
    """Tests for AffineTransformation class."""

    def test_default_initialization(self):
        """Test default initialization."""
        tf = transforms.AffineTransformation(device="cpu", dtype=torch.float32)
        assert tf.b.shape == (3, 3)
        assert tf.t.shape == (3,)

    def test_transform_identity(self):
        """Test that default transform is essentially identity."""
        tf = transforms.AffineTransformation(device="cpu", dtype=torch.float32)
        tf.t = torch.zeros(3)  # Default is ones, set to zeros for identity
        points = torch.randn(10, 3)
        transformed = tf.transform(points)
        assert torch.allclose(points, transformed)

    def test_transform_shape(self):
        """Test that transform preserves shape."""
        tf = transforms.AffineTransformation(device="cpu", dtype=torch.float32)
        points = torch.randn(15, 3)
        transformed = tf.transform(points)
        assert transformed.shape == points.shape


class TestNonRigidTransformation:
    """Tests for NonRigidTransformation class."""

    def test_initialization(self):
        """Test initialization creates RBF kernel."""
        points = torch.randn(20, 3)
        w = torch.zeros(20, 3)
        tf = transforms.NonRigidTransformation(w=w, points=points, beta=2.0)
        assert tf.g.shape == (20, 20)
        assert tf.w is None or torch.equal(tf.w, w)

    def test_transform_with_zero_weights(self):
        """Test that zero weights give identity transform."""
        points = torch.randn(20, 3)
        w = torch.zeros(20, 3)
        tf = transforms.NonRigidTransformation(w=w, points=points, beta=2.0)
        tf.w = w
        transformed = tf.transform(points)
        assert torch.allclose(points, transformed, atol=1e-5)

    def test_transform_shape(self):
        """Test that transform preserves shape."""
        points = torch.randn(20, 3)
        w = torch.randn(20, 3) * 0.01
        tf = transforms.NonRigidTransformation(w=w, points=points, beta=2.0)
        tf.w = w
        transformed = tf.transform(points)
        assert transformed.shape == points.shape


class TestCombinedTransformation:
    """Tests for CombinedTransformation class."""

    def test_default_initialization(self):
        """Test default initialization."""
        tf = transforms.CombinedTransformation()
        assert tf.rigid_trans is not None
        assert tf.v == 0.0

    def test_transform_shape(self):
        """Test that transform preserves shape."""
        tf = transforms.CombinedTransformation()
        points = torch.randn(15, 3)
        transformed = tf.transform(points)
        assert transformed.shape == points.shape


class TestTPSTransformation:
    """Tests for TPSTransformation class."""

    def test_initialization(self):
        """Test TPS initialization."""
        a = torch.randn(4, 3)
        v = torch.randn(10, 3)
        control_pts = torch.randn(10, 3)
        tf = transforms.TPSTransformation(a=a, v=v, control_pts=control_pts)
        assert tf.a is not None
        assert tf.v is not None
        assert tf.control_pts is not None

    def test_prepare_output_shape(self):
        """Test prepare method output shape."""
        a = torch.randn(4, 3)
        v = torch.randn(10, 3)
        control_pts = torch.randn(10, 3)
        tf = transforms.TPSTransformation(a=a, v=v, control_pts=control_pts)
        
        landmarks = torch.randn(5, 3)
        basis, kernel = tf.prepare(landmarks)
        assert basis.shape[0] == 5  # number of landmarks


class TestTransformPointsHomogeneous:
    """Tests for transform_points_homogeneous function."""

    def test_identity_transform(self):
        """Test identity transform doesn't change points."""
        from zreg.dataset import zRegPointCloud
        
        pc = zRegPointCloud(pos=torch.randn(10, 3))
        identity = torch.eye(4)
        
        result = transforms.transform_points_homogeneous(pc, identity)
        assert torch.allclose(result["pos"], pc["pos"], atol=1e-5)

    def test_translation_transform(self):
        """Test translation via homogeneous transform."""
        from zreg.dataset import zRegPointCloud
        
        pc = zRegPointCloud(pos=torch.zeros(5, 3))
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        
        result = transforms.transform_points_homogeneous(pc, transform)
        expected = torch.tensor([[1.0, 2.0, 3.0]]).expand(5, 3)
        assert torch.allclose(result["pos"], expected, atol=1e-5)

    def test_tensor_input_raises_error(self):
        """Test that raw tensor input raises TypeError."""
        points = torch.randn(10, 3)
        transform = torch.eye(4)
        
        with pytest.raises(TypeError):
            transforms.transform_points_homogeneous(points, transform)
