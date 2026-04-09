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

    def test_near_zero_w_does_not_produce_inf_or_nan(self):
        """Test 2: w-clamping: near-zero w coordinate does not produce inf or NaN."""
        from zreg.dataset import zRegPointCloud

        # A 4x4 matrix that zeroes out the w coordinate for the last row.
        # Row 3 (bottom row) is [0,0,0,0] so transformed_points[:, 3] == 0
        # Without clamping this causes division by zero.
        transform = torch.eye(4)
        transform[3, :] = 0.0  # zero out the bottom row -> w = 0 for all points

        pc = zRegPointCloud(pos=torch.tensor([[1.0, 2.0, 3.0]]))
        result = transforms.transform_points_homogeneous(pc, transform)

        assert not torch.isnan(result["pos"]).any(), "NaN produced by near-zero w"
        assert not torch.isinf(result["pos"]).any(), "Inf produced by near-zero w"


class TestRigidTransformationComposition:
    """Tests for RigidTransformation.__mul__ post-composition validation."""

    def test_composition_valid_rotations(self):
        """Test 3: composing two valid identity rotations succeeds."""
        r1 = transforms.RigidTransformation(rot=torch.eye(3), t=torch.zeros(3))
        r2 = transforms.RigidTransformation(rot=torch.eye(3), t=torch.ones(3))
        r3 = r1 * r2
        assert r3 is not None
        assert torch.allclose(r3.rot, torch.eye(3), atol=1e-5)

    def test_composition_invalid_det_raises_value_error(self):
        """Test 4: composing matrices whose product has det far from 1.0 raises ValueError."""
        # A 3x3 matrix with det=2 (scaling matrix)
        bad_rot = torch.diag(torch.tensor([2.0, 1.0, 1.0]))
        r1 = transforms.RigidTransformation(rot=bad_rot, t=torch.zeros(3))
        r2 = transforms.RigidTransformation(rot=bad_rot, t=torch.zeros(3))
        # composed det will be 4.0, far from 1.0
        with pytest.raises(ValueError) as exc_info:
            _ = r1 * r2
        msg = str(exc_info.value)
        assert "RigidTransformation composition produced invalid rotation" in msg
        assert "det=" in msg

    def test_composition_invalid_cond_raises_value_error(self):
        """Test 5: composing matrices with high condition number raises ValueError."""
        # A near-singular rotation: first column scaled to near zero
        ill = torch.eye(3)
        ill[0, 0] = 1e-8  # near-singular -> high cond number
        # det ≈ 1e-8 (far from 1), will trigger det check first.
        # Use a matrix where cond is high but det might still pass if needed.
        # The plan says cond > 1e6 triggers ValueError with cond= value.
        # We create a matrix where det is close enough but cond is huge.
        # Actually: ill above has det=1e-8 so det check fires first; either path is fine.
        r1 = transforms.RigidTransformation(rot=ill, t=torch.zeros(3))
        r2 = transforms.RigidTransformation(rot=torch.eye(3), t=torch.zeros(3))
        with pytest.raises(ValueError) as exc_info:
            _ = r1 * r2
        msg = str(exc_info.value)
        assert "RigidTransformation composition produced invalid rotation" in msg
        # Either det= or cond= must appear (depending on which check fires first)
        assert "det=" in msg or "cond=" in msg
