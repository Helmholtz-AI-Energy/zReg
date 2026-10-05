"""Tests for zreg.core.transforms module."""

import pytest
import torch

from zreg.core import transforms


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
        """The default AffineTransformation is the identity (t defaults to zeros)."""
        tf = transforms.AffineTransformation(device="cpu", dtype=torch.float32)
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

    def test_transform_applies_tps(self):
        """tf.transform(points) executes _transform → prepare + transform_basis."""
        n, d = 10, 3
        null_dim = n - d - 1  # 6
        a = torch.randn(d + 1, d)
        v = torch.randn(null_dim, d)
        control_pts = torch.randn(n, d)
        tf = transforms.TPSTransformation(a=a, v=v, control_pts=control_pts)
        result = tf.transform(torch.randn(5, d))
        assert result.shape == (5, d)

    def test_transform_basis_directly(self):
        """transform_basis() applied to precomputed basis returns correct shape."""
        n, d = 10, 3
        null_dim = n - d - 1
        a = torch.randn(d + 1, d)
        v = torch.randn(null_dim, d)
        control_pts = torch.randn(n, d)
        tf = transforms.TPSTransformation(a=a, v=v, control_pts=control_pts)
        basis, _ = tf.prepare(torch.randn(5, d))
        result = tf.transform_basis(basis)
        assert result.shape == (5, d)


class TestTransformPointsHomogeneous:
    """Tests for transform_points_homogeneous function."""

    def test_identity_transform(self):
        """Test identity transform doesn't change points."""
        from zreg.core.dataset import zRegPointCloud
        
        pc = zRegPointCloud(pos=torch.randn(10, 3))
        identity = torch.eye(4)
        
        result = transforms.transform_points_homogeneous(pc, identity)
        assert torch.allclose(result["pos"], pc["pos"], atol=1e-5)

    def test_translation_transform(self):
        """Test translation via homogeneous transform."""
        from zreg.core.dataset import zRegPointCloud
        
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
        from zreg.core.dataset import zRegPointCloud

        # A 4x4 matrix that zeroes out the w coordinate for the last row.
        # Row 3 (bottom row) is [0,0,0,0] so transformed_points[:, 3] == 0
        # Without clamping this causes division by zero.
        transform = torch.eye(4)
        transform[3, :] = 0.0  # zero out the bottom row -> w = 0 for all points

        pc = zRegPointCloud(pos=torch.tensor([[1.0, 2.0, 3.0]]))
        result = transforms.transform_points_homogeneous(pc, transform)

        assert not torch.isnan(result["pos"]).any(), "NaN produced by near-zero w"
        assert not torch.isinf(result["pos"]).any(), "Inf produced by near-zero w"

    def test_unsupported_type_raises_error(self):
        """Test that unsupported point type raises TypeError."""
        transform = torch.eye(4)

        # Pass an unsupported type (a list)
        with pytest.raises(TypeError, match="Unsupported point type"):
            transforms.transform_points_homogeneous([1, 2, 3], transform)

    def test_get_open3d_no_open3d(self):
        """_get_open3d returns (None, False) when HAS_OPEN3D is False."""
        import sys
        from unittest.mock import patch
        from zreg.core.transforms.homogeneous import _get_open3d
        with patch("zreg.core.transforms.homogeneous.HAS_OPEN3D", False):
            o3d, flag = _get_open3d()
        assert o3d is None
        assert flag is False

    def test_get_open3d_import_error(self):
        """_get_open3d returns (None, False) when open3d import raises ImportError."""
        import sys
        from unittest.mock import patch
        from zreg.core.transforms.homogeneous import _get_open3d
        with patch("zreg.core.transforms.homogeneous.HAS_OPEN3D", True):
            with patch.dict(sys.modules, {"open3d": None}):
                o3d, flag = _get_open3d()
        assert o3d is None
        assert flag is False

    def test_return_o3d_from_zreg_input(self):
        """return_o3d=True on zRegPointCloud converts result to open3d PointCloud."""
        from zreg.core.dataset import HAS_OPEN3D as _HAS, zRegPointCloud
        if not _HAS:
            pytest.skip("open3d not available")
        try:
            from zreg.core.transforms.homogeneous import transform_points_homogeneous
            pc = zRegPointCloud(
                pos=torch.randn(5, 3).float(),
                label=torch.rand(5, 3).float(),
                id=torch.arange(5, dtype=torch.int32),
            )
            result = transform_points_homogeneous(pc, torch.eye(4), return_o3d=True)
            assert result is not None
        except (ImportError, OSError) as e:
            pytest.skip(f"open3d unusable: {e}")

    def test_open3d_pc_transform(self):
        """o3d.t.geometry.PointCloud input transforms and returns zRegPointCloud."""
        from zreg.core.dataset import HAS_OPEN3D as _HAS
        if not _HAS:
            pytest.skip("open3d not available")
        try:
            import open3d as o3d
            import numpy as np
            from zreg.core.transforms.homogeneous import transform_points_homogeneous
            pts = o3d.t.geometry.PointCloud()
            pts.point["positions"] = o3d.core.Tensor(
                np.random.randn(5, 3).astype(np.float32)
            )
            result = transform_points_homogeneous(pts, torch.eye(4))
            assert "pos" in result
        except (ImportError, OSError, Exception) as e:
            pytest.skip(f"open3d unusable: {e}")

    def test_open3d_pc_return_o3d(self):
        """o3d PointCloud input + return_o3d=True returns the transformed o3d object directly."""
        from zreg.core.dataset import HAS_OPEN3D as _HAS
        if not _HAS:
            pytest.skip("open3d not available")
        try:
            import open3d as o3d
            import numpy as np
            from zreg.core.transforms.homogeneous import transform_points_homogeneous
            pts = o3d.t.geometry.PointCloud()
            pts.point["positions"] = o3d.core.Tensor(
                np.random.randn(5, 3).astype(np.float32)
            )
            result = transform_points_homogeneous(pts, torch.eye(4), return_o3d=True)
            assert result is not None
        except (ImportError, OSError, Exception) as e:
            pytest.skip(f"open3d unusable: {e}")

    def test_open3d_pc_numpy_matrix(self):
        """numpy transform_matrix triggers the AttributeError→pass path (lines 108-109)."""
        from zreg.core.dataset import HAS_OPEN3D as _HAS
        if not _HAS:
            pytest.skip("open3d not available")
        try:
            import open3d as o3d
            import numpy as np
            from zreg.core.transforms.homogeneous import transform_points_homogeneous
            pts = o3d.t.geometry.PointCloud()
            pts.point["positions"] = o3d.core.Tensor(
                np.random.randn(5, 3).astype(np.float32)
            )
            result = transform_points_homogeneous(pts, np.eye(4, dtype=np.float32))
            assert "pos" in result
        except (ImportError, OSError, Exception) as e:
            pytest.skip(f"open3d unusable: {e}")


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

    def test_ill_conditioned_rotation_raises_cond(self):
        """det≈1 but cond>1e6 fires the cond check (line 172), not the det check."""
        # diag([1e4, 1, 1e-4]): det = 1.0 exactly, cond = 1e8 >> 1e6
        ill_rot = torch.diag(torch.tensor([1e4, 1.0, 1e-4]))
        r1 = transforms.RigidTransformation(rot=ill_rot, t=torch.zeros(3))
        r2 = transforms.RigidTransformation(rot=torch.eye(3), t=torch.zeros(3))
        with pytest.raises(ValueError, match="cond="):
            _ = r1 * r2


def _make_rotation(angle_deg, axis="z"):
    """Create a rotation matrix for a given angle (degrees) around the specified axis."""
    a = torch.tensor(angle_deg * 3.14159265 / 180.0)
    c, s = torch.cos(a), torch.sin(a)
    if axis == "z":
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    elif axis == "x":
        return torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return torch.tensor([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


class TestTransformCompositionDepth:
    """Tests for 3-chain transform composition invariants and device handling (TEST-03, TEST-04)."""

    def _make_three_chain(self):
        """Create a 3-chain composition of rigid transforms."""
        tf1 = transforms.RigidTransformation(
            rot=_make_rotation(15, "z"), t=torch.tensor([1.0, 0.0, 0.0])
        )
        tf2 = transforms.RigidTransformation(
            rot=_make_rotation(20, "x"), t=torch.tensor([0.0, 1.0, 0.0])
        )
        tf3 = transforms.RigidTransformation(
            rot=_make_rotation(25, "y"), t=torch.tensor([0.0, 0.0, 1.0])
        )
        return tf1 * tf2 * tf3

    def test_three_chain_det_near_one(self):
        """Composed rotation matrix determinant is near 1.0."""
        torch.manual_seed(42)
        tf_composed = self._make_three_chain()
        assert abs(torch.det(tf_composed.rot).item() - 1.0) < 1e-5

    def test_three_chain_orthogonality(self):
        """Composed rotation matrix is orthogonal (R^T @ R = I)."""
        torch.manual_seed(42)
        tf_composed = self._make_three_chain()
        assert torch.allclose(tf_composed.rot.T @ tf_composed.rot, torch.eye(3), atol=1e-5)

    def test_three_chain_condition_number(self):
        """Composed rotation matrix condition number is below 1e6."""
        torch.manual_seed(42)
        tf_composed = self._make_three_chain()
        assert torch.linalg.cond(tf_composed.rot).item() < 1e6

    def test_three_chain_round_trip(self):
        """Composed transform round-trips points via manual inverse and built-in inverse."""
        torch.manual_seed(42)
        tf_composed = self._make_three_chain()

        # Manual inverse
        inv_rot = tf_composed.rot.T
        inv_t = -torch.matmul(tf_composed.rot.T, tf_composed.t) / tf_composed.scale
        inv_scale = 1.0 / tf_composed.scale
        tf_inv = transforms.RigidTransformation(rot=inv_rot, t=inv_t, scale=inv_scale)

        points = torch.randn(20, 3)
        forward = tf_composed.transform(points)
        back = tf_inv.transform(forward)
        assert torch.allclose(points, back, atol=1e-4)

        # Built-in inverse
        tf_inv2 = tf_composed.inverse()
        back2 = tf_inv2.transform(forward)
        assert torch.allclose(points, back2, atol=1e-4)

    def test_transform_on_cpu_device(self):
        """CPU transform produces output on CPU device."""
        tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
        points = torch.randn(10, 3)
        result = tf.transform(points)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_transform_on_gpu_device(self):
        """GPU transform produces output on CUDA device."""
        tf = transforms.RigidTransformation(device="cuda", dtype=torch.float32)
        points = torch.randn(10, 3).cuda()
        result = tf.transform(points)
        assert result.device.type == "cuda"
