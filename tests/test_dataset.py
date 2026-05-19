"""Tests for zreg.dataset module."""

import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

from zreg.dataset import zRegPointCloud, load_shah_from_csv, HAS_OPEN3D


class TestZRegPointCloud:
    """Tests for zRegPointCloud class."""

    def test_default_initialization(self):
        """Test default initialization creates empty point cloud."""
        pc = zRegPointCloud()
        assert pc["pos"] is None
        assert pc["color"] is None
        assert pc["id"] is None
        assert pc["fps-idx"] is None

    def test_initialization_with_data(self):
        """Test initialization with position data."""
        pos = torch.randn(10, 3)
        color = torch.randn(10, 3)
        ids = torch.arange(10)
        
        pc = zRegPointCloud(pos=pos, color=color, id=ids)
        
        assert torch.equal(pc["pos"], pos)
        assert torch.equal(pc["color"], color)
        assert torch.equal(pc["id"], ids)

    def test_dict_behavior(self):
        """Test that zRegPointCloud behaves like a dict."""
        pc = zRegPointCloud()
        pc["custom_key"] = torch.tensor([1, 2, 3])
        assert "custom_key" in pc
        assert torch.equal(pc["custom_key"], torch.tensor([1, 2, 3]))

    def test_to_device(self):
        """Test moving point cloud to device."""
        pos = torch.randn(10, 3)
        pc = zRegPointCloud(pos=pos)
        
        # Move to CPU (should work on any machine)
        pc_cpu = pc.to("cpu")
        assert pc_cpu["pos"].device.type == "cpu"

    def test_to_device_with_none_values(self):
        """Test to() handles None values gracefully."""
        pc = zRegPointCloud(pos=torch.randn(10, 3))  # color, id, fps-idx are None
        pc_cpu = pc.to("cpu")
        assert pc_cpu["color"] is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self):
        """Test moving point cloud to CUDA."""
        pos = torch.randn(10, 3)
        pc = zRegPointCloud(pos=pos)
        
        pc_cuda = pc.to("cuda")
        assert pc_cuda["pos"].device.type == "cuda"


class TestLoadShahFromCSV:
    """Tests for load_shah_from_csv function."""

    def test_load_simple_csv(self):
        """Test loading a simple CSV file."""
        # Create a temporary CSV file
        csv_content = """x,y,z,t,layer,id
1.0,2.0,3.0,1,0,1
4.0,5.0,6.0,1,1,2
7.0,8.0,9.0,2,0,3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            pcs = load_shah_from_csv(temp_path, device="cpu")
            
            # Should have 2 time points (t=1 becomes 0, t=2 becomes 1)
            assert 0 in pcs
            assert 1 in pcs
            
            # First time point should have 2 points
            assert pcs[0]["pos"].shape[0] == 2
            
            # Second time point should have 1 point
            assert pcs[1]["pos"].shape[0] == 1
            
            # Check position values
            assert torch.allclose(pcs[0]["pos"][0], torch.tensor([1.0, 2.0, 3.0]))
        finally:
            os.unlink(temp_path)

    def test_load_csv_returns_zregpointcloud(self):
        """Test that loaded data is zRegPointCloud instances."""
        csv_content = """x,y,z,t,layer,id
1.0,2.0,3.0,1,0,1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            pcs = load_shah_from_csv(temp_path, device="cpu")
            assert isinstance(pcs[0], zRegPointCloud)
        finally:
            os.unlink(temp_path)

    def test_load_csv_device(self):
        """Test that data is loaded to correct device."""
        csv_content = """x,y,z,t,layer,id
1.0,2.0,3.0,1,0,1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            pcs = load_shah_from_csv(temp_path, device="cpu")
            assert pcs[0]["pos"].device.type == "cpu"
        finally:
            os.unlink(temp_path)


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
class TestOpen3DConversions:
    """Tests for Open3D conversion functions."""

    @pytest.fixture
    def sample_pointcloud(self):
        """Create a sample point cloud for testing."""
        return zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.randn(10, 3),
            id=torch.arange(10),
        )

    def test_zreg_to_open3d(self, sample_pointcloud):
        """Test conversion from zRegPointCloud to Open3D."""
        try:
            from zreg.dataset import zreg_to_open3d
            import open3d as o3d
            
            o3d_pc = zreg_to_open3d(sample_pointcloud)
            
            # Check it's an Open3D point cloud
            assert isinstance(o3d_pc, o3d.t.geometry.PointCloud)
            
            # Check positions were copied
            assert o3d_pc.point.positions.shape[0] == 10
        except ImportError:
            pytest.skip("Open3D not installed")

    def test_open3d_to_zreg(self, sample_pointcloud):
        """Test conversion from Open3D to zRegPointCloud."""
        try:
            from zreg.dataset import zreg_to_open3d, open3d_to_zreg

            # Convert to Open3D and back
            o3d_pc = zreg_to_open3d(sample_pointcloud)
            restored = open3d_to_zreg(o3d_pc, device="cpu")

            # Check positions match
            assert torch.allclose(
                restored["pos"],
                sample_pointcloud["pos"],
                atol=1e-5
            )
        except ImportError:
            pytest.skip("Open3D not installed")

    def test_open3d_to_zreg_numpy(self, sample_pointcloud):
        """Test open3d_to_zreg with to_torch=False returns numpy arrays."""
        from zreg.dataset import zreg_to_open3d, open3d_to_zreg
        import numpy as np

        o3d_pc = zreg_to_open3d(sample_pointcloud)
        result = open3d_to_zreg(o3d_pc, to_torch=False)

        assert isinstance(result["pos"], np.ndarray)
        assert isinstance(result["color"], np.ndarray)

    def test_zreg_to_open3d_with_fps_idx(self):
        """Test zreg_to_open3d preserves fps-idx when set."""
        from zreg.dataset import zreg_to_open3d
        import open3d as o3d

        pc = zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.randn(10, 3),
            id=torch.arange(10),
        )
        pc["fps-idx"] = torch.arange(5)

        o3d_pc = zreg_to_open3d(pc)
        assert isinstance(o3d_pc, o3d.t.geometry.PointCloud)
        assert "fps_idx" in o3d_pc.point

    def test_open3d_to_zreg_no_labels(self):
        """Test open3d_to_zreg when PointCloud has no labels key."""
        from zreg.dataset import open3d_to_zreg
        import open3d as o3d
        import open3d.core as o3c
        import numpy as np

        o3d_pc = o3d.t.geometry.PointCloud({
            "positions": o3c.Tensor(torch.randn(5, 3).numpy().astype(np.float32)),
            "colors": o3c.Tensor(torch.randn(5, 3).numpy().astype(np.float32)),
        })
        result = open3d_to_zreg(o3d_pc, device="cpu")

        assert result["id"] is None
        assert result["pos"].shape == (5, 3)

    def test_get_open3d_pc_method(self, sample_pointcloud):
        """Test the get_open3d_pc method on zRegPointCloud."""
        try:
            import open3d as o3d
            
            o3d_pc = sample_pointcloud.get_open3d_pc()
            
            assert isinstance(o3d_pc, o3d.t.geometry.PointCloud)
            assert o3d_pc.point.positions.shape[0] == 10
        except ImportError:
            pytest.skip("Open3D not installed")


class TestLoadDataFromTracklets:
    """Tests for load_data_from_tracklets function (requires MATLAB files)."""

    def test_function_exists(self):
        """Test that the function is importable."""
        from zreg.dataset import load_data_from_tracklets
        assert callable(load_data_from_tracklets)

    def test_missing_file_raises_error(self):
        """Test that missing file raises appropriate error."""
        from zreg.dataset import load_data_from_tracklets

        with pytest.raises(FileNotFoundError):
            load_data_from_tracklets("nonexistent_file.mat")


class TestLoadDataFromTrackletsWithMock:
    """Tests for load_data_from_tracklets body via mocked scipy.io.loadmat."""

    def _make_fake_mat(self):
        """Return a minimal fake loadmat result mimicking the tracklet format."""
        return {
            "trackletsPerTimePoint": [None, None],  # len=2 time points
            "tracklets": [
                {
                    "startTime": 1,
                    "endTime": 2,
                    "pos": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    "color": [0.5, 0.6, 0.7],
                    "id": 1,
                },
                {
                    "startTime": 2,
                    "endTime": 2,
                    "pos": [[7.0, 8.0, 9.0]],
                    "color": [0.1, 0.2, 0.3],
                    "id": 2,
                },
            ],
        }

    def test_load_executes_full_body(self):
        """Mocked loadmat executes lines 129-163 of load_data_from_tracklets."""
        from zreg.dataset import load_data_from_tracklets
        import numpy as np

        fake = self._make_fake_mat()
        # Wrap lists as numpy arrays to match real loadmat output
        for t in fake["tracklets"]:
            t["pos"] = np.array(t["pos"])
            t["color"] = np.array(t["color"])

        with patch("zreg.dataset.sio.loadmat", return_value=fake):
            pc, tracklets = load_data_from_tracklets("fake.mat", device="cpu")

        assert 0 in pc
        assert 1 in pc
        assert isinstance(pc[0], zRegPointCloud)
        assert pc[0]["pos"].shape[1] == 3

    def test_load_cuda_not_available_falls_back_to_cpu(self):
        """device='cuda' with CUDA unavailable logs warning and uses cpu (lines 125-127)."""
        from zreg.dataset import load_data_from_tracklets
        import numpy as np

        fake = self._make_fake_mat()
        for t in fake["tracklets"]:
            t["pos"] = np.array(t["pos"])
            t["color"] = np.array(t["color"])

        with patch("zreg.dataset.sio.loadmat", return_value=fake), \
             patch("torch.cuda.is_available", return_value=False):
            pc, _ = load_data_from_tracklets("fake.mat", device="cuda")

        assert pc[0]["pos"].device.type == "cpu"


class TestDatasetNoOpen3D:
    """Tests for RuntimeError paths when Open3D is unavailable (lines 44, 187, 261)."""

    def test_get_open3d_pc_raises_without_open3d(self):
        """get_open3d_pc raises RuntimeError when HAS_OPEN3D is False (line 44)."""
        import zreg.dataset as ds
        pc = zRegPointCloud(pos=torch.randn(5, 3))
        with patch.object(ds, "HAS_OPEN3D", False):
            with pytest.raises(RuntimeError, match="open3d is not available"):
                pc.get_open3d_pc()

    def test_zreg_to_open3d_raises_without_open3d(self):
        """zreg_to_open3d raises RuntimeError when HAS_OPEN3D is False (line 187)."""
        import zreg.dataset as ds
        from zreg.dataset import zreg_to_open3d
        pc = zRegPointCloud(pos=torch.randn(5, 3), color=torch.randn(5, 3), id=torch.arange(5))
        with patch.object(ds, "HAS_OPEN3D", False):
            with pytest.raises(RuntimeError, match="open3d is not available"):
                zreg_to_open3d(pc)

    def test_open3d_to_zreg_raises_without_open3d(self):
        """open3d_to_zreg raises RuntimeError when HAS_OPEN3D is False (line 261)."""
        import zreg.dataset as ds
        from zreg.dataset import open3d_to_zreg
        mock_pc = MagicMock()
        with patch.object(ds, "HAS_OPEN3D", False):
            with pytest.raises(RuntimeError, match="open3d is not available"):
                open3d_to_zreg(mock_pc)


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
class TestZRegToOpen3DWithOpen3DTensors:
    """Test zreg_to_open3d when pc values are already Open3D tensors (lines 205-209)."""

    def test_zreg_to_open3d_from_non_torch_values(self):
        """Values that are o3c.Tensor (not torch.Tensor) take the else branch (lines 205-209)."""
        import open3d.core as o3c
        import numpy as np
        from zreg.dataset import zreg_to_open3d
        import open3d as o3d

        pc = zRegPointCloud()
        pc["pos"] = o3c.Tensor(np.zeros((5, 3), dtype=np.float32))
        pc["color"] = o3c.Tensor(np.zeros((5, 3), dtype=np.float32))
        pc["id"] = o3c.Tensor(np.arange(5, dtype=np.int32))

        result = zreg_to_open3d(pc)
        assert isinstance(result, o3d.t.geometry.PointCloud)
        assert result.point.positions.shape[0] == 5

    def test_zreg_to_open3d_from_non_torch_with_fps_idx(self):
        """Non-torch values with fps-idx set also use the else branch (line 209: fps_idx added)."""
        import open3d.core as o3c
        import numpy as np
        from zreg.dataset import zreg_to_open3d
        import open3d as o3d

        pc = zRegPointCloud()
        pc["pos"] = o3c.Tensor(np.zeros((5, 3), dtype=np.float32))
        pc["color"] = o3c.Tensor(np.zeros((5, 3), dtype=np.float32))
        pc["id"] = o3c.Tensor(np.arange(5, dtype=np.int32))
        pc["fps-idx"] = o3c.Tensor(np.arange(3, dtype=np.int32))

        result = zreg_to_open3d(pc)
        assert isinstance(result, o3d.t.geometry.PointCloud)
        assert "fps_idx" in result.point


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
class TestOpen3DToZRegNonPointCloud:
    """Tests for the non-o3dtgeo.PointCloud else branch in open3d_to_zreg (lines 276-285)."""

    def test_non_pointcloud_open3d_input(self):
        """Passing a mock with .positions/.colors/.labels attributes uses the else branch."""
        from zreg.dataset import open3d_to_zreg
        import numpy as np

        mock_pc = MagicMock()
        mock_pc.positions.cpu().numpy.return_value = np.zeros((5, 3), dtype=np.float32)
        mock_pc.colors.cpu().numpy.return_value = np.zeros((5, 3), dtype=np.float32)
        mock_pc.labels.cpu().numpy.return_value = np.arange(5, dtype=np.int32)
        mock_pc.fps_idx.cpu().numpy.return_value = np.arange(5, dtype=np.int32)

        # Must not be recognized as o3dtgeo.PointCloud — MagicMock is not
        result = open3d_to_zreg(mock_pc, device="cpu")

        assert result["pos"].shape == (5, 3)
        assert result["id"] is not None

    def test_non_pointcloud_missing_labels(self):
        """Non-PointCloud input without labels attribute yields id=None (line 279-280)."""
        from zreg.dataset import open3d_to_zreg
        import numpy as np

        mock_pc = MagicMock()
        mock_pc.positions.cpu().numpy.return_value = np.zeros((5, 3), dtype=np.float32)
        mock_pc.colors.cpu().numpy.return_value = np.zeros((5, 3), dtype=np.float32)
        mock_pc.labels.cpu().numpy.side_effect = KeyError("labels")
        mock_pc.fps_idx.cpu().numpy.side_effect = KeyError("fps_idx")

        result = open3d_to_zreg(mock_pc, device="cpu")
        assert result["id"] is None
        assert result["fps-idx"] is None


class TestImportOpen3D:
    """Tests for _import_open3d in dataset (lines 22-23)."""

    def test_import_open3d_import_error(self):
        """_import_open3d returns (None, None, False) when open3d import raises ImportError."""
        import sys
        from unittest.mock import patch
        from zreg.dataset import _import_open3d
        with patch("zreg.dataset.HAS_OPEN3D", True):
            with patch.dict(sys.modules, {"open3d": None}):
                o3dtgeo, o3c, flag = _import_open3d()
        assert o3dtgeo is None
        assert o3c is None
        assert flag is False
