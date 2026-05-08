"""Tests for zreg.dataset module."""

import pytest
import torch
import tempfile
import os

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
