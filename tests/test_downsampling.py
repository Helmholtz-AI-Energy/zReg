"""Tests for zreg.downsampling module."""

import pytest
import torch

from zreg.dataset import zRegPointCloud
from zreg import downsampling

@pytest.fixture
def sample_pointcloud():
    """Create a sample point cloud for testing."""
    return zRegPointCloud(
        pos=torch.randn(100, 3),
        color=torch.randn(100, 3),
        id=torch.arange(100),
    )


@pytest.fixture
def two_pointclouds():
    """Create two point clouds of different sizes."""
    pc1 = zRegPointCloud(
        pos=torch.randn(100, 3),
        color=torch.randn(100, 3),
        id=torch.arange(100),
    )
    pc2 = zRegPointCloud(
        pos=torch.randn(50, 3),
        color=torch.randn(50, 3),
        id=torch.arange(50),
    )
    return pc1, pc2


class TestPrecomputeFPS:
    """Tests for precompute_fps function."""

    def test_precompute_adds_fps_idx(self, sample_pointcloud):
        """Test that precompute_fps adds fps-idx to point cloud."""
        result = downsampling.precompute_fps(sample_pointcloud)
        assert result["fps-idx"] is not None
        assert len(result["fps-idx"]) == len(result["pos"])

    def test_precompute_returns_same_dict(self, sample_pointcloud):
        """Test that precompute_fps modifies dict in place and returns it."""
        result = downsampling.precompute_fps(sample_pointcloud)
        assert result is sample_pointcloud


class TestFarthestPointDownSample:
    """Tests for farthest_point_down_sample function."""

    def test_downsample_to_same_size(self, two_pointclouds):
        """Test that downsampling produces same-sized clouds."""
        pc1, pc2 = two_pointclouds
        x, y = downsampling.farthest_point_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == y["pos"].shape[0]

    def test_downsample_equal_clouds_unchanged(self):
        """Test that equal-sized clouds are not changed."""
        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        x, y = downsampling.farthest_point_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == 50
        assert y["pos"].shape[0] == 50

    def test_downsample_with_fixed_points(self, two_pointclouds):
        """Test downsampling to a fixed number of points."""
        pc1, pc2 = two_pointclouds
        target_points = 30
        x, y = downsampling.farthest_point_down_sample(
            pc1, pc2, points=target_points, use_precomputed_indexes=False
        )
        assert x["pos"].shape[0] == target_points
        assert y["pos"].shape[0] == target_points

    def test_downsample_preserves_data_structure(self, two_pointclouds):
        """Test that downsampling preserves data structure."""
        pc1, pc2 = two_pointclouds
        x, y = downsampling.farthest_point_down_sample(pc1, pc2)
        
        # Check all keys exist
        for key in ["pos", "color", "id"]:
            assert key in x
            assert key in y

    def test_downsample_with_precomputed_indexes(self, two_pointclouds):
        """Test downsampling with precomputed FPS indexes."""
        pc1, pc2 = two_pointclouds
        # Precompute FPS indexes
        downsampling.precompute_fps(pc1)
        downsampling.precompute_fps(pc2)
        
        x, y = downsampling.farthest_point_down_sample(
            pc1, pc2, use_precomputed_indexes=True
        )
        assert x["pos"].shape[0] == y["pos"].shape[0]


class TestRandomDownSample:
    """Tests for random_down_sample function."""

    def test_random_downsample_to_same_size(self, two_pointclouds):
        """Test that random downsampling produces same-sized clouds."""
        pc1, pc2 = two_pointclouds
        x, y = downsampling.random_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == y["pos"].shape[0]

    def test_random_downsample_smaller_is_unchanged(self, two_pointclouds):
        """Test that the smaller cloud is unchanged."""
        pc1, pc2 = two_pointclouds  # pc1=100, pc2=50
        original_pc2_size = pc2["pos"].shape[0]
        x, y = downsampling.random_down_sample(pc1, pc2)
        assert y["pos"].shape[0] == original_pc2_size
        assert x["pos"].shape[0] == original_pc2_size

    def test_random_downsample_with_fixed_points(self, two_pointclouds):
        """Test random downsampling to fixed number of points."""
        pc1, pc2 = two_pointclouds
        target_points = 30
        x, y = downsampling.random_down_sample(pc1, pc2, points=target_points)
        assert x["pos"].shape[0] == target_points
        assert y["pos"].shape[0] == target_points

    def test_random_downsample_invalid_points_raises(self, two_pointclouds):
        """Test that requesting too many points raises error."""
        pc1, pc2 = two_pointclouds  # pc2 has 50 points
        with pytest.raises(RuntimeError):
            downsampling.random_down_sample(pc1, pc2, points=200)

    def test_random_downsample_equal_clouds(self):
        """Test random downsampling with equal-sized clouds."""
        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        x, y = downsampling.random_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == 50
        assert y["pos"].shape[0] == 50


class TestUniformDownSample:
    """Tests for uniform_down_sample function."""

    def test_uniform_downsample_to_same_size(self, two_pointclouds):
        """Test that uniform downsampling produces same-sized clouds."""
        pc1, pc2 = two_pointclouds
        x, y = downsampling.uniform_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == y["pos"].shape[0]

    def test_uniform_downsample_preserves_structure(self, two_pointclouds):
        """Test that uniform downsampling preserves data structure."""
        pc1, pc2 = two_pointclouds
        x, y = downsampling.uniform_down_sample(pc1, pc2)
        
        for key in ["pos", "color", "id"]:
            assert key in x
            assert key in y

    def test_uniform_downsample_equal_clouds(self):
        """Test uniform downsampling with equal-sized clouds."""
        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        x, y = downsampling.uniform_down_sample(pc1, pc2)
        assert x["pos"].shape[0] == 50
        assert y["pos"].shape[0] == 50


class TestRemoveOutliersKNN:
    """Tests for remove_outliers_knn function."""

    def test_remove_outliers_returns_pointcloud(self, sample_pointcloud):
        """Test that remove_outliers returns a point cloud."""
        result = downsampling.remove_outliers_knn(sample_pointcloud, k=2, threshold=5.0)
        assert isinstance(result, zRegPointCloud)

    def test_remove_outliers_inplace(self, sample_pointcloud):
        """Test inplace outlier removal."""
        original_id = id(sample_pointcloud)
        result = downsampling.remove_outliers_knn(
            sample_pointcloud, k=2, threshold=5.0, inplace=True
        )
        assert id(result) == original_id

    def test_remove_outliers_not_inplace(self, sample_pointcloud):
        """Test non-inplace outlier removal creates new object."""
        original_size = sample_pointcloud["pos"].shape[0]
        result = downsampling.remove_outliers_knn(
            sample_pointcloud, k=2, threshold=5.0, inplace=False
        )
        # Original should be unchanged
        assert sample_pointcloud["pos"].shape[0] == original_size

    def test_remove_outliers_with_outlier(self):
        """Test that actual outliers are removed."""
        # Create points clustered around origin with one outlier
        pos = torch.cat([
            torch.randn(50, 3) * 0.1,  # Clustered points
            torch.tensor([[100.0, 100.0, 100.0]])  # Obvious outlier
        ])
        pc = zRegPointCloud(
            pos=pos,
            color=torch.randn(51, 3),
            id=torch.arange(51),
        )
        
        result = downsampling.remove_outliers_knn(pc, k=2, threshold=3.0)
        
        # Should have fewer points (outlier removed)
        assert result["pos"].shape[0] < 51

    def test_remove_outliers_preserves_structure(self, sample_pointcloud):
        """Test that outlier removal preserves data structure."""
        result = downsampling.remove_outliers_knn(sample_pointcloud, k=2, threshold=5.0)
        
        for key in ["pos", "color", "id"]:
            assert key in result
            if result[key] is not None:
                # All arrays should have same length
                assert result[key].shape[0] == result["pos"].shape[0]


class TestFPSAndKNNExplicitMode:
    """Tests for fps/knn_graph with explicit use_torch_cluster flag (branch coverage)."""

    def test_knn_explicit_false_covers_branch(self):
        """knn_graph with use_torch_cluster=False explicitly skips the None-check branch."""
        pos = torch.randn(20, 3)
        edge_index = downsampling.knn_graph(pos, k=2, use_torch_cluster=False)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 20 * 2

    def test_knn_mixed_batch_logs_warning(self, caplog):
        """knn_graph with a heterogeneous batch vector logs a warning (line 207)."""
        import logging
        pos = torch.randn(20, 3)
        batch = torch.tensor([0] * 10 + [1] * 10)
        with caplog.at_level(logging.WARNING, logger="zreg.downsampling"):
            edge_index = downsampling.knn_graph(pos, k=2, batch=batch, use_torch_cluster=False)
        assert any("single batch" in str(r.message) for r in caplog.records)

    @pytest.mark.skipif(downsampling.TORCH_CLUSTER_AVAILABLE, reason="torch_cluster IS installed")
    def test_fps_use_torch_cluster_true_raises_importerror(self):
        """fps with use_torch_cluster=True raises ImportError when torch_cluster missing."""
        pos = torch.randn(20, 3)
        with pytest.raises(ImportError, match="torch_cluster"):
            downsampling.fps(pos, ratio=0.5, use_torch_cluster=True)

    @pytest.mark.skipif(downsampling.TORCH_CLUSTER_AVAILABLE, reason="torch_cluster IS installed")
    def test_knn_use_torch_cluster_true_raises_importerror(self):
        """knn_graph with use_torch_cluster=True raises ImportError when torch_cluster missing."""
        pos = torch.randn(20, 3)
        with pytest.raises(ImportError, match="torch_cluster"):
            downsampling.knn_graph(pos, k=2, use_torch_cluster=True)


class TestRandomDownsampleXSmallerThanY:
    """Test random_down_sample when x has fewer points than y (line 383)."""

    def test_x_smaller_downsample_y(self):
        """When x has fewer points, y is downsampled to x's size (line 383)."""
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            color=torch.randn(30, 3),
            id=torch.arange(30),
        )
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            color=torch.randn(80, 3),
            id=torch.arange(80),
        )
        x, y = downsampling.random_down_sample(pc_small, pc_large)
        assert x["pos"].shape[0] == 30
        assert y["pos"].shape[0] == 30

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_random_downsample_return_o3d_without_open3d(self):
        """return_o3d=True raises RuntimeError when Open3D is unavailable (lines 397-398)."""
        pc1 = zRegPointCloud(pos=torch.randn(30, 3), color=torch.randn(30, 3), id=torch.arange(30))
        pc2 = zRegPointCloud(pos=torch.randn(80, 3), color=torch.randn(80, 3), id=torch.arange(80))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.random_down_sample(pc1, pc2, return_o3d=True)


class TestUniformDownsampleEdgeCases:
    """Edge cases for uniform_down_sample (lines 427, 429, 435, 442, 459-460)."""

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_uniform_downsample_return_o3d_equal_sizes_without_open3d(self):
        """Equal-sized clouds with return_o3d=True raise RuntimeError without Open3D (line 435)."""
        pc1 = zRegPointCloud(pos=torch.randn(50, 3), color=torch.randn(50, 3), id=torch.arange(50))
        pc2 = zRegPointCloud(pos=torch.randn(50, 3), color=torch.randn(50, 3), id=torch.arange(50))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.uniform_down_sample(pc1, pc2, return_o3d=True)

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_uniform_downsample_return_o3d_unequal_without_open3d(self):
        """Unequal-sized clouds with return_o3d=True raise RuntimeError without Open3D (lines 459-460)."""
        pc1 = zRegPointCloud(pos=torch.randn(30, 3), color=torch.randn(30, 3), id=torch.arange(30))
        pc2 = zRegPointCloud(pos=torch.randn(80, 3), color=torch.randn(80, 3), id=torch.arange(80))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.uniform_down_sample(pc1, pc2, return_o3d=True)

    def test_uniform_downsample_x_smaller_than_y(self):
        """When x has fewer points, y is downsampled to match x (line 442: target = y)."""
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            color=torch.randn(30, 3),
            id=torch.arange(30),
        )
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            color=torch.randn(80, 3),
            id=torch.arange(80),
        )
        x, y = downsampling.uniform_down_sample(pc_small, pc_large)
        assert x["pos"].shape[0] == y["pos"].shape[0]


@pytest.mark.skipif(not downsampling.HAS_OPEN3D, reason="Open3D not available")
class TestDownsamplingWithOpen3D:
    """Tests for downsampling functions with Open3D point clouds."""

    @pytest.fixture
    def open3d_pointclouds(self):
        """Create Open3D point clouds for testing."""
        try:
            from zreg.dataset import zreg_to_open3d
            
            pc1 = zRegPointCloud(
                pos=torch.randn(100, 3),
                color=torch.randn(100, 3),
                id=torch.arange(100),
            )
            pc2 = zRegPointCloud(
                pos=torch.randn(50, 3),
                color=torch.randn(50, 3),
                id=torch.arange(50),
            )
            return zreg_to_open3d(pc1), zreg_to_open3d(pc2)
        except ImportError:
            pytest.skip("Open3D not installed")

    def test_fps_with_open3d(self, open3d_pointclouds):
        """Test FPS downsampling with Open3D inputs."""
        try:
            o3d_pc1, o3d_pc2 = open3d_pointclouds
            x, y = downsampling.farthest_point_down_sample(o3d_pc1, o3d_pc2)
            assert x["pos"].shape[0] == y["pos"].shape[0]
        except ImportError:
            pytest.skip("Open3D not installed")

    def test_random_with_open3d(self, open3d_pointclouds):
        """Test random downsampling with Open3D inputs."""
        try:
            o3d_pc1, o3d_pc2 = open3d_pointclouds
            x, y = downsampling.random_down_sample(o3d_pc1, o3d_pc2)
            assert x["pos"].shape[0] == y["pos"].shape[0]
        except ImportError:
            pytest.skip("Open3D not installed")

    @pytest.fixture
    def open3d_equal_pointclouds(self):
        """Create two equal-sized Open3D point clouds for testing early-return paths."""
        from zreg.dataset import zreg_to_open3d

        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            color=torch.randn(50, 3),
            id=torch.arange(50),
        )
        return zreg_to_open3d(pc1), zreg_to_open3d(pc2)

    def test_fps_return_o3d_unequal(self, open3d_pointclouds):
        """FPS with return_o3d=True on unequal clouds returns Open3D PointClouds (lines 287-288)."""
        o3d_pc1, o3d_pc2 = open3d_pointclouds
        result = downsampling.farthest_point_down_sample(o3d_pc1, o3d_pc2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)
        assert result[0].point.positions.shape[0] == result[1].point.positions.shape[0]

    def test_fps_return_o3d_equal(self, open3d_equal_pointclouds):
        """FPS with return_o3d=True on equal-size clouds hits the early-return path (line 265)."""
        eq1, eq2 = open3d_equal_pointclouds
        result = downsampling.farthest_point_down_sample(eq1, eq2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)

    def test_random_return_o3d_unequal(self, open3d_pointclouds):
        """random_down_sample with return_o3d=True on unequal clouds returns Open3D PointClouds (lines 377-378)."""
        o3d_pc1, o3d_pc2 = open3d_pointclouds
        result = downsampling.random_down_sample(o3d_pc1, o3d_pc2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)
        assert result[0].point.positions.shape[0] == result[1].point.positions.shape[0]

    def test_random_return_o3d_equal(self, open3d_equal_pointclouds):
        """random_down_sample with return_o3d=True on equal-size clouds hits early-return (line 346)."""
        eq1, eq2 = open3d_equal_pointclouds
        result = downsampling.random_down_sample(eq1, eq2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)

    def test_uniform_return_o3d_unequal(self, open3d_pointclouds):
        """uniform_down_sample with return_o3d=True on unequal clouds returns Open3D PointClouds (lines 439-440)."""
        o3d_pc1, o3d_pc2 = open3d_pointclouds
        result = downsampling.uniform_down_sample(o3d_pc1, o3d_pc2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)
        assert result[0].point.positions.shape[0] == result[1].point.positions.shape[0]

    def test_uniform_return_o3d_equal(self, open3d_equal_pointclouds):
        """uniform_down_sample with return_o3d=True on equal-size clouds hits early-return (line 415)."""
        eq1, eq2 = open3d_equal_pointclouds
        result = downsampling.uniform_down_sample(eq1, eq2, return_o3d=True)
        assert isinstance(result[0], downsampling.o3d.t.geometry.PointCloud)
        assert isinstance(result[1], downsampling.o3d.t.geometry.PointCloud)
