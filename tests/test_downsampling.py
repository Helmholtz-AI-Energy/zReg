"""Tests for zreg.preprocessing.downsampling module."""

import pytest
import torch

from zreg.core.dataset import zRegPointCloud
from zreg.preprocessing import downsampling

@pytest.fixture
def sample_pointcloud():
    """Create a sample point cloud for testing."""
    return zRegPointCloud(
        pos=torch.randn(100, 3),
        label=torch.randn(100, 3),
        id=torch.arange(100),
    )


@pytest.fixture
def two_pointclouds():
    """Create two point clouds of different sizes."""
    pc1 = zRegPointCloud(
        pos=torch.randn(100, 3),
        label=torch.randn(100, 3),
        id=torch.arange(100),
    )
    pc2 = zRegPointCloud(
        pos=torch.randn(50, 3),
        label=torch.randn(50, 3),
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
            label=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
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
        for key in ["pos", "label", "id"]:
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
            label=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
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
        
        for key in ["pos", "label", "id"]:
            assert key in x
            assert key in y

    def test_uniform_downsample_equal_clouds(self):
        """Test uniform downsampling with equal-sized clouds."""
        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
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
            label=torch.randn(51, 3),
            id=torch.arange(51),
        )
        
        result = downsampling.remove_outliers_knn(pc, k=2, threshold=3.0)
        
        # Should have fewer points (outlier removed)
        assert result["pos"].shape[0] < 51

    def test_remove_outliers_preserves_structure(self, sample_pointcloud):
        """Test that outlier removal preserves data structure."""
        result = downsampling.remove_outliers_knn(sample_pointcloud, k=2, threshold=5.0)
        
        for key in ["pos", "label", "id"]:
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
        with caplog.at_level(logging.WARNING, logger="zreg.preprocessing.downsampling"):
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


class TestFPSNumpyBreakPath:
    """Test that _fps_numpy's break condition (line 51) is exercised."""

    def test_fps_numpy_break_condition_ratio_one(self):
        """All-zero points cause argmax to always return 0 (already selected), triggering break.

        With pos=torch.zeros(5,3) and ratio=1.0: num_samples=5, selected=[0],
        min_distances=[0,0,0,0,0] so argmax=0 which is already selected -> break.
        Covers line 51 (_fps_numpy break statement).
        """
        pos = torch.zeros(5, 3)
        result = downsampling.fps(pos, ratio=1.0, use_torch_cluster=False)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert len(result) >= 1
        assert result[0].item() == 0


class TestRandomDownsampleXSmallerThanY:
    """Test random_down_sample when x has fewer points than y (line 383)."""

    def test_x_smaller_downsample_y(self):
        """When x has fewer points, y is downsampled to x's size (line 383)."""
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            label=torch.randn(30, 3),
            id=torch.arange(30),
        )
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            label=torch.randn(80, 3),
            id=torch.arange(80),
        )
        x, y = downsampling.random_down_sample(pc_small, pc_large)
        assert x["pos"].shape[0] == 30
        assert y["pos"].shape[0] == 30

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_random_downsample_return_o3d_without_open3d(self):
        """return_o3d=True raises RuntimeError when Open3D is unavailable (lines 397-398)."""
        pc1 = zRegPointCloud(pos=torch.randn(30, 3), label=torch.randn(30, 3), id=torch.arange(30))
        pc2 = zRegPointCloud(pos=torch.randn(80, 3), label=torch.randn(80, 3), id=torch.arange(80))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.random_down_sample(pc1, pc2, return_o3d=True)


class TestUniformDownsampleEdgeCases:
    """Edge cases for uniform_down_sample (lines 427, 429, 435, 442, 459-460)."""

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_uniform_downsample_return_o3d_equal_sizes_without_open3d(self):
        """Equal-sized clouds with return_o3d=True raise RuntimeError without Open3D (line 435)."""
        pc1 = zRegPointCloud(pos=torch.randn(50, 3), label=torch.randn(50, 3), id=torch.arange(50))
        pc2 = zRegPointCloud(pos=torch.randn(50, 3), label=torch.randn(50, 3), id=torch.arange(50))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.uniform_down_sample(pc1, pc2, return_o3d=True)

    @pytest.mark.skipif(downsampling.HAS_OPEN3D, reason="Open3D available — no RuntimeError expected")
    def test_uniform_downsample_return_o3d_unequal_without_open3d(self):
        """Unequal-sized clouds with return_o3d=True raise RuntimeError without Open3D (lines 459-460)."""
        pc1 = zRegPointCloud(pos=torch.randn(30, 3), label=torch.randn(30, 3), id=torch.arange(30))
        pc2 = zRegPointCloud(pos=torch.randn(80, 3), label=torch.randn(80, 3), id=torch.arange(80))
        with pytest.raises(RuntimeError, match="open3d is not available"):
            downsampling.uniform_down_sample(pc1, pc2, return_o3d=True)

    def test_uniform_downsample_x_smaller_than_y(self):
        """When x has fewer points, y is downsampled to match x (line 442: target = y)."""
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            label=torch.randn(30, 3),
            id=torch.arange(30),
        )
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            label=torch.randn(80, 3),
            id=torch.arange(80),
        )
        x, y = downsampling.uniform_down_sample(pc_small, pc_large)
        assert x["pos"].shape[0] == y["pos"].shape[0]


class TestFarthestPointXSmallerThanY:
    """Test farthest_point_down_sample when x has fewer points than y (line 282)."""

    def test_fps_x_smaller_downsamples_y(self):
        """When x is smaller, y is downsampled to x's size (covers line 282)."""
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            label=torch.randn(30, 3),
            id=torch.arange(30),
        )
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            label=torch.randn(80, 3),
            id=torch.arange(80),
        )
        x, y = downsampling.farthest_point_down_sample(pc_small, pc_large)
        assert x["pos"].shape[0] == 30
        assert y["pos"].shape[0] == 30
        assert x["pos"].shape[0] == y["pos"].shape[0]


class TestFPSPrecomputeWhenNone:
    """Test farthest_point_down_sample with use_precomputed_indexes=True but fps-idx=None (lines 258, 260)."""

    def test_precompute_triggered_for_x_and_y_when_fps_idx_is_none(self):
        """Both x and y have fps-idx=None with use_precomputed=True → precompute_fps called (lines 258, 260)."""
        pc1 = zRegPointCloud(
            pos=torch.randn(100, 3),
            label=torch.randn(100, 3),
            id=torch.arange(100),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
            id=torch.arange(50),
        )
        # Confirm fps-idx is None (default)
        assert pc1["fps-idx"] is None
        assert pc2["fps-idx"] is None

        x, y = downsampling.farthest_point_down_sample(
            pc1, pc2, use_precomputed_indexes=True
        )
        assert x["pos"].shape[0] == y["pos"].shape[0]


class TestFPSInternalRoundoffCorrection:
    """Test _farthest_point_ds_internal roundoff corrections (lines 287, 290).

    shape=39, points=31 is the smallest pair where:
      int(31/39 * 39) = 30 < 31  →  line 287 (perc_keep += 1/shape)
      int(32/39 * 39) = 32 > 31  →  line 290 (perc_keep -= 0.5/shape)
    """

    def test_roundoff_correction_applied(self):
        """farthest_point_down_sample with shape=39→31 triggers both roundoff corrections."""
        pc_large = zRegPointCloud(
            pos=torch.randn(80, 3),
            label=torch.randn(80, 3),
            id=torch.arange(80),
        )
        pc_target = zRegPointCloud(
            pos=torch.randn(39, 3),
            label=torch.randn(39, 3),
            id=torch.arange(39),
        )
        # With x=80pts, y=39pts and points=31, _farthest_point_ds_internal is called
        # with shape=39, points=31 — triggers both roundoff guards
        x, y = downsampling.farthest_point_down_sample(
            pc_large, pc_target, points=31, use_precomputed_indexes=False
        )
        assert x["pos"].shape[0] == 31
        assert y["pos"].shape[0] == 31


class TestUniformDownsampleXLargerThanY:
    """Test uniform_down_sample when x has more points than y (lines 423-426)."""

    def test_uniform_x_larger_downsamples_x(self):
        """When x is larger, x is downsampled to y's size (covers lines 423-426)."""
        pc_large = zRegPointCloud(
            pos=torch.randn(100, 3),
            label=torch.randn(100, 3),
            id=torch.arange(100),
        )
        pc_small = zRegPointCloud(
            pos=torch.randn(30, 3),
            label=torch.randn(30, 3),
            id=torch.arange(30),
        )
        x, y = downsampling.uniform_down_sample(pc_large, pc_small)
        assert y["pos"].shape[0] == 30
        assert x["pos"].shape[0] == y["pos"].shape[0]
        assert x["pos"].shape[0] < 100


@pytest.mark.skipif(not downsampling.HAS_OPEN3D, reason="Open3D not available")
class TestDownsamplingWithOpen3D:
    """Tests for downsampling functions with Open3D point clouds."""

    @pytest.fixture
    def open3d_pointclouds(self):
        """Create Open3D point clouds for testing."""
        try:
            from zreg.core.dataset import zreg_to_open3d
            
            pc1 = zRegPointCloud(
                pos=torch.randn(100, 3),
                label=torch.randn(100, 3),
                id=torch.arange(100),
            )
            pc2 = zRegPointCloud(
                pos=torch.randn(50, 3),
                label=torch.randn(50, 3),
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
        from zreg.core.dataset import zreg_to_open3d

        pc1 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
            id=torch.arange(50),
        )
        pc2 = zRegPointCloud(
            pos=torch.randn(50, 3),
            label=torch.randn(50, 3),
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


class TestGetOpen3DDownsampling:
    """Tests for the _get_open3d helper in downsampling (lines 13, 17-18, 25)."""

    def test_no_open3d(self):
        """Returns (None, False) when HAS_OPEN3D is False."""
        from unittest.mock import patch
        from zreg.preprocessing.downsampling import _get_open3d
        with patch("zreg.preprocessing.downsampling.HAS_OPEN3D", False):
            o3d, flag = _get_open3d()
        assert o3d is None and flag is False

    def test_import_error(self):
        """Returns (None, False) when open3d import raises ImportError."""
        import sys
        from unittest.mock import patch
        from zreg.preprocessing.downsampling import _get_open3d
        with patch("zreg.preprocessing.downsampling.HAS_OPEN3D", True):
            with patch.dict(sys.modules, {"open3d": None}):
                o3d, flag = _get_open3d()
        assert o3d is None and flag is False

    def test_getattr_o3d_no_open3d(self):
        """downsampling.o3d raises AttributeError when HAS_OPEN3D is False."""
        import zreg.preprocessing.downsampling
        from unittest.mock import patch
        with patch("zreg.preprocessing.downsampling.HAS_OPEN3D", False):
            with pytest.raises(AttributeError, match="open3d not installed"):
                _ = zreg.preprocessing.downsampling.o3d


def _fresh_pair_id_none(n_x=40, n_y=30):
    """Two fresh clouds of different sizes without ids (the functions mutate inputs)."""
    x = zRegPointCloud(pos=torch.randn(n_x, 3), label=torch.randn(n_x, 3), id=None)
    y = zRegPointCloud(pos=torch.randn(n_y, 3), label=torch.randn(n_y, 3), id=None)
    return x, y


class TestIdNoneDownsampling:
    """id=None must survive downsampling (Phase 64, D-01) instead of raising TypeError."""

    def test_random_id_none_size_match(self):
        x, y = _fresh_pair_id_none()
        x_ds, y_ds = downsampling.random_down_sample(x, y)
        assert x_ds["pos"].shape[0] == 30
        assert y_ds["pos"].shape[0] == 30
        assert x_ds["id"] is None
        assert y_ds["id"] is None
        assert x_ds["label"].shape[0] == 30

    def test_random_id_none_points_branch(self):
        x, y = _fresh_pair_id_none()
        x_ds, y_ds = downsampling.random_down_sample(x, y, points=20)
        assert x_ds["pos"].shape[0] == 20
        assert y_ds["pos"].shape[0] == 20
        assert x_ds["id"] is None
        assert y_ds["id"] is None

    def test_uniform_id_none(self):
        x, y = _fresh_pair_id_none()
        x_ds, y_ds = downsampling.uniform_down_sample(x, y)
        assert x_ds["pos"].shape[0] == y_ds["pos"].shape[0] == 30
        assert x_ds["id"] is None
        assert y_ds["id"] is None

    def test_fps_id_none(self):
        x, y = _fresh_pair_id_none()
        x_ds, y_ds = downsampling.farthest_point_down_sample(x, y)
        assert x_ds["pos"].shape[0] == y_ds["pos"].shape[0] == 30
        assert x_ds["id"] is None
        assert y_ds["id"] is None

    def test_random_id_none_return_o3d(self):
        """return_o3d=True converts id=None clouds without a TypeError (no "labels" attribute)."""
        pytest.importorskip("open3d")
        x, y = _fresh_pair_id_none()
        x_o3d, y_o3d = downsampling.random_down_sample(x, y, return_o3d=True)
        assert x_o3d.point.positions.shape[0] == y_o3d.point.positions.shape[0] == 30
        assert "labels" not in x_o3d.point
        assert "labels" not in y_o3d.point
        assert x_o3d.point.colors.shape[0] == 30

    def test_zreg_to_open3d_id_and_label_none(self):
        """zreg_to_open3d only maps optional fields that are present (torch and numpy input)."""
        pytest.importorskip("open3d")
        from zreg.core.dataset import zreg_to_open3d

        pc = zRegPointCloud(pos=torch.randn(5, 3), label=None, id=None)
        pc["fps-idx"] = None
        out = zreg_to_open3d(pc)
        assert out.point.positions.shape[0] == 5
        assert "colors" not in out.point
        assert "labels" not in out.point

        import open3d as o3d

        pc_np = zRegPointCloud(pos=o3d.core.Tensor(pc["pos"].numpy()), label=None, id=None)
        pc_np["fps-idx"] = None
        out_np = zreg_to_open3d(pc_np)
        assert out_np.point.positions.shape[0] == 5
        assert "labels" not in out_np.point

    @pytest.mark.parametrize("to_torch", [True, False])
    def test_open3d_round_trip_label_and_id_none(self, to_torch):
        """A label=None / id=None cloud survives zreg_to_open3d -> open3d_to_zreg (no KeyError on colors)."""
        pytest.importorskip("open3d")
        from zreg.core.dataset import open3d_to_zreg, zreg_to_open3d

        pos = torch.randn(5, 3)
        pc = zRegPointCloud(pos=pos.clone(), label=None, id=None)
        pc["fps-idx"] = None
        back = open3d_to_zreg(zreg_to_open3d(pc), to_torch=to_torch)
        assert back["label"] is None
        assert back["id"] is None
        assert back["fps-idx"] is None
        back_pos = back["pos"] if to_torch else torch.from_numpy(back["pos"])
        assert torch.equal(back_pos, pos)

        # the point-attribute TensorMap branch tolerates the missing colors too
        back_map = open3d_to_zreg(zreg_to_open3d(pc).point, to_torch=to_torch)
        assert back_map["label"] is None
        assert back_map["id"] is None

    @pytest.mark.parametrize("points", [-1, 20])
    def test_random_ids_follow_positions(self, points):
        """With ids present, returned ids still index the returned positions row-for-row."""
        orig_x = torch.randn(40, 3)
        orig_y = torch.randn(30, 3)
        x = zRegPointCloud(pos=orig_x.clone(), label=None, id=torch.arange(40))
        y = zRegPointCloud(pos=orig_y.clone(), label=None, id=torch.arange(30))
        x_ds, y_ds = downsampling.random_down_sample(x, y, points=points)
        assert torch.equal(x_ds["pos"], orig_x[x_ds["id"]])
        assert torch.equal(y_ds["pos"], orig_y[y_ds["id"]])

    def test_dtw_random_downsample_id_none(self):
        from zreg.algorithms.dtw import DynamicTimeWarping
        from zreg.data_generation import generate_trajectory

        traj_x = generate_trajectory(n_points=30, n_frames=4, seed=0)
        traj_y = generate_trajectory(n_points=30, n_frames=5, seed=1)
        dtw = DynamicTimeWarping(
            x=traj_x,
            y=traj_y,
            distance_metric="swd",
            downsample_method="random",
            cpd_type="rigid",
            window=10,
        )
        result = dtw.compute()
        assert result.distance >= 0
