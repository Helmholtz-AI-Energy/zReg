"""Tests for zreg.dtw module."""

import pytest
import torch

from zreg import dtw
from zreg.dataset import zRegPointCloud


@pytest.fixture
def trajectory_pair():
    """Create two simple trajectories for testing."""
    # Create trajectory x with 5 time points
    x = {}
    for i in range(5):
        x[i] = zRegPointCloud(
            pos=torch.randn(20, 3) + i * 0.5,  # Slight drift over time
            color=torch.rand(20, 3),
            id=torch.arange(20),
        )
    
    # Create trajectory y with 5 time points (similar pattern)
    y = {}
    for i in range(5):
        y[i] = zRegPointCloud(
            pos=torch.randn(20, 3) + i * 0.5,
            color=torch.rand(20, 3),
            id=torch.arange(20),
        )
    
    return x, y


@pytest.fixture
def small_trajectory_pair():
    """Create very small trajectories for fast testing."""
    x = {}
    for i in range(3):
        x[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.rand(10, 3),
            id=torch.arange(10),
        )
    
    y = {}
    for i in range(3):
        y[i] = zRegPointCloud(
            pos=torch.randn(10, 3),
            color=torch.rand(10, 3),
            id=torch.arange(10),
        )
    
    return x, y


class TestSanitizeDTWMatrix:
    """Tests for _sanitize_dtw_matrix helper function."""

    def test_single_distance_metric(self, small_trajectory_pair):
        """Test sanitization with single distance metric."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = dtw._sanitize_dtw_matrix(
            distance_kwargs=None,
            distance_metrics="euclidean",
            downsample_method=None,
            x=x,
            y=y,
        )
        
        assert len(distance_fns) == 1
        assert callable(distance_fns[0])

    def test_multiple_distance_metrics(self, small_trajectory_pair):
        """Test sanitization with multiple distance metrics."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = dtw._sanitize_dtw_matrix(
            distance_kwargs=[None, None],
            distance_metrics=["euclidean", "manhattan"],
            downsample_method=None,
            x=x,
            y=y,
        )
        
        assert len(distance_fns) == 2

    def test_swd_requires_downsampling(self, small_trajectory_pair):
        """Test that SWD metrics require downsampling."""
        x, y = small_trajectory_pair
        
        with pytest.raises(RuntimeError):
            dtw._sanitize_dtw_matrix(
                distance_kwargs=None,
                distance_metrics="swd",
                downsample_method=None,  # Should fail
                x=x,
                y=y,
            )

    def test_swd_with_downsampling(self, small_trajectory_pair):
        """Test SWD with downsampling method."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = dtw._sanitize_dtw_matrix(
            distance_kwargs=None,
            distance_metrics="swd",
            downsample_method="random",
            x=x,
            y=y,
        )
        
        assert len(distance_fns) == 1
        assert ds_method == "random"
        assert callable(ds_fn)

    def test_cpd_distance_metric(self, small_trajectory_pair):
        """Test CPD as distance metric returns None function."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = dtw._sanitize_dtw_matrix(
            distance_kwargs=None,
            distance_metrics="cpd",
            downsample_method=None,
            x=x,
            y=y,
        )
        
        assert distance_fns[0] is None

    def test_invalid_distance_metric(self, small_trajectory_pair):
        """Test invalid distance metric raises error."""
        x, y = small_trajectory_pair
        
        with pytest.raises(ValueError):
            dtw._sanitize_dtw_matrix(
                distance_kwargs=None,
                distance_metrics="invalid_metric",
                downsample_method=None,
                x=x,
                y=y,
            )

    def test_downsampling_methods(self, small_trajectory_pair):
        """Test various downsampling methods."""
        x, y = small_trajectory_pair
        
        for method in ["random", "uniform", "farthest"]:
            _, ds_method, ds_fn = dtw._sanitize_dtw_matrix(
                distance_kwargs=None,
                distance_metrics="euclidean",
                downsample_method=method,
                x=x,
                y=y,
            )
            assert ds_method == method
            assert callable(ds_fn)

    def test_invalid_downsampling_method(self, small_trajectory_pair):
        """Test invalid downsampling method raises error."""
        x, y = small_trajectory_pair
        
        with pytest.raises(ValueError):
            dtw._sanitize_dtw_matrix(
                distance_kwargs=None,
                distance_metrics="euclidean",
                downsample_method="invalid_method",
                x=x,
                y=y,
            )


class TestCreateDTWMatrix:
    """Tests for create_dtw_matrix function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic DTW matrix creation."""
        x, y = small_trajectory_pair
        
        dtw_matrix, rots = dtw.create_dtw_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        
        # Shape should be (num_metrics, x_samples+1, y_samples+1)
        assert dtw_matrix.shape == (1, 3, 3)

    def test_matrix_shape_with_window(self, small_trajectory_pair):
        """Test DTW matrix with window constraint."""
        x, y = small_trajectory_pair
        
        dtw_matrix, rots = dtw.create_dtw_matrix(
            x, y,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)
        # Values outside window should be inf
        # Note: window=1 means we compute i-1 to i+1

    def test_multiple_distance_metrics(self, small_trajectory_pair):
        """Test DTW with multiple distance metrics."""
        x, y = small_trajectory_pair
        
        dtw_matrix, rots = dtw.create_dtw_matrix(
            x, y,
            normalize=True,
            distance_metric=["euclidean", "manhattan"],
            distance_kwargs=[None, None],
        )
        
        # Should have 2 metrics
        assert dtw_matrix.shape[0] == 2

    def test_with_downsampling(self, small_trajectory_pair):
        """Test DTW with downsampling."""
        x, y = small_trajectory_pair
        
        dtw_matrix, rots = dtw.create_dtw_matrix(
            x, y,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)

    def test_with_cpd(self, small_trajectory_pair):
        """Test DTW with CPD registration."""
        x, y = small_trajectory_pair
        
        dtw_matrix, rots = dtw.create_dtw_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)
        # Should have collected rotations
        assert len(rots) > 0

    def test_normalization_effect(self, small_trajectory_pair):
        """Test that normalization affects results."""
        x, y = small_trajectory_pair
        
        dtw_normalized, _ = dtw.create_dtw_matrix(
            x, y, normalize=True, distance_metric="euclidean"
        )
        
        dtw_unnormalized, _ = dtw.create_dtw_matrix(
            x, y, normalize=False, distance_metric="euclidean"
        )
        
        # Results should be different
        assert not torch.allclose(dtw_normalized, dtw_unnormalized)

    def test_finite_values(self, small_trajectory_pair):
        """Test that computed values are finite."""
        x, y = small_trajectory_pair
        
        dtw_matrix, _ = dtw.create_dtw_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
        )
        
        # All computed values should be finite (not inf)
        assert torch.isfinite(dtw_matrix).all()


class TestCreateDTWMatrixGivenRigidRot:
    """Tests for create_dtw_matrix_given_rigid_rot function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic DTW matrix creation with given rotation."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        dtw_matrix = dtw.create_dtw_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)

    def test_with_rotation(self, small_trajectory_pair):
        """Test DTW with non-identity rotation."""
        x, y = small_trajectory_pair
        
        # 90 degree rotation around z
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = torch.tensor([1.0, 2.0, 0.0])
        
        dtw_matrix = dtw.create_dtw_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)
        assert torch.isfinite(dtw_matrix).all()

    def test_with_window(self, small_trajectory_pair):
        """Test DTW with window constraint."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        dtw_matrix = dtw.create_dtw_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)

    def test_with_downsampling(self, small_trajectory_pair):
        """Test DTW with downsampling."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        dtw_matrix = dtw.create_dtw_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        
        assert dtw_matrix.shape == (1, 3, 3)


class TestDTWExportedFunctions:
    """Tests for exported functions in __all__."""

    def test_exports(self):
        """Test that expected functions are exported."""
        assert "create_dtw_matrix" in dtw.__all__
        assert "create_dtw_matrix_given_rigid_rot" in dtw.__all__
