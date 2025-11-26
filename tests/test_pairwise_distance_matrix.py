"""Tests for zreg.pairwise_distance_matrix module."""

import pytest
import torch

from zreg import pairwise_distance_matrix
from zreg.dataset import zRegPointCloud


@pytest.fixture
def trajectory_pair():
    """Create two simple trajectories for testing."""
    x = {}
    for i in range(5):
        x[i] = zRegPointCloud(
            pos=torch.randn(20, 3) + i * 0.5,
            color=torch.rand(20, 3),
            id=torch.arange(20),
        )
    
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


class TestSanitizePairwiseDistanceMatrix:
    """Tests for _sanitize_pairwise_distance_matrix helper function."""

    def test_single_distance_metric(self, small_trajectory_pair):
        """Test sanitization with single distance metric."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="swd",
                downsample_method=None,
                x=x,
                y=y,
            )

    def test_swd_with_downsampling(self, small_trajectory_pair):
        """Test SWD with downsampling method."""
        x, y = small_trajectory_pair
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
        
        distance_fns, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
            _, ds_method, ds_fn = pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
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
            pairwise_distance_matrix._sanitize_pairwise_distance_matrix(
                distance_kwargs=None,
                distance_metrics="euclidean",
                downsample_method="invalid_method",
                x=x,
                y=y,
            )


class TestCreatePairwiseDistanceMatrix:
    """Tests for create_pairwise_distance_matrix function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic pairwise distance matrix creation."""
        x, y = small_trajectory_pair
        
        distance_matrix, rots = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            downsample_method=None,
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_matrix_shape_with_window(self, small_trajectory_pair):
        """Test pairwise distance matrix with window constraint."""
        x, y = small_trajectory_pair
        
        distance_matrix, rots = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_multiple_distance_metrics(self, small_trajectory_pair):
        """Test pairwise distance matrix with multiple distance metrics."""
        x, y = small_trajectory_pair
        
        distance_matrix, rots = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric=["euclidean", "manhattan"],
            distance_kwargs=[None, None],
        )
        
        assert distance_matrix.shape[0] == 2

    def test_with_downsampling(self, small_trajectory_pair):
        """Test pairwise distance matrix with downsampling."""
        x, y = small_trajectory_pair
        
        distance_matrix, rots = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_with_cpd(self, small_trajectory_pair):
        """Test pairwise distance matrix with CPD registration."""
        x, y = small_trajectory_pair
        
        distance_matrix, rots = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
            cpd_type="rigid",
            downsample_method="random",
        )
        
        assert distance_matrix.shape == (1, 3, 3)
        assert len(rots) > 0

    def test_normalization_effect(self, small_trajectory_pair):
        """Test that normalization affects results."""
        x, y = small_trajectory_pair
        
        matrix_normalized, _ = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y, normalize=True, distance_metric="euclidean"
        )
        
        matrix_unnormalized, _ = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y, normalize=False, distance_metric="euclidean"
        )
        
        assert not torch.allclose(matrix_normalized, matrix_unnormalized)

    def test_finite_values(self, small_trajectory_pair):
        """Test that computed values are finite."""
        x, y = small_trajectory_pair
        
        distance_matrix, _ = pairwise_distance_matrix.create_pairwise_distance_matrix(
            x, y,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert torch.isfinite(distance_matrix).all()


class TestCreatePairwiseDistanceMatrixGivenRigidRot:
    """Tests for create_pairwise_distance_matrix_given_rigid_rot function."""

    def test_basic_creation(self, small_trajectory_pair):
        """Test basic pairwise distance matrix creation with given rotation."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_with_rotation(self, small_trajectory_pair):
        """Test pairwise distance matrix with non-identity rotation."""
        x, y = small_trajectory_pair
        
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = torch.tensor([1.0, 2.0, 0.0])
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            scale=1.0,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)
        assert torch.isfinite(distance_matrix).all()

    def test_with_window(self, small_trajectory_pair):
        """Test pairwise distance matrix with window constraint."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            window=1,
            normalize=True,
            distance_metric="euclidean",
        )
        
        assert distance_matrix.shape == (1, 3, 3)

    def test_with_downsampling(self, small_trajectory_pair):
        """Test pairwise distance matrix with downsampling."""
        x, y = small_trajectory_pair
        
        rotation = torch.eye(3)
        translation = torch.zeros(3)
        
        distance_matrix = pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
            x, y,
            rotation=rotation,
            translation=translation,
            normalize=True,
            distance_metric="swd",
            downsample_method="random",
        )
        
        assert distance_matrix.shape == (1, 3, 3)


class TestPairwiseDistanceMatrixExportedFunctions:
    """Tests for exported functions in __all__."""

    def test_exports(self):
        """Test that expected functions are exported."""
        assert "create_pairwise_distance_matrix" in pairwise_distance_matrix.__all__
        assert "create_pairwise_distance_matrix_given_rigid_rot" in pairwise_distance_matrix.__all__
