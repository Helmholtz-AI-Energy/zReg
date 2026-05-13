"""
Conftest.py for zreg tests.

This file contains shared fixtures and configuration for pytest.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest

# open3d must be imported before torch to avoid libomp conflict on macOS ARM.
# zreg.dataset triggers this import internally; doing it here at conftest import
# time ensures the correct library initialisation order for the entire test session.
try:
    import open3d.t.geometry  # noqa: F401
    import open3d.core  # noqa: F401
except (ImportError, OSError):
    pass

import torch

from zreg.dataset import zRegPointCloud


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_points_3d():
    """Create sample 3D points."""
    return torch.randn(50, 3)


@pytest.fixture
def sample_pointcloud():
    """Create a sample zRegPointCloud."""
    return zRegPointCloud(
        pos=torch.randn(100, 3),
        color=torch.randn(100, 3),
        id=torch.arange(100),
    )


@pytest.fixture
def point_cloud_pair():
    """Create a pair of point clouds for registration testing."""
    source = torch.randn(50, 3)
    
    # Create target as transformed version of source
    rotation = torch.tensor([
        [0.866, -0.5, 0.0],
        [0.5, 0.866, 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = torch.tensor([1.0, 2.0, 0.0])
    target = source @ rotation.T + translation
    
    return source, target


@pytest.fixture
def trajectory_data():
    """Create sample trajectory data (dict of time points to point clouds)."""
    trajectory = {}
    for t in range(5):
        trajectory[t] = zRegPointCloud(
            pos=torch.randn(30, 3) + t * 0.5,
            color=torch.rand(30, 3),
            id=torch.arange(30),
        )
    return trajectory


# Configure pytest to show more detailed output
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Skip CUDA tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip CUDA tests when unavailable."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
