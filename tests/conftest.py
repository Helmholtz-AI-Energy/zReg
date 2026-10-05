"""
Conftest.py for zreg tests.

This file contains shared fixtures and configuration for pytest.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import importlib.util
import sys
from pathlib import Path

# Allow `from eval.tracking import ...` in all test files without per-file boilerplate
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_repo_root))

import pytest

# zreg (and scipy) must be imported before torch on macOS ARM to avoid
# duplicate libomp initialisation (SIGABRT). No open3d dependency.
from zreg.core.dataset import zRegPointCloud

import torch


@pytest.fixture
def device():  # pragma: no cover
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_points_3d():  # pragma: no cover
    """Create sample 3D points."""
    return torch.randn(50, 3)


@pytest.fixture
def sample_pointcloud():  # pragma: no cover
    """Create a sample zRegPointCloud."""
    return zRegPointCloud(
        pos=torch.randn(100, 3),
        label=torch.randn(100, 3),
        id=torch.arange(100),
    )


@pytest.fixture
def point_cloud_pair():  # pragma: no cover
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
def trajectory_data():  # pragma: no cover
    """Create sample trajectory data (dict of time points to point clouds)."""
    trajectory = {}
    for t in range(5):
        trajectory[t] = zRegPointCloud(
            pos=torch.randn(30, 3) + t * 0.5,
            label=torch.rand(30, 3),
            id=torch.arange(30),
        )
    return trajectory


# Configure pytest to show more detailed output
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "open3d: needs Open3D (e.g. ICP); skipped where it is not installed"
    )


# Skip CUDA tests if CUDA is not available, and Open3D tests if Open3D is not installed
# (it has no wheel for some platforms, e.g. aarch64 on JUPITER).
def pytest_collection_modifyitems(config, items):  # pragma: no cover
    """Modify test collection to skip CUDA / Open3D tests when unavailable."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
    if importlib.util.find_spec("open3d") is None:
        skip_o3d = pytest.mark.skip(reason="Open3D not available")
        for item in items:
            if item.get_closest_marker("open3d") is not None:
                item.add_marker(skip_o3d)


# Phase 39: ICP Registration fixtures
@pytest.fixture
def eval_config_with_icp(tmp_path):  # pragma: no cover
    """EvalConfig with alignment_method='icp' for ICP testing."""
    from eval.config import EvalConfig
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        alignment_method="icp",
    )


@pytest.fixture
def eval_config_with_cpd(tmp_path):  # pragma: no cover
    """EvalConfig with alignment_method='cpd' (explicit, for comparison)."""
    from eval.config import EvalConfig
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        alignment_method="cpd",
    )
