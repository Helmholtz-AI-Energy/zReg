"""Tests for zreg.models._ops — FPS, ball-query, radius-graph geometry ops.

Wave-0 test file (45-DESIGN.md's "Downstream Wave-0 Test Obligations"): covers
correctness of the three Open3D-wrapped ops, the isolated-point ball-query guard
(Pitfall 2, 47-RESEARCH.md), the D-03 sub-50ms benchmark at Phase 46's actual
100-300pt point-count regime, and device-agnostic output placement.
"""

import time

# zreg (and scipy) must be imported before torch/open3d/torch_geometric on macOS
# ARM to avoid duplicate libomp initialisation (SIGABRT) — mirrors
# tests/conftest.py:20-24 and 47-RESEARCH.md's Pitfall 1 import-order convention.
from zreg.dataset import zRegPointCloud  # noqa: F401

import pytest
import torch

from zreg.models._ops import ball_query, build_radius_graph, farthest_point_sample


@pytest.mark.parametrize("n_points", [100, 200, 300, 600])
def test_farthest_point_sample_unique_in_range(n_points):
    """FPS returns n_samples unique, in-range indices for N in 100..600."""
    pos = torch.randn(n_points, 3)
    n_samples = n_points // 2

    idx = farthest_point_sample(pos, n_samples)

    assert idx.shape == (n_samples,)
    assert idx.dtype == torch.long
    assert idx.unique().numel() == n_samples
    assert idx.min() >= 0
    assert idx.max() < n_points


def test_ball_query_isolated_point_self_fallback():
    """A center with zero neighbours within radius falls back to [i] (self)."""
    n_clustered = 49
    cluster = torch.randn(n_clustered, 3) * 0.01
    isolated = torch.tensor([[10.0, 10.0, 10.0]])
    pos = torch.cat([cluster, isolated], dim=0)
    isolated_idx = n_clustered  # last row is the deliberately isolated point

    radius = 0.5  # smaller than the gap between the cluster and the isolated point
    center_idx = torch.tensor([isolated_idx])
    groups = ball_query(pos, center_idx, radius, max_neighbors=16)

    assert len(groups) == 1
    assert groups[0].numel() == 1
    assert groups[0].tolist() == [isolated_idx]


def test_ball_query_never_returns_empty_tensor():
    """ball_query never returns an empty neighbour tensor, even at a tiny radius."""
    pos = torch.randn(100, 3)
    center_idx = torch.arange(100)

    groups = ball_query(pos, center_idx, radius=0.01, max_neighbors=16)

    assert len(groups) == 100
    assert all(g.numel() >= 1 for g in groups)


def test_build_radius_graph_shape_and_valid_indices():
    """build_radius_graph returns a [2, E] edge_index with valid endpoints."""
    pos = torch.randn(200, 3)

    edge_index = build_radius_graph(pos, radius=0.3, max_neighbors=16)

    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long
    assert edge_index.min() >= 0
    assert edge_index.max() < pos.shape[0]


@pytest.mark.parametrize("n_points", [100, 200, 300])
def test_ops_benchmark_under_50ms(n_points):
    """D-03: combined FPS + ball-query wall-clock stays under 50ms at 100-300 pts.

    Generous margin over the ~3ms measured live in 47-RESEARCH.md — a soft perf
    smoke test, not a strict gate.
    """
    pos = torch.randn(n_points, 3)
    n_centers = n_points // 4

    start = time.perf_counter()
    idx = farthest_point_sample(pos, n_centers)
    ball_query(pos, idx, radius=0.3, max_neighbors=16)
    elapsed = time.perf_counter() - start

    print(f"D-03 benchmark: n_points={n_points} elapsed={elapsed * 1000:.2f}ms")
    assert elapsed < 0.050


def test_ops_device_agnostic_cpu():
    """All three ops place their outputs on pos.device (cpu at minimum)."""
    pos = torch.randn(150, 3, device="cpu")

    idx = farthest_point_sample(pos, 40)
    assert idx.device == pos.device

    groups = ball_query(pos, idx, radius=0.3, max_neighbors=16)
    assert all(g.device == pos.device for g in groups)

    edge_index = build_radius_graph(pos, radius=0.3, max_neighbors=16)
    assert edge_index.device == pos.device
