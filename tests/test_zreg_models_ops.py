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
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import pytest
import torch

from zreg.models import _ops
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


def test_ball_query_zero_max_neighbors_triggers_isolated_guard():
    """max_neighbors=0 truncates all radius results → isolated-point guard fires → [i] (line 85)."""
    pos = torch.randn(10, 3)
    center_idx = torch.tensor([0])
    groups = ball_query(pos, center_idx, radius=5.0, max_neighbors=0)
    assert len(groups) == 1
    assert groups[0].tolist() == [0]


_requires_open3d = pytest.mark.skipif(not _ops._HAS_OPEN3D, reason="Open3D not available")


@pytest.fixture
def torch_fallback(monkeypatch):
    """Force the pure-torch fallback even where Open3D is installed."""
    monkeypatch.setattr(_ops, "_HAS_OPEN3D", False)


@_requires_open3d
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_fallback_matches_open3d(seed, monkeypatch):
    """The no-Open3D fallback reproduces Open3D's FPS, ball-query and radius-graph output."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(300, 3, generator=gen)

    def run():
        idx = farthest_point_sample(pos, 75)
        return (idx, ball_query(pos, idx, radius=0.15, max_neighbors=16),
                build_radius_graph(pos, 0.12, 8), _ops.knn(pos[idx], pos, 3))

    idx_o3d, groups_o3d, edges_o3d, knn_o3d = run()
    monkeypatch.setattr(_ops, "_HAS_OPEN3D", False)
    idx_t, groups_t, edges_t, knn_t = run()

    assert torch.equal(idx_t, idx_o3d)
    assert [g.tolist() for g in groups_t] == [g.tolist() for g in groups_o3d]
    assert torch.equal(edges_t, edges_o3d)
    assert knn_t[0] == knn_o3d[0]
    assert torch.allclose(torch.tensor(knn_t[1]), torch.tensor(knn_o3d[1]), rtol=1e-12, atol=1e-15)


@pytest.mark.usefixtures("torch_fallback")
def test_torch_fallback_contracts():
    """Fallback keeps the public contracts: unique FPS indices, never-empty groups, no self-edges."""
    pos = torch.randn(200, 3)
    idx = farthest_point_sample(pos, 50)
    assert idx.dtype == torch.long and idx.unique().numel() == 50
    assert idx.tolist() == sorted(idx.tolist())

    far = torch.cat([pos, torch.full((1, 3), 100.0)])
    groups = ball_query(far, torch.tensor([200, 0]), radius=0.3, max_neighbors=4)
    assert groups[0].tolist() == [200]  # isolated-point guard
    assert 1 <= len(groups[1]) <= 4

    edge_index = build_radius_graph(far, radius=0.5, max_neighbors=6)
    src, tgt = edge_index
    assert edge_index.shape[0] == 2
    assert torch.unique(tgt).numel() == far.shape[0]  # every node has an incoming edge
    self_loop_targets = set(tgt[src == tgt].tolist())
    assert 200 in self_loop_targets  # isolated-point guard
    for i in self_loop_targets:  # a self-loop is only ever a node's sole incoming edge
        assert int((tgt == i).sum()) == 1

    idx, d2 = _ops.knn(pos[:20], pos[:5], 3)
    assert [row[0] for row in idx] == [0, 1, 2, 3, 4]  # each query is its own nearest point
    assert all(row == sorted(row) for row in d2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.usefixtures("torch_fallback")
def test_torch_fallback_stays_on_cuda():
    """The fallback computes on pos.device and returns CUDA tensors for CUDA input."""
    pos = torch.randn(150, 3, device="cuda")
    idx = farthest_point_sample(pos, 40)
    assert idx.device == pos.device
    assert all(g.device == pos.device for g in ball_query(pos, idx, radius=0.3, max_neighbors=16))
    assert build_radius_graph(pos, radius=0.3, max_neighbors=16).device == pos.device
