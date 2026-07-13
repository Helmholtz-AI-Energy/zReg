"""Open3D-wrapped, non-differentiable geometry ops shared by PointNet++ and eGNN.

Provides ``farthest_point_sample``, ``ball_query``, and ``build_radius_graph`` —
the shared indexing layer both point-cloud model architectures consume. Follows
the same extract-positions -> build Open3D structure -> call native op -> convert
back to ``torch.Tensor`` wrapper pattern as ``src/zreg/registration/icp.py``.

All three functions read ``pos.device`` and place their outputs there via
``torch.as_tensor(..., device=pos.device)`` — no ``device`` kwarg, no bare CUDA
device calls, per 47-CONTEXT.md D-01 (device-agnostic code).

Benchmarked live at Phase 46's actual 100-300 point regime (47-RESEARCH.md
Pattern 1): FPS 0.4-0.9ms, ball-query (n/4 centers) 1.5-6.2ms, well under the
D-03 50ms threshold — no vectorization needed at this scale.
"""

import open3d as o3d
import torch
import numpy as np

__all__ = ["farthest_point_sample", "ball_query", "build_radius_graph"]


def farthest_point_sample(pos: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Return indices of n_samples farthest-point-sampled rows of pos (N, 3).

    Verified (47-RESEARCH.md, this session's live run): recovered indices are
    unique and in [0, N) for n=100..600, n_samples up to n//2 -- no duplicate
    -index collisions observed.

    Parameters
    ----------
    pos : torch.Tensor
        [N, 3] point positions.
    n_samples : int
        Number of points to sample.

    Returns
    -------
    torch.Tensor
        [n_samples] torch.long indices into pos, on pos.device.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.detach().cpu().numpy().astype(np.float64))
    sampled = pcd.farthest_point_down_sample(n_samples)
    tree = o3d.geometry.KDTreeFlann(pcd)
    idx = [tree.search_knn_vector_3d(p, 1)[1][0] for p in sampled.points]
    return torch.as_tensor(idx, dtype=torch.long, device=pos.device)


def ball_query(
    pos: torch.Tensor, center_idx: torch.Tensor, radius: float, max_neighbors: int
) -> list[torch.Tensor]:
    """Return, for each center point, indices of neighbours within radius.

    Falls back to [i] (self) when a center has zero neighbours within radius --
    prevents empty-tensor max-pool crashes downstream in a PointNet++
    SetAbstraction layer (47-RESEARCH.md Pitfall 2: sample_bowl's sparser rim
    regions can leave isolated points with no neighbours at small radii).

    Parameters
    ----------
    pos : torch.Tensor
        [N, 3] point positions.
    center_idx : torch.Tensor
        [M] indices of query centers into pos.
    radius : float
        Search radius.
    max_neighbors : int
        Maximum neighbours to keep per center.

    Returns
    -------
    list[torch.Tensor]
        One torch.long tensor per center, on pos.device. Never empty.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.detach().cpu().numpy().astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    groups = []
    for i in center_idx.tolist():
        _, idx, _ = tree.search_radius_vector_3d(pcd.points[i], radius)
        idx = list(idx)[:max_neighbors]
        if len(idx) == 0:
            idx = [i]  # isolated-point guard
        groups.append(torch.as_tensor(idx, dtype=torch.long, device=pos.device))
    return groups


def build_radius_graph(pos: torch.Tensor, radius: float, max_neighbors: int) -> torch.Tensor:
    """Build a directed radius graph over pos as a [2, E] edge_index.

    For each point i, finds neighbours within ``radius`` (up to
    ``max_neighbors``, excluding i itself) and adds directed edges j -> i
    (source_to_target: message flows from neighbour j into center i, matching
    torch_geometric.nn.MessagePassing's default flow convention). Self-loops
    are excluded by design: EGNNConv's own forward() already applies a residual
    node-feature update (``h + node_mlp(...)``), so a self-loop edge would
    double-count the center node's own contribution.

    Falls back to a self-loop (i -> i) for a point with zero neighbours within
    radius, guaranteeing every node has at least one incoming edge -- mirrors
    ball_query's isolated-point guard.

    Parameters
    ----------
    pos : torch.Tensor
        [N, 3] point positions.
    radius : float
        Search radius.
    max_neighbors : int
        Maximum neighbours to keep per center (excluding self).

    Returns
    -------
    torch.Tensor
        [2, E] torch.long edge_index, on pos.device. edge_index[0] are source
        (neighbour) indices, edge_index[1] are target (center) indices.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.detach().cpu().numpy().astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    n = pos.shape[0]
    sources: list[int] = []
    targets: list[int] = []
    for i in range(n):
        _, idx, _ = tree.search_radius_vector_3d(pcd.points[i], radius)
        neighbours = [j for j in idx if j != i][:max_neighbors]
        if len(neighbours) == 0:
            neighbours = [i]  # isolated-point guard, mirrors ball_query
        for j in neighbours:
            sources.append(j)
            targets.append(i)
    return torch.as_tensor([sources, targets], dtype=torch.long, device=pos.device)
