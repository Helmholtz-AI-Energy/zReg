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

Where Open3D cannot be imported (it has no wheel for some platforms, e.g. aarch64 on
JUPITER), each op falls back to an equivalent pure-torch implementation on pos.device
that reproduces Open3D's conventions: FPS starts at index 0 and returns indices in
ascending order, radius neighbours are sorted by distance, and a point exactly at
``radius`` is excluded.
"""

import importlib.util

import torch
import numpy as np

__all__ = ["farthest_point_sample", "ball_query", "build_radius_graph", "knn"]

_HAS_OPEN3D = importlib.util.find_spec("open3d") is not None


def _fps_torch(pos: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Greedy farthest-point sampling from index 0 in float64, indices sorted ascending."""
    p = pos.detach().to(torch.float64)
    selected = torch.empty(n_samples, dtype=torch.long, device=pos.device)
    min_d2 = torch.full((p.shape[0],), float("inf"), dtype=torch.float64, device=pos.device)
    cur = torch.zeros((), dtype=torch.long, device=pos.device)
    for i in range(n_samples):
        selected[i] = cur
        min_d2 = torch.minimum(min_d2, ((p - p[cur]) ** 2).sum(dim=1))
        cur = torch.argmax(min_d2)
    return torch.sort(selected).values


def _radius_neighbours_torch(pos: torch.Tensor, centers: torch.Tensor, radius: float) -> list[list[int]]:
    """For each center index, the indices with squared distance < radius**2, nearest first."""
    p = pos.detach().to(torch.float64)
    d2 = torch.cdist(p[centers], p, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
    order = torch.argsort(d2, dim=1, stable=True)
    inside = torch.gather(d2, 1, order) < radius * radius
    return [row[mask].tolist() for row, mask in zip(order.cpu(), inside.cpu())]


def knn(ref: torch.Tensor, query: torch.Tensor, k: int) -> tuple[list[list[int]], list[list[float]]]:
    """Return the k nearest ``ref`` rows of every ``query`` row, nearest first.

    Parameters
    ----------
    ref : torch.Tensor
        [M, 3] reference positions searched.
    query : torch.Tensor
        [N, 3] query positions.
    k : int
        Number of neighbours (at most M).

    Returns
    -------
    tuple[list[list[int]], list[list[float]]]
        Per query: neighbour indices into ``ref`` and their squared distances
        (float64), both sorted by increasing distance.
    """
    if not _HAS_OPEN3D:
        d2 = torch.cdist(
            query.detach().to(torch.float64), ref.detach().to(torch.float64),
            compute_mode="donot_use_mm_for_euclid_dist",  # exact, no |a|^2+|b|^2-2ab cancellation
        ).pow(2)
        order = torch.argsort(d2, dim=1, stable=True)[:, :k]
        return order.cpu().tolist(), torch.gather(d2, 1, order).cpu().tolist()
    import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ref.detach().cpu().numpy().astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    indices, dists2 = [], []
    for p in query.detach().cpu().numpy().astype(np.float64):
        _, idx, dist2 = tree.search_knn_vector_3d(p, k)
        indices.append(list(idx))
        dists2.append(list(dist2))
    return indices, dists2


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
    if not _HAS_OPEN3D:
        return _fps_torch(pos, n_samples)
    import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
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
    if not _HAS_OPEN3D:
        neighbours = _radius_neighbours_torch(pos, center_idx.to(pos.device), radius)
        return [
            torch.as_tensor(idx[:max_neighbors] or [i], dtype=torch.long, device=pos.device)
            for i, idx in zip(center_idx.tolist(), neighbours)
        ]
    import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
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
    n = pos.shape[0]
    if _HAS_OPEN3D:
        import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos.detach().cpu().numpy().astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd)
        radius_hits = [list(tree.search_radius_vector_3d(pcd.points[i], radius)[1]) for i in range(n)]
    else:
        radius_hits = _radius_neighbours_torch(pos, torch.arange(n, device=pos.device), radius)
    sources: list[int] = []
    targets: list[int] = []
    for i, idx in enumerate(radius_hits):
        neighbours = [j for j in idx if j != i][:max_neighbors]
        if len(neighbours) == 0:
            neighbours = [i]  # isolated-point guard, mirrors ball_query
        for j in neighbours:
            sources.append(j)
            targets.append(i)
    return torch.as_tensor([sources, targets], dtype=torch.long, device=pos.device)
