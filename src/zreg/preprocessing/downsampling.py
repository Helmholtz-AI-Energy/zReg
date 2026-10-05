from ..core.dataset import open3d_to_zreg, zreg_to_open3d, zRegPointCloud, HAS_OPEN3D
import logging
import torch
import copy
from scipy.spatial import cKDTree
import numpy as np

from .. import utils


def _get_open3d():
    if not HAS_OPEN3D:
        return None, False
    try:
        import open3d as o3d
        return o3d, True
    except (ImportError, OSError):
        return None, False


def __getattr__(name: str):
    if name == "o3d":
        o3d_mod, _ = _get_open3d()
        if o3d_mod is None:
            raise AttributeError(f"module {__name__!r} has no attribute 'o3d' (open3d not installed)")
        return o3d_mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Optional torch_cluster support for GPU-accelerated operations
try:
    from torch_cluster import fps as torch_cluster_fps, knn_graph as torch_cluster_knn_graph
    TORCH_CLUSTER_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    TORCH_CLUSTER_AVAILABLE = False  # pragma: no cover
    torch_cluster_fps = None  # pragma: no cover
    torch_cluster_knn_graph = None  # pragma: no cover


log = logging.getLogger(__name__)

__all__ = [
    "farthest_point_down_sample",
    "random_down_sample",
    "uniform_down_sample",
    "precompute_fps",
    "remove_outliers_knn",
    "TORCH_CLUSTER_AVAILABLE",
]


def _fps_numpy(pos: torch.Tensor, ratio: float) -> torch.Tensor:
    """Farthest point sampling using numpy (greedy, deterministic from point 0)."""
    num_samples = max(1, int(pos.shape[0] * ratio))
    device = pos.device
    points = pos.cpu().numpy()

    # Start from a deterministic seed for stable tests and reproducible behavior.
    selected_indices = [0]
    min_distances = np.linalg.norm(points - points[0], axis=1)

    while len(selected_indices) < num_samples:
        farthest_index = int(np.argmax(min_distances))
        if farthest_index in selected_indices:
            break
        selected_indices.append(farthest_index)

        candidate = points[farthest_index]
        distances = np.linalg.norm(points - candidate, axis=1)
        min_distances = np.minimum(min_distances, distances)

    return torch.tensor(selected_indices, dtype=torch.long, device=device)


def fps(pos: torch.Tensor, ratio: float, use_torch_cluster: bool = None) -> torch.Tensor:
    """Farthest point sampling.

    Uses torch_cluster if available and requested (GPU path), otherwise falls
    back to the built-in numpy implementation.

    Parameters
    ----------
    pos : torch.Tensor
        Point positions of shape (N, 3)
    ratio : float
        Ratio of points to sample (0, 1]
    use_torch_cluster : bool, optional
        If True, use torch_cluster (requires installation).
        If None (default), use torch_cluster only if available and pos is on GPU.

    Returns
    -------
    torch.Tensor
        Indices of sampled points
    """
    if use_torch_cluster is None:
        use_torch_cluster = TORCH_CLUSTER_AVAILABLE and pos.is_cuda

    if use_torch_cluster:
        log.debug("Using torch_cluster FPS: %d points", pos.shape[0])
        if not TORCH_CLUSTER_AVAILABLE:
            raise ImportError(
                "torch_cluster is not installed. Install it with:\n"
                "  pip install torch_cluster -f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html\n"
                "Or set use_torch_cluster=False to use the numpy fallback instead."
            )
        return torch_cluster_fps(pos, ratio=ratio)  # pragma: no cover
    else:
        log.debug("Using numpy FPS: %d points", pos.shape[0])
        return _fps_numpy(pos, ratio)


def _knn_scipy(pos: torch.Tensor, k: int) -> torch.Tensor:
    """K-nearest neighbors using scipy's cKDTree.

    Parameters
    ----------
    pos : torch.Tensor
        Point positions of shape (N, 3)
    k : int
        Number of neighbors

    Returns
    -------
    torch.Tensor
        Edge index tensor of shape (2, N*k) where edge_index[0] are source nodes
        and edge_index[1] are target nodes
    """
    device = pos.device
    points = pos.cpu().numpy()

    tree = cKDTree(points)
    # k+1 because query includes self, we exclude it
    distances, indices = tree.query(points, k=k + 1)

    # Exclude self (first column)
    neighbor_indices = indices[:, 1:]  # Shape: (N, k)

    n_points = pos.shape[0]
    # Create edge index: source (neighbors) -> target (center points)
    source = neighbor_indices.flatten()  # neighbor indices
    target = np.repeat(np.arange(n_points), k)  # center point indices

    edge_index = torch.tensor(
        np.stack([source, target], axis=0),
        dtype=torch.long,
        device=device
    )

    return edge_index


def knn_graph(pos: torch.Tensor, k: int, batch: torch.Tensor = None, loop: bool = False,
              use_torch_cluster: bool = None) -> torch.Tensor:
    """Compute k-nearest neighbors graph.

    Uses torch_cluster if available and requested, otherwise falls back to scipy.

    Parameters
    ----------
    pos : torch.Tensor
        Point positions of shape (N, 3)
    k : int
        Number of neighbors
    batch : torch.Tensor, optional
        Batch vector (ignored in scipy implementation, assumes single batch)
    loop : bool, optional
        Whether to include self-loops (ignored in scipy implementation)
    use_torch_cluster : bool, optional
        If True, use torch_cluster (requires installation). If False, use scipy.
        If None (default), use torch_cluster only if available and pos is on GPU.

    Returns
    -------
    torch.Tensor
        Edge index tensor of shape (2, N*k)
    """
    if use_torch_cluster is None:
        # Auto-detect: use torch_cluster if available and on GPU
        use_torch_cluster = TORCH_CLUSTER_AVAILABLE and pos.is_cuda

    if use_torch_cluster:
        log.debug("Using torch_cluster KNN (k=%d): %d points", k, pos.shape[0])
    else:
        log.debug("Using scipy KNN (k=%d, torch_cluster %s, CUDA %s): %d points",
                  k,
                  "available" if TORCH_CLUSTER_AVAILABLE else "unavailable",
                  "yes" if pos.is_cuda else "no",
                  pos.shape[0])

    if use_torch_cluster:
        if not TORCH_CLUSTER_AVAILABLE:
            raise ImportError(
                "torch_cluster is not installed. Install it with:\n"
                "  pip install torch_cluster -f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html\n"
                "Or set use_torch_cluster=False to use scipy instead."
            )
        return torch_cluster_knn_graph(pos, k=k, batch=batch, loop=loop)  # pragma: no cover
    else:
        if batch is not None and not torch.all(batch == batch[0]):
            log.warning("scipy KNN implementation does not support batched point clouds. "
                       "Processing as single batch.")
        return _knn_scipy(pos, k)


def precompute_fps(pc: zRegPointCloud) -> zRegPointCloud:
    """Precomputes the farthest points from a random start.

    These points will be reused until deleted or if specified in the downsample operation.

    Parameters
    ----------
    pc : zRegPointCloud
        A point cloud containing the data.

    Returns
    -------
    zRegPointCloud
        The point cloud with the precomputed farthest points indices added.
    """
    indexes = fps(pc["pos"], ratio=1.0)
    pc["fps-idx"] = indexes
    return pc


PointCloudType = "zRegPointCloud | o3d.t.geometry.PointCloud"


def farthest_point_down_sample(
    x: PointCloudType,
    y: PointCloudType,
    return_o3d: bool = False,
    use_precomputed_indexes: bool = False,
    points: int = None,
) -> tuple[PointCloudType, PointCloudType]:
    """Downsamples the point clouds using farthest point sampling.

    This function takes two point clouds and downsamples them to the same number of
    points using farthest point sampling. The downsampling is done using the
    torch_cluster.fps function.

    Parameters
    ----------
    x : zRegPointCloud or o3d.t.geometry.PointCloud
        The first point cloud.
    y : zRegPointCloud or o3d.t.geometry.PointCloud
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False
    use_precomputed_indexes : bool, optional
        Whether to use precomputed farthest point indices, by default True

    Returns
    -------
    tuple[PointCloudType, PointCloudType]
        The downsampled point clouds.
    """
    o3d, HAS_OPEN3D = _get_open3d()
    # o3d is slow, moving to torch cluster for this
    if HAS_OPEN3D and isinstance(x, o3d.t.geometry.PointCloud):
        x = open3d_to_zreg(x)
    if HAS_OPEN3D and isinstance(y, o3d.t.geometry.PointCloud):
        y = open3d_to_zreg(y)
    # early out
    if x["pos"].shape[0] == y["pos"].shape[0]:
        if not return_o3d:
            return x, y
        else:
            return zreg_to_open3d(x), zreg_to_open3d(y)

    if use_precomputed_indexes:
        if x["fps-idx"] is None:
            precompute_fps(x)
        if y["fps-idx"] is None:
            precompute_fps(y)

    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]
    # perc_keep = min(xshape, yshape) / max(xshape, yshape)

    if points is not None:
        x = _farthest_point_ds_internal(target=x, points=points, use_precomputed_indexes=use_precomputed_indexes)
        y = _farthest_point_ds_internal(target=y, points=points, use_precomputed_indexes=use_precomputed_indexes)

    elif xshape < yshape:  # downsample y to size of x
        y = _farthest_point_ds_internal(target=y, points=xshape, use_precomputed_indexes=use_precomputed_indexes)
    else:  # xshape > yshape — downsample x to size of y
        x = _farthest_point_ds_internal(target=x, points=yshape, use_precomputed_indexes=use_precomputed_indexes)

    if return_o3d:
        x = zreg_to_open3d(x)
        y = zreg_to_open3d(y)
    return x, y


def _farthest_point_ds_internal(target, points, use_precomputed_indexes):
    if not use_precomputed_indexes:
        shape = target["pos"].shape[0]
        perc_keep = points / shape
        if int(perc_keep * shape) < points:
            # if roundoff error
            perc_keep += 1 / (shape)
        if int(perc_keep * shape) > points:
            # overcorrected
            perc_keep -= 0.5 / (shape)
        indexes = fps(target["pos"], ratio=perc_keep)[:points]
    else:
        indexes = target["fps-idx"][:points]
    target["pos"] = target["pos"][indexes]
    target["label"] = target["label"][indexes] if target["label"] is not None else None
    target["id"] = target["id"][indexes] if target["id"] is not None else None
    target["fps-idx"] = target["fps-idx"][indexes] if target["fps-idx"] is not None else None
    return target


def random_down_sample(
    x: PointCloudType,
    y: PointCloudType,
    points: int = -1,
    return_o3d: bool = False,
) -> tuple[PointCloudType, PointCloudType]:
    """Downsamples the point clouds by randomly selecting points.

    This function takes two point clouds and downsamples them to the same number of
    points by randomly selecting points.

    Parameters
    ----------
    x : zRegPointCloud or o3d.t.geometry.PointCloud
        The first point cloud.
    y : zRegPointCloud or o3d.t.geometry.PointCloud
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False

    Returns
    -------
    tuple[PointCloudType, PointCloudType]
        The downsampled point clouds.
    """
    o3d, HAS_OPEN3D = _get_open3d()
    if HAS_OPEN3D and isinstance(x, o3d.t.geometry.PointCloud):
        x = open3d_to_zreg(x)
    if HAS_OPEN3D and isinstance(y, o3d.t.geometry.PointCloud):
        y = open3d_to_zreg(y)

    if x["pos"].shape[0] == y["pos"].shape[0]:
        if not return_o3d:
            return x, y
        else:
            return zreg_to_open3d(x), zreg_to_open3d(y)

    # expect point clouds and will randomly sample down to the number needed
    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]

    if points > 0 and (points > xshape or points > yshape):
        raise RuntimeError(f"points given excited yshape ({yshape}) or xshape ({xshape})")
    elif points > 0:
        for target in [x, y]:
            keep = torch.randperm(target["pos"].shape[0], device=target["pos"].device)[:points]
            target["pos"] = target["pos"][keep]
            target["label"] = target["label"][keep] if target["label"] is not None else None
            target["id"] = target["id"][keep] if target["id"] is not None else None
            target["fps-idx"] = target["fps-idx"][keep] if target["fps-idx"] is not None else None
    else:
        if xshape < yshape:  # downsample y
            target = y
            keep = torch.randperm(yshape, device=target["pos"].device)[:xshape]
        else:  # if xshape > yshape:
            target = x
            keep = torch.randperm(xshape, device=target["pos"].device)[:yshape]
        target["pos"] = target["pos"][keep]
        target["label"] = target["label"][keep] if target["label"] is not None else None
        target["id"] = target["id"][keep] if target["id"] is not None else None
        target["fps-idx"] = target["fps-idx"][keep] if target["fps-idx"] is not None else None

    if return_o3d:
        x = zreg_to_open3d(x)
        y = zreg_to_open3d(y)
    return x, y


def uniform_down_sample(
    x: PointCloudType,
    y: PointCloudType,
    return_o3d: bool = False,
) -> tuple[PointCloudType, PointCloudType]:
    """Downsamples the point clouds uniformly.

    This function takes two point clouds and downsamples them to the same number of
    points by uniformly removing points.

    Parameters
    ----------
    x : zRegPointCloud or o3d.t.geometry.PointCloud
        The first point cloud.
    y : zRegPointCloud or o3d.t.geometry.PointCloud
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False

    Returns
    -------
    tuple[PointCloudType, PointCloudType]
        The downsampled point clouds.
    """
    if not isinstance(x, zRegPointCloud):
        x = open3d_to_zreg(x)
    if not isinstance(y, zRegPointCloud):
        y = open3d_to_zreg(y)

    if x["pos"].shape[0] == y["pos"].shape[0]:
        if not return_o3d:
            return x, y
        else:
            return zreg_to_open3d(x), zreg_to_open3d(y)

    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]
    num_to_remove = abs(xshape - yshape)

    if xshape < yshape:  # downsample y
        target = y
    else:  # xshape > yshape — downsample x
        target = x

    remainder = target["pos"].shape[0] % num_to_remove
    step = target["pos"].shape[0] // num_to_remove
    indices_to_remove = torch.arange(remainder, target["pos"].shape[0], step)

    mask = torch.ones(target["pos"].shape[0], dtype=torch.bool, device=target["pos"].device)
    mask[indices_to_remove] = 0
    target["pos"] = target["pos"][mask]
    target["label"] = target["label"][mask] if target["label"] is not None else None
    target["id"] = target["id"][mask] if target["id"] is not None else None

    target["fps-idx"] = target["fps-idx"][mask] if target["fps-idx"] is not None else None

    if return_o3d:
        x = zreg_to_open3d(x)
        y = zreg_to_open3d(y)
    return x, y


def remove_outliers_knn(pc: zRegPointCloud, k=2, threshold=5.0, inplace: bool = False):
    pos_base = pc["pos"]
    if pos_base.max() > 5:  # arbitrary upper bound to test if the pc is normalized
        normed = True
        pos, norms = utils.normalize_point_cloud(pos_base, byaxis=False)
    else:
        pos = copy.deepcopy(pos_base)
        normed = False
    # use KNN to detemine what to drop
    edge_index = knn_graph(pos, k=k, batch=torch.zeros(pos.shape[0]), loop=False)
    # print(edge_index)
    # Calculate the mean distance to neighbors for each point
    pair, base = edge_index[0], edge_index[1]
    distances = torch.norm(pos[pair] - pos[base], dim=1).view((-1, k))
    # print(edge_index[:, :10], distances.shape)
    mean_distances = distances.mean(dim=1)

    # Calculate the standard deviation of the mean distances
    std_dev = torch.std(mean_distances)
    # print(std_dev)

    # Identify outliers as points with mean distances greater than the threshold
    outliers = mean_distances > threshold * std_dev

    # Ensure we don't remove all points - keep at least the inliers or all points if all are outliers
    if outliers.all():
        # If all points would be removed, keep all of them (no outlier removal)
        outliers = torch.zeros_like(outliers, dtype=torch.bool)

    # Remove outliers from the point cloud
    pos = pos[~outliers]
    if normed:
        pos = utils.undo_normalize(pos, minvals=norms[0], maxvals=norms[1])

    if inplace:
        pc["pos"] = pos
        for key in pc:
            if key == "pos":
                continue
            if pc[key] is not None:
                pc[key] = pc[key][~outliers]
        return pc
    else:
        ret = zRegPointCloud()
        ret["pos"] = pos
        for key in pc:
            if key == "pos":
                continue
            if pc[key] is not None:
                ret[key] = pc[key][~outliers]
            else:
                ret[key] = pc[key]
        return ret
