from .dataset import open3d_to_zreg, zreg_to_open3d, zRegPointCloud
import open3d as o3d
import logging
from torch_cluster import fps
import torch
from typing import Tuple


log = logging.getLogger(__name__)

__all__ = ["farthest_point_down_sample", "random_down_sample", "uniform_down_sample", "precompute_fps"]


def _preserve_labels(new_points, old_points):
    indices = (
        (new_points.point.positions.numpy()[:, None] == old_points.point.positions.numpy())
        .all(axis=-1)
        .any(axis=0)
        .nonzero()
    )
    new_points.point.labels = old_points.point.labels[indices]
    return new_points


def precompute_fps(pc: dict) -> dict:
    """Precomputes the farthest points from a random start.

    These points will be reused until deleted or if specified in the downsample operation.

    Parameters
    ----------
    pc : dict
        A dictionary containing the point cloud data.

    Returns
    -------
    dict
        The point cloud dictionary with the precomputed farthest points indices added.
    """
    indexes = fps(pc["pos"], ratio=1.0)
    pc["fps-idx"] = indexes
    return pc


def farthest_point_down_sample(
    x: dict,
    y: dict,
    return_o3d: bool = False,
    use_precomputed_indexes: bool = False,
    points: int = None,
) -> Tuple[dict, dict]:
    """Downsamples the point clouds using farthest point sampling.

    This function takes two point clouds and downsamples them to the same number of
    points using farthest point sampling. The downsampling is done using the
    torch_cluster.fps function.

    Parameters
    ----------
    x : dict
        The first point cloud.
    y : dict
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False
    use_precomputed_indexes : bool, optional
        Whether to use precomputed farthest point indices, by default True

    Returns
    -------
    Tuple[dict, dict]
        The downsampled point clouds.
    """
    # o3d is slow, moving to torch cluster for this
    if isinstance(x, o3d.t.geometry.PointCloud):
        x = open3d_to_zreg(x)
    if isinstance(y, o3d.t.geometry.PointCloud):
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
    elif xshape > yshape:  # downsample x to size of y
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
    target["color"] = target["color"][indexes]
    target["id"] = target["id"][indexes]
    target["fps-idx"] = target["fps-idx"][indexes] if target["fps-idx"] is not None else None
    return target


def random_down_sample(x: dict, y: dict, return_o3d: bool = False) -> Tuple[dict, dict]:
    """Downsamples the point clouds by randomly selecting points.

    This function takes two point clouds and downsamples them to the same number of
    points by randomly selecting points.

    Parameters
    ----------
    x : dict
        The first point cloud.
    y : dict
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False

    Returns
    -------
    Tuple[dict, dict]
        The downsampled point clouds.
    """
    if isinstance(x, o3d.t.geometry.PointCloud):
        x = open3d_to_zreg(x)
    if isinstance(y, o3d.t.geometry.PointCloud):
        y = open3d_to_zreg(y)

    if x["pos"].shape[0] == y["pos"].shape[0]:
        if not return_o3d:
            return x, y
        else:
            return open3d_to_zreg(x), open3d_to_zreg(y)

    # expect point clouds and will randomly sample down to the number needed
    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]

    if xshape < yshape:  # downsample y
        target = y
        keep = torch.randperm(yshape, device=target["pos"].device)[:xshape]
    elif xshape > yshape:
        target = x
        keep = torch.randperm(xshape, device=target["pos"].device)[:yshape]
    target["pos"] = target["pos"][keep]
    target["color"] = target["color"][keep]
    target["id"] = target["id"][keep]
    target["fps-idx"] = target["fps-idx"][keep] if target["fps-idx"] is not None else None

    if return_o3d:
        x = zreg_to_open3d(x)
        y = zreg_to_open3d(y)
    return x, y


def uniform_down_sample(x: dict, y: dict, return_o3d: bool = False) -> Tuple[dict, dict]:
    """Downsamples the point clouds uniformly.

    This function takes two point clouds and downsamples them to the same number of
    points by uniformly removing points.

    Parameters
    ----------
    x : dict
        The first point cloud.
    y : dict
        The second point cloud.
    return_o3d : bool, optional
        Whether to return the point clouds as Open3D objects, by default False

    Returns
    -------
    Tuple[dict, dict]
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
            return open3d_to_zreg(x), open3d_to_zreg(y)

    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]
    num_to_remove = abs(xshape - yshape)

    if xshape < yshape:  # downsample y
        target = y
    elif xshape > yshape:  # downsample x
        target = x

    remainder = target["pos"].shape[0] % num_to_remove
    step = target["pos"].shape[0] // num_to_remove
    indices_to_remove = torch.arange(remainder, target["pos"].shape[0], step)

    mask = torch.ones(target["pos"].shape[0], dtype=torch.bool, device=target["pos"].device)
    mask[indices_to_remove] = 0
    target["pos"] = target["pos"][mask]
    target["color"] = target["color"][mask]
    target["id"] = target["id"][mask]

    target["fps-idx"] = target["fps-idx"][mask] if target["fps-idx"] is not None else None

    if return_o3d:
        x = zreg_to_open3d(x)
        y = zreg_to_open3d(y)
    return x, y


# TODO: check out how to make this work in the future
# Open3d source: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.voxel_down_sample
# def voxel_down_sample(pc: Union[o3d.t.geometry.PointCloud, torch.Tensor], num_samples: int, return_o3d: bool = False):
#     if not isinstance(pc, o3d.t.geometry.PointCloud):
#         pc = torch_to_open3d(pc)

#     downsampled = pc.farthest_point_down_sample(num_samples)

#     if not return_o3d:
#         return open3d_to_torch(downsampled)
#     return downsampled
