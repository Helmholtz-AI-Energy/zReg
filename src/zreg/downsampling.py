from .dataset import open3d_to_torch, torch_to_open3d
import open3d as o3d
import logging
import torch

log = logging.getLogger(__name__)


def _preserve_labels(new_points, old_points):
    indices = (
        (new_points.point.positions.numpy()[:, None] == old_points.point.positions.numpy())
        .all(axis=-1)
        .any(axis=0)
        .nonzero()
    )
    new_points.point.labels = old_points.point.labels[indices]
    return new_points


def farthest_point_down_sample(x, y, return_o3d: bool = False, preserve_labels: bool = False):
    if not isinstance(x, o3d.t.geometry.PointCloud):
        x = torch_to_open3d(x)
    if not isinstance(y, o3d.t.geometry.PointCloud):
        y = torch_to_open3d(y)

    if preserve_labels:
        log.info("Preseriving labels for points results in poor performance")
    # expect point clouds and will randomly sample down to the number needed
    xshape = x.point.positions.shape[0]
    yshape = y.point.positions.shape[0]
    num_samples = min(xshape, yshape)

    # this downsample drops the labels and colors from the point cloud and need to get them back...
    if xshape < yshape:  # downsample y
        new_points = y.farthest_point_down_sample(num_samples)
        if preserve_labels:
            y = _preserve_labels(new_points=new_points, old_points=y)
        else:
            y = new_points
    elif xshape > yshape:
        new_points = x.farthest_point_down_sample(num_samples)
        if preserve_labels:
            x = _preserve_labels(new_points=new_points, old_points=y)
        else:
            x = new_points

    if not return_o3d:
        x = open3d_to_torch(x)
        y = open3d_to_torch(y)
    return x, y


def random_down_sample(x, y, preserve_labels: bool = False, return_o3d: bool = False):
    if isinstance(x, o3d.t.geometry.PointCloud):
        x = open3d_to_torch(x)
    if isinstance(y, o3d.t.geometry.PointCloud):
        y = open3d_to_torch(y)

    # TODO: rewrite with torch to make this cleaner?

    # expect point clouds and will randomly sample down to the number needed
    xshape = x["pos"].shape[0]
    yshape = y["pos"].shape[0]
    # percent_keep = 1 - (abs(xshape - yshape) / max(xshape, yshape))

    if xshape < yshape:  # downsample y
        target = y
        keep = torch.randperm(yshape, device=target["pos"].device)[:xshape]
    elif xshape > yshape:
        target = x
        keep = torch.randperm(xshape, device=target["pos"].device)[:yshape]
    target["pos"] = target["pos"][keep]
    target["color"] = target["color"][keep]
    target["id"] = target["id"][keep]

    if return_o3d:
        x = torch_to_open3d(x)
        y = torch_to_open3d(y)
    return x, y


def uniform_down_sample(x, y, preserve_labels: bool = False, return_o3d: bool = False):
    # this is simpler in torch than in open3d
    if not isinstance(x, dict):
        x = open3d_to_torch(x)
    if not isinstance(y, dict):
        y = open3d_to_torch(y)

    if x["pos"].shape[0] == y["pos"].shape[0]:
        if not return_o3d:
            return x, y
        else:
            return open3d_to_torch(x), open3d_to_torch(y)

    # expect point clouds and will randomly sample down to the number needed
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

    if return_o3d:
        x = torch_to_open3d(x)
        y = torch_to_open3d(y)
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
