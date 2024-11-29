from .dataset import open3d_to_torch, torch_to_open3d
import open3d as o3d
import logging

log = logging.getLogger(__name__)


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
        indices = (
            (new_points.point.positions.numpy()[:, None] == y.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = y.point.labels[indices]
        y = new_points
    elif xshape > yshape:
        new_points = x.farthest_point_down_sample(num_samples)
        indices = (
            (new_points.point.positions.numpy()[:, None] == x.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = x.point.labels[indices]
        x = new_points

    if not return_o3d:
        x = open3d_to_torch(x)
        y = open3d_to_torch(y)
    return x, y


def random_down_sample(x, y, return_o3d: bool = False):
    if not isinstance(x, o3d.t.geometry.PointCloud):
        x = torch_to_open3d(x)
    if not isinstance(y, o3d.t.geometry.PointCloud):
        y = torch_to_open3d(y)

    # expect point clouds and will randomly sample down to the number needed
    xshape = x.point.positions.shape[0]
    yshape = y.point.positions.shape[0]
    percent_keep = 1 - (abs(xshape - yshape) / max(xshape, yshape))

    if xshape < yshape:  # downsample y
        new_points = y.random_down_sample(sampling_ratio=percent_keep)
        indices = (
            (new_points.point.positions.numpy()[:, None] == y.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = y.point.labels[indices]
        y = new_points
    elif xshape > yshape:
        new_points = x.random_down_sample(sampling_ratio=percent_keep)
        indices = (
            (new_points.point.positions.numpy()[:, None] == x.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = x.point.labels[indices]
        x = new_points

    if not return_o3d:
        x = open3d_to_torch(x)
        y = open3d_to_torch(y)
    return x, y


def uniform_down_sample(x, y, return_o3d: bool = False):
    if not isinstance(x, o3d.t.geometry.PointCloud):
        x = torch_to_open3d(x)
    if not isinstance(y, o3d.t.geometry.PointCloud):
        y = torch_to_open3d(y)

    # expect point clouds and will randomly sample down to the number needed
    xshape = x.point.positions.shape[0]
    yshape = y.point.positions.shape[0]
    samples_to_remove = abs(xshape - yshape)

    # uniform down sample will remove every kth point
    # this is a

    if xshape < yshape:  # downsample y
        new_points = y.uniform_down_sample(samples_to_remove)
        indices = (
            (new_points.point.positions.numpy()[:, None] == y.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = y.point.labels[indices]
        y = new_points
    elif xshape > yshape:
        new_points = x.uniform_down_sample(samples_to_remove)
        indices = (
            (new_points.point.positions.numpy()[:, None] == x.point.positions.numpy())
            .all(axis=-1)
            .any(axis=0)
            .nonzero()
        )
        new_points.point.labels = x.point.labels[indices]
        x = new_points

    if not return_o3d:
        x = open3d_to_torch(x)
        y = open3d_to_torch(y)
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
