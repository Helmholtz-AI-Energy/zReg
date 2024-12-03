from .. import utils

__all__ = ["euclidean_distance", "manhatten_distance", "minkowski_distance"]


def euclidean_distance(x, y, normalize: bool = False):
    return minkowski_distance(x, y, p=2, normalize=normalize)


def manhatten_distance(x, y, normalize: bool = False):
    return minkowski_distance(x, y, p=1, normalize=normalize)


def minkowski_distance(x, y, p: float = 2, normalize: bool = False):
    # returns an array of shape x[0] by y[0]
    # the entries of which are the distance between the vectors of x and y
    if normalize:
        x = utils.scale_point_cloud(x)
        y = utils.scale_point_cloud(y)

    if p == 1:
        return (x[None, :, :] - y[:, None, :]).abs().sum(dim=2)

    if p // 2:  # even, no need for abs
        return (x[None, :, :] - y[:, None, :]).pow(p).sum(dim=2).pow(1 / p)
    # else need to have abs and pows
    return (x[None, :, :] - y[:, None, :]).abs().pow(p).sum(dim=2).pow(1 / p)
