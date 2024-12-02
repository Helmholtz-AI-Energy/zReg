import torch


def squared_kernel_sum(x: torch.Tensor, y: torch.Tensor) -> float:
    return squared_kernel(x, y).sum() / (x.shape[0] * x.shape[1] * y.shape[0])


def squared_kernel(x, y):
    # this is the l2**2 norm of the rows with respect to all the other rows
    # NOTE: this may need to be transposed. the result will be [y.shape[0], x.shape[0]]
    #       easy to fix, but need to check later
    return (x[None, :, :] - y[:, None, :]).pow(2).sum(dim=2)


def rbf_kernel(x, y, beta: float):
    # i found that this almost requires a re-scaling of the data. if the vals are too large, then it will just return an I matrix
    x = scale_point_cloud(x)
    y = scale_point_cloud(y)
    diff2 = squared_kernel(x, y)
    # return torch.exp(-diff2 / (2.0 * beta))
    return torch.exp(-diff2 / (2.0 * beta))


def tps_kernel_2d(x, y):
    eps = 1e-9
    diff2 = squared_kernel(x, y)
    return torch.where(diff2 > eps, diff2 * torch.log(torch.sqrt(diff2)), 0.0)


def tps_kernel_3d(x, y):
    diff2 = squared_kernel(x, y)
    return -diff2.sqrt()


def tps_kernel(x, y):
    assert x.shape[1] == y.shape[1], "x and y must have same dimensions."
    if x.shape[1] == 2:
        return tps_kernel_2d(x, y)
    elif x.shape[1] == 3:
        return tps_kernel_3d(x, y)
    else:
        raise ValueError("Invalid dimension of x: %d." % x.shape[1])


def inverse_multiquadric_kernel(x, y, c: float):
    diff2 = squared_kernel(x, y)
    return 1.0 / (diff2 + c).sqrt()


def scale_point_cloud(points):
    """
    Scales the points of a point cloud to be between -1 and 1.

    Args:
        points: A numpy array of shape (n, d) representing the point cloud,
                where n is the number of points and d is the dimensionality.

    Returns:
        A numpy array of the same shape as points, with the points scaled
        to be between -1 and 1.
    """

    # Find the minimum and maximum values along each dimension
    min_vals = torch.min(points, dim=0)[0]
    max_vals = torch.max(points, dim=0)[0]

    # Calculate the range of each dimension
    ranges = max_vals - min_vals

    # Scale the points
    scaled_points = 2 * (points - min_vals) / ranges - 1

    return scaled_points
