import torch
from typing import Tuple

__all__ = [
    "squared_kernel",
    "squared_kernel_sum",
    "rbf_kernel",
    "tps_kernel",
    "inverse_multiquadric_kernel",
    "normalize_point_cloud",
    "normalize_to_larger_pc",
]


def squared_kernel_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of the squared kernel between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, d).
    y : torch.Tensor
        Second tensor with shape (m, d).

    Returns
    -------
    torch.Tensor
        The sum of the squared kernel divided by (x.shape[0] * x.shape[1] * y.shape[0])
    """
    return squared_kernel(x, y).sum() / (x.shape[0] * x.shape[1] * y.shape[0])


def squared_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared kernel between two tensors.

    This function calculates the squared L2 norm between all pairs of rows in x and y.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, d).
    y : torch.Tensor
        Second tensor with shape (m, d).

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the squared kernel.
        Note that this is M by N not N by M!
    """
    return (x[None, :, :] - y[:, None, :]).pow(2).sum(dim=2)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Computes the Radial Basis Function (RBF) kernel between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, d).
    y : torch.Tensor
        Second tensor with shape (m, d).
    beta : float
        Bandwidth parameter for the RBF kernel.

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the RBF kernel.
    """
    # Scale the point clouds to prevent numerical instability
    x, _ = normalize_point_cloud(x)
    y, _ = normalize_point_cloud(y)
    diff2 = squared_kernel(x, y)
    return torch.exp(-diff2 / (2.0 * beta))


def _tps_kernel_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Thin Plate Spline (TPS) kernel in 2D between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, 2).
    y : torch.Tensor
        Second tensor with shape (m, 2).

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the TPS kernel.
    """
    eps = 1e-9
    diff2 = squared_kernel(x, y)
    return torch.where(diff2 > eps, diff2 * torch.log(torch.sqrt(diff2)), 0.0)


def _tps_kernel_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Thin Plate Spline (TPS) kernel in 3D between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, 3).
    y : torch.Tensor
        Second tensor with shape (m, 3).

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the TPS kernel.
    """
    diff2 = squared_kernel(x, y)
    return -diff2.sqrt()


def tps_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Thin Plate Spline (TPS) kernel between two tensors.

    The function automatically selects the 2D or 3D version of the kernel based on the
    dimensionality of the input tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, d).
    y : torch.Tensor
        Second tensor with shape (m, d).

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the TPS kernel.

    Raises
    ------
    ValueError
        If the dimensionality of x is not 2 or 3.
    """
    assert x.shape[1] == y.shape[1], "x and y must have same dimensions."
    if x.shape[1] == 2:
        return _tps_kernel_2d(x, y)
    elif x.shape[1] == 3:
        return _tps_kernel_3d(x, y)
    else:
        raise ValueError("Invalid dimension of x: %d." % x.shape[1])


def inverse_multiquadric_kernel(x, y, c: float):
    diff2 = squared_kernel(x, y)
    return 1.0 / (diff2 + c).sqrt()


def normalize_point_cloud(
    points: torch.Tensor, max_vals: torch.Tensor = None, min_vals: torch.Tensor = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Scales the points of a point cloud to be between -1 and 1.

    Parameters
    ----------
    points : torch.Tensor
        A tensor of shape (n, d) representing the point cloud,
        where n is the number of points and d is the dimensionality.
    max_vals : torch.Tensor, optional
        A tensor of shape (d,) representing the maximum values along each dimension.
        If None, the maximum values are computed from the input points.
    min_vals : torch.Tensor, optional
        A tensor of shape (d,) representing the minimum values along each dimension.
        If None, the minimum values are computed from the input points.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape as points, with the points scaled
        to be between -1 and 1.
    """

    # Find the minimum and maximum values along each dimension if not given
    if max_vals is None:
        max_vals = torch.max(points, dim=0)[0]
    if min_vals is None:
        min_vals = torch.min(points, dim=0)[0]

    # Calculate the range of each dimension
    ranges = max_vals - min_vals

    # Scale the points
    scaled_points = 2 * (points - min_vals) / ranges - 1

    return scaled_points, (min_vals, max_vals)


def normalize_to_larger_pc(pointx, pointy) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # TODO: fix normalize to use the points dicts not just the torch dicts
    if pointx.shape[0] > pointy.shape[0]:
        xi, (minv, maxv) = normalize_point_cloud(pointx)
        yi, _ = normalize_point_cloud(pointy, min_vals=minv, max_vals=maxv)
    else:
        yi, (minv, maxv) = normalize_point_cloud(pointy)
        xi, _ = normalize_point_cloud(pointx, min_vals=minv, max_vals=maxv)

    return xi, yi, (minv, maxv)
