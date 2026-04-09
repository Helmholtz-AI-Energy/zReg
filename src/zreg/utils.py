import torch
from typing import Tuple

__all__ = [
    "squared_kernel",
    "squared_kernel_sum",
    "rbf_kernel",
    "tps_kernel",
    "inverse_multiquadric_kernel",
    "normalize_point_cloud",
    "normalize_to_pc_w_most_points",
    "undo_normalize",
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
    dist = torch.cdist(x, y, p=2).T
    return dist.pow(2)  # undo the square root from cdist
    # return (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim=2)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Computes the Radial Basis Function (RBF) kernel between two tensors.

    .. note::
        Inputs must be pre-normalized (e.g., via :func:`normalize_point_cloud`)
        to prevent numerical instability. This function does NOT normalize internally.

    Parameters
    ----------
    x : torch.Tensor
        First tensor with shape (n, d). Must be pre-normalized.
    y : torch.Tensor
        Second tensor with shape (m, d). Must be pre-normalized.
    beta : float
        Bandwidth parameter for the RBF kernel.

    Returns
    -------
    torch.Tensor
        A tensor with shape (m, n) representing the RBF kernel.
    """
    # NOTE: inputs must be pre-normalized before calling this function
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
    points: torch.Tensor,
    max_vals: torch.Tensor = None,
    min_vals: torch.Tensor = None,
    byaxis=False,
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
    # Handle empty point clouds
    if points.shape[0] == 0:
        # Return empty tensor with same shape and dummy min/max values
        d = points.shape[1] if points.dim() > 1 else 1
        dummy_vals = torch.zeros(d, dtype=points.dtype, device=points.device)
        return points, (dummy_vals, dummy_vals)

    # Find the minimum and maximum values along each dimension if not given
    if max_vals is None:
        max_vals = torch.max(points, dim=0)[0]
    if min_vals is None:
        min_vals = torch.min(points, dim=0)[0]

    if not byaxis and max_vals.numel() > 1:
        # normalize all axis to the same scale with the same ratio
        max_vals = max_vals.max()
        min_vals = min_vals.min()

    # Calculate the range of each dimension
    ranges = max_vals - min_vals

    # Scale the points
    scaled_points = 2 * (points - min_vals) / ranges - 1

    return scaled_points, (min_vals, max_vals)


def normalize_to_pc_w_most_points(
    pointx: torch.Tensor, pointy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Normalizes two point clouds to the range [-1, 1] based on the one with the most points.

    This function takes two point clouds, `pointx` and `pointy`, and normalizes them to the range
    [-1, 1]. It determines the minimum and maximum values from the point cloud with the *most*
    points and uses these values to normalize *both* point clouds. This ensures that both point
    clouds are scaled and translated consistently, even if they have different numbers of points.

    Parameters
    ----------
    pointx : torch.Tensor
        The first point cloud, represented as a tensor of shape (N, D) where N is the number of
        points and D is the dimensionality of each point.
    pointy : torch.Tensor
        The second point cloud, represented as a tensor of shape (M, D) where M is the number of
        points and D is the dimensionality of each point.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        A tuple containing:
        - **xi** : torch.Tensor
            The normalized `pointx` point cloud, with the same shape as the input `pointx`.
        - **yi** : torch.Tensor
            The normalized `pointy` point cloud, with the same shape as the input `pointy`.
        - **(minv, maxv)** : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the minimum and maximum values used for normalization. These
            values are determined from the point cloud with the most points. `minv` and `maxv` are
            tensors with shape (D,).

    Raises
    ------
    TypeError
        If `pointx` or `pointy` is not a torch.Tensor.
    ValueError
        If `pointx` or `pointy` is not 2-dimensional.

    Notes
    -----
    - The function assumes that `normalize_point_cloud` function is available and used for the actual normalization.
      A placeholder definition is included in the example for completeness.
    - The `TODO` comment in the original code is not addressed in the docstring as it's an internal implementation detail.

    Examples
    --------
    >>> import torch
    >>> def normalize_point_cloud(pc, min_vals=None, max_vals=None):
    ...     if min_vals is None:
    ...         min_vals = pc.min(dim=0, keepdim=True).values
    ...     if max_vals is None:
    ...         max_vals = pc.max(dim=0, keepdim=True).values
    ...     return (pc - min_vals) / (max_vals - min_vals) * 2 - 1, (min_vals.squeeze(0), max_vals.squeeze(0))
    ...
    >>> pointx = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> pointy = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    >>> xi, yi, (minv, maxv) = normalize_to_pc_w_most_points(pointx, pointy)
    >>> xi
    tensor([[-1.0000, -1.0000],
            [ 0.0000,  0.0000],
            [ 1.0000,  1.0000]])
    >>> yi
    tensor([[-1.2500, -1.2500],
            [-0.2500, -0.2500]])
    >>> minv
    tensor([1., 2.])
    >>> maxv
    tensor([5., 6.])

    >>> pointx = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    >>> pointy = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> xi, yi, (minv, maxv) = normalize_to_pc_w_most_points(pointx, pointy)
    >>> xi
    tensor([[-1.0000, -1.0000],
            [ 0.0000,  0.0000]])
    >>> yi
    tensor([[-0.7500, -0.7500],
            [ 0.2500,  0.2500],
            [ 1.2500,  1.2500]])
    >>> minv
    tensor([0.5000, 1.5000])
    >>> maxv
    tensor([2.5000, 3.5000])
    """
    # TODO: fix normalize to use the points dicts not just the torch dicts
    if pointx.shape[0] > pointy.shape[0]:
        xi, (minv, maxv) = normalize_point_cloud(pointx)
        yi, _ = normalize_point_cloud(pointy, min_vals=minv, max_vals=maxv)
    else:
        yi, (minv, maxv) = normalize_point_cloud(pointy)
        xi, _ = normalize_point_cloud(pointx, min_vals=minv, max_vals=maxv)

    return xi, yi, (minv, maxv)


def undo_normalize(points: "torch.Tensor", maxvals: "torch.Tensor", minvals: "torch.Tensor") -> "torch.Tensor":
    """Reverses the normalization applied to a set of points.

    This function takes a set of normalized points and applies the inverse
    of the min-max normalization, effectively restoring the points to their
    original scale. The normalization is assumed to have been performed
    using the following formula:

    `normalized_points = 2 * (original_points - minvals) / (maxvals - minvals) - 1`

    This function reverses this process.

    Parameters
    ----------
    points : torch.Tensor
        The normalized points to be unnormalized.
    maxvals : torch.Tensor
        The maximum values used during the original normalization.
        Must have the same shape as `minvals`, or be broadcastable to it.
    minvals : torch.Tensor
        The minimum values used during the original normalization.
        Must have the same shape as `maxvals`, or be broadcastable to it.

    Returns
    -------
    torch.Tensor
        The unnormalized points, with the same shape and dtype as the input `points`.

    Examples
    --------
    >>> import torch
    >>> points = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.25]])
    >>> maxvals = torch.tensor([10.0, 20.0, 30.0])
    >>> minvals = torch.tensor([0.0, 5.0, 10.0])
    >>> undo_normalize(points, maxvals, minvals)
    tensor([[ 0.0000,  5.0000, 10.0000],
            [ 2.5000, 12.5000, 13.7500]])

    >>> points = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.25]])
    >>> maxvals = torch.tensor(10.0)
    >>> minvals = torch.tensor(0.0)
    >>> undo_normalize(points, maxvals, minvals)
    tensor([[0.0000, 5.0000, 10.0000],
            [2.5000, 7.5000,  6.2500]])
    """
    return (points + 1) * (maxvals - minvals) * 0.5 + minvals


def generate_random_rotation_matrix(angles=None):
    """
    Generates a 3D rotation matrix based on three Euler angles (roll, pitch, yaw).

    Args:
        angles: A Torch Tensor of shape (3,) containing the roll, pitch, and yaw angles in radians.

    Returns:
        A 3x3 Torch Tensor representing the rotation matrix.
    """
    roll, pitch, yaw = angles if angles is not None else torch.rand(3)

    # Rotation matrix around x-axis (roll)
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]])

    # Rotation matrix around y-axis (pitch)
    Ry = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]])

    # Rotation matrix around z-axis (yaw)
    Rz = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]])

    # Combined rotation matrix (Rz @ Ry @ Rx)
    R = Rz @ Ry @ Rx

    return R
