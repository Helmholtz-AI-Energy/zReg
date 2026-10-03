import torch

__all__ = [
    "squared_kernel",
    "squared_kernel_sum",
    "rbf_kernel",
    "tps_kernel",
    "inverse_multiquadric_kernel",
    "normalize_point_cloud",
    "normalize_to_pc_w_most_points",
    "undo_normalize",
    "shared_bounds",
    "registration_bounds",
    "normalization_matrix",
    "denormalization_matrix",
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    # Guard against zero-range axes (would produce NaN/Inf)
    eps = torch.finfo(points.dtype).eps
    ranges = torch.clamp(ranges, min=eps)

    # Scale the points
    scaled_points = 2 * (points - min_vals) / ranges - 1

    return scaled_points, (min_vals, max_vals)


def normalize_to_pc_w_most_points(
    pointx: torch.Tensor, pointy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        A tuple containing:
        - **xi** : torch.Tensor
            The normalized `pointx` point cloud, with the same shape as the input `pointx`.
        - **yi** : torch.Tensor
            The normalized `pointy` point cloud, with the same shape as the input `pointy`.
        - **(minv, maxv)** : tuple[torch.Tensor, torch.Tensor]
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


def shared_bounds(*clouds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one scalar bounds pair covering every given point cloud.

    Registration wrappers (ICP, SWD) normalise source and target with the
    same bounds so that a rigid transform found in the normalised frame stays
    rigid after denormalisation.

    Parameters
    ----------
    *clouds : torch.Tensor
        One or more ``(N, D)`` point tensors, all on the same device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(lo, hi)`` as 0-d tensors: the minimum and maximum coordinate over
        all clouds and all axes. They live on the clouds' device and dtype.

    Examples
    --------
    >>> import torch
    >>> lo, hi = shared_bounds(torch.tensor([[0.0, 1.0]]), torch.tensor([[-2.0, 5.0]]))
    >>> lo.item(), hi.item()
    (-2.0, 5.0)
    """
    if not clouds:
        raise ValueError("shared_bounds requires at least one point cloud")
    lo = torch.stack([c.min() for c in clouds]).min()
    hi = torch.stack([c.max() for c in clouds]).max()
    return lo, hi


def registration_bounds(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a registration pair and return its shared scalar bounds.

    Validating wrapper around ``shared_bounds`` used by the rigid aligners
    (ICP, SWD). Checks, in this order, for each cloud:

    1. shape is ``(N, 3)``;
    2. the cloud is not empty;
    3. every coordinate is finite (no NaN/inf);
    4. the cloud has a non-zero overall extent, i.e. the largest per-axis
       range over the points is not exactly zero. Clouds whose points all
       coincide (including a single point) leave the rotation undefined.

    Zero range on one or two axes (planar or collinear clouds) is accepted,
    and so are small-but-nonzero clouds: there is no eps threshold.

    Parameters
    ----------
    source, target : torch.Tensor
        ``(N, 3)`` and ``(M, 3)`` point tensors on the same device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(lo, hi)`` exactly as ``shared_bounds(source, target)``.

    Raises
    ------
    ValueError
        If a cloud is not ``(N, 3)``, is empty, contains NaN/inf, has zero
        extent (all points coincide), or the clouds live on different devices.
        The message names the offending cloud (``source['pos']`` or
        ``target['pos']``).

    Examples
    --------
    >>> import torch
    >>> src = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
    >>> tgt = torch.tensor([[-2.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    >>> lo, hi = registration_bounds(src, tgt)
    >>> lo.item(), hi.item()
    (-2.0, 5.0)
    """
    from .validation import _validate_tensors

    names = ["source['pos']", "target['pos']"]
    clouds = [source, target]
    for c, name in zip(clouds, names):
        if c.dim() != 2 or c.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape (N, 3), but has shape {tuple(c.shape)}"
            )
    for c, name in zip(clouds, names):
        if c.shape[0] == 0:
            raise ValueError(f"{name} is empty (shape {tuple(c.shape)}); cannot register")
    _validate_tensors(source, target, names=names)
    for c, name in zip(clouds, names):
        spread = (c.max(dim=0).values - c.min(dim=0).values).max()
        if spread == 0:
            raise ValueError(
                f"{name} has zero extent ({c.shape[0]} point(s)): "
                "all points coincide; the rotation is undefined"
            )
    return shared_bounds(source, target)


def _bounds_on(lo, hi, dtype, device):
    """Convert ``lo``/``hi`` to detached tensors on one device and dtype."""
    if device is None:
        device = lo.device if isinstance(lo, torch.Tensor) else torch.device("cpu")
    lo_t = torch.as_tensor(lo).detach().to(device=device, dtype=dtype)
    hi_t = torch.as_tensor(hi).detach().to(device=device, dtype=dtype)
    rng = torch.clamp(hi_t - lo_t, min=torch.finfo(dtype).eps)
    return lo_t, rng, device


def normalization_matrix(
    lo, hi, *, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """Build the 4x4 homogeneous matrix of ``normalize_point_cloud``.

    Maps ``x`` to ``2 * (x - lo) / (hi - lo) - 1`` on every axis. The range is
    clamped to ``torch.finfo(dtype).eps`` exactly like ``normalize_point_cloud``
    so the matrix cannot diverge from the applied normalisation.

    Parameters
    ----------
    lo, hi : float or torch.Tensor
        Scalar lower and upper bound (e.g. from ``shared_bounds``).
    dtype : torch.dtype, default torch.float32
        dtype of the returned matrix.
    device : torch.device or str, optional
        Device of the returned matrix. ``None`` means the device of ``lo`` if
        it is a tensor, else CPU. ``lo``/``hi`` are detached and moved here, so
        every operand lives on one device.

    Returns
    -------
    torch.Tensor
        ``[4, 4]`` matrix ``D`` with ``[x_norm, 1] = D @ [x, 1]``.

    Examples
    --------
    >>> import torch
    >>> D = normalization_matrix(2.0, 10.0)
    >>> (D @ torch.tensor([2.0, 6.0, 10.0, 1.0]))[:3]
    tensor([-1.,  0.,  1.])
    """
    lo_t, rng, device = _bounds_on(lo, hi, dtype, device)
    m = torch.eye(4, dtype=dtype, device=device)
    idx = torch.arange(3, device=device)
    m[idx, idx] = 2.0 / rng
    m[:3, 3] = -2.0 * lo_t / rng - 1.0
    return m


def denormalization_matrix(
    lo, hi, *, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """Build the exact inverse of ``normalization_matrix`` (matrix form of ``undo_normalize``).

    Maps ``x_norm`` to ``(x_norm + 1) * (hi - lo) / 2 + lo``, i.e. scale
    ``range / 2`` and translation ``lo + range / 2``.

    Parameters
    ----------
    lo, hi : float or torch.Tensor
        Scalar lower and upper bound used for the normalisation.
    dtype : torch.dtype, default torch.float32
        dtype of the returned matrix.
    device : torch.device or str, optional
        Device of the returned matrix (same rules as ``normalization_matrix``).

    Returns
    -------
    torch.Tensor
        ``[4, 4]`` matrix ``D_inv`` with ``D_inv @ normalization_matrix(lo, hi) == I``.

    Examples
    --------
    >>> import torch
    >>> D_inv = denormalization_matrix(2.0, 10.0)
    >>> (D_inv @ torch.tensor([-1.0, 0.0, 1.0, 1.0]))[:3]
    tensor([ 2.,  6., 10.])
    """
    lo_t, rng, device = _bounds_on(lo, hi, dtype, device)
    m = torch.eye(4, dtype=dtype, device=device)
    idx = torch.arange(3, device=device)
    m[idx, idx] = rng / 2.0
    m[:3, 3] = lo_t + rng / 2.0
    return m


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
