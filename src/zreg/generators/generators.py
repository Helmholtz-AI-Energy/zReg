"""From-scratch trajectory factory for synthetic point cloud sequences.

Generates multi-frame 3D point cloud trajectories where each frame is an
independent Gaussian-blob sample (``N(0, I)`` distribution). Labels and
structured corruptions are added separately by ``labels.py`` and
``corruption.py``.

Seed contract: pass ``seed: int | None = 42`` to any stochastic function.
``seed=None`` means the caller controls the RNG state; the function will NOT
call ``torch.manual_seed``.
"""

import torch

from zreg.dataset import zRegPointCloud

__all__ = ["generate_trajectory"]


def generate_trajectory(
    n_points: int,
    n_frames: int,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Generate a multi-frame synthetic point cloud trajectory.

    Each frame is an independent sample from a 3-D isotropic Gaussian
    distribution ``N(0, I)``.  The ``color`` and ``id`` fields of every
    returned ``zRegPointCloud`` are ``None``; use ``generate_labels`` from
    ``zreg.generators.labels`` to assign integer class labels afterwards.

    Parameters
    ----------
    n_points : int
        Number of points per frame.  Must be >= 1.
    n_frames : int
        Number of frames in the trajectory.  Must be >= 1.
    seed : int or None, optional
        Random seed for reproducibility.  When an integer, ``torch.manual_seed``
        is called before any stochastic operation.  When ``None`` the caller
        controls RNG state and the function does not touch it.  Default: 42.

    Returns
    -------
    dict[int, zRegPointCloud]
        A mapping from frame index (0 … n_frames-1) to a ``zRegPointCloud``
        with ``pos`` of shape ``(n_points, 3)`` and ``dtype=torch.float32``.
        Fields ``color``, ``id``, and ``fps-idx`` are all ``None``.

    Raises
    ------
    ValueError
        If ``n_points < 1`` or ``n_frames < 1``.

    Examples
    --------
    >>> traj = generate_trajectory(n_points=100, n_frames=5, seed=0)
    >>> len(traj)
    5
    >>> traj[0]["pos"].shape
    torch.Size([100, 3])
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    if seed is not None:
        torch.manual_seed(seed)

    return {
        i: zRegPointCloud(pos=torch.randn(n_points, 3))
        for i in range(n_frames)
    }
