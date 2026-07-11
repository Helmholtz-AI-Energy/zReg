"""From-scratch trajectory factory for synthetic point cloud sequences.

Generates multi-frame 3D point cloud trajectories where each frame is an
independent Gaussian-blob sample (``N(0, I)`` distribution). Labels and
structured corruptions are added separately by ``labels.py`` and
``corruption.py``.

Also provides single-frame geometry samplers (``sample_ball``, ``sample_bowl``)
that produce bowl/ball-shaped point clouds resembling the real Kobitski/Shah
embryo shapes, ported (simplified to a single fixed radius, no per-frame
growth) from the standalone rejection-sampling math in
``scripts/generate_datasets.py``.

Seed contract: pass ``seed: int | None = 42`` to any stochastic function.
``seed=None`` means the caller controls the RNG state; the function will NOT
call ``torch.manual_seed``.

The single-frame geometry samplers (``sample_ball``, ``sample_bowl``) deviate
from this torch-based contract: their geometry math is numpy-native (inverse-
CDF sampling and rejection sampling are easiest to express with
``numpy.random.Generator``), so they seed a fresh ``np.random.default_rng(seed)``
instead of calling ``torch.manual_seed``. The "seed=None means caller-
controlled/nondeterministic" semantic is preserved: passing ``seed=None`` calls
``np.random.default_rng()`` with no seed.
"""

import numpy as np
import torch

from zreg.dataset import zRegPointCloud

__all__ = ["generate_trajectory", "sample_ball"]


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


def _sample_ball_shell(
    n: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform sample in spherical shell [r_inner, r_outer] (volume-correct).

    Ported from ``scripts/generate_datasets.py``'s ``sample_ball_shell``.
    """
    r = rng.uniform(r_inner**3, r_outer**3, n) ** (1.0 / 3)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    return np.stack(
        [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t],
        axis=1,
    ).astype(np.float32)


def sample_ball(
    n_points: int,
    seed: int | None = 42,
    radius: float = 1.0,
) -> torch.Tensor:
    """Generate a single-frame solid-ball point cloud.

    Points are sampled uniformly (volume-correct) inside a sphere of the
    given ``radius``, centred at the origin. This is a simplified,
    single-frame (non-growth) adaptation of ``scripts/generate_datasets.py``'s
    ``sample_ball_shell`` with ``R_inner=0.0`` (solid ball rather than a
    shell). No labels, ids, or growth-over-frames logic are added — the
    caller (see 46-03) wraps the returned tensor in a ``zRegPointCloud``.

    Parameters
    ----------
    n_points : int
        Number of points to sample. Must be >= 1.
    seed : int or None, optional
        Random seed for reproducibility. When an integer, a fresh
        ``numpy.random.default_rng(seed)`` is created and used for all
        sampling. When ``None``, ``numpy.random.default_rng()`` is created
        with no seed (caller-nondeterministic). This deviates from
        ``generate_trajectory``'s ``torch.manual_seed`` contract because
        this sampler's rejection/inverse-CDF geometry math is numpy-native;
        the "seed=None means nondeterministic" semantic is preserved.
        Default: 42.
    radius : float, optional
        Radius of the ball. Must be > 0. Default: 1.0.

    Returns
    -------
    torch.Tensor
        A ``(n_points, 3)`` float32 tensor of positions, all with L2 norm
        <= ``radius``.

    Raises
    ------
    ValueError
        If ``n_points < 1``.

    Examples
    --------
    >>> pos = sample_ball(n_points=100, seed=0)
    >>> pos.shape
    torch.Size([100, 3])
    >>> bool(pos.norm(dim=1).max() <= 1.0 + 1e-5)
    True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    rng = np.random.default_rng(seed)
    arr = _sample_ball_shell(n_points, 0.0, radius, rng)
    return torch.from_numpy(arr.astype(np.float32))
