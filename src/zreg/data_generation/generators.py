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

import math

import numpy as np
import torch

from zreg.core.dataset import zRegPointCloud

__all__ = ["generate_trajectory", "sample_ball", "sample_bowl"]


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


# Phase 62 U6-4 (Review cycles 1-3): bounds for ``sample_bowl``'s rejection
# sampler. The acceptance rate is roughly 0.39 * d_ratio for small d_ratio
# (about 0.04% at the lower bound), so smaller d_ratio values are rejected.
# ``_MAX_BOWL_CANDIDATES`` bounds the TOTAL number of candidate points drawn
# over all rounds (independent of n_points; about 1.2 GB of float64 drawn in
# total at worst, never resident at once). ``_MAX_BOWL_BATCH`` bounds the
# candidates drawn in one round, i.e. the peak allocation (24 MB of float64).
_MIN_BOWL_D_RATIO = 1e-3
_MAX_BOWL_CANDIDATES = 50_000_000
_MAX_BOWL_BATCH = 1_000_000


def _in_bowl(pts: np.ndarray, R: float, d: float) -> np.ndarray:
    """Boolean mask: True for points inside bowl(R, d).

    Bowl = { p : ‖p‖ <= R  AND  ‖p-(0,0,d)‖^2 > R^2+d^2  AND  z <= 0 }

    Ported from ``scripts/generate_datasets.py``'s ``in_bowl``. The carving
    sphere (radius sqrt(R^2+d^2), centred at (0,0,d)) intersects the outer
    sphere exactly at z = 0, so the bowl opening sits at the equatorial
    plane with zero wall thickness there.
    """
    norm_sq = (pts**2).sum(axis=1)
    carve_sq = pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - d) ** 2
    return (norm_sq <= R**2) & (carve_sq > R**2 + d**2) & (pts[:, 2] <= 0.0)


def sample_bowl(
    n_points: int,
    seed: int | None = 42,
    radius: float = 1.0,
    d_ratio: float = 0.5,
) -> torch.Tensor:
    """Generate a single-frame bowl (lower-hemisphere-shell) point cloud.

    Points are sampled uniformly (via rejection sampling) inside the bowl
    region: the outer sphere of the given ``radius``, carved by an inner
    sphere of radius ``sqrt(radius**2 + d**2)`` centred at ``(0, 0, d)``
    where ``d = d_ratio * radius``, restricted to the lower hemisphere
    (``z <= 0``). This is a simplified, single-frame (non-growth) adaptation
    of ``scripts/generate_datasets.py``'s ``sample_bowl_frame``/``in_bowl``.
    No labels, ids, or growth-over-frames logic are added — the caller (see
    46-03) wraps the returned tensor in a ``zRegPointCloud``.

    Parameters
    ----------
    n_points : int
        Number of points to sample. Must be >= 1.
    seed : int or None, optional
        Random seed for reproducibility. Same numpy-native seed contract as
        ``sample_ball`` (see its docstring): an integer seeds a fresh
        ``numpy.random.default_rng(seed)``; ``None`` calls
        ``numpy.random.default_rng()`` unseeded. Default: 42.
    radius : float, optional
        Outer sphere radius R. Must be finite and > 0. Default: 1.0.
    d_ratio : float, optional
        Ratio used to compute the carving-sphere offset ``d = d_ratio *
        radius``. Must be finite and >= 1e-3: the rejection acceptance rate
        is roughly ``0.39 * d_ratio``, and ``d_ratio <= 0`` describes an
        empty region. Default: 0.5.

    Returns
    -------
    torch.Tensor
        A ``(n_points, 3)`` float32 tensor of positions, all satisfying the
        bowl predicate (lower-hemisphere shell).

    Raises
    ------
    ValueError
        If ``n_points < 1``; if ``radius`` is not finite or ``<= 0``; if
        ``d_ratio`` is not finite or ``< 1e-3``; if ``n_points`` exceeds
        the candidate budget (50 million candidate points, unsatisfiable
        even at 100% acceptance).
    RuntimeError
        If fewer than ``n_points`` points are accepted within the candidate
        budget of 50 million candidate points summed over all rounds. Each
        round draws at most ``max(n_points * 10, 2_000)`` candidates, capped
        at 1 million per round (peak allocation) and at the remaining budget.

    Examples
    --------
    >>> pos = sample_bowl(n_points=100, seed=0)
    >>> pos.shape
    torch.Size([100, 3])
    >>> bool((pos[:, 2] <= 1e-6).all())
    True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be a finite float > 0, got {radius!r}")
    if not math.isfinite(d_ratio) or d_ratio < _MIN_BOWL_D_RATIO:
        raise ValueError(
            f"d_ratio must be a finite float >= {_MIN_BOWL_D_RATIO} (acceptance rate "
            f"~0.39*d_ratio makes smaller values impractical), got {d_ratio!r}"
        )
    if n_points > _MAX_BOWL_CANDIDATES:
        raise ValueError(
            f"n_points={n_points} exceeds the sample_bowl candidate budget "
            f"_MAX_BOWL_CANDIDATES={_MAX_BOWL_CANDIDATES}"
        )

    rng = np.random.default_rng(seed)
    R = radius
    d = d_ratio * radius
    # The per-round cap bounds peak allocation. numpy draws the candidates
    # sequentially from one stream, so chunking does not change the
    # accepted points.
    batch = min(max(n_points * 10, 2_000), _MAX_BOWL_BATCH)

    collected: list[np.ndarray] = []
    total = 0
    drawn = 0
    while total < n_points and drawn < _MAX_BOWL_CANDIDATES:
        current = min(batch, _MAX_BOWL_CANDIDATES - drawn)
        cands = rng.uniform(-R, R, (current, 3))
        drawn += current
        keep = cands[_in_bowl(cands, R, d)]
        collected.append(keep)
        total += len(keep)

    if total < n_points:
        raise RuntimeError(
            f"sample_bowl accepted only {total} of n_points={n_points} points after "
            f"drawn={drawn} candidates (budget _MAX_BOWL_CANDIDATES="
            f"{_MAX_BOWL_CANDIDATES}); radius={radius!r}, d_ratio={d_ratio!r}"
        )

    arr = np.concatenate(collected, axis=0)[:n_points]
    return torch.from_numpy(arr.astype(np.float32))
