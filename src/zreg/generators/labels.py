"""Label generation and removal utilities for synthetic trajectories.

Provides Voronoi-based synthetic label generation (``generate_labels``) and
label removal (``remove_labels``). Both utilities are immutable — they accept
a ``dict[int, zRegPointCloud]`` and return a new dict. The input dict is never
modified (deep-copy contract, D-03).

Labels are stored in ``zRegPointCloud["label"]`` as ``torch.long`` tensors of
shape ``(N,)``, one integer class ID per point (D-06, D-07). This format is
directly compatible with ``zreg.metrics.compute_f1`` without conversion.
"""

import copy

import torch

from zreg.dataset import zRegPointCloud

__all__ = ["generate_labels", "remove_labels"]


def generate_labels(
    trajectory: dict[int, zRegPointCloud],
    n_classes: int,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Assign Voronoi-based integer labels to every frame in a trajectory.

    For each frame, ``n_classes`` seed points are sampled from ``N(0, I)``
    in R³. Each point is assigned to the nearest seed (L2 distance via
    ``torch.cdist``), producing spatially coherent class clusters. The label
    tensor is stored in ``pc["label"]`` with ``dtype=torch.long`` and shape
    ``(N,)`` so it is directly passable to ``zreg.metrics.compute_f1``.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    n_classes : int
        Number of distinct class labels. Must be ``>= 1``.
    seed : int | None, optional
        Seed for ``torch.manual_seed``. ``None`` means the caller controls
        the RNG state. Default: ``42``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict where every frame's ``pc["label"]`` is a
        ``torch.long`` tensor of shape ``(N,)`` with values in
        ``[0, n_classes)``.

    Raises
    ------
    ValueError
        If ``n_classes`` is less than 1.

    Notes
    -----
    Voronoi assignment is deterministic given a fixed ``seed`` and
    ``dict.values()`` insertion order. The seed is set once at function
    entry; successive ``torch.randn`` calls for each frame are consumed
    in that order.

    Empty frames (``N == 0``) are handled transparently: ``torch.cdist``
    returns shape ``(0, n_classes)`` and ``argmin`` returns a ``(0,)``
    ``torch.long`` tensor, which is assigned without special-casing.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if seed is not None:
        torch.manual_seed(seed)
    result = copy.deepcopy(trajectory)
    for pc in result.values():
        pos = pc["pos"]  # shape (N, 3)
        seeds = torch.randn(n_classes, 3, dtype=pos.dtype, device=pos.device)
        dists = torch.cdist(pos, seeds, p=2)  # (N, n_classes)
        labels = dists.argmin(dim=1).to(torch.long)  # (N,), values in [0, n_classes)
        pc["label"] = labels
    return result


def remove_labels(
    trajectory: dict[int, zRegPointCloud],
) -> dict[int, zRegPointCloud]:
    """Remove labels from every frame in a trajectory.

    Sets ``pc["label"] = None`` for every frame, matching the default
    ``zRegPointCloud.__init__`` behaviour where unset fields are ``None``
    (D-08). Callers check ``if pc["label"] is None`` to detect unlabelled
    frames.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict where every frame's ``pc["label"]`` is ``None``.
    """
    result = copy.deepcopy(trajectory)
    for pc in result.values():
        pc["label"] = None
    return result
