"""Corruption wrappers: Gaussian noise and outlier injection.

Both wrappers are immutable — they accept a ``dict[int, zRegPointCloud]``
and return a new dict with the transformation applied. The input dict is
never modified (deep-copy contract, D-03).
"""

import copy

import torch

from zreg.dataset import zRegPointCloud

__all__ = ["add_gaussian_noise", "add_outliers"]


def add_gaussian_noise(
    trajectory: dict[int, zRegPointCloud],
    sigma: float = 0.01,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Add independent Gaussian noise to every frame's ``pos`` field.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    sigma : float, optional
        Standard deviation of the additive noise. Must be ``>= 0``.
        When ``sigma == 0.0`` the returned ``pos`` tensors are
        element-wise equal to the input (no-op via exact zero addition).
        Default: ``0.01``.
    seed : int | None, optional
        Seed for ``torch.manual_seed``. ``None`` means the caller
        controls the RNG state; ``torch.manual_seed`` is not called.
        Default: ``42``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict whose every frame's ``pos`` has additive
        ``N(0, sigma^2)`` noise applied.

    Raises
    ------
    ValueError
        If ``sigma`` is negative.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if seed is not None:
        torch.manual_seed(seed)
    result = copy.deepcopy(trajectory)
    for pc in result.values():
        noise = torch.randn_like(pc["pos"]) * sigma
        pc["pos"] = pc["pos"] + noise
    return result


def add_outliers(
    trajectory: dict[int, zRegPointCloud],
    n_outliers: int,
    scale: float = 3.0,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Inject outlier points into every frame of a trajectory.

    Outliers are sampled from ``N(0, scale^2 * I)`` and appended to
    the end of each frame's ``pos`` tensor. When the frame carries
    auxiliary per-point fields (``id``, 1-D ``color``, ``fps-idx``),
    those fields are extended with sentinel ``-1`` values so that every
    per-point field remains length-consistent with ``pos``.

    For 2-D ``color`` tensors (e.g. RGB shape ``(N, 3)``), the extension
    uses zeros of the same column width to preserve the invariant
    ``pc["pos"].shape[0] == pc["color"].shape[0]``.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    n_outliers : int
        Number of outlier points to append per frame. Must be ``>= 0``.
        ``n_outliers == 0`` is allowed (no-op code path, useful in sweeps).
    scale : float, optional
        Standard deviation of the outlier point distribution. Default: ``3.0``.
    seed : int | None, optional
        Seed for ``torch.manual_seed``. ``None`` means the caller owns
        the RNG state. Default: ``42``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict whose every frame has ``n_outliers`` extra
        points appended.

    Raises
    ------
    ValueError
        If ``n_outliers`` is negative.
    """
    if n_outliers < 0:
        raise ValueError(f"n_outliers must be >= 0, got {n_outliers}")
    if seed is not None:
        torch.manual_seed(seed)
    result = copy.deepcopy(trajectory)
    for pc in result.values():
        pos = pc["pos"]
        outliers = torch.randn(
            n_outliers, 3, dtype=pos.dtype, device=pos.device
        ) * scale
        pc["pos"] = torch.cat([pos, outliers], dim=0)

        # Extend id field with sentinel -1
        if pc["id"] is not None:
            sentinel_ids = torch.full(
                (n_outliers,), -1, dtype=pc["id"].dtype, device=pc["id"].device
            )
            pc["id"] = torch.cat([pc["id"], sentinel_ids], dim=0)

        # Extend color field — 1-D labels get sentinel -1; 2-D RGB gets zeros
        if pc["color"] is not None:
            if pc["color"].dim() == 1:
                sentinel_color = torch.full(
                    (n_outliers,),
                    -1,
                    dtype=pc["color"].dtype,
                    device=pc["color"].device,
                )
                pc["color"] = torch.cat([pc["color"], sentinel_color], dim=0)
            else:
                # 2-D color (e.g. RGB): extend with zeros to keep row count consistent
                extra = torch.zeros(
                    (n_outliers, pc["color"].shape[1]),
                    dtype=pc["color"].dtype,
                    device=pc["color"].device,
                )
                pc["color"] = torch.cat([pc["color"], extra], dim=0)

        # Extend fps-idx field with sentinel -1
        if pc["fps-idx"] is not None:
            sentinel_fps = torch.full(
                (n_outliers,),
                -1,
                dtype=pc["fps-idx"].dtype,
                device=pc["fps-idx"].device,
            )
            pc["fps-idx"] = torch.cat([pc["fps-idx"], sentinel_fps], dim=0)

    return result
