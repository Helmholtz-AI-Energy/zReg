"""Label generation and removal utilities for synthetic trajectories.

Provides Voronoi-based synthetic label generation (``generate_labels``),
label removal (``remove_labels``), spherical-cap binary labelling
(``assign_cap_labels``), and soft Gaussian-probability Bernoulli labelling
(``assign_gaussian_labels``). All utilities are immutable — they accept a
``dict[int, zRegPointCloud]`` and return a new dict. The input dict is never
modified (deep-copy contract, D-03).

Labels are stored in ``zRegPointCloud["label"]`` as ``torch.long`` tensors of
shape ``(N,)``, one integer class ID per point (D-06, D-07). This format is
directly compatible with ``zreg.metrics.compute_f1`` without conversion.
"""

import copy
from typing import Literal, Sequence

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from zreg.core.dataset import zRegPointCloud

__all__ = ["generate_labels", "remove_labels", "assign_cap_labels", "assign_gaussian_labels"]


class LabelComponentSpec(BaseModel):
    """One region component (voronoi/blob/cone) contributing to a label.

    A ``LabelSpec`` combines one or more ``LabelComponentSpec`` instances
    into a single label via a weighted ``logsumexp`` mixture (D-11). This
    model only defines the vocabulary — scoring math lives in
    ``_component_score``/``_label_scores``.

    Parameters
    ----------
    shape : {"voronoi", "blob", "cone"}
        Region shape. ``"voronoi"`` = nearest-center partition;
        ``"blob"`` = isotropic Euclidean Gaussian (D-08); ``"cone"`` =
        origin-anchored angular Gaussian (D-09).
    center : list[float]
        Exactly 3 elements. Euclidean center for ``voronoi``/``blob``; pole
        direction vector for ``cone`` (need not be unit length — normalised
        internally by ``_angular_distance_deg``).
    sigma : float | None, optional
        Euclidean isotropic sigma for ``blob``; angular sigma-in-degrees
        for ``cone``. Required and must be ``> 0`` for both shapes. Ignored
        (not an error) for ``voronoi``. Default: ``None``.
    temperature : float | None, optional
        Softmax temperature, meaningful only for ``voronoi`` in
        probabilistic mode. Ignored (not an error) for ``blob``/``cone`` and
        for ``voronoi`` in deterministic mode. Default: ``None``.
    weight : float, optional
        Mixing coefficient/prior (D-11). Not required to sum to 1 across a
        label's components — a relative multiplier. Must be ``> 0``.
        Default: ``1.0``.
    """

    model_config = ConfigDict(extra="forbid")

    shape: Literal["voronoi", "blob", "cone"]
    center: list[float]
    sigma: float | None = None
    temperature: float | None = None
    weight: float = 1.0

    @field_validator("center")
    @classmethod
    def _validate_center(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"center must have exactly 3 elements, got {len(v)}")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"weight must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_sigma(self) -> "LabelComponentSpec":
        if self.shape in ("blob", "cone") and (self.sigma is None or self.sigma <= 0):
            raise ValueError(f"sigma must be > 0 for shape={self.shape!r}, got {self.sigma}")
        return self


class LabelSpec(BaseModel):
    """A single label defined by one or more mixed-shape region components.

    Parameters
    ----------
    label_id : int
        The integer label value this spec's components jointly define.
    components : list[LabelComponentSpec]
        One or more components (D-11 mixture). Must be non-empty.
    """

    model_config = ConfigDict(extra="forbid")

    label_id: int
    components: list[LabelComponentSpec]

    @field_validator("components")
    @classmethod
    def _validate_components(cls, v: list[LabelComponentSpec]) -> list[LabelComponentSpec]:
        if len(v) == 0:
            raise ValueError("components must be non-empty")
        return v


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
    result = copy.deepcopy(trajectory)
    if seed is not None:
        torch.manual_seed(seed)
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


def _angular_distance_deg(
    pos: torch.Tensor,
    pole: Sequence[float],
) -> torch.Tensor:
    """Return per-point angular distance in degrees from a pole direction.

    Parameters
    ----------
    pos : torch.Tensor
        Shape ``(N, 3)``, float dtype.
    pole : sequence of float
        3-element direction vector (need not be unit length).

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``, float, values in ``[0, 180]``.
    """
    pole_vec = torch.tensor(pole, dtype=pos.dtype, device=pos.device)
    pole_norm = pole_vec.norm()
    if pole_norm == 0:
        raise ValueError("pole must be a non-zero vector")
    pole_unit = pole_vec / pole_norm
    eps = torch.finfo(pos.dtype).eps
    norms = pos.norm(dim=1).clamp(min=eps)
    pos_unit = pos / norms.unsqueeze(1)
    cos = (pos_unit @ pole_unit).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def assign_cap_labels(
    trajectory: dict[int, zRegPointCloud],
    pole: Sequence[float],
    theta_deg: float,
    label_inside: int = 2,
    label_outside: int = 1,
    seed: int | None = None,
) -> dict[int, zRegPointCloud]:
    """Assign hard spherical-cap binary labels to every frame in a trajectory.

    Each point is classified by its angular distance from ``pole``. Points
    whose angular distance is strictly less than ``theta_deg`` receive
    ``label_inside``; all others receive ``label_outside``.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    pole : sequence of float
        3-element direction vector defining the cap centre. Need not be
        unit length — normalised internally.
    theta_deg : float
        Half-opening angle of the cap in degrees (strict ``<`` boundary).
    label_inside : int, optional
        Label assigned to points inside the cap. Default: ``2``.
    label_outside : int, optional
        Label assigned to points outside the cap. Default: ``1``.
    seed : int | None, optional
        Seed for ``torch.manual_seed``. Cap assignment is deterministic and
        consumes no RNG; this parameter exists for API symmetry. Default: ``None``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict where every frame's ``pc["label"]`` is a
        ``torch.long`` tensor of shape ``(N,)``.

    Notes
    -----
    The deep-copy contract (D-03) is honoured: the input ``trajectory`` is
    never modified. Empty frames (``N == 0``) produce a ``(0,)`` ``torch.long``
    tensor without special-casing.
    """
    result = copy.deepcopy(trajectory)
    if seed is not None:
        torch.manual_seed(seed)
    for pc in result.values():
        pos = pc["pos"]
        theta = _angular_distance_deg(pos, pole)
        labels = torch.where(theta < theta_deg, label_inside, label_outside)
        pc["label"] = labels.to(torch.long)
    return result


def assign_gaussian_labels(
    trajectory: dict[int, zRegPointCloud],
    pole: Sequence[float],
    sigma_deg: float,
    label_inside: int = 2,
    label_outside: int = 1,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Assign Gaussian-probability Bernoulli labels to every frame in a trajectory.

    For each point, the probability of receiving ``label_inside`` is::

        p = exp(-theta_deg² / (2 · sigma_deg²))

    where ``theta_deg`` is the angular distance from ``pole``. A
    ``torch.bernoulli`` draw then assigns the label: success (1) →
    ``label_inside``, failure (0) → ``label_outside``.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    pole : sequence of float
        3-element direction vector defining the distribution centre. Need not
        be unit length — normalised internally.
    sigma_deg : float
        Width of the Gaussian in degrees. Smaller values produce a sharper
        boundary.
    label_inside : int, optional
        Label assigned on Bernoulli success. Default: ``2``.
    label_outside : int, optional
        Label assigned on Bernoulli failure. Default: ``1``.
    seed : int | None, optional
        Seed for ``torch.manual_seed`` set once at function entry. Same seed
        reproduces identical labels. Default: ``42``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict where every frame's ``pc["label"]`` is a
        ``torch.long`` tensor of shape ``(N,)``.

    Notes
    -----
    Success probability decreases monotonically with angular distance from
    ``pole``. The deep-copy contract (D-03) is honoured. Empty frames produce
    a ``(0,)`` ``torch.long`` tensor without special-casing.
    """
    if sigma_deg <= 0:
        raise ValueError(f"sigma_deg must be > 0, got {sigma_deg}")
    result = copy.deepcopy(trajectory)
    if seed is not None:
        torch.manual_seed(seed)
    for pc in result.values():
        pos = pc["pos"]
        theta = _angular_distance_deg(pos, pole)
        prob = torch.exp(-(theta ** 2) / (2.0 * sigma_deg ** 2))
        draw = torch.bernoulli(prob)
        labels = torch.where(draw.bool(), label_inside, label_outside)
        pc["label"] = labels.to(torch.long)
    return result
