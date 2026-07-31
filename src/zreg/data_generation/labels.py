"""Label generation and removal utilities for synthetic trajectories.

Provides a single, config-driven entry point for region-based labelling
(``generate_labels``) and label removal (``remove_labels``). A label is
defined by one or more ``LabelComponentSpec`` region components
(``voronoi``/``blob``/``cone``) combined into a ``LabelSpec`` mixture
(D-11). ``generate_labels`` supports a simple path (``n_labels``:
auto-random Voronoi, drop-in replacement for the old ``n_classes``-based
call) and a full path (``label_specs``: arbitrary shapes/mixtures/label
IDs), with either deterministic (argmax) or probabilistic (sampled)
assignment (D-12). All utilities are immutable — they accept a
``dict[int, zRegPointCloud]`` and return a new dict. The input dict is never
modified (deep-copy contract, D-03). Region/component centers are resolved
exactly once per ``generate_labels`` call, before the per-frame loop, and
reused unchanged across every frame (D-07).

Labels are stored in ``zRegPointCloud["label"]`` as ``torch.long`` tensors of
shape ``(N,)``, one integer label ID per point. This format is directly
compatible with ``zreg.metrics.compute_f1`` without conversion.
"""

import copy
import math
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


def _assign_deterministic(
    scores: torch.Tensor,
    label_ids: torch.Tensor,
) -> torch.Tensor:
    """Assign each point the label with the highest mixture score (argmax).

    Parameters
    ----------
    scores : torch.Tensor
        Shape ``(N, n_labels)``, per-label mixture scores (e.g. the
        ``dim=1``-stacked output of ``_label_scores`` across labels).
    label_ids : torch.Tensor
        Shape ``(n_labels,)``, ``torch.long``. The label ID corresponding
        to each column of ``scores``.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``, ``torch.long``. ``label_ids[scores.argmax(dim=1)]``.
    """
    return label_ids[scores.argmax(dim=1)]


def _assign_probabilistic(
    scores: torch.Tensor,
    label_ids: torch.Tensor,
) -> torch.Tensor:
    """Sample each point's label from a softmax categorical over labels.

    Parameters
    ----------
    scores : torch.Tensor
        Shape ``(N, n_labels)``, per-label mixture scores.
    label_ids : torch.Tensor
        Shape ``(n_labels,)``, ``torch.long``. The label ID corresponding
        to each column of ``scores``.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``, ``torch.long``. Each point's label is sampled from
        ``torch.multinomial(torch.softmax(scores, dim=1), num_samples=1)``,
        then mapped through ``label_ids``.
    """
    probs = torch.softmax(scores, dim=1)
    idx = torch.multinomial(probs, num_samples=1).squeeze(1)
    return label_ids[idx]


def generate_labels(
    trajectory: dict[int, zRegPointCloud],
    n_labels: int | None = None,
    label_specs: list["LabelSpec"] | None = None,
    mode: Literal["deterministic", "probabilistic"] = "deterministic",
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]:
    """Assign config-driven, region-based integer labels to a trajectory.

    Supports two mutually-exclusive paths:

    - **Simple path** (``n_labels``): ``n_labels`` Voronoi centers are drawn
      from ``N(0, I)`` in R³ and each point is assigned to its nearest
      center — a drop-in replacement for the old ``n_classes``-based
      behaviour.
    - **Full path** (``label_specs``): arbitrary ``LabelSpec`` instances
      (each a mixture of ``voronoi``/``blob``/``cone`` region components,
      D-11) with caller-chosen ``label_id`` values.

    In both paths, ``mode`` selects how per-point scores become a label:
    ``"deterministic"`` takes the argmax over labels; ``"probabilistic"``
    samples from a softmax categorical over labels (D-12). Region/component
    centers are resolved exactly **once** per call, before the per-frame
    loop, and reused unchanged across every frame — this is the D-07 fix
    for the previous per-frame center-redraw bug.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory. Never mutated.
    n_labels : int | None, optional
        Number of auto-random Voronoi labels (simple path). Must be
        ``>= 1``. Exactly one of ``n_labels``/``label_specs`` must be
        given. Default: ``None``.
    label_specs : list[LabelSpec] | None, optional
        Explicit label specs (full path). Exactly one of
        ``n_labels``/``label_specs`` must be given. Default: ``None``.
    mode : {"deterministic", "probabilistic"}, optional
        Assignment mode, applied uniformly across all labels (D-12). The
        simple ``n_labels`` path only supports ``"deterministic"`` (D-10 —
        voronoi has no natural probability scale without an explicit
        temperature, which the simple path does not expose). Default:
        ``"deterministic"``.
    seed : int | None, optional
        Seed for ``torch.manual_seed``, set once at function entry.
        ``None`` means the caller controls the RNG state. Default: ``42``.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory dict where every frame's ``pc["label"]`` is a
        ``torch.long`` tensor of shape ``(N,)``.

    Raises
    ------
    ValueError
        If neither or both of ``n_labels``/``label_specs`` are given; if
        ``n_labels`` is given and is less than 1; if ``n_labels`` is given
        together with ``mode="probabilistic"``.

    Notes
    -----
    Region/component centers are resolved once, immediately after
    ``torch.manual_seed(seed)`` and before the per-frame loop, then reused
    unchanged across every frame (D-07). Two frames with identical ``pos``
    therefore produce identical ``pc["label"]`` output.

    Empty frames (``N == 0``) are handled transparently on every path/mode
    combination: per-component/per-label scores are shape ``(0,)``,
    stacking produces ``(0, n_labels)``, and both assignment functions
    return a ``(0,)`` ``torch.long`` tensor without special-casing.
    """
    if (n_labels is None) == (label_specs is None):
        raise ValueError(
            "exactly one of n_labels or label_specs must be provided "
            f"(got n_labels={n_labels!r}, label_specs={label_specs!r})"
        )
    if n_labels is not None and n_labels < 1:
        raise ValueError(f"n_labels must be >= 1, got {n_labels}")
    if n_labels is not None and mode == "probabilistic":
        raise ValueError(
            "the simple n_labels path only supports mode='deterministic'; "
            "pass label_specs with an explicit voronoi temperature for "
            "probabilistic mode"
        )

    result = copy.deepcopy(trajectory)
    if seed is not None:
        torch.manual_seed(seed)

    # D-07 fix: resolve all region centers exactly once, before the
    # per-frame loop, so they are reused unchanged across every frame.
    if n_labels is not None:
        centers = torch.randn(n_labels, 3)
        resolved_specs: list[LabelSpec] = [
            LabelSpec(
                label_id=i,
                components=[
                    LabelComponentSpec(shape="voronoi", center=centers[i].tolist())
                ],
            )
            for i in range(n_labels)
        ]
    else:
        resolved_specs = label_specs

    label_ids_tensor = torch.tensor(
        [spec.label_id for spec in resolved_specs], dtype=torch.long
    )

    for pc in result.values():
        pos = pc["pos"]  # shape (N, 3)
        stacked = torch.stack(
            [_label_scores(pos, spec, mode) for spec in resolved_specs], dim=1
        )  # (N, n_labels)
        if mode == "deterministic":
            labels = _assign_deterministic(stacked, label_ids_tensor)
        else:
            labels = _assign_probabilistic(stacked, label_ids_tensor)
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


def _component_score(
    pos: torch.Tensor,
    component: "LabelComponentSpec",
    mode: str,
) -> torch.Tensor:
    """Return a per-point score for a single region component.

    Higher scores indicate a point is more likely to belong to this
    component. Scores are unnormalised log-space quantities suitable for
    combination via ``torch.logsumexp`` in ``_label_scores``.

    Parameters
    ----------
    pos : torch.Tensor
        Shape ``(N, 3)``, float dtype.
    component : LabelComponentSpec
        The region component to score against.
    mode : str
        ``"deterministic"`` or ``"probabilistic"``. Only affects the
        ``voronoi`` shape (D-10): deterministic mode ignores
        ``temperature``; probabilistic mode requires it.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``mode == "probabilistic"`` and ``component.shape == "voronoi"``
        but ``component.temperature`` is missing or non-positive, or if
        ``component.shape`` is not one of ``"voronoi"``/``"blob"``/``"cone"``.
    """
    if component.shape == "voronoi":
        center_tensor = torch.tensor(component.center, dtype=pos.dtype, device=pos.device)
        dist_sq = (pos - center_tensor).pow(2).sum(dim=1)
        if mode == "probabilistic":
            if component.temperature is None or component.temperature <= 0:
                raise ValueError(
                    "temperature is required for voronoi components in "
                    f"probabilistic mode, got {component.temperature}"
                )
            return -dist_sq / component.temperature
        return -dist_sq
    if component.shape == "blob":
        center_tensor = torch.tensor(component.center, dtype=pos.dtype, device=pos.device)
        dist_sq = (pos - center_tensor).pow(2).sum(dim=1)
        return -dist_sq / (2.0 * component.sigma ** 2)
    if component.shape == "cone":
        theta = _angular_distance_deg(pos, component.center)
        return -(theta ** 2) / (2.0 * component.sigma ** 2)
    raise ValueError(f"unknown shape {component.shape!r}")


def _label_scores(
    pos: torch.Tensor,
    label_spec: "LabelSpec",
    mode: str,
) -> torch.Tensor:
    """Return a per-point mixture score for a label's components.

    Combines each component's ``_component_score`` with its ``weight`` via
    a weighted ``logsumexp`` mixture (D-11):
    ``log(sum(weight_i * exp(score_i))) = logsumexp(score_i + log(weight_i))``.

    Parameters
    ----------
    pos : torch.Tensor
        Shape ``(N, 3)``, float dtype.
    label_spec : LabelSpec
        The label whose components are combined.
    mode : str
        Forwarded to ``_component_score`` (see its docstring).

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``. Empty frames (``N == 0``) are handled transparently:
        each component score is shape ``(0,)``, ``torch.stack`` produces
        ``(0, n_components)``, and ``torch.logsumexp`` produces ``(0,)``.
    """
    per_component = torch.stack(
        [_component_score(pos, c, mode) + math.log(c.weight) for c in label_spec.components],
        dim=1,
    )
    return torch.logsumexp(per_component, dim=1)


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
