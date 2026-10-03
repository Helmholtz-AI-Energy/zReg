"""Label transfer functionality for assigning cell type information between aligned point clouds.

This module provides methods to transfer label/cell type information from a source
point cloud to a target point cloud after spatial alignment.
"""

from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple
import math
import warnings
import torch
import torch.nn as nn
import logging

from .algorithms.cpd import EstepResult

from .core.dataset import zRegPointCloud

if TYPE_CHECKING:
    from .models.egnn import EGNNLabelTransfer
    from .models.pointnet2 import PointNet2LabelTransfer

log = logging.getLogger(__name__)

_PMAT_LAYOUTS = ("receiver_provider", "provider_receiver")

# Phase 62 RD-3 (59-REVIEW IN-09c): a frame in which more than this fraction of
# receiver rows needs the nearest-provider fallback is no longer a
# posterior-weighted transfer and raises instead of being scored as one.
MAX_PMAT_FALLBACK_FRACTION: float = 0.5

__all__ = [
    "transfer_labels",
    "LabelTransferMethod",
    "repair_pmat_rows",
    "PmatRowRepair",
    "MAX_PMAT_FALLBACK_FRACTION",
]


class PmatRowRepair(NamedTuple):
    """Result of :func:`repair_pmat_rows` (one frame, receiver_provider layout).

    Attributes
    ----------
    pmat : torch.Tensor
        Repaired copy, shape ``(n_receiver, n_provider)``: bad rows set to 1.0
        so the row normalisation stays finite.  Their weighted value is a
        placeholder only and must be overwritten from ``fallback_idx``.
    bad_rows : torch.Tensor
        Bool mask, shape ``(n_receiver,)``.
    fallback_idx : torch.Tensor
        Long, shape ``(n_bad,)``, aligned with ``bad_rows.nonzero()``: index of
        the nearest provider point with a finite position, or ``-1`` when the
        receiver position itself is non-finite (no label can be assigned).
    n_bad : int
        Number of bad rows.
    n_nonfinite_pos : int
        Number of bad rows whose receiver position is non-finite.
    message : str
        Human-readable report (empty when ``n_bad == 0``).
    """

    pmat: torch.Tensor
    bad_rows: torch.Tensor
    fallback_idx: torch.Tensor
    n_bad: int
    n_nonfinite_pos: int
    message: str


def repair_pmat_rows(
    pmat: torch.Tensor,
    provider_pos: torch.Tensor,
    receiver_pos: torch.Tensor,
    *,
    max_fallback_fraction: float = MAX_PMAT_FALLBACK_FRACTION,
    context: str = "",
) -> PmatRowRepair:
    """Apply the single zero-row policy for CPD posterior label transfer.

    This is the only place that decides which posterior rows are unusable and
    what replaces them (Phase 62 RD-1..RD-3, from the Phase 59 WR-02 stage
    policy); the library ``cpd_weighted`` path and ``LabelTransferStage`` both
    call it.

    Policy:

    - A receiver row is *bad* when its posterior mass is zero, negative or
      non-finite (e.g. float underflow far from every provider point), or
      when the receiver position is non-finite.
    - Every row bad -> ``ValueError`` ("CPD posterior has zero or non-finite
      mass for every receiver point ... in {context}").
    - ``n_bad / n_receiver > max_fallback_fraction`` (strictly greater) ->
      ``ValueError`` naming ``max_fallback_fraction=<value>`` and the context.
    - Otherwise a bad row with a finite receiver position falls back to its
      nearest provider point (Euclidean ``torch.cdist`` + first minimum, the
      ``nearest_neighbor`` method's metric and tie-break); provider points
      with non-finite positions are never chosen.  A bad row whose receiver
      position is non-finite gets ``-1`` (no label).

    Parameters
    ----------
    pmat : torch.Tensor
        Posterior in receiver_provider layout, shape ``(n_receiver, n_provider)``;
        columns are provider points (= source labels).  Not modified.
    provider_pos : torch.Tensor
        Provider positions, shape ``(n_provider, d)``.
    receiver_pos : torch.Tensor
        Receiver positions, shape ``(n_receiver, d)``.
    max_fallback_fraction : float, optional
        Upper bound in ``(0, 1]`` on the fraction of fallback rows.
        Default :data:`MAX_PMAT_FALLBACK_FRACTION` (0.5).
    context : str, optional
        Location used in messages, e.g. ``"frame 7"``.

    Returns
    -------
    PmatRowRepair
        With zero receivers: an empty repair (``n_bad == 0``, empty message).

    Raises
    ------
    ValueError
        On malformed inputs (pmat not 2-D, coordinate width mismatch, shape
        not ``(n_receiver, n_provider)``, tensors on different devices,
        ``max_fallback_fraction`` outside ``(0, 1]``), when every row is bad,
        when the fallback fraction exceeds the bound, or when a fallback is
        needed but no provider position is finite.
    """
    if pmat.ndim != 2:
        raise ValueError(
            f"pmat must be 2-D (n_receiver, n_provider), got shape {tuple(pmat.shape)}"
        )
    if provider_pos.ndim != 2 or receiver_pos.ndim != 2:
        raise ValueError(
            "provider_pos and receiver_pos must be 2-D (n_points, n_coordinates), got shapes "
            f"{tuple(provider_pos.shape)} and {tuple(receiver_pos.shape)}"
        )
    if provider_pos.shape[1] != receiver_pos.shape[1]:
        raise ValueError(
            "provider_pos and receiver_pos coordinate widths differ: "
            f"{provider_pos.shape[1]} vs {receiver_pos.shape[1]}"
        )
    expected = (receiver_pos.shape[0], provider_pos.shape[0])
    if tuple(pmat.shape) != expected:
        raise ValueError(
            f"pmat shape {tuple(pmat.shape)} does not match expected {expected} "
            "(n_receiver, n_provider)"
        )
    if not (pmat.device == provider_pos.device == receiver_pos.device):
        raise ValueError(
            "pmat, provider_pos and receiver_pos must share one device, got "
            f"{pmat.device}, {provider_pos.device}, {receiver_pos.device}"
        )
    try:
        frac_bound = float(max_fallback_fraction)
    except (TypeError, ValueError):
        raise ValueError(
            f"max_fallback_fraction must be in (0, 1], got {max_fallback_fraction!r}"
        ) from None
    if not (0.0 < frac_bound <= 1.0):
        raise ValueError(f"max_fallback_fraction must be in (0, 1], got {max_fallback_fraction!r}")

    if receiver_pos.shape[0] == 0:
        # Zero-receiver contract: return before the every-row-bad check, which
        # is vacuously true on an empty mask.
        return PmatRowRepair(
            pmat=pmat.clone(),
            bad_rows=torch.zeros(0, dtype=torch.bool, device=pmat.device),
            fallback_idx=torch.zeros(0, dtype=torch.long, device=pmat.device),
            n_bad=0,
            n_nonfinite_pos=0,
            message="",
        )

    n_receiver = receiver_pos.shape[0]
    row_mass = pmat.sum(dim=1)
    nonfinite_pos = ~torch.isfinite(receiver_pos).all(dim=1)
    bad_rows = ~torch.isfinite(row_mass) | (row_mass <= 0) | nonfinite_pos
    n_bad = int(bad_rows.sum().item())
    where = f" in {context}" if context else ""

    if n_bad == n_receiver:
        raise ValueError(
            "method='cpd_weighted': CPD posterior has zero or non-finite mass for every "
            f"receiver point ({n_bad}){where}; cannot weight provider labels "
            "(consider knn_voting or check alignment)"
        )
    if n_bad / n_receiver > frac_bound:
        raise ValueError(
            f"method='cpd_weighted': {n_bad} of {n_receiver} receiver point(s) "
            f"(fraction {n_bad / n_receiver:.3f}) have zero or non-finite posterior mass "
            f"or a non-finite position, above max_fallback_fraction={max_fallback_fraction}"
            f"{where}; a mostly nearest-neighbour-labelled frame is not a cpd_weighted result"
        )

    bad_idx = bad_rows.nonzero(as_tuple=True)[0]
    bad_nonfinite = nonfinite_pos[bad_idx]
    n_nonfinite_pos = int(bad_nonfinite.sum().item())
    fallback_idx = torch.full((n_bad,), -1, dtype=torch.long, device=pmat.device)
    searchable = ~bad_nonfinite
    if bool(searchable.any()):
        provider_finite = torch.isfinite(provider_pos).all(dim=1)
        if not bool(provider_finite.any()):
            raise ValueError(
                "method='cpd_weighted': no provider point has a finite position"
                f"{where}; cannot apply the nearest-neighbour fallback"
            )
        query = receiver_pos[bad_idx[searchable]]
        distances = torch.cdist(query, provider_pos, p=2)
        distances = distances.masked_fill(~provider_finite.unsqueeze(0), float("inf"))
        _, nearest = torch.min(distances, dim=1)
        fallback_idx[searchable] = nearest

    repaired = pmat.clone()
    message = ""
    if n_bad > 0:
        repaired[bad_rows] = 1.0
        message = (
            f"cpd_weighted: {n_bad} of {n_receiver} receiver point(s){where} had zero or "
            "non-finite posterior mass; nearest-neighbour (k=1) fallback"
        )
        if n_nonfinite_pos > 0:
            message += f"; {n_nonfinite_pos} with non-finite position left unlabelled (-1)"

    return PmatRowRepair(
        pmat=repaired,
        bad_rows=bad_rows,
        fallback_idx=fallback_idx,
        n_bad=n_bad,
        n_nonfinite_pos=n_nonfinite_pos,
        message=message,
    )


def _weighted_colours_from_repair(repair: PmatRowRepair, source_colors: torch.Tensor) -> torch.Tensor:
    """Posterior-weighted source colours with the repair's bad rows overwritten.

    Good rows: row-normalised ``repair.pmat @ source_colors``.  Bad rows: the
    fallback provider's colour row, or an all-NaN row ("no label") when the
    fallback index is ``-1`` (62-REVIEW WR-03: an all-zero row would argmax
    to class 0 and be indistinguishable from a real label).  1-D ``source_colors`` (one label channel,
    e.g. ``zRegPointCloud['label']``) give a 1-D result (62-REVIEW WR-01).
    """
    squeeze = source_colors.ndim == 1
    cols = source_colors.unsqueeze(1) if squeeze else source_colors
    prob_matrix = repair.pmat / repair.pmat.sum(dim=1, keepdim=True)
    # 62-REVIEW WR-02: colours follow the posterior's float dtype (float64 pmat).
    dtype = prob_matrix.dtype if prob_matrix.is_floating_point() else torch.float32
    transferred = torch.matmul(prob_matrix.to(dtype), cols.to(dtype))
    if repair.n_bad > 0:
        idx = repair.fallback_idx
        fill = cols[idx.clamp(min=0)].to(dtype)
        fill = torch.where((idx >= 0).unsqueeze(1), fill, torch.full_like(fill, float("nan")))
        transferred[repair.bad_rows] = fill
    return transferred[:, 0] if squeeze else transferred


class LabelTransferMethod(Enum):
    """Enumeration of available label transfer methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CPD_WEIGHTED = "cpd_weighted"
    KNN_VOTING = "knn_voting"
    GAUSSIAN_KERNEL = "gaussian_kernel"
    EGNN = "egnn"
    POINTNET2 = "pointnet2"


def transfer_labels(
    source: zRegPointCloud | torch.Tensor,
    target: zRegPointCloud | torch.Tensor,
    method: str | LabelTransferMethod = LabelTransferMethod.NEAREST_NEIGHBOR,
    source_colors: torch.Tensor | None = None,
    target_colors: torch.Tensor | None = None,
    estep_result: EstepResult | None = None,
    model: "nn.Module | None" = None,
    pmat_layout: Literal["receiver_provider", "provider_receiver"] | None = None,
    max_fallback_fraction: float = MAX_PMAT_FALLBACK_FRACTION,
    pmat_repair: PmatRowRepair | None = None,
    **kwargs
) -> torch.Tensor:
    """Transfer labels from source to target point cloud.

    Parameters
    ----------
    source : zRegPointCloud or torch.Tensor
        Source point cloud data. If zRegPointCloud, uses 'pos' and 'label' fields.
        If torch.Tensor, should be positions of shape (n_points, n_dims).
    target : zRegPointCloud or torch.Tensor
        Target point cloud data. If zRegPointCloud, uses 'pos' field.
        If torch.Tensor, should be positions of shape (m_points, n_dims).
    method : str, optional
        Label transfer method. Options: 'nearest_neighbor', 'cpd_weighted', 'knn_voting', 'gaussian_kernel'.
        Default: 'nearest_neighbor'.
    source_colors : torch.Tensor, optional
        Source labels of shape (n_points, n_label_channels). Required if source
        is torch.Tensor. If source is a zRegPointCloud, an explicit
        ``source_colors`` wins; otherwise ``source['label']`` is used, and a
        ``ValueError`` is raised when neither is available.  For 'egnn' and
        'pointnet2' it may also be 1-D integer class ids ``(n_points,)``;
        2-D labels must then have width ``model.n_classes``.
    target_colors : torch.Tensor, optional
        Target labels of shape (m_points, n_label_channels). Not used in current methods.
    estep_result : EstepResult, optional
        Result from CPD E-step containing posterior probabilities. Required for
        'cpd_weighted' method.
    pmat_layout : {"receiver_provider", "provider_receiver"}, optional
        Orientation of ``estep_result.pmat``; required for 'cpd_weighted' and
        never inferred from the shape (a square pmat would be ambiguous).
        ``"receiver_provider"``: pmat shaped ``(n_target, n_source)``, used as is.
        ``"provider_receiver"``: pmat shaped ``(n_source, n_target)`` -- the raw
        ``expectation_step`` output when the provider (source) is the CPD
        moving set -- transposed internally.
    max_fallback_fraction : float, optional
        'cpd_weighted' only.  Bound in ``(0, 1]`` on the fraction of receiver
        rows that may use the nearest-provider fallback; default
        :data:`MAX_PMAT_FALLBACK_FRACTION` (0.5).  See :func:`repair_pmat_rows`.
    pmat_repair : PmatRowRepair, optional
        'cpd_weighted' only, mutually exclusive with ``estep_result``.  Pass it
        when you already ran :func:`repair_pmat_rows` (receiver_provider
        layout) and reported its message; the library then neither repairs
        again nor warns.
    model : nn.Module, optional
        Pre-loaded EGNNLabelTransfer or PointNet2LabelTransfer instance.
        Required for method="egnn" or method="pointnet2".
    **kwargs
        Additional method-specific parameters. For 'knn_voting': 'k' (int, default 5).
        For 'gaussian_kernel': 'sigma' (float, default 1.0).

    Returns
    -------
    torch.Tensor
        Transferred labels for target point cloud of shape (m_points, n_label_channels).

    Raises
    ------
    ValueError
        If method is not supported or required parameters are missing.

    Notes
    -----
    **cpd_weighted zero-row policy** (one policy for every pmat consumer, see
    :func:`repair_pmat_rows`): a receiver row with zero or non-finite
    posterior mass takes the colour row of its nearest provider point and a
    single ``RuntimeWarning`` is emitted; an all-NaN output row means no
    label could be assigned (non-finite receiver position).  The call raises
    when every row is bad or more than ``max_fallback_fraction`` of the rows
    fall back.  NaN rows appear only for non-finite receiver positions; a
    zero-mass posterior row never produces NaN.  ``argmax`` on the raw
    output is unsafe for such rows (``torch.argmax`` of an all-NaN row is
    0): mask them with ``torch.isnan(out).any(dim=-1)`` first.
    """
    # Extract positions and labels
    if isinstance(source, zRegPointCloud):
        source_pos = source["pos"]
        # U6-5: an explicit source_colors argument wins over source['label'].
        if source_colors is None:
            source_colors = source.get("label")
        if source_colors is None:
            raise ValueError("source labels missing: pass source_colors or provide source['label']")
    else:
        source_pos = source
        if source_colors is None:
            raise ValueError("source_colors must be provided when source is torch.Tensor")

    if isinstance(target, zRegPointCloud):
        target_pos = target["pos"]
    else:
        target_pos = target

    # Validate inputs
    if source_pos.shape[0] == 0:
        raise ValueError("source has 0 points — cannot transfer labels from an empty point cloud")

    if source_pos.shape[1] != target_pos.shape[1]:
        raise ValueError(f"Source and target must have same dimensionality: {source_pos.shape[1]} vs {target_pos.shape[1]}")

    # Normalize method to enum (accepts both string and enum)
    if isinstance(method, str):
        try:
            method = LabelTransferMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown label transfer method: '{method}'. "
                f"Valid options: {[m.value for m in LabelTransferMethod]}"
            )

    if method == LabelTransferMethod.NEAREST_NEIGHBOR:
        return _transfer_labels_nearest_neighbor(source_pos, target_pos, source_colors)
    elif method == LabelTransferMethod.CPD_WEIGHTED:
        return _transfer_labels_cpd_weighted(
            source_pos, target_pos, source_colors, estep_result, pmat_layout,
            max_fallback_fraction=max_fallback_fraction, pmat_repair=pmat_repair,
        )
    elif method == LabelTransferMethod.KNN_VOTING:
        k = kwargs.get('k', 5)
        return _transfer_labels_knn_voting(source_pos, target_pos, source_colors, k)
    elif method == LabelTransferMethod.GAUSSIAN_KERNEL:
        sigma = kwargs.get('sigma', 1.0)
        return _transfer_labels_gaussian_kernel(source_pos, target_pos, source_colors, sigma)
    elif method == LabelTransferMethod.EGNN:
        if model is None:
            raise ValueError("model is required for EGNN method")
        return _transfer_labels_model(source_pos, target_pos, source_colors, model)
    elif method == LabelTransferMethod.POINTNET2:
        if model is None:
            raise ValueError("model is required for POINTNET2 method")
        return _transfer_labels_model(source_pos, target_pos, source_colors, model)
    else:  # pragma: no cover
        # Should not reach here if normalization above is correct, but guard anyway
        raise ValueError(
            f"Unknown label transfer method: '{method}'. "
            f"Valid options: {[m.value for m in LabelTransferMethod]}"
        )


def _transfer_labels_nearest_neighbor(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor
) -> torch.Tensor:
    """Transfer labels using nearest neighbor assignment.

    For each target point, finds the closest source point and copies its label.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).
    """
    log.debug(f"Transferring labels using nearest neighbor: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find nearest neighbors
    _, nearest_indices = torch.min(distances, dim=1)  # (m_points,)

    # Assign labels
    transferred_colors = source_colors[nearest_indices]  # (m_points, n_label_channels)

    return transferred_colors


def _transfer_labels_cpd_weighted(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    estep_result: "EstepResult | None",
    pmat_layout: str | None = None,
    max_fallback_fraction: float = MAX_PMAT_FALLBACK_FRACTION,
    pmat_repair: PmatRowRepair | None = None,
) -> torch.Tensor:
    """Transfer labels using CPD posterior probabilities.

    For each target point, computes weighted average of source labels
    based on correspondence probabilities from CPD.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    estep_result : EstepResult or None
        Result from CPD E-step containing posterior probabilities.  Exactly
        one of ``estep_result`` and ``pmat_repair`` must be given.
    pmat_layout : {"receiver_provider", "provider_receiver"}
        Declared orientation of ``estep_result.pmat`` (see ``transfer_labels``).
        The exact expected shape of the declared layout is checked; the
        orientation is never inferred from the shape.
    max_fallback_fraction : float
        Passed to :func:`repair_pmat_rows`.
    pmat_repair : PmatRowRepair or None
        Precomputed repair (receiver_provider layout).  Used as is: no second
        repair and no warning (the caller owns reporting, RD-1b).

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).  Bad rows
        hold the nearest provider's colour row, or NaN ("no label assigned")
        for a non-finite receiver position (62-REVIEW WR-03).

    Raises
    ------
    ValueError
        If ``pmat_layout`` is missing or unknown, the pmat shape does not
        match the declared layout, ``source_colors`` rows do not match the
        source points, both or neither of ``estep_result``/``pmat_repair`` are
        given, or :func:`repair_pmat_rows` rejects the frame.
    """
    log.debug(f"Transferring labels using CPD weights: {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    n_target = target_pos.shape[0]
    n_source = source_pos.shape[0]
    if source_colors.shape[0] != n_source:
        raise ValueError(
            f"source_colors has {source_colors.shape[0]} rows but the source has "
            f"{n_source} points (pmat columns are source labels)"
        )

    if pmat_repair is not None:
        if estep_result is not None:
            raise ValueError(
                "pass either estep_result (with pmat_layout) or pmat_repair for "
                "cpd_weighted, not both"
            )
        if tuple(pmat_repair.pmat.shape) != (n_target, n_source):
            raise ValueError(
                f"pmat_repair.pmat has shape {tuple(pmat_repair.pmat.shape)} but expected "
                f"(n_target={n_target}, n_source={n_source})"
            )
        if tuple(pmat_repair.bad_rows.shape) != (n_target,):
            raise ValueError(
                f"pmat_repair.bad_rows has shape {tuple(pmat_repair.bad_rows.shape)} but "
                f"expected ({n_target},)"
            )
        # RD-1b: the caller already repaired and reported this frame.
        return _weighted_colours_from_repair(pmat_repair, source_colors)

    if estep_result is None:
        raise ValueError(
            "estep_result is required for CPD-weighted method "
            "(or pass a precomputed pmat_repair)"
        )
    if pmat_layout is None:
        raise ValueError(
            "pmat_layout is required for cpd_weighted: pass 'receiver_provider' "
            "(pmat shaped (n_target, n_source)) or 'provider_receiver' "
            "(pmat shaped (n_source, n_target), the raw expectation_step output)"
        )
    if pmat_layout not in _PMAT_LAYOUTS:
        raise ValueError(
            f"unknown pmat_layout {pmat_layout!r}; expected one of {_PMAT_LAYOUTS}"
        )

    pmat = estep_result.pmat
    expected = (n_target, n_source) if pmat_layout == "receiver_provider" else (n_source, n_target)
    if tuple(pmat.shape) != expected:
        raise ValueError(
            f"pmat has shape {tuple(pmat.shape)} but pmat_layout={pmat_layout!r} expected "
            f"{expected} (n_target={n_target}, n_source={n_source})"
        )
    # Orientation is declared, never inferred (U6-9): a square pmat is ambiguous.
    prob_matrix = pmat if pmat_layout == "receiver_provider" else pmat.T

    # Phase 62 RD-1..RD-3: the shared zero-row policy (never a NaN row).
    repair = repair_pmat_rows(
        prob_matrix, source_pos, target_pos,
        max_fallback_fraction=max_fallback_fraction, context="cpd_weighted pmat",
    )
    if repair.n_bad > 0:
        warnings.warn(repair.message, RuntimeWarning, stacklevel=3)
    return _weighted_colours_from_repair(repair, source_colors)


def _transfer_labels_knn_voting(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    k: int
) -> torch.Tensor:
    """Transfer labels using K-nearest neighbors with majority voting.

    For each target point, finds K nearest source points and assigns the most
    common label among them.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).

    Raises
    ------
    ValueError
        If n_label_channels != 1 (labels must be class indices).
    """
    log.debug(f"Transferring labels using KNN voting (k={k}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    if source_colors.shape[1] != 1:
        raise ValueError("KNN voting assumes single-channel labels (class indices)")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Find K nearest neighbors
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)  # (m_points, k)

    # Get labels for nearest neighbors
    selected_colors = source_colors[indices].squeeze(-1)  # (m_points, k)

    # Compute mode for each target point
    transferred_colors = torch.mode(selected_colors, dim=1).values.unsqueeze(-1)  # (m_points, 1)

    return transferred_colors


def _transfer_labels_gaussian_kernel(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """Transfer labels using Gaussian kernel interpolation.

    Treats labels as a continuous field and interpolates using Gaussian kernels
    centered at source points.

    Parameters
    ----------
    source_pos : torch.Tensor
        Source positions of shape (n_points, n_dims).
    target_pos : torch.Tensor
        Target positions of shape (m_points, n_dims).
    source_colors : torch.Tensor
        Source labels of shape (n_points, n_label_channels).
    sigma : float
        Standard deviation of the Gaussian kernel; must be finite and > 0.

    Returns
    -------
    torch.Tensor
        Transferred labels of shape (m_points, n_label_channels).

    Notes
    -----
    The weights are ``softmax(-d^2 / (2 sigma^2))`` over source points, which
    equals ``exp(...) / sum(exp(...))`` but cannot underflow to ``0/0`` when
    every source point is far from a target point (U6-3): the nearest source
    then dominates instead of the row becoming NaN.
    """
    try:
        sigma_f = float(sigma)
    except (TypeError, ValueError):
        raise ValueError(f"sigma must be a finite float > 0; got {sigma!r}") from None
    if not math.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError(f"sigma must be a finite float > 0; got {sigma!r}")
    log.debug(f"Transferring labels using Gaussian kernel (sigma={sigma}): {source_pos.shape[0]} -> {target_pos.shape[0]} points")

    # Compute pairwise distances
    distances = torch.cdist(target_pos, source_pos, p=2)  # (m_points, n_points)

    # Gaussian weights normalised per target point; softmax is the numerically
    # stable form of exp(.) / sum(exp(.)) (U6-3).
    weights = torch.softmax(-distances**2 / (2 * sigma_f**2), dim=1)  # (m_points, n_points)

    # Compute weighted average of labels
    transferred_colors = torch.matmul(weights, source_colors.float())  # (m_points, n_label_channels)

    return transferred_colors


def _transfer_labels_model(
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    source_colors: torch.Tensor,
    model: "nn.Module",
) -> torch.Tensor:
    """Transfer labels using a pre-trained neural network (EGNN or PointNet2).

    Builds a joint source+target cloud, encodes source labels as one-hot features
    with unknown_flag=0, encodes target points as unknown_flag=1, and runs
    model.forward(). Returns softmax probabilities for target points only.

    Parameters
    ----------
    source_pos, target_pos : torch.Tensor
        Positions, shapes ``(n_source, d)`` and ``(n_target, d)``.
    source_colors : torch.Tensor
        Either 1-D integer class ids of shape ``(n_source,)`` with values in
        ``[0, model.n_classes)`` (one-hot encoded here), or 2-D one-hot/soft
        labels of shape ``(n_source, model.n_classes)``.  A 2-D tensor of any
        other width is rejected (a ``(n, 1)`` column is not read as class ids).
    model : nn.Module
        Model exposing ``n_classes``; the feature width is taken from the
        model, never from the data (U6-1).

    Returns
    -------
    torch.Tensor
        Softmax probabilities, shape ``(n_target, model.n_classes)``.

    Raises
    ------
    ValueError
        If the model has no ``n_classes`` attribute, or the labels do not fit
        ``model.n_classes`` (wrong width, non-integer or out-of-range ids).
    """
    if not hasattr(model, "n_classes"):
        raise ValueError(
            f"model {type(model).__name__} has no n_classes attribute; label-transfer "
            "models must expose n_classes"
        )
    n_classes = int(model.n_classes)
    n_source = source_pos.shape[0]
    device = source_pos.device

    if source_colors.ndim == 1:
        ids = source_colors
        if ids.dtype.is_floating_point:
            if not bool(torch.isfinite(ids).all()) or not bool((ids == ids.round()).all()):
                raise ValueError(
                    "1-D source labels must be integer class ids in [0, n_classes) "
                    f"with model.n_classes={n_classes}"
                )
        if ids.numel() > 0 and (int(ids.min()) < 0 or int(ids.max()) >= n_classes):
            raise ValueError(
                f"source label ids span [{int(ids.min())}, {int(ids.max())}] but "
                f"model.n_classes={n_classes} requires values in [0, {n_classes})"
            )
        source_feat = torch.nn.functional.one_hot(ids.long(), num_classes=n_classes).float()
    elif source_colors.ndim == 2 and tuple(source_colors.shape) == (n_source, n_classes):
        source_feat = source_colors.float()
    else:
        raise ValueError(
            f"source labels shape {tuple(source_colors.shape)} incompatible with "
            f"model.n_classes={n_classes}: expected 1-D class ids ({n_source},) or "
            f"({n_source}, {n_classes})"
        )
    if source_feat.shape[0] != n_source:
        raise ValueError(
            f"source labels have {source_feat.shape[0]} rows but the source has {n_source} points"
        )

    joint_pos = torch.cat([source_pos, target_pos], dim=0)

    # joint_feat: [n_joint, n_classes + 1]
    # source rows: one-hot / soft labels + unknown_flag=0
    # target rows: zeros + unknown_flag=1
    joint_feat = torch.zeros(
        joint_pos.shape[0], n_classes + 1,
        dtype=torch.float32, device=device,
    )
    joint_feat[:n_source, :n_classes] = source_feat.to(device)
    joint_feat[n_source:, n_classes] = 1.0

    model.eval()
    with torch.no_grad():
        logits = model(joint_pos, joint_feat)  # [n_joint, n_classes]

    target_logits = logits[n_source:]
    return torch.softmax(target_logits, dim=-1)
