"""Frozen pydantic v2 result models for the zReg evaluation framework.

This module defines the six typed result containers produced and consumed by
the evaluation framework (FRAME-04):

- ``AlignResult``    — output of Phase 19 ``AlignmentStage``
- ``LabelResult``    — output of Phase 20 ``LabelTransferStage``
- ``StageMetrics``   — raw + normalized per-stage metric scores (D-03)
- ``Trial``          — single hyperparameter-search trial (Phase 22)
- ``SearchResult``   — aggregated hyperparameter-search outcome (Phase 22)
- ``EvalReport``     — final per-run report serialised to JSON in Phase 21

All six models use ``ConfigDict(frozen=True, arbitrary_types_allowed=True)``
per D-02.  ``frozen=True`` signals read-only output objects;
``arbitrary_types_allowed=True`` is required because several fields hold
``torch.Tensor`` or ``zRegPointCloud`` (a ``dict`` subclass), neither of
which has a native pydantic validator.

``D-01``: pydantic ``BaseModel`` is preferred over ``@dataclass`` for
consistency with ``EvalConfig`` (Phase 17) and to get ``.model_dump()`` /
``.model_dump_json()`` for free in Phase 21.

Notes
-----
**Pitfall 1 — pydantic ``frozen=True`` is shallow.**  The frozen flag blocks
*attribute reassignment* (``sm.chamfer_distance = 0.0`` raises
``pydantic.ValidationError``) but does NOT prevent mutation of the underlying
dict/list/tensor referenced by an attribute (``sm.normalized["chamfer"] = 0.99``
silently succeeds and corrupts the instance).  Consumers MUST treat normalized,
params_used, aligned_cloud, transferred_labels, etc. as read-only and use
``model.model_copy(update={"field": new_value})`` whenever an "update" is needed.

**Pitfall 2 — ``model_dump_json()`` cannot serialise ``torch.Tensor`` fields
out-of-the-box.**  ``AlignResult.aligned_cloud`` (contains tensors via
``zRegPointCloud``) and ``LabelResult.transferred_labels`` (raw tensors) will
raise ``PydanticSerializationError`` on ``.model_dump_json()`` — only
``.model_dump()`` (dict form) is safe for these models.  Phase 21
``EvalReport`` JSON writer will need ``@field_serializer`` handlers to coerce
tensors to lists; Phase 18 does NOT solve this — only ``StageMetrics``,
``Trial``, ``SearchResult``, and ``EvalReport`` (whose tensor fields are
absent) JSON-serialise cleanly.
"""

from typing import Any, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field

# zreg.dataset MUST precede import torch (libomp SIGABRT lesson from Phase 12;
# enforced in tests/conftest.py:20-24 and eval/data_factory.py:19-35).
from zreg.dataset import zRegPointCloud

# torch AFTER zreg.* imports
import torch

__all__ = [
    "AlignResult",
    "LabelResult",
    "StageMetrics",
    "Trial",
    "SearchResult",
    "EvalReport",
    "StageResult",  # Phase 19 D-01
]


class AlignResult(BaseModel):
    """Output of an alignment stage (Phase 19 ``AlignmentStage.run``).

    Holds the per-frame aligned point clouds plus the DTW warping path,
    distance, change-point count, and the hyperparameters that produced
    this result.  Frozen and arbitrary-type-allowed per D-02.

    Parameters
    ----------
    aligned_cloud : dict[int, zRegPointCloud]
        One aligned point cloud per frame, keyed by integer frame index.
        Mirrors the shape returned by ``DataFactory.load_real()``.
    warp_path : list[tuple[int, int]]
        Optimal DTW alignment path as ``(source_idx, target_idx)`` pairs.
        Matches ``zreg.dtw.DTWResult.warping_path`` exactly.
    dtw_distance : float
        Total DTW accumulated cost at the end of the path.  Matches
        ``zreg.dtw.DTWResult.distance``.
    n_changepoints : int
        Number of change-points detected by the CPD stage of the pipeline.
    params_used : dict[str, Any]
        Hyperparameters that produced this result.  Heterogeneous values
        (int / float / str / bool) — see module Pitfall 8.

    Attributes
    ----------
    aligned_cloud : dict[int, zRegPointCloud]
    warp_path : list[tuple[int, int]]
    dtw_distance : float
    n_changepoints : int
    params_used : dict[str, Any]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    aligned_cloud: dict[int, zRegPointCloud]  # pass-through ref — callers must not mutate dataset after run() returns
    warp_path: list[tuple[int, int]]
    dtw_distance: float
    n_changepoints: int
    params_used: dict[str, Any]


class LabelResult(BaseModel):
    """Output of a label-transfer stage (Phase 20 ``LabelTransferStage.run``).

    Holds per-frame transferred label tensors plus the hyperparameters that
    produced this result.  Frozen and arbitrary-type-allowed per D-02.

    Parameters
    ----------
    transferred_labels : dict[int, torch.Tensor]
        One 1-D label tensor per frame, keyed by integer frame index.
        Mirrors the per-frame shape of ``AlignResult.aligned_cloud``.
    params_used : dict[str, Any]
        Hyperparameters that produced this result.  Heterogeneous values
        (int / float / str / bool) — see module Pitfall 8.

    Attributes
    ----------
    transferred_labels : dict[int, torch.Tensor]
    params_used : dict[str, Any]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    transferred_labels: dict[int, torch.Tensor]
    params_used: dict[str, Any]


class StageMetrics(BaseModel):
    """Per-stage raw + normalized metric scores (D-03).

    Carries the six raw metric values produced by ``MetricsEngine`` plus a
    ``normalized`` dict whose keys are the **short canonical names** used by
    ``EvalConfig.metric_weights`` (Pitfall 4).  Frozen and
    arbitrary-type-allowed per D-02.

    Parameters
    ----------
    chamfer_distance : float
        Chamfer distance between source and aligned target point clouds.
        Lower is better.  Source: ``zreg.metrics.chamfer``.
    hausdorff_distance : float
        95th-percentile Hausdorff distance.  Lower is better.  Source:
        ``zreg.metrics.hausdorff``.
    path_smoothness : float
        Variance of slope changes along the DTW warping path.  Lower is
        better.  Source: ``zreg.metrics.path_smoothness``.
    temporal_stability : float
        Mean Frobenius norm of consecutive transform differences.  Lower
        is better.  Source: ``zreg.metrics.temporal_stability``.
    f1_score : float
        Weighted F1 score of transferred labels vs ground truth, already in
        ``[0, 1]``.  Higher is better.  Source: ``zreg.metrics.compute_f1``.
    knn_consistency : float
        Fraction of k-NN-consistent labels, already in ``[0, 1]``.  Higher
        is better.  Source: ``zreg.metrics.knn_consistency``.
    normalized : dict[str, float]
        Populated by ``MetricsEngine.normalize`` (Plan 18-02).  Default
        empty dict.  Canonical short-name key set (Pitfall 4): ``"chamfer"``,
        ``"hausdorff"``, ``"path_smoothness"``, ``"temporal_stability"``,
        ``"f1"``, ``"knn_consistency"``.  These keys match
        ``EvalConfig.metric_weights`` so ``compute_score`` is a clean dot
        product.

    Attributes
    ----------
    chamfer_distance : float
    hausdorff_distance : float
    path_smoothness : float
    temporal_stability : float
    f1_score : float
    knn_consistency : float
    normalized : dict[str, float]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chamfer_distance: float = Field(ge=0)
    hausdorff_distance: float = Field(ge=0)
    path_smoothness: float = Field(ge=0)
    temporal_stability: float = Field(ge=0)
    f1_score: float
    knn_consistency: float
    normalized: dict[str, float] = Field(default_factory=dict)


class Trial(BaseModel):
    """Single hyperparameter-search trial (Phase 22 ``HyperparamOptimizer``).

    Records the parameters tried, the scalar score produced by
    ``MetricsEngine.compute_score``, the full ``StageMetrics`` from this
    trial, and the search-tier label (sanity / dev / full).  Frozen and
    arbitrary-type-allowed per D-02.

    Parameters
    ----------
    params : dict[str, Any]
        Hyperparameters used in this trial.  Heterogeneous values (int /
        float / str / bool) — see module Pitfall 8.
    score : float
        Scalar score from ``MetricsEngine.compute_score`` (auto-rescaled
        weighted sum of the normalized metrics).  Phase 22 maximises this.
    metrics : StageMetrics
        Full per-stage raw + normalized metric values produced during this
        trial.
    tier : str
        Search-tier label: one of ``"sanity"``, ``"dev"``, ``"full"``.

    Attributes
    ----------
    params : dict[str, Any]
    score : float
    metrics : StageMetrics
    tier : str
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    params: dict[str, Any]
    score: float
    metrics: StageMetrics
    tier: str


class SearchResult(BaseModel):
    """Aggregated outcome of a hyperparameter search (Phase 22).

    Records the best parameters / score discovered, the full per-trial
    history, and the search-tier label.  Frozen and arbitrary-type-allowed
    per D-02.

    Parameters
    ----------
    best_params : dict[str, Any]
        Parameter assignment that maximised ``score`` across ``history``.
        Heterogeneous values (int / float / str / bool) — see module
        Pitfall 8.
    best_score : float
        Maximum ``score`` value across ``history``.
    history : list[Trial]
        All trials in the search, in execution order.
    tier : str
        Search-tier label: one of ``"sanity"``, ``"dev"``, ``"full"``.

    Attributes
    ----------
    best_params : dict[str, Any]
    best_score : float
    history : list[Trial]
    tier : str
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    best_params: dict[str, Any]
    best_score: float
    history: list[Trial]
    tier: str


class EvalReport(BaseModel):
    """Final per-run evaluation report (Phase 21 ``EvaluationRunner`` output).

    Aggregates per-dataset and overall metrics, optional plot paths, and
    sanity-check flags into a single object that Phase 21 serialises to
    ``eval_report.json``.  Frozen and arbitrary-type-allowed per D-02.

    Parameters
    ----------
    params : dict[str, Any]
        Final parameter assignment used for this report.  Heterogeneous
        values (int / float / str / bool) — see module Pitfall 8.
    metrics : StageMetrics
        Headline per-run raw + normalized metric values.
    aggregated_metrics : dict[str, dict[str, float]]
        Per-metric ``{mean, std, min, max}`` summaries produced by
        ``MetricsEngine.aggregate`` (Plan 18-02).
    per_dataset : dict[str, dict[str, float]]
        Per-dataset breakdown of the same summary statistics.
    plot_paths : list[str]
        Filesystem paths to figures written by Phase 21 ``viz`` helpers.
        Default empty list.
    sanity_flags : list[str]
        Warning strings emitted by ``MetricsEngine.sanity_check`` for
        degenerate inputs (Plan 18-02).  Default empty list.

    Attributes
    ----------
    params : dict[str, Any]
    metrics : StageMetrics
    aggregated_metrics : dict[str, dict[str, float]]
    per_dataset : dict[str, dict[str, float]]
    plot_paths : list[str]
    sanity_flags : list[str]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    params: dict[str, Any]
    metrics: StageMetrics
    aggregated_metrics: dict[str, dict[str, float]]
    per_dataset: dict[str, dict[str, float]]
    plot_paths: list[str] = Field(default_factory=list)
    sanity_flags: list[str] = Field(default_factory=list)


StageResult: TypeAlias = Union[AlignResult, LabelResult]
"""Union of the two stage-output types (Phase 19 D-01).

Used by ``eval.stages.base.PipelineStage.run`` to type the return value uniformly
across ``AlignmentStage`` (returns ``AlignResult``) and ``LabelTransferStage``
(returns ``LabelResult``).  A TypeAlias declaration is used here for explicit
IDE-friendly annotation and to signal intent to static type-checkers (mypy, pyright).
"""
