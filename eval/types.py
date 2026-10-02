"""Frozen pydantic v2 result models for the zReg evaluation framework.

This module defines the typed result containers produced and consumed by
the evaluation framework (FRAME-04):

- ``AlignResult``    — output of Phase 19 ``AlignmentStage``
- ``LabelResult``    — output of Phase 20 ``LabelTransferStage``
- ``StageMetrics``   — raw + normalized per-stage metric scores (D-03)
- ``Trial``          — single hyperparameter-search trial (Phase 22)
- ``SearchResult``   — aggregated hyperparameter-search outcome (Phase 22)
- ``EvalReport``     — final per-run report serialised to JSON in Phase 21
- ``TrainingTriple``  — one (source, target) training example for Phase 47's
  learned label-transfer training loop (Phase 46, D-02)
- ``MethodBenchmarkResult`` — per-method/per-dataset-source benchmark outcome
  for Phase 49's label-transfer method comparison (Phase 49, RESEARCH Pattern 6)
- ``BenchmarkReport``       — aggregated multi-method benchmark report (Phase 49,
  RESEARCH Pattern 6)

All models use ``ConfigDict(frozen=True, arbitrary_types_allowed=True)``
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
from zreg.core.dataset import zRegPointCloud
from zreg.algorithms.cpd import EstepResult

# torch AFTER zreg.* imports
import torch

__all__ = [
    "AlignResult",
    "LabelResult",
    "StageMetrics",
    "Trial",
    "SearchResult",
    "EvalReport",
    "TrainingTriple",  # Phase 46 D-02
    "StageResult",  # Phase 19 D-01
    "MethodBenchmarkResult",  # Phase 49
    "BenchmarkReport",  # Phase 49
]


class AlignResult(BaseModel):
    """Output of an alignment stage (Phase 19 ``AlignmentStage.run``).

    Holds the per-frame aligned point clouds plus the DTW warping path,
    distance, change-point count, and the hyperparameters that produced
    this result.  Frozen and arbitrary-type-allowed per D-02.

    Parameters
    ----------
    aligned_cloud : dict[int, zRegPointCloud]
        CPD-transformed (or DTW-resampled) source trajectory keyed by full
        target keys.  When ``cpd_penalty=None``, contains temporally-resampled
        source frames (no spatial registration).  When ``cpd_penalty`` is set,
        each frame is spatially registered to its paired target frame via CPD.
        ``Keys == set(target.keys())``.
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
    estep_results : dict[int, EstepResult]
        CPD E-step posterior (Phase 44, D-09), keyed the same way as
        ``aligned_cloud`` (by target frame key).  Only populated for frames
        that went through CPD registration (``alignment_method="cpd"`` and
        ``cpd_penalty is not None``, D-01/D-03); empty otherwise (icp, swd,
        temporal-only). A read-only diagnostic of registration quality — not
        part of the transform itself.

    Attributes
    ----------
    aligned_cloud : dict[int, zRegPointCloud]
    warp_path : list[tuple[int, int]]
    dtw_distance : float
    n_changepoints : int
    params_used : dict[str, Any]
    estep_results : dict[int, EstepResult]
        CPD E-step posterior keyed by target frame key, empty unless
        ``alignment_method="cpd"`` and ``cpd_penalty is not None``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # CPD-transformed (or DTW-resampled) source trajectory keyed by full target keys.
    # When cpd_penalty=None, contains temporally-resampled source frames (no spatial
    # registration). When cpd_penalty is set, each frame is spatially registered to
    # its paired target frame via CPD. Keys == set(target.keys()).
    aligned_cloud: dict[int, zRegPointCloud]
    warp_path: list[tuple[int, int]]
    dtw_distance: float
    n_changepoints: int
    params_used: dict[str, Any]
    velocity_landmarks: list[int] = Field(default_factory=list)
    # CPD E-step posterior (Phase 44, D-09), keyed the same way as aligned_cloud
    # (by target frame key). Empty unless alignment_method="cpd" and
    # cpd_penalty is not None (D-01/D-03).
    estep_results: dict[int, EstepResult] = Field(default_factory=dict)


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
    pre_transfer_alignment : float
        Mean Chamfer distance between source and target before label transfer.
        Computed per-frame pair using the same sequential pairing as ``run()``.
        Value of 0.0 indicates identical clouds or unchecked (default).

    Attributes
    ----------
    transferred_labels : dict[int, torch.Tensor]
    params_used : dict[str, Any]
    pre_transfer_alignment : float
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    transferred_labels: dict[int, torch.Tensor]
    params_used: dict[str, Any]
    pre_transfer_alignment: float = Field(default=0.0, ge=0.0)


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
    coverage_flags : list[str]
        Human-readable frame-coverage problems found while computing
        chamfer/hausdorff (no shared frame keys, non-dict inputs, partial
        key overlap, skipped degenerate frames); reported by
        ``MetricsEngine.sanity_check``.  Every entry starts with
        ``"frame coverage:"``.  Default empty list.

    Attributes
    ----------
    chamfer_distance : float
    hausdorff_distance : float
    path_smoothness : float
    temporal_stability : float
    f1_score : float
    knn_consistency : float
    normalized : dict[str, float]
    coverage_flags : list[str]
        Human-readable frame-coverage problems found while computing
        chamfer/hausdorff; reported by ``MetricsEngine.sanity_check``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chamfer_distance: float = Field(ge=0)
    hausdorff_distance: float = Field(ge=0)
    path_smoothness: float = Field(ge=0)
    temporal_stability: float = Field(ge=0)
    f1_score: float
    knn_consistency: float
    normalized: dict[str, float] = Field(default_factory=dict)
    coverage_flags: list[str] = Field(default_factory=list)


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
    trajectory_paths : list[str]
        Filesystem paths to CSV and metadata JSON files written by
        eval.tracking.export_trajectory (Phase 24 EXT-01). Default empty list.
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
    trajectory_paths : list[str]
    sanity_flags : list[str]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    params: dict[str, Any]
    metrics: StageMetrics
    aggregated_metrics: dict[str, dict[str, float]]
    per_dataset: dict[str, dict[str, float]]
    plot_paths: list[str] = Field(default_factory=list)
    trajectory_paths: list[str] = Field(default_factory=list)
    sanity_flags: list[str] = Field(default_factory=list)


class TrainingTriple(BaseModel):
    """One (source, target) training example for learned label transfer (Phase 46, D-02).

    Produced by ``DataFactory``'s seed-driven training-triple generation method
    (Phase 46 Plan 03) and consumed by Phase 47's training loop (not built in
    this phase).  Frozen and arbitrary-type-allowed per D-02, matching
    ``AlignResult``/``LabelResult`` conventions exactly.

    Parameters
    ----------
    source_cloud : zRegPointCloud
        Labeled source point cloud — ``pos`` and ``label`` populated, ``id``
        is ``None``.
    target_cloud : zRegPointCloud
        Transformed target point cloud produced from ``source_cloud`` via
        ``DataFactory.generate_target``.  Its ``label`` field is the
        correctly-propagated per-point target label (already threaded through
        every transform's index selection by ``augment()`` — no separate
        correspondence-tracking is needed).
    seed : int
        The seed that produced this triple.  Callers use this to verify
        train/val seed-set membership (D-03) — seeds are partitioned into
        disjoint train/val ranges upstream of this model.

    Attributes
    ----------
    source_cloud : zRegPointCloud
    target_cloud : zRegPointCloud
    seed : int

    Notes
    -----
    **Label vs id discipline (45-DESIGN.md).**  ``source_labels`` and
    ``target_labels`` read exclusively from ``pc["label"]`` — the small,
    fixed-vocabulary categorical class assigned by ``generate_labels``'s
    Voronoi partition.  They NEVER read ``pc["id"]`` (the monotonically
    growing per-cell identity used for ground-truth correspondence
    elsewhere in ``DataFactory``).  Training against ``id`` instead of
    ``label`` would silently grow the effective number of classes over a
    trajectory — see 45-DESIGN.md "label vs id Discipline".
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    source_cloud: zRegPointCloud
    target_cloud: zRegPointCloud
    seed: int

    @property
    def source_labels(self) -> torch.Tensor:
        """Ground-truth source class labels, read from ``pc["label"]`` (never ``pc["id"]``)."""
        return self.source_cloud["label"]

    @property
    def target_labels(self) -> torch.Tensor:
        """Ground-truth target class labels, read from ``pc["label"]`` (never ``pc["id"]``)."""
        return self.target_cloud["label"]


class MethodBenchmarkResult(BaseModel):
    """Per-method, per-dataset-source benchmark outcome (Phase 49, RESEARCH Pattern 6).

    One instance per ``(method, dataset_source)`` combination evaluated by the
    Phase 49 benchmark runner (not built in this plan).  Frozen and
    arbitrary-type-allowed per D-02, matching ``AlignResult``/``LabelResult``
    conventions exactly.

    Parameters
    ----------
    method : str
        Label-transfer method name evaluated, e.g. ``"knn_voting"``,
        ``"cpd_weighted"``, ``"pointnet2"``, or ``"egnn"``.
    dataset_source : str
        Identifier for the dataset this result was computed against, e.g.
        ``"synthetic_holdout"`` or ``"real_shah"`` (49-CONTEXT D-02 item 2 —
        generalization across dataset sources).
    f1_score : float or None
        Weighted F1 score of transferred labels vs ground truth, in
        ``[0, 1]``.  ``None`` means no ground truth was available for this
        dataset source (the real-data dimension — RESEARCH Pitfall 1) or the
        run failed (see ``error``).
    knn_consistency : float or None
        Fraction of k-NN-consistent labels, in ``[0, 1]``.  ``None`` under the
        same conditions as ``f1_score``.
    latency_seconds : float or None
        Wall-clock inference time, amortized per frame pair (RESEARCH
        Pattern 4 — CPU inference-latency benchmarking, 49-CONTEXT D-02
        item 3).  ``None`` if not measured or the run failed.
    n_pairs : int
        Number of source/target frame pairs evaluated to produce this result.
        Defaults to ``0``.
    error : str or None
        Non-``None`` when this ``(method, dataset_source)`` combination
        failed (graceful degradation, 49-CONTEXT D-03) — in that case all
        other metric fields are ``None`` and this field carries a short
        diagnostic message.  ``None`` on success.

    Attributes
    ----------
    method : str
    dataset_source : str
    f1_score : float or None
    knn_consistency : float or None
    latency_seconds : float or None
    n_pairs : int
    error : str or None
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: str
    dataset_source: str
    f1_score: float | None = None
    knn_consistency: float | None = None
    latency_seconds: float | None = None
    n_pairs: int = 0
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Aggregated multi-method benchmark report (Phase 49, RESEARCH Pattern 6).

    Collects one ``MethodBenchmarkResult`` per ``(method, dataset_source)``
    combination evaluated by the Phase 49 benchmark runner (not built in this
    plan), plus the run parameters and free-form notes (e.g. D-03's
    point-count scale-mismatch findings).  Frozen and arbitrary-type-allowed
    per D-02, matching ``AlignResult``/``LabelResult`` conventions exactly.

    Parameters
    ----------
    params : dict[str, Any]
        Run parameters that produced this report.  Heterogeneous values
        (int / float / str / bool) — see module Pitfall 8.  Must stay
        JSON-primitive so ``model_dump()`` round-trips through
        ``json.dump`` (T-49-02).
    results : list[MethodBenchmarkResult]
        One entry per ``(method, dataset_source)`` combination evaluated.
    notes : list[str]
        Free-form diagnostic strings (e.g. D-03 point-count scale-mismatch
        findings).  Default empty list.

    Attributes
    ----------
    params : dict[str, Any]
    results : list[MethodBenchmarkResult]
    notes : list[str]
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    params: dict[str, Any]
    results: list[MethodBenchmarkResult]
    notes: list[str] = Field(default_factory=list)


StageResult: TypeAlias = Union[AlignResult, LabelResult]
"""Union of the two stage-output types (Phase 19 D-01).

Used by ``eval.stages.base.PipelineStage.run`` to type the return value uniformly
across ``AlignmentStage`` (returns ``AlignResult``) and ``LabelTransferStage``
(returns ``LabelResult``).  A TypeAlias declaration is used here for explicit
IDE-friendly annotation and to signal intent to static type-checkers (mypy, pyright).
"""
