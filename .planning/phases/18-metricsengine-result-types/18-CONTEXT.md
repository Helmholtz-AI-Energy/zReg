# Phase 18: MetricsEngine & Result Types - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `eval/types.py` (six pydantic `BaseModel` result types: `AlignResult`, `LabelResult`, `StageMetrics`, `Trial`, `SearchResult`, `EvalReport`) and `eval/metrics.py` (`MetricsEngine` wrapping all existing `zreg.metrics.*` functions with `normalize`, `aggregate`, `compute_score`, and `sanity_check`). Also extends `eval/config.py` with a `metric_weights` field. Unit tests in `tests/test_metrics.py`.

All downstream phases (19–23) are out of scope for this phase.

</domain>

<decisions>
## Implementation Decisions

### Result Types — eval/types.py
- **D-01:** All 6 result types (`AlignResult`, `LabelResult`, `StageMetrics`, `Trial`, `SearchResult`, `EvalReport`) use **pydantic `BaseModel`** (not stdlib `@dataclass`). Consistent with `EvalConfig`; `.model_dump()` available for JSON serialization in Phase 21 (`EvalReport` → `eval_report.json`).
- **D-02:** All result models use `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)`. Frozen signals read-only output objects; `arbitrary_types_allowed` required for `dict[int, zRegPointCloud]`, `torch.Tensor`, and other non-pydantic types.
- **D-03:** `StageMetrics` carries both raw typed fields **and** a `normalized: dict[str, float]` field. Raw fields: `chamfer_distance: float`, `hausdorff_distance: float`, `path_smoothness: float`, `temporal_stability: float`, `f1_score: float`, `knn_consistency: float`. The `normalized` dict is populated by `MetricsEngine.normalize()` and then stored on the model. Matches FRAME-03 "alignment + label fields + normalized dict".

### Normalization Strategy — MetricsEngine.normalize()
- **D-04:** `normalize(self, metrics: StageMetrics) -> dict[str, float]`. Accepts a `StageMetrics` object (not raw dict) for strong typing.
- **D-05:** Lower-is-better metrics (`chamfer_distance`, `hausdorff_distance`, `path_smoothness`, `temporal_stability`) are normalized via `1 / (1 + x)` — always in `(0, 1]`, single-sample safe, no dataset context required. Works correctly in Phase 22 HPO where each trial produces one `StageMetrics` independently.
- **D-06:** Higher-is-better metrics (`f1_score`, `knn_consistency`) are already in `[0, 1]` and are passed through unchanged (no transformation applied).

### Score Weights — MetricsEngine.compute_score()
- **D-07:** `metric_weights` is a new **`EvalConfig` field** (extending `eval/config.py` from Phase 17). Default weights: `{"chamfer": 0.35, "hausdorff": 0.15, "path_smoothness": 0.10, "temporal_stability": 0.10, "f1": 0.20, "knn_consistency": 0.10}`. Phase 22 optimizer can override weights per experiment via YAML.
- **D-08:** `MetricsEngine.__init__` accepts an `EvalConfig` and stores it. `compute_score(self, metrics: StageMetrics) -> float` reads `self.config.metric_weights`, **auto-normalizes** the weights by dividing each by their sum (`total = sum(w.values())`), then returns `sum(norm[k] * w[k] / total for k in w)`. User can set arbitrary relative weights in YAML without needing them to sum to 1.0 exactly.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FRAME-03 — Full `MetricsEngine` spec: `normalize(dict) → dict`, `aggregate(list[StageMetrics]) → dict`, `compute_score(StageMetrics) → float`, `sanity_check(result) → list[str]`; test gate criteria
- `.planning/REQUIREMENTS.md` §FRAME-04 — Full result type spec: field lists for all 6 dataclasses; `tests/test_metrics.py` gate criteria

### Existing Metrics (must be wrapped, not reimplemented)
- `src/zreg/metrics/__init__.py` — 6 public symbols: `chamfer`, `hausdorff`, `path_smoothness`, `compute_f1`, `knn_consistency`, `temporal_stability`
- `src/zreg/metrics/alignment.py` — `chamfer(source, target, squared)`, `hausdorff(source, target, percentile)`, `path_smoothness(path)` — all lower-is-better
- `src/zreg/metrics/label_transfer.py` — `compute_f1(y_true, y_pred, average, zero_division)`, `knn_consistency(points, labels, k)` — higher-is-better; `temporal_stability(transforms)` — lower-is-better

### EvalConfig (Phase 17 — must be extended)
- `eval/config.py` — Add `metric_weights: dict[str, float]` field with `Field(default_factory=lambda: {...})`. Must not break existing `EvalConfig` tests.

### eval/ Patterns
- `eval/config.py` — Pattern: pydantic `BaseModel`, `ConfigDict(extra="forbid")`, `EvalConfigError(ValueError)` wrapper
- `eval/data_factory.py` — Pattern: `MetricsEngine.__init__(config: EvalConfig)` should mirror `DataFactory.__init__(config: EvalConfig)` structure
- `eval/tracking/__init__.py` — Pattern for importable eval subpackage with `__all__`
- `tests/conftest.py` — `sys.path.insert(0, repo_root)` already present; `from eval.types import ...` and `from eval.metrics import ...` work without changes

### Testing
- `tests/test_eval_metrics.py` — Existing metric unit tests (reference for fixture patterns and assert style)
- `tests/test_metrics.py` — New file this phase; gate criteria per FRAME-04: normalize direction, sanity_check triggers, compute_score output

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `zreg.metrics.chamfer`, `hausdorff`, `path_smoothness` — call directly from `MetricsEngine`; all accept `torch.Tensor` inputs and return scalar tensors or `float`
- `zreg.metrics.compute_f1`, `knn_consistency`, `temporal_stability` — call directly; already return `float` or `torch.Tensor`
- `eval/config.py` `EvalConfig` — extend with `metric_weights` field; `EvalConfigError` already handles `ValidationError` wrapping

### Established Patterns
- `eval/` is a namespace directory — `eval/types.py` and `eval/metrics.py` are standalone modules; no `eval/__init__.py` needed or allowed
- `model_config = ConfigDict(extra="forbid")` — apply to `EvalConfig` extension to reject unknown YAML keys
- NumPy-style docstrings on all public functions and classes — `Parameters`, `Returns`, `Raises`
- `__all__` defined in every new module
- `tests/test_eval_metrics.py` uses `class Test*` pattern with `torch.allclose(..., atol=1e-5)` for float comparisons

### Integration Points
- `MetricsEngine` constructor takes `EvalConfig` — downstream phases (19–21) construct it as `MetricsEngine(config)` where `config` came from `EvalConfig.from_yaml(path)`
- `StageMetrics.normalized` dict is consumed by `MetricsEngine.compute_score()` and `MetricsEngine.aggregate()` — must be populated before `compute_score` is called
- `EvalReport` uses `.model_dump()` in Phase 21 for JSON serialization — `arbitrary_types_allowed=True` must handle any non-serializable fields (e.g., torch.Tensor) with `exclude` or JSON encoders

</code_context>

<specifics>
## Specific Ideas

- Default weights confirmed: `chamfer=0.35, hausdorff=0.15, path_smoothness=0.10, temporal_stability=0.10, f1=0.20, knn_consistency=0.10` (sum=1.0, but auto-normalization makes exact sum irrelevant)
- `compute_score` auto-normalizes weights by their sum — user sets relative weights in YAML without needing them to be exact

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 18-MetricsEngine & Result Types*
*Context gathered: 2026-05-27*
