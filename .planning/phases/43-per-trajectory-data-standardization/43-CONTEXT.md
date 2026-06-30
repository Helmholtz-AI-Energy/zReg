# Phase 43: Per-Trajectory Data Standardization - Context

**Gathered:** 2026-06-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Add a `DataPreprocessingConfig` Pydantic sub-model to `EvalConfig` and wire three scaling methods (z-score standardize, min-max normalize, robust median+IQR) into `DataFactory.load_real()` and `DataFactory.load_target()`. Standardization is applied per-trajectory (global statistics across all frames combined) to the `pos` field only, post-load and post-subsampling. Target reuses source statistics in paired mode. Standardization is ON by default for all configs, including existing ones.

`AlignmentStage`, `LabelTransferStage`, `EvaluationRunner`, `HyperparamOptimizer`, and all other stages — untouched.

</domain>

<decisions>
## Implementation Decisions

### Default Behavior
- **D-01:** `data_preprocessing` defaults to `DataPreprocessingConfig(method="standardize")` — NOT `None`. This means ALL existing configs (including `configs/*.yaml`) silently inherit z-score standardization after this change. This is intentional; no need to add `data_preprocessing: null` to existing configs.
- **D-02:** Users can opt out by setting `data_preprocessing: null` in their YAML (pydantic `| None` field type), or by selecting a different method.

### Source-Target Coupling (Paired Mode)
- **D-03:** In paired mode, `load_target()` reuses the statistics computed by `load_real()`. Target is NOT standardized with its own independent mean/std — it inherits the source's stats so both trajectories live in the same standardized coordinate space for downstream alignment.
- **D-04:** Statistics are stored as a private attribute `self._preprocessing_stats: dict | None` on `DataFactory` (initialized to `None`). `load_real()` computes and stores them; `load_target()` checks and reuses them. Follows the existing `_real_dataset` / `_target_dataset` caching pattern.
- **D-05:** If `load_target()` is called without a prior `load_real()` call (no cached stats), it computes its own stats. This handles edge cases like target-only evaluation without breaking.

### Scaling Methods
- **D-06:** Three distinct methods via `method` field (Literal enum):
  - `"standardize"` — z-score: subtract per-dimension mean, divide by per-dimension std (default)
  - `"normalize"` — min-max: scale each dimension to [0, 1]
  - `"robust"` — median + IQR: subtract per-dimension median, divide by per-dimension IQR, then clip values outside ±`robust_outlier_threshold` × IQR
- **D-07:** `robust_outlier_threshold: float = 3.0` — only consulted when `method: "robust"`. Silently ignored for other methods (not an error — Pydantic sub-model carries the field regardless of method).
- **D-08:** `robust_outlier_threshold` default is `3.0` (standard robust clip boundary; 3×IQR retains ~99.3% of normal-ish data).

### Per-Trajectory Scope
- **D-09:** "Per-trajectory" = concatenate all frames' `pos` tensors for a single trajectory, compute one mean (and one std/IQR) per XYZ dimension across all concatenated points, then apply those statistics uniformly to every frame in the trajectory. Inter-frame scale relationships are preserved.
- **D-10:** Statistics are computed per XYZ dimension independently (shape: `[3]` vectors for mean, std, IQR).

### Which Fields Are Scaled
- **D-11:** Only `zRegPointCloud["pos"]` (3D spatial coordinates). Fields `"label"`, `"id"`, and `"fps-idx"` are NOT touched — they are categorical IDs and index bookkeeping, not spatial data.

### Claude's Discretion
- Whether `DataPreprocessingConfig` lives in `eval/config.py` alongside `AlignmentPreprocessingConfig` or in a separate module — co-location in `config.py` is consistent with Phase 41 but planner can decide.
- Whether to log the applied preprocessing method and computed stats at `logging.debug` when standardization runs.
- Whether to guard against zero-std / zero-IQR (numeric stability): e.g., add `eps=1e-8` to denominator or raise a `ValueError`. Guard is recommended.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS-v1.4.md` §DATA-02 — Full DATA-02 requirements (DATA-02-01 through DATA-02-07); includes config key names, method list, and acceptance criteria

### Config & Sub-Model Pattern
- `eval/config.py` — `EvalConfig` (Pydantic v2 BaseModel with `extra="forbid"`); `AlignmentPreprocessingConfig` (sub-model pattern to follow for `DataPreprocessingConfig`); add `data_preprocessing: DataPreprocessingConfig | None = DataPreprocessingConfig()` field here
- Note: `AlignmentPreprocessingConfig | None = None` is the Phase 41 precedent — Phase 43 changes the default to a non-None value (`DataPreprocessingConfig(method="standardize")`)

### DataFactory Integration
- `eval/data_factory.py` — `DataFactory`; `load_real()` and `load_target()` (wire standardization here, after `_subsample_to_max`); `_subsample_to_max` (existing post-load hook — same structural position for standardization); private cache attributes `_real_dataset`, `_target_dataset` (pattern for `_preprocessing_stats`)

### Data Model
- `src/zreg/dataset.py` — `zRegPointCloud` class (dict subclass with `"pos"`, `"label"`, `"id"`, `"fps-idx"` keys); `"pos"` is a `torch.Tensor` of shape `[N, 3]` — the only field that gets scaled

### Tests
- `tests/test_data_factory.py` — existing DataFactory unit tests; add `TestDataPreprocessing` class here following existing structure
- `tests/test_config.py` or `tests/test_eval_config.py` — add tests for `DataPreprocessingConfig` field validation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/config.py::AlignmentPreprocessingConfig` — copy the `model_config = ConfigDict(extra="forbid")` pattern; use `Literal["standardize", "normalize", "robust"]` for the `method` field
- `eval/data_factory.py::DataFactory._subsample_to_max` — insert `dataset = self._standardize(dataset)` immediately after `dataset = self._subsample_to_max(dataset)` in both `load_real()` and `load_target()`. New private method `_standardize(dataset, stats=None) -> dict[int, zRegPointCloud]` mirrors `_subsample_to_max`'s shape.
- `eval/data_factory.py` private cache pattern — `self._real_dataset: dict[int, zRegPointCloud] | None = None` → add `self._preprocessing_stats: dict | None = None` in `__init__`

### Established Patterns
- **Sub-model with `extra="forbid"`**: `AlignmentPreprocessingConfig` (Phase 41) sets this; `DataPreprocessingConfig` must follow. Unknown keys in YAML raise validation errors at parse time.
- **Post-load hooks in DataFactory**: `_subsample_to_max` runs after every `load_*` call. Standardization slots into the same position — consistent ordering: load → subsample → standardize.
- **`zreg.*` before `torch` import order**: enforced in `eval/data_factory.py:36-37` (libomp SIGABRT lesson from Phase 12). `DataPreprocessingConfig` is in `eval/config.py` which is already after the `zreg` imports, so no new risk.
- **`__all__` exports**: `eval/config.py` currently exports `["EvalConfig", "EvalConfigError", "AlignmentPreprocessingConfig"]` — add `"DataPreprocessingConfig"`.

### Integration Points
- `eval/config.py` line 16: `__all__` — add `"DataPreprocessingConfig"`
- `eval/config.py` line 230: `alignment_preprocessing: AlignmentPreprocessingConfig | None = None` — add `data_preprocessing` field below this, with a non-None default
- `eval/data_factory.py::DataFactory.__init__`: add `self._preprocessing_stats: dict | None = None`
- `eval/data_factory.py::load_real()` line ~120: after `_subsample_to_max`, call `self._standardize(dataset)` and store stats
- `eval/data_factory.py::load_target()` line ~171: after `_subsample_to_max`, call `self._standardize(dataset, stats=self._preprocessing_stats)` to reuse source stats if available

</code_context>

<specifics>
## Specific Ideas

- `robust_outlier_threshold` default of `3.0` mirrors the `sobol_seed = 42` convention — use a sensible, well-known default rather than an arbitrary one.
- The `_preprocessing_stats` dict should store: `{"mean": tensor[3], "std": tensor[3], "median": tensor[3], "iqr": tensor[3]}` — compute all stats in one pass so switching method doesn't require re-loading.
- Numeric stability guard: if std or IQR is zero for any dimension (constant point cloud), add `eps = 1e-8` to the denominator rather than raising — a point cloud where all points share a coordinate is valid data, not an error.
- `DataPreprocessingConfig` fields: `method: Literal["standardize", "normalize", "robust"] = "standardize"` and `robust_outlier_threshold: float = 3.0`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 43-per-trajectory-data-standardization*
*Context gathered: 2026-06-30*
