---
phase: 18-metricsengine-result-types
plan: 01
subsystem: eval-framework
tags:
  - pydantic
  - result-types
  - eval-framework
  - frame-04
requirements:
  - FRAME-04
dependency_graph:
  requires:
    - "Phase 17: eval/config.py (EvalConfig pattern), eval/data_factory.py (import-order convention)"
    - "Phase 13: zreg.metrics.* (return type knowledge for Plan 18-02)"
    - "Phase 8: zreg.dtw.DTWResult.warping_path: list[tuple[int, int]] (AlignResult.warp_path type source)"
    - "Pre-v1.2: zreg.dataset.zRegPointCloud (AlignResult.aligned_cloud value type)"
  provides:
    - "eval.types: 6 frozen pydantic result models for the eval framework"
    - "tests/test_metrics.py scaffold: 6 test classes (2 populated, 4 stubbed for Plan 18-02)"
  affects:
    - "Plan 18-02: extends EvalConfig with metric_weights, implements MetricsEngine, populates 4 stubbed test classes"
    - "Phase 19: AlignmentStage.run will return AlignResult"
    - "Phase 20: LabelTransferStage.run will return LabelResult"
    - "Phase 21: EvaluationRunner constructs Trial / SearchResult / EvalReport"
tech_stack:
  added: []
  patterns:
    - "pydantic ConfigDict(frozen=True, arbitrary_types_allowed=True) (D-02)"
    - "Field(default_factory=dict) and Field(default_factory=list) for collection defaults"
    - "macOS-ARM import order: zreg.dataset before torch (Phase 12 lesson)"
    - "namespace dir invariant: no eval/__init__.py (Phase 17 D-04)"
key_files:
  created:
    - eval/types.py
    - tests/test_metrics.py
  modified: []
decisions:
  - "All 6 result models use ConfigDict(frozen=True, arbitrary_types_allowed=True) per D-02 (one definition per class, 6 total)"
  - "StageMetrics.normalized uses canonical short-name keys (chamfer, hausdorff, path_smoothness, temporal_stability, f1, knn_consistency) per Pitfall 4; documented in class docstring"
  - "Pitfall 1 (shallow frozen) and Pitfall 2 (torch.Tensor JSON serialisation) documented in module docstring so Plan 18-02 / Phase 21 consumers do not re-discover them"
  - "Trial.metrics typed as StageMetrics (RESEARCH Q5 recommendation); SearchResult.history typed as list[Trial] (RESEARCH Q6 recommendation)"
  - "Test scaffold imports do NOT include eval.metrics — that module is added in Plan 18-02; the 4 stub classes use pytest.skip('populated in Plan 18-02') to keep the suite green"
metrics:
  duration_seconds: 243
  completed_at: "2026-05-28T06:43:53Z"
  tasks: 2
  files_created: 2
  files_modified: 0
  tests_added_active: 4
  tests_added_skipped: 4
---

# Phase 18 Plan 01: MetricsEngine Result Types Summary

Created `eval/types.py` (309 lines) with all 6 frozen pydantic result models — AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport — and scaffolded `tests/test_metrics.py` (190 lines) with TestResultTypesImportable and TestStageMetricsFrozen fully populated plus 4 placeholder classes (TestNormalize, TestComputeScore, TestSanityCheck, TestAggregate) skipped pending Plan 18-02.

## Files Added

| Path | Lines | Purpose |
|------|-------|---------|
| `eval/types.py` | 309 | 6 frozen pydantic v2 result models (FRAME-04) |
| `tests/test_metrics.py` | 190 | Test scaffold: 2 populated + 4 stubbed test classes |

## Files Modified

None.

## Result Models — Field Signatures

### `AlignResult` (Phase 19 output)
| Field | Type |
|-------|------|
| `aligned_cloud` | `dict[int, zRegPointCloud]` |
| `warp_path` | `list[tuple[int, int]]` (matches `DTWResult.warping_path`) |
| `dtw_distance` | `float` |
| `n_changepoints` | `int` |
| `params_used` | `dict[str, Any]` |

### `LabelResult` (Phase 20 output)
| Field | Type |
|-------|------|
| `transferred_labels` | `dict[int, torch.Tensor]` |
| `params_used` | `dict[str, Any]` |

### `StageMetrics` (D-03 — six raw float fields + normalized dict)
| Field | Type | Default |
|-------|------|---------|
| `chamfer_distance` | `float` | required |
| `hausdorff_distance` | `float` | required |
| `path_smoothness` | `float` | required |
| `temporal_stability` | `float` | required |
| `f1_score` | `float` | required |
| `knn_consistency` | `float` | required |
| `normalized` | `dict[str, float]` | `Field(default_factory=dict)` |

### `Trial` (Phase 22 — one HPO trial)
| Field | Type |
|-------|------|
| `params` | `dict[str, Any]` |
| `score` | `float` |
| `metrics` | `StageMetrics` |
| `tier` | `str` |

### `SearchResult` (Phase 22 — aggregated HPO outcome)
| Field | Type |
|-------|------|
| `best_params` | `dict[str, Any]` |
| `best_score` | `float` |
| `history` | `list[Trial]` |
| `tier` | `str` |

### `EvalReport` (Phase 21 — final per-run report)
| Field | Type | Default |
|-------|------|---------|
| `params` | `dict[str, Any]` | required |
| `metrics` | `StageMetrics` | required |
| `aggregated_metrics` | `dict[str, dict[str, float]]` | required |
| `per_dataset` | `dict[str, dict[str, float]]` | required |
| `plot_paths` | `list[str]` | `Field(default_factory=list)` |
| `sanity_flags` | `list[str]` | `Field(default_factory=list)` |

All six models use `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)` per D-02.

## Test Scaffold — Populated vs Stubbed

| Class | Status | Tests | Gate |
|-------|--------|-------|------|
| `TestResultTypesImportable` | Populated (Plan 18-01) | 1 active | FRAME-04 gate 1 |
| `TestStageMetricsFrozen` | Populated (Plan 18-01) | 3 active | D-02 frozen + arbitrary types |
| `TestNormalize` | Stubbed (Plan 18-02) | 1 skip | FRAME-03 / D-05 / D-06 / Pitfall 4 |
| `TestComputeScore` | Stubbed (Plan 18-02) | 1 skip | FRAME-03 / D-07 / D-08 |
| `TestSanityCheck` | Stubbed (Plan 18-02) | 1 skip | FRAME-03 / Pitfall 3 (5 cases) |
| `TestAggregate` | Stubbed (Plan 18-02) | 1 skip | FRAME-03 / Pitfall 7 |

Stubbed classes carry full subtest enumeration in their docstrings so Plan 18-02 has an explicit roadmap.

## Pre / Post-Plan Test Counts

| Snapshot | Passed | Skipped | Notes |
|----------|--------|---------|-------|
| Pre-plan (Phase 17 baseline) | 579 | 17 | per STATE.md |
| Post-plan | 583 | 21 | +4 new active tests, +4 stub skips |

Full suite executed via `python -m pytest tests/ -q` — exits 0.

## Pitfall Encounters

- **Pitfall 1 (shallow frozen)** — Documented in `eval/types.py` module docstring with explicit guidance: consumers must use `model.model_copy(update={"field": new_value})` because pydantic's `frozen=True` does not block mutation of dict/list/tensor contents referenced by an attribute. Verified locally: `sm.chamfer_distance = 0.0` raises `pydantic.ValidationError`; `sm.normalized["x"] = 1.0` would silently succeed (not currently tested — Plan 18-02 may add a regression test).
- **Pitfall 2 (torch.Tensor JSON)** — Documented in module docstring. Phase 18 deliberately does not solve this; it is deferred to Phase 21's `@field_serializer` work for `LabelResult.transferred_labels` and `AlignResult.aligned_cloud`. `StageMetrics`, `Trial`, `SearchResult`, and `EvalReport` JSON-serialise cleanly because their value graphs contain no tensors directly (only via nested AlignResult/LabelResult, which Phase 21 will encode separately).
- **Pitfall 4 (short-name normalized keys)** — `StageMetrics.normalized` documented in its `Attributes` block to use the canonical short-name set so that Plan 18-02 `MetricsEngine.normalize()` and `EvalConfig.metric_weights` share keys for the `compute_score` dot product.
- **Pitfall 5 (`.item()` coercion)** — Deferred to Plan 18-02 where `MetricsEngine.compute_stage_metrics` lives. `eval/types.py` only declares `float` field types; the call-site responsibility for coercing `torch.Tensor` scalars (from `chamfer`, `hausdorff`, `temporal_stability`) sits with the engine, not the types module.
- **Pitfall 6 (`eval/` not pip-discoverable)** — Already known (Phase 17 CR-01, STATE.md). Tests work via `tests/conftest.py:16` `sys.path.insert(0, repo_root)`. Phase 18 does NOT attempt a fix — flagged for pre-Phase-23 attention.

## Deviations from Plan

None — both tasks executed exactly as written. Two trivial environment notes (not deviations):

- `python -c` from the repo root (without `PYTHONPATH=src`) cannot find `zreg` because the project uses `package_dir = =src`; the smoke test in the plan's `<verify><automated>` block was therefore executed via the test runner instead. The 4 active tests in `tests/test_metrics.py` cover the same import and frozen-instance assertions and pass cleanly.
- `ruff` is not installed in the active Python environment, so the `ruff check` gate was substituted with a manual line-length scan (`awk 'length > 120'` returned no matches for either file) and a manual style review. Both files use double quotes throughout per `pyproject.toml:36`.

## Authentication Gates

None — plan was fully autonomous as declared.

## Known Stubs

The 4 stubbed test classes in `tests/test_metrics.py` (TestNormalize, TestComputeScore, TestSanityCheck, TestAggregate) intentionally call `pytest.skip("populated in Plan 18-02")`. These are not bugs — they are explicit hooks for the next plan in the phase and prevent test-collection failures since `eval.metrics` does not yet exist. Resolution is locked into Plan 18-02 (named in plan acceptance criteria and in this summary's `decisions` field).

## Open Hooks for Plan 18-02

1. **Populate 4 stubbed test classes** — TestNormalize (3 tests), TestComputeScore (3 tests), TestSanityCheck (5 tests), TestAggregate (3 tests). Class docstrings enumerate the intended subtests.
2. **Extend `EvalConfig` with `metric_weights: dict[str, float] = Field(default_factory=lambda: {...})`** — default values per D-07, insertion point after `val_split` at `eval/config.py:103`. Existing `TestEvalConfigFromYAML` tests must still pass (no schema-breaking changes).
3. **Create `eval/metrics.py` with `MetricsEngine` class** — `__init__(self, config: EvalConfig)`; public methods `normalize`, `compute_score`, `aggregate`, `sanity_check`; private `compute_stage_metrics` helper that wraps `zreg.metrics.*` and applies Pitfall 5 `.item()` coercion. Import `from eval.types import StageMetrics` (and possibly `AlignResult`, `LabelResult` for `sanity_check`).
4. **Add `eval.metrics` import to `tests/test_metrics.py`** — currently absent (noted in module docstring); Plan 18-02 inserts `from eval.metrics import MetricsEngine` immediately after the `from eval.types import (...)` block, preserving macOS-ARM import order.
5. **Pitfall 1 regression test (optional)** — `sm.normalized["x"] = 1.0` mutation should be explicitly tested as a known caveat; consider adding to TestStageMetricsFrozen with `pytest.warns` or as a documentation-only test.

## Self-Check: PASSED

**Created files verified:**
- FOUND: `eval/types.py` (309 lines)
- FOUND: `tests/test_metrics.py` (190 lines)
- FOUND: `eval/__init__.py` correctly does NOT exist (namespace dir invariant)

**Commits verified:**
- FOUND: `44fa6f6` — feat(18-01): add eval/types.py with 6 frozen pydantic result models
- FOUND: `3f08491` — test(18-01): scaffold tests/test_metrics.py with 2 populated + 4 stubbed classes

**Behavior gates verified:**
- All 6 result types import from `eval.types` (TestResultTypesImportable green)
- `StageMetrics` frozen — attribute assignment raises `pydantic.ValidationError` matching "frozen"
- `AlignResult` accepts `dict[int, zRegPointCloud]`
- `LabelResult` accepts `dict[int, torch.Tensor]`
- `tests/test_metrics.py` collects 8 tests (1 + 3 active + 4 stub placeholders) across 6 classes
- Full suite: 583 passed (+4 vs 579 baseline), 21 skipped (+4 stubs vs 17), 0 failed
