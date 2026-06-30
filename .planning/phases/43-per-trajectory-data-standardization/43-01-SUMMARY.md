---
phase: 43
plan: "01"
subsystem: data-preprocessing
tags: [data-factory, config, standardization, pydantic, tdd]
requirements-completed: [DATA-02-01, DATA-02-02, DATA-02-03, DATA-02-04, DATA-02-05, DATA-02-06, DATA-02-07]
dependency-graph:
  requires: []
  provides: [DataPreprocessingConfig, DataFactory._standardize]
  affects: [eval/config.py, eval/data_factory.py]
tech-stack:
  added: []
  patterns: [pydantic-sub-model, per-trajectory-stats-cache, eps-numeric-stability]
key-files:
  created:
    - tests/test_data_preprocessing_config.py
  modified:
    - eval/config.py
    - eval/data_factory.py
    - tests/test_data_factory.py
decisions:
  - "DataPreprocessingConfig defaults to method='standardize' — all existing YAML configs inherit z-score silently (D-01)"
  - "EvalConfig.data_preprocessing defaults to live DataPreprocessingConfig() instance, not None, so no YAML change required for adoption"
  - "_preprocessing_stats cache is set only in the stats=None path; target call (paired mode) leaves cache unchanged (D-03)"
  - "eps=1e-8 applied to all three scaling denominator expressions to prevent ZeroDivisionError on constant-pos inputs"
  - "4 existing identity assertions (result is mock_ds) updated to key-equality assertions — _standardize legitimately creates new dicts"
metrics:
  duration_minutes: 12
  tasks_completed: 3
  files_modified: 4
  files_created: 1
  tests_added: 22
  tests_total: 1154
  completed_date: "2026-06-30"
---

# Phase 43 Plan 01: Per-Trajectory Data Standardization Summary

**One-liner:** DataPreprocessingConfig Pydantic sub-model with three scaling methods (z-score, min-max, robust) wired into DataFactory via _standardize() with per-trajectory stats caching and paired-mode reuse.

## What Was Built

**eval/config.py** — `DataPreprocessingConfig` Pydantic sub-model added between `AlignmentPreprocessingConfig` and `EvalConfig`:
- `method: Literal["standardize", "normalize", "robust"] = "standardize"` — z-score default
- `robust_outlier_threshold: float = 3.0` — only consulted for robust method
- `model_config = ConfigDict(extra="forbid")` — typos caught at parse time
- Exported in `__all__`

`EvalConfig.data_preprocessing: DataPreprocessingConfig | None = DataPreprocessingConfig()` added after `alignment_preprocessing`. Default is a live instance (ON by default, D-01). Set to `None` in YAML to opt out (D-02).

**eval/data_factory.py** — `_standardize()` method and wiring:
- `self._preprocessing_stats: dict | None = None` in `__init__`
- `_standardize(dataset, stats=None)` computes six statistics (mean, std, median, iqr, min, max) as shape-[3] tensors; stores in `self._preprocessing_stats` when `stats=None`; uses provided stats when not None (paired mode, D-03)
- Three scaling paths: z-score (`(pos - mean) / (std + eps)`), min-max (`(pos - min) / (max - min + eps)`), robust (`(pos - median) / (iqr + eps)` then clamp)
- `eps = 1e-8` prevents ZeroDivisionError on constant-pos inputs
- Only `pos` is scaled; `label`, `id`, `fps-idx` passed through unchanged
- `load_real()` calls `_standardize(dataset)` after `_subsample_to_max()`
- `load_target()` calls `_standardize(dataset, stats=self._preprocessing_stats)` — reuses source stats in paired mode (D-03), computes own stats in standalone mode (D-05)
- `import logging` added; debug log emitted after stats determination

**tests/test_data_preprocessing_config.py** — 10 Pydantic contract tests (module-level functions, mirrors alignment_preprocessing_config style).

**tests/test_data_factory.py** — `TestDataPreprocessing` class (12 tests) covering all three methods, zero-std stability, paired stats reuse, opt-out passthrough, label/id immutability, and two integration scenarios.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed 4 existing identity assertions broken by _standardize**
- **Found during:** Task 2
- **Issue:** `TestLoadReal.test_dispatches_tracklets`, `TestLoadTarget.test_dispatches_tracklets`, `TestLoadTarget.test_target_format_csv_overrides_source_tracklets`, `TestLoadTarget.test_target_format_tracklets_overrides_source_csv` all asserted `result is mock_ds`. Before this plan, `_subsample_to_max` returned the input dict by reference when `max_points_per_frame=None`. After adding `_standardize`, a new dict is always created (pos is a new tensor). The identity check correctly failed.
- **Fix:** Changed `assert result is mock_ds` to `assert set(result.keys()) == set(mock_ds.keys())` — tests still verify correct dispatch but no longer require object identity
- **Files modified:** tests/test_data_factory.py (4 lines)
- **Commit:** 8e6ef86

## Verification Results

All plan verification checks passed:
- Config smoke: `EvalConfig(data_path='x').data_preprocessing.method == 'standardize'` → OK
- Opt-out: `EvalConfig(data_path='x', data_preprocessing=None).data_preprocessing is None` → OK
- Backward compat: `EvalConfig.from_yaml('configs/alignment_dev.yaml').data_preprocessing is not None` → OK
- `__all__` export: `'DataPreprocessingConfig' in eval.config.__all__` → OK
- New test suites: 22 tests passed
- Full suite: 1154 passed, 18 skipped, 1 xpassed (1132 baseline + 22 new)

## Known Stubs

None — all data paths are fully wired.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- eval/config.py — contains `class DataPreprocessingConfig(BaseModel):`
- eval/data_factory.py — contains `def _standardize` and `self._preprocessing_stats`
- tests/test_data_preprocessing_config.py — 10 tests, all passing
- tests/test_data_factory.py::TestDataPreprocessing — 12 tests, all passing
- Commits: 78ccd2d (Task 1), 8e6ef86 (Task 2), 82ebaef (Task 3)
