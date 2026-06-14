---
phase: 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
verified: 2026-06-14T12:00:00Z
status: passed
score: 14/14
overrides_applied: 0
re_verification: false
---

# Phase 31: Synthetic Pipeline Mode Verification Report

**Phase Goal:** Add `pipeline_mode = "synthetic"` end-to-end: `EvalConfig` gains `transform_spec: dict | None = None`; `DataFactory.generate_target(dataset, transform_spec)` applies the specified transform and returns a distinct target trajectory; `DataFactory.get_synthetic_ground_truth()` returns per-frame cell-identity labels derived from the known deterministic correspondence; `EvaluationRunner.run()` in synthetic mode calls `generate_target()` instead of `load_target()`; `HyperparamOptimizer._objective()` uses GT F1 from `get_synthetic_ground_truth()` as the calibration signal when `pipeline_mode == "synthetic"`; add `configs/synthetic_mode.yaml` scenario config.
**Verified:** 2026-06-14T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EvalConfig accepts `transform_spec: dict \| None = None` field without raising EvalConfigError | VERIFIED | `eval/config.py` line 159: `transform_spec: dict \| None = None`; behavioral spot-check confirms `EvalConfig(data_path="x").transform_spec is None` |
| 2 | EvalConfig.from_yaml() loads YAML containing transform_spec block without error | VERIFIED | `configs/synthetic_mode.yaml` loads cleanly; spot-check `cfg.transform_spec == {"type": "rigid", "rotation_deg": 30.0, "rotation_axis": [0, 0, 1]}` confirmed |
| 3 | DataFactory.generate_target() returns a dict[int, zRegPointCloud] with positions differing from the input | VERIFIED | Method at `eval/data_factory.py` line 175; behavioral spot-check confirms positions differ (`not torch.equal(out[0]["pos"], ds[0]["pos"])` is True) |
| 4 | DataFactory.generate_target() restores self.config.augmentation_params after the call, even if augment() raises | VERIFIED | `try/finally` pattern at lines 245-249; behavioral spot-check confirms config restored to `{"sigma": 0.5}` after `generate_target` with `{"sigma": 0.1}` |
| 5 | DataFactory.generate_target() raises ValueError when transform_spec is None, empty, or type-only | VERIFIED | Guards at lines 229-242; `TestGenerateTarget` tests 6, 7, 8 all pass |
| 6 | DataFactory.get_synthetic_ground_truth() raises RuntimeError when called before generate_target() | VERIFIED | Guard at line 498; behavioral spot-check confirms message contains "generate_target"; `TestGetSyntheticGroundTruth::test_raises_runtime_error_before_generate_target` passes |
| 7 | DataFactory.get_synthetic_ground_truth() returns dict[int, torch.Tensor] with dtype=torch.long for every frame | VERIFIED | Lines 503-508; behavioral spot-check confirms `gt[0].dtype == torch.int64`; id-or-ordinal fallback both enforce `torch.long` |
| 8 | EvaluationRunner.run() in synthetic mode calls factory.generate_target() instead of factory.load_target() | VERIFIED | `eval/runners/eval_runner.py` lines 191-192: `target = self.factory.generate_target(source, self.config.transform_spec)` in `else` branch; CR-03 placeholder removed; `TestEvaluationRunnerSyntheticMode::test_synthetic_mode_calls_generate_target` passes |
| 9 | EvaluationRunner._run_single() in synthetic mode calls factory.get_synthetic_ground_truth() instead of factory.get_ground_truth(source) | VERIFIED | Lines 328-329: `if self.config.pipeline_mode == "synthetic": gt = self.factory.get_synthetic_ground_truth()`; `TestEvaluationRunnerSyntheticMode::test_synthetic_mode_calls_get_synthetic_ground_truth` passes |
| 10 | HyperparamOptimizer._objective() sanity tier in synthetic mode uses _apply_transform_to_dataset() and does NOT call self._factory.generate_target() | VERIFIED | Module-level helper at `eval/runners/optimizer.py` lines 105-136; sanity branch at line 371 calls `_apply_transform_to_dataset()`; `TestOptimizerSyntheticMode::test_sanity_synthetic_mode_does_not_call_main_factory_generate_target` passes |
| 11 | HyperparamOptimizer._objective() dev/full tier in synthetic mode uses self._factory._synthetic_target and self._factory.get_synthetic_ground_truth() | VERIFIED | Lines 376-380: `_synthetic_target` sliced to tier_dataset keys; line 420: `y_true = self._factory.get_synthetic_ground_truth()[...]`; `TestOptimizerSyntheticMode::test_dev_synthetic_mode_uses_synthetic_target_and_get_synthetic_ground_truth` passes |
| 12 | Paired-mode behaviour is unchanged — all existing tests still pass | VERIFIED | Full test suite: 869 passed, 18 skipped; paired branches preserved at optimizer.py lines 382-385 and eval_runner.py line 190 |
| 13 | configs/synthetic_mode.yaml exists with pipeline_mode=synthetic, transform_spec set, target_data_path absent | VERIFIED | File exists; `pipeline_mode: synthetic` at line 10; `transform_spec:` at line 11; no `target_data_path` key; EvalConfig.from_yaml() succeeds |
| 14 | Full test suite passes with 869 tests (839 existing + ~30 new) | VERIFIED | `python -m pytest tests/ -q --tb=no` reports `869 passed, 18 skipped in 41.87s` |

**Score:** 14/14 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/config.py` | `transform_spec: dict \| None = None` field on EvalConfig | VERIFIED | Line 159; also documented in class docstring at line 108 |
| `eval/data_factory.py` | `generate_target()`, `get_synthetic_ground_truth()`, 3 new instance attrs | VERIFIED | Methods at lines 175 and 452; attrs at lines 82-84 in `__init__` |
| `eval/runners/eval_runner.py` | Synthetic branch calling `factory.generate_target()` and `factory.get_synthetic_ground_truth()` | VERIFIED | Lines 191-192 and 328-329; CR-03 placeholder removed |
| `eval/runners/optimizer.py` | `_apply_transform_to_dataset()` module-level helper; `_objective()` branches on `pipeline_mode == "synthetic"` | VERIFIED | Helper at lines 105-136; two `pipeline_mode` branches at lines 367 and 408 |
| `configs/synthetic_mode.yaml` | Exists, `pipeline_mode: synthetic`, `transform_spec` set | VERIFIED | File exists with correct content; loads via EvalConfig.from_yaml() cleanly |
| `tests/test_data_factory.py` | TestEvalConfigTransformSpec (5 tests), TestGenerateTarget (10 tests), TestGetSyntheticGroundTruth (6 tests) | VERIFIED | Classes at lines 139, 300, 573; 21 new test methods confirmed; all pass |
| `tests/test_eval_runner.py` | TestEvaluationRunnerSyntheticMode (4 tests) | VERIFIED | Class at line 577; 4 test methods (lines 588, 616, 644, 664); all pass |
| `tests/test_optimizer.py` | TestOptimizerSyntheticMode (4 tests) | VERIFIED | Class at line 295; 4 test methods (lines 305, 323, 342, 392); all pass |
| `tests/test_cli.py` | test_synthetic_mode_yaml_loads_and_declares_synthetic_mode | VERIFIED | Method at line 313 in TestScenarioConfigs; passes |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `eval/runners/eval_runner.py:run` | `eval/data_factory.py:generate_target` | `self.factory.generate_target(source, self.config.transform_spec)` in synthetic branch | WIRED | Line 192; exact call confirmed |
| `eval/runners/eval_runner.py:_run_single` | `eval/data_factory.py:get_synthetic_ground_truth` | `self.factory.get_synthetic_ground_truth()` in synthetic branch | WIRED | Lines 328-329; exact call confirmed |
| `eval/runners/optimizer.py:_objective sanity branch` | `_apply_transform_to_dataset helper` | Module-level function with scratch DataFactory (D-11 isolation) | WIRED | Line 371-373; `_apply_transform_to_dataset(tier_dataset, self.config.transform_spec, self.config)` |
| `eval/runners/optimizer.py:_objective dev/full branch` | `self._factory._synthetic_target sliced to tier_dataset keys` | Dict comprehension filtering by tier_dataset key membership | WIRED | Lines 376-380 |
| `eval/data_factory.py:generate_target` | `eval/data_factory.py:augment` | `try:` block with temporary override of `self.config.augmentation_params` | WIRED | Lines 245-248; try/finally at lines 245-249 |
| `eval/data_factory.py:get_synthetic_ground_truth` | `self._source_dataset` (set by generate_target) | Iteration over stored source dataset frames at line 504 | WIRED | Lines 503-508 |
| `eval/config.py:transform_spec` | `EvalConfig.from_yaml()` via pydantic | `dict \| None` field with `extra="forbid"` still enforced | WIRED | Line 159; YAML round-trip confirmed by test and spot-check |
| `configs/synthetic_mode.yaml` | `EvalConfig.transform_spec` | YAML `transform_spec:` block → pydantic field | WIRED | `from_yaml()` succeeds; `cfg.transform_spec` is `{"type": "rigid", ...}` |
| `tests/test_cli.py:test_synthetic_mode_yaml_loads_and_declares_synthetic_mode` | `configs/synthetic_mode.yaml` | `EvalConfig.from_yaml(_REPO_ROOT / "configs" / "synthetic_mode.yaml")` | WIRED | Line 313; exact path reference confirmed |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `eval/runners/eval_runner.py:run` | `target` | `factory.generate_target(source, config.transform_spec)` → `augment()` → zreg transforms | Yes — augment() applies real geometric transformations | FLOWING |
| `eval/runners/eval_runner.py:_run_single` | `gt` (dict[int, Tensor]) | `factory.get_synthetic_ground_truth()` → `_source_dataset` with id-or-ordinal fallback | Yes — derives from stored source dataset | FLOWING |
| `eval/runners/optimizer.py:_objective` | `tier_target` (synthetic) | `_apply_transform_to_dataset()` → scratch DataFactory → `augment()` (sanity) or `_factory._synthetic_target` slice (dev/full) | Yes — real transform applied | FLOWING |
| `eval/runners/optimizer.py:_objective` | `y_true` (synthetic) | `factory.get_synthetic_ground_truth()[last_key]` (dev/full) or `pc["color"]` / ordinal (sanity) | Yes — from stored GT | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_apply_transform_to_dataset` importable | `from eval.runners.optimizer import _apply_transform_to_dataset; print(...)` | `<function _apply_transform_to_dataset at ...>` | PASS |
| `synthetic_mode.yaml` loads and asserts structural invariants | `EvalConfig.from_yaml("configs/synthetic_mode.yaml")` with all 7 assertions | "OK — all assertions passed" | PASS |
| `DataFactory` 3 new attrs are None at init | `factory._synthetic_target is None and ...` | True | PASS |
| `get_synthetic_ground_truth()` raises RuntimeError before `generate_target()` | `factory.get_synthetic_ground_truth()` | RuntimeError with "generate_target" in message | PASS |
| `generate_target()` returns positions distinct from input | `not torch.equal(out[0]["pos"], ds[0]["pos"])` | True | PASS |
| GT dtype is torch.long | `gt[0].dtype == torch.long` | True (`torch.int64`) | PASS |
| Config restored after `generate_target` | `factory.config.augmentation_params == {"sigma": 0.5}` after call | True | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MODE-02 | 31-01, 31-02 | Synthetic mode target generation — EvalConfig.transform_spec, DataFactory.generate_target(), EvaluationRunner synthetic branch | SATISFIED | `transform_spec` field in config.py; `generate_target()` in data_factory.py; `run()` synthetic branch in eval_runner.py |
| MODE-03 | 31-01, 31-02, 31-03 | GT-aware HPO in synthetic mode — get_synthetic_ground_truth(), HyperparamOptimizer GT objective, synthetic_mode.yaml | SATISFIED | `get_synthetic_ground_truth()` in data_factory.py; `_objective()` synthetic branches in optimizer.py; `configs/synthetic_mode.yaml` present |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `eval/runners/optimizer.py` | 282 | `# placeholder` comment (in `zero-filled StageMetrics placeholder`) | Info | Informational docstring comment about design decision (Open Question 2 / option a); not a stub — `StageMetrics` is fully initialized with real fields. No impact on phase goal. |

No blocking or warning-level anti-patterns found. The only occurrence of the word "placeholder" is in a code comment describing a documented design decision (PropulateSearch StageMetrics zero-fill per Open Question 2), not a stub implementation.

---

## Human Verification Required

None required. All truths are verifiable programmatically. The full pytest suite confirms functional correctness across 869 tests. Behavioral spot-checks confirm correct runtime behavior.

---

## Gaps Summary

No gaps. All 14 must-haves are VERIFIED.

**Phase 31 goal fully achieved:**

- `EvalConfig.transform_spec: dict | None = None` — field declared and tested
- `DataFactory.generate_target()` — applies augment dispatch via try/finally, stores instance state, validates inputs
- `DataFactory.get_synthetic_ground_truth()` — returns torch.long tensors via id-or-ordinal fallback, guards against pre-call invocation
- `EvaluationRunner.run()` — synthetic branch wired; CR-03 placeholder removed
- `EvaluationRunner._run_single()` — synthetic GT extraction via `get_synthetic_ground_truth()`
- `HyperparamOptimizer._objective()` — `_apply_transform_to_dataset` helper isolates sanity tier; dev/full uses `_synthetic_target` + `get_synthetic_ground_truth()`
- `configs/synthetic_mode.yaml` — rigid rotation scenario config, loads cleanly
- Full pytest suite: **869 passed, 18 skipped** (30 new tests added vs 839 baseline)

---

_Verified: 2026-06-14T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
