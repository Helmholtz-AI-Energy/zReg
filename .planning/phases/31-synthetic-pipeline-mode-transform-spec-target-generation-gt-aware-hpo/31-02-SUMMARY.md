---
phase: 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
plan: "02"
subsystem: eval-runners
tags: [synthetic-mode, pipeline-mode, hyperparameter-optimization, evaluation-runner, gt-aware-hpo]

requires:
  - "31-01 (DataFactory.generate_target, DataFactory.get_synthetic_ground_truth, EvalConfig.transform_spec)"
provides:
  - "EvaluationRunner.run() synthetic branch calling generate_target()"
  - "_run_single() synthetic GT via get_synthetic_ground_truth()"
  - "HyperparamOptimizer._apply_transform_to_dataset() module-level helper (D-11)"
  - "HyperparamOptimizer._objective() synthetic branches for tier_target and y_true"
affects:
  - "eval/runners/eval_runner.py"
  - "eval/runners/optimizer.py"
  - "tests/test_eval_runner.py"
  - "tests/test_optimizer.py"

tech-stack:
  added: []
  patterns:
    - "pipeline_mode if/else branching in run() and _run_single()"
    - "scratch DataFactory with model_copy for D-11 isolation in _apply_transform_to_dataset()"
    - "_synthetic_target dict-comprehension slicing for tier key membership (Pitfall 7)"
    - "pc['color'] fallback path for sanity toy dataset GT (Pitfall 5)"

key-files:
  created: []
  modified:
    - "eval/runners/eval_runner.py"
    - "eval/runners/optimizer.py"
    - "tests/test_eval_runner.py"
    - "tests/test_optimizer.py"

key-decisions:
  - "Used pipeline_mode check directly in _run_single() rather than passing gt_override parameter — avoids signature change that would break test mocks"
  - "_apply_transform_to_dataset() placed as module-level function (not staticmethod) to be importable for tests"
  - "Added generate_synthetic mock to dev-tier test to ensure tier_dataset is non-empty (data_path does not exist in test, so _tier_dataset falls back to generate_synthetic)"

requirements-completed:
  - MODE-02
  - MODE-03

duration: "~28 minutes"
completed: "2026-06-14"
---

# Phase 31 Plan 02: Synthetic Mode End-to-End Wiring Summary

EvaluationRunner and HyperparamOptimizer wired for `pipeline_mode='synthetic'`: runner calls `generate_target()` and `get_synthetic_ground_truth()`; optimizer uses `_apply_transform_to_dataset()` scratch-factory helper for D-11 isolation and `_synthetic_target`/`get_synthetic_ground_truth()` for dev/full tiers.

**Duration:** ~28 minutes
**Tasks:** 2/2 complete
**Files modified:** 4
**New tests:** 8 (4 in TestEvaluationRunnerSyntheticMode + 4 in TestOptimizerSyntheticMode)
**Full suite:** 868 passed, 18 skipped

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wire synthetic branch in EvaluationRunner.run() and _run_single() | bbc2cff | eval/runners/eval_runner.py, tests/test_eval_runner.py |
| 2 | Add _apply_transform_to_dataset helper and synthetic branches in HyperparamOptimizer._objective() | 2d11ae3 | eval/runners/optimizer.py, tests/test_optimizer.py |

## What Was Built

### Task 1 — EvaluationRunner synthetic mode (eval/runners/eval_runner.py)

1. **`run()` synthetic branch** — Replaced the CR-03 placeholder (`target = source`) with `target = self.factory.generate_target(source, self.config.transform_spec)`. The stale comment block was removed. The paired branch (`load_target()`) is unchanged.

2. **`_run_single()` GT injection** — Replaced the single `gt = self.factory.get_ground_truth(source)` line with a `pipeline_mode` branch:
   - `synthetic`: `gt = self.factory.get_synthetic_ground_truth()`
   - `paired` (else): original `get_ground_truth(source)` call preserved

3. **TestEvaluationRunnerSyntheticMode** (4 tests):
   - `test_synthetic_mode_calls_generate_target`: asserts `generate_target` called once with correct args, `load_target` not called
   - `test_synthetic_mode_calls_get_synthetic_ground_truth`: asserts `get_synthetic_ground_truth` called, `get_ground_truth` not called
   - `test_paired_mode_unchanged`: regression guard — `load_target` called, `generate_target` not called
   - `test_synthetic_mode_run_returns_eval_report`: smoke test — `isinstance(report, EvalReport)`

### Task 2 — HyperparamOptimizer synthetic mode (eval/runners/optimizer.py)

1. **`_apply_transform_to_dataset()` module-level helper** (before `class HyperparamOptimizer`):
   - Strips `"type"` discriminator key from `transform_spec`
   - Constructs scratch `EvalConfig` via `config.model_copy(update={"augmentation_params": augment_params})`
   - Constructs scratch `DataFactory(scratch_cfg)`
   - Returns `scratch_factory.augment(dataset)` — caller's factory state is never touched (D-11)

2. **`_objective()` tier_target selection** — Replaced existing paired-only `if tier_name == "sanity"` block with `pipeline_mode` outer branch:
   - `synthetic/sanity`: `tier_target = _apply_transform_to_dataset(tier_dataset, self.config.transform_spec, self.config)` (D-11, Pitfall 3)
   - `synthetic/dev+full`: `tier_target = {k: self._factory._synthetic_target[k] for k in tier_dataset if k in self._factory._synthetic_target}` (D-10, Pitfall 7)
   - `paired/sanity`: `tier_target = tier_dataset` (preserved)
   - `paired/dev+full`: `tier_target = self._factory.load_target()` (preserved)

3. **`_objective()` y_true / GT key block** — Replaced CR-04 block with `pipeline_mode` outer branch:
   - `synthetic/sanity`: reads `pc["color"]` from toy tier_dataset (Pitfall 5), fallback to `torch.arange(n)` if None
   - `synthetic/dev+full`: `y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]` (D-09)
   - `paired` (else): original CR-04 block preserved verbatim (`gt_key = "id" if sample_pc["id"] is not None else "color"`)

4. **TestOptimizerSyntheticMode** (4 tests):
   - `test_apply_transform_returns_distinct_pos`: at least one frame has different `pos` after noise transform
   - `test_apply_transform_does_not_touch_caller_factory`: outer factory `_synthetic_target` remains `None` after helper call (D-11)
   - `test_sanity_synthetic_mode_does_not_call_main_factory_generate_target`: `mock_factory.generate_target` not called during sanity-tier run (D-11, Pitfall 3)
   - `test_dev_synthetic_mode_uses_synthetic_target_and_get_synthetic_ground_truth`: `get_synthetic_ground_truth.called is True`, `load_target` not called

## Deviations from Plan

### Auto-fixed Issues

**[Rule 1 - Bug] Dev tier test needed generate_synthetic mock**
- **Found during:** Task 2 — Test 4 failing with all trials returning 0.0
- **Issue:** The test's `data_path` points to a non-existent file, so `_tier_dataset("dev")` falls back to `generate_synthetic()`. Without mocking `generate_synthetic`, the mock returns a default `MagicMock` object that iterates as empty, producing an empty `tier_dataset`. This caused the dev tier objective to always fail silently.
- **Fix:** Added `mock_factory.generate_synthetic.return_value = synthetic_dataset` to the dev tier test. This matches the pattern from existing `TestHyperparamOptimizerSanityTier` which also stubs `generate_synthetic`.
- **Files modified:** `tests/test_optimizer.py`
- **Commit:** 2d11ae3

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test setup). **Impact:** None on production code; test reflects correct mock setup for dev tier.

## Verification Results

All acceptance criteria verified:

```
PASS: grep "factory.generate_target(source, self.config.transform_spec)" eval/runners/eval_runner.py → line 192
PASS: grep "factory.get_synthetic_ground_truth()" eval/runners/eval_runner.py → line 329 (inside _run_single)
PASS: grep CR-03 placeholder → 0 matches (removed)
PASS: grep "factory.load_target()" eval/runners/eval_runner.py → line 190 (paired branch preserved)
PASS: grep "factory.get_ground_truth(source)" eval/runners/eval_runner.py → line 331 (paired GT preserved)
PASS: test_eval_runner.py contains TestEvaluationRunnerSyntheticMode with 4 methods
PASS: pytest tests/test_eval_runner.py::TestEvaluationRunnerSyntheticMode → 4 passed
PASS: pytest tests/test_eval_runner.py → 25 passed
PASS: grep "^def _apply_transform_to_dataset" eval/runners/optimizer.py → line 105
PASS: grep "scratch_factory = DataFactory" eval/runners/optimizer.py → line 135
PASS: grep 'self.config.pipeline_mode == "synthetic"' eval/runners/optimizer.py → 2 matches (367, 408)
PASS: grep "self._factory._synthetic_target" eval/runners/optimizer.py → lines 377, 379
PASS: grep "self._factory.get_synthetic_ground_truth()" eval/runners/optimizer.py → line 420
PASS: grep "self._factory.load_target()" eval/runners/optimizer.py → line 385 (paired preserved)
PASS: grep CR-04 gt_key pattern eval/runners/optimizer.py → line 427 (paired preserved)
PASS: test_optimizer.py contains TestOptimizerSyntheticMode with 4 methods
PASS: pytest tests/test_optimizer.py::TestOptimizerSyntheticMode → 4 passed
PASS: pytest tests/test_optimizer.py → 9 passed
PASS: pytest tests/test_eval_runner.py tests/test_optimizer.py tests/test_data_factory.py → 109 passed
PASS: python -c "from eval.runners.optimizer import _apply_transform_to_dataset; print(_apply_transform_to_dataset)" → function object
PASS: pytest tests/ -q → 868 passed, 18 skipped
```

## Known Stubs

None — all synthetic mode branches are fully wired with real behavior. No placeholder comments remain.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced in this plan. The `_apply_transform_to_dataset` helper creates a scratch DataFactory in-process with no I/O side effects. Cross-module `_synthetic_target` access (T-31-04 in PLAN.md threat register) was pre-assessed and accepted.

## Self-Check: PASSED

- eval/runners/eval_runner.py: FOUND
- eval/runners/optimizer.py: FOUND
- tests/test_eval_runner.py (TestEvaluationRunnerSyntheticMode): FOUND
- tests/test_optimizer.py (TestOptimizerSyntheticMode): FOUND
- Commit bbc2cff: FOUND (`git log --oneline -5` confirms)
- Commit 2d11ae3: FOUND (`git log --oneline -5` confirms)
- Full test suite: 868 passed, 18 skipped
