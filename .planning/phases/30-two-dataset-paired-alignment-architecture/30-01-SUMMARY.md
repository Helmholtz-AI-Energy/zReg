---
phase: 30-two-dataset-paired-alignment-architecture
plan: "01"
subsystem: eval-framework
tags: [paired-alignment, eval-config, data-factory, pipeline-stage, dtw, label-transfer]
dependency_graph:
  requires: []
  provides: [EvalConfig.pipeline_mode, EvalConfig.target_data_path, DataFactory.load_target, PipelineStage.run(source,target,params), AlignmentStage.run(source,target,params), LabelTransferStage.run(source,target,params)]
  affects: [eval/runners/eval_runner.py, tests/test_eval_runner.py, tests/test_optimizer.py, tests/test_trajectory_export.py]
tech_stack:
  added: []
  patterns: [pydantic-Literal-validation, lazy-cache-mirror, two-input-stage-signature, sequential-cross-trajectory-pairing]
key_files:
  created: []
  modified:
    - eval/config.py
    - eval/data_factory.py
    - eval/stages/base.py
    - eval/stages/alignment.py
    - eval/stages/label_transfer.py
    - tests/test_data_factory.py
    - tests/test_alignment_stage.py
    - tests/test_label_transfer_stage.py
decisions:
  - "D-05: EvalConfigError raised at load_target() call time, not at EvalConfig construction — consistent with existing error-at-use-time convention"
  - "D-06: AlignResult.aligned_cloud = source pass-through; target consumed by DTW but not returned"
  - "D-02: Frame-0 pass-through deleted — sequential source[k]->target[k] pairing replaces it (Phase 30 contract)"
  - "Rule 1 fix: test_run_with_distinct_source_target_returns_align_result used torch.equal() instead of dict != dict to avoid RuntimeError on ambiguous tensor boolean comparison"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-12"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 8
  tests_before: 826
  tests_after: 851
  tests_added: 9
  tests_deleted: 1
---

# Phase 30 Plan 01: Contract Layer — EvalConfig + DataFactory + Stage Signatures Summary

Two-input stage contract layer landed atomically: `EvalConfig` gains `pipeline_mode` and `target_data_path`; `DataFactory` gains `load_target()`; all three stage classes (`PipelineStage`, `AlignmentStage`, `LabelTransferStage`) change to `run(source, target, params)`; test suites updated and green.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend EvalConfig + DataFactory; update DataFactory tests | 977267b | eval/config.py, eval/data_factory.py, tests/test_data_factory.py |
| 2 | Change PipelineStage ABC + stages to run(source, target, params); update stage tests | 45a23b6 | eval/stages/base.py, eval/stages/alignment.py, eval/stages/label_transfer.py, tests/test_alignment_stage.py, tests/test_label_transfer_stage.py |

## Files Modified

| File | Lines | Key Changes |
|------|-------|-------------|
| eval/config.py | 185 | Added `from typing import Literal`; `pipeline_mode: Literal["paired","synthetic"] = "paired"`; `target_data_path: str \| None = None` |
| eval/data_factory.py | 536 | Extended import to `EvalConfig, EvalConfigError`; `_target_dataset` cache in `__init__`; `load_target()` method with D-05 guard + format dispatch |
| eval/stages/base.py | 114 | ABC `run` signature changed to `(source, target, params)` with updated docstring |
| eval/stages/alignment.py | 276 | `run(source, target, params)`; symmetric strided sub-dicts; `DynamicTimeWarping(x=source_sub, y=target_sub)`; `aligned_cloud=source`; module docstring updated |
| eval/stages/label_transfer.py | 230 | `run(source, target, params)`; two empty guards; `source_keys`/`target_keys`; `n_pairs = min(...)`; sequential `source[k]->target[k]`; frame-0 pass-through deleted; module docstring updated |
| tests/test_data_factory.py | 768 | `TestEvalConfigFromYAML` extended (+3 tests); `TestDataFactoryConstruction` checks `_target_dataset is None`; `TestLoadTarget` class added (+4 tests) |
| tests/test_alignment_stage.py | 278 | All 7 existing `stage.run(...)` callsites updated to 3-arg form; `TestAlignmentStageTwoInput` added (+3 tests) |
| tests/test_label_transfer_stage.py | 533 | `test_run_frame0_passthrough` deleted (-1); all run callsites updated to 3-arg form; `test_run_all_tensors_are_1d` renamed + covers all keys; FRAME-06 gate tests updated |

## Test Counts

| Scope | Before | After | Delta |
|-------|--------|-------|-------|
| tests/test_data_factory.py | 54 | 61 | +7 |
| tests/test_alignment_stage.py | 24 | 27 | +3 |
| tests/test_label_transfer_stage.py | 29 | 28 | -1 (pass-through deleted) + 0 net new |
| Three-file total | 107 | 116 | +9 |

## Signature Changes Confirmed

```
PipelineStage.run:     (self, source, target, params) -> StageResult
AlignmentStage.run:    (self, source, target, params) -> AlignResult
LabelTransferStage.run:(self, source, target, params) -> LabelResult
```

Verified via `inspect.signature` in final smoke test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] dict inequality comparison fails on tensor-valued dicts**
- **Found during:** Task 2, `TestAlignmentStageTwoInput.test_run_with_distinct_source_target_returns_align_result`
- **Issue:** `result.aligned_cloud != synthetic_dataset_b` raises `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` because Python dict `__ne__` on zRegPointCloud values triggers tensor element-wise comparison with no boolean coercion
- **Fix:** Replaced `assert result.aligned_cloud != synthetic_dataset_b` with `assert result.aligned_cloud is not synthetic_dataset_b` (identity) + `assert not torch.equal(result.aligned_cloud[0]["pos"], synthetic_dataset_b[0]["pos"])` (structural verification at a specific frame)
- **Files modified:** tests/test_alignment_stage.py
- **Commit:** 45a23b6

## Expected Downstream Failures (Plans 30-02 / 30-03)

As designed, `tests/test_eval_runner.py`, `tests/test_optimizer.py`, and `tests/test_trajectory_export.py` now fail with `TypeError: AlignmentStage.run() missing 1 required positional argument: 'params'` — the runner still calls `run(dataset, params)` (two args). These are fixed in Plan 30-02.

## Known Stubs

None — all production code paths are functional. `pipeline_mode="synthetic"` is accepted but treated as a no-op by `DataFactory` (deferred to Phase 31 per CONTEXT deferred ideas).

## Threat Flags

No new threat surface introduced beyond what the plan's threat model already covers:
- T-30-01 (Tampering/pipeline_mode): mitigated — `Literal["paired","synthetic"]` validated at parse time
- T-30-02 (target_data_path traversal): accepted — inherits same posture as `data_path`
- T-30-03 (load_target caching DoS): mitigated — `_target_dataset` instance cache prevents unbounded reloading

## Self-Check: PASSED

- eval/config.py: FOUND
- eval/data_factory.py: FOUND (load_target method present)
- eval/stages/base.py: FOUND
- eval/stages/alignment.py: FOUND (y=target_sub present, y=x_sub absent)
- eval/stages/label_transfer.py: FOUND (n_pairs = min present, frame-0 pass-through absent)
- tests/test_data_factory.py: FOUND (TestLoadTarget, 3 new EvalConfigFromYAML tests)
- tests/test_alignment_stage.py: FOUND (TestAlignmentStageTwoInput)
- tests/test_label_transfer_stage.py: FOUND (test_run_frame0_passthrough absent)
- Commit 977267b: FOUND
- Commit 45a23b6: FOUND
- pytest tests/test_alignment_stage.py tests/test_label_transfer_stage.py tests/test_data_factory.py: 142 passed, 0 failed
