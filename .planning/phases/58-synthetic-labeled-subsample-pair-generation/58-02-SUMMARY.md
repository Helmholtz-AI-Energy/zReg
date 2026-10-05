---
phase: 58-synthetic-labeled-subsample-pair-generation
plan: 02
subsystem: evaluation-framework
tags: [pytorch, pydantic, ground-truth, label-transfer, data-factory, eval-runner]

# Dependency graph
requires:
  - phase: 58-01
    provides: "DataFactory.generate_subsample_pair(dataset, transform_spec) -> (source_view,
      target_view), producer-contract-compatible with get_synthetic_ground_truth()"
provides:
  - "EvaluationRunner.run() three-way dispatch: paired -> load_real()+load_target();
    transform_spec['type'] == 'subsample_pair' -> generate_subsample_pair() (skips
    load_real() when synthesize=True, D-02); else -> load_real()+generate_target()
    (Phase 31 MODE-02, unchanged)"
  - "TestEvaluationRunnerSubsamplePair test class (dataset mode, synthesize mode,
    GT-extraction reuse proof, real end-to-end F1 smoke test)"
affects: [58-03, 58-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Three-way if/elif/else dispatch in EvaluationRunner.run() keyed on
      pipeline_mode + transform_spec['type'], mirroring the existing
      pipeline_mode-only if/else shape used elsewhere in the codebase"
    - "Conditional load_real() call (base = None if synthesize else load_real())
      so a subsample_pair run can execute with zero real data on disk (D-02)"

key-files:
  created: []
  modified:
    - eval/runners/eval_runner.py
    - tests/test_eval_runner.py

key-decisions:
  - "_run_single's GT-extraction branch (pipeline_mode == 'synthetic' ->
    get_synthetic_ground_truth()) required NO code change — Plan 58-01's
    generate_subsample_pair() already populates DataFactory state
    (_source_dataset/_correspondence_idx) identically to generate_target()'s
    producer contract, confirmed by both mocked tests and the real
    end-to-end smoke test"
  - "subsample_pair branch checked via elif before the rigid/noise else branch,
    keeping both pre-existing branches (paired, rigid/noise) textually
    unchanged in shape — only the unconditional load_real() at the top of
    run() was removed and pushed into each specific branch"

requirements-completed: [GT-04, GT-06]

# Metrics
duration: ~15min
completed: 2026-08-04
---

# Phase 58 Plan 02: EvaluationRunner Subsample-Pair Dispatch Summary

**`EvaluationRunner.run()` now dispatches to `DataFactory.generate_subsample_pair()` for `transform_spec["type"] == "subsample_pair"`, skipping `load_real()` entirely in synthesize mode, with zero changes needed to the existing GT-extraction path.**

## Performance

- **Duration:** ~15 min active work
- **Started:** 2026-08-04
- **Completed:** 2026-08-04
- **Tasks:** 2 completed
- **Files modified:** 2 (eval/runners/eval_runner.py, tests/test_eval_runner.py)

## Accomplishments

- `EvaluationRunner.run()` three-way dispatch replacing the previous unconditional
  `load_real()` + `if paired/else` two-way branch:
  1. `pipeline_mode == "paired"` -> `load_real()` + `load_target()` (unchanged)
  2. `transform_spec.get("type") == "subsample_pair"` -> NEW branch: `base = None if
     transform_spec.get("synthesize", False) else self.factory.load_real()`, then
     `source, target = self.factory.generate_subsample_pair(base, transform_spec)`
  3. else (rigid/noise `transform_spec`, Phase 31 MODE-02) -> `load_real()` +
     `generate_target()` (unchanged)
- `load_real()` is never called when `transform_spec["synthesize"]` is `True`,
  satisfying D-02's "no real data available" requirement — verified by a mocked
  test asserting `mock_factory.load_real.called is False`
- `_run_single`'s GT-extraction branch confirmed to need no changes: both mocked
  tests assert `get_synthetic_ground_truth.called is True` and
  `get_ground_truth.called is False` after `run()`, and the real end-to-end smoke
  test produces a correct, non-degenerate F1 score through the unmodified branch
- `TestEvaluationRunnerSubsamplePair` test class (3 new tests): dataset-mode
  dispatch proof, synthesize-mode `load_real()`-skip proof, and one real
  (non-mocked) end-to-end run through `EvaluationRunner(config, full_params).run()`
  asserting `EvalReport.metrics.f1_score` is a finite float in `[0.0, 1.0]`
- Existing `TestEvaluationRunnerSyntheticMode` (rigid/noise) and paired-mode tests
  pass unmodified — zero regressions across the full 32-test `test_eval_runner.py`
  suite

## Task Commits

Each task was committed atomically:

1. **Task 1: EvaluationRunner.run() subsample_pair dispatch branch** - `d96307c` (feat)
2. **Task 2: Tests for EvaluationRunner subsample_pair wiring** - `e758789` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `eval/runners/eval_runner.py` - `run()` method's dataset-construction block replaced
  with a three-way `if pipeline_mode == "paired": ... elif transform_spec.get("type")
  == "subsample_pair": ... else: ...` dispatch; docstring's step-2 bullet updated to
  describe the three-way dispatch (Phase 58 GT-04/GT-06 note added)
- `tests/test_eval_runner.py` - New `TestEvaluationRunnerSubsamplePair` class (3 tests)
  placed after `TestEvaluationRunnerSyntheticMode`, mirroring its
  `@patch("eval.runners.eval_runner.DataFactory")` mocking convention for the two
  mocked tests, plus one fully real (non-mocked) end-to-end smoke test

## Decisions Made

- Followed the plan's exact three-way dispatch shape (Task 1 action items 1-5)
  verbatim, including the one-line comment citing GT-04/GT-05/GT-06 and D-02
- Kept the `paired` and rigid/noise `else` branches textually byte-for-byte
  unchanged in their internal call sequence (`load_real()` then
  `load_target()`/`generate_target()`) — only the position of `load_real()`
  moved from a single unconditional call at the top of `run()` into each
  specific branch, per acceptance criteria
- Test fixtures reused the module-level `synthetic_dataset`/`full_params`
  fixtures already present in `tests/test_eval_runner.py` rather than adding
  new ones, consistent with `TestEvaluationRunnerSyntheticMode`'s convention

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Full `tests/test_eval_runner.py` suite (32 tests) and `tests/test_data_factory.py`
(133 tests, unmodified by this plan) both pass with zero regressions.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `eval/runners/eval_runner.py`'s `run()` dispatch is now the reference pattern
  Plan 58-03 needs to replicate across `eval/runners/optimizer.py`'s three
  `pipeline_mode == "synthetic"` call sites (per 58-PATTERNS.md's integration-point
  mapping)
- `TestEvaluationRunnerSubsamplePair`'s real end-to-end smoke test proves
  `generate_subsample_pair()`'s synthesize-mode output is fully compatible with
  `AlignmentStage`/`LabelTransferStage`/`MetricsEngine` — no further compatibility
  work needed for Plan 58-03/58-04 to build on
- Full regression suite for the touched files (`test_eval_runner.py`,
  `test_data_factory.py`) shows zero new failures

---
*Phase: 58-synthetic-labeled-subsample-pair-generation*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: eval/runners/eval_runner.py (contains "generate_subsample_pair(" literal call,
  verified via `grep -n "generate_subsample_pair" eval/runners/eval_runner.py`)
- FOUND: tests/test_eval_runner.py (TestEvaluationRunnerSubsamplePair class, 3 tests,
  all passing, verified via `pytest tests/test_eval_runner.py -k "SubsamplePair" -q`)
- FOUND commit d96307c in `git log --oneline --all`
- FOUND commit e758789 in `git log --oneline --all`
