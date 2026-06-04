---
phase: 25-visualisation-refactor
plan: 02
subsystem: eval
tags: [tdd, test-cleanup, eval_runner, alignment_trajectory, point_cloud_removal]

# Dependency graph
requires:
  - phase: 25-plan-01
    provides: plot_trajectory, plot_metrics, EvalConfig.label_names, EvaluationRunner wired to new API
provides:
  - tests/test_eval_runner.py: zero point_cloud.pdf references; alignment_trajectory.pdf assertions
  - TestEvaluationRunnerSavePlots: 5 tests green against refactored runner
affects: [EXT-02, phase-25-completion]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "test_run_alignment_false_* checks alignment_trajectory.pdf absent (not point_cloud.pdf)"
    - "Semantic assertion upgrade: trivially-true guard replaced with meaningful behaviour check"

key-files:
  created: []
  modified:
    - tests/test_eval_runner.py

key-decisions:
  - "Assertion in test_run_alignment_false upgraded from checking point_cloud.pdf absent (trivially true) to checking alignment_trajectory.pdf absent (semantically correct for run_alignment=False)"
  - "Docstring on line 474 reworded to remove historical migration note — keeps only current-state description"

# Metrics
duration: 4min
completed: 2026-06-04
---

# Phase 25 Plan 02: EvaluationRunner Test Cleanup Summary

**Eliminated all 4 stale point_cloud.pdf references from TestEvaluationRunnerSavePlots — renamed test methods and upgraded assertion semantics to match the alignment_trajectory.pdf API introduced in Plan 25-01**

## Performance

- **Duration:** ~4 min
- **Completed:** 2026-06-04
- **Tasks:** 2 (Task 1 already done by Wave 1; Task 2 executed here)
- **Files modified:** 1

## Accomplishments

- Renamed `test_run_alignment_false_omits_point_cloud_pdf` to `test_run_alignment_false_omits_alignment_trajectory_pdf`
- Updated docstring at line 474 to remove stale "Phase 25 replaces point_cloud.pdf" migration note
- Updated docstring at line 535 to reference alignment_trajectory.pdf semantics
- Upgraded assertion line 550: now checks `not any("alignment_trajectory.pdf" in p ...)` instead of trivially-true `not any("point_cloud.pdf" in p ...)`
- grep -c "point_cloud.pdf" tests/test_eval_runner.py: 0 (was 4)
- All 5 TestEvaluationRunnerSavePlots tests pass; full suite 789 passed, 17 skipped

## Task Commits

1. **Task 1: Update EvaluationRunner.run() (Wave 1 deviation — already done by 25-01 executor)** — `cfcf73c`
2. **Task 2: Update TestEvaluationRunnerSavePlots assertions** — `c44a165`

## Files Created/Modified

- `tests/test_eval_runner.py` — Renamed 1 test method; updated 2 docstrings; upgraded 1 assertion from `point_cloud.pdf` to `alignment_trajectory.pdf`

## Decisions Made

- The assertion change from `not any("point_cloud.pdf" in p ...)` to `not any("alignment_trajectory.pdf" in p ...)` is a semantic improvement: the old check was trivially true (point_cloud.pdf is never produced), while the new check verifies that when run_alignment=False, plot_trajectory produces no alignment_trajectory.pdf — which is the actual behaviour contract.

## Deviations from Plan

### Wave 1 pre-completion

**1. [Prior Wave Deviation - Context] Task 1 fully completed by Wave 1 executor**
- **Found during:** Plan start (prior_wave_context in prompt)
- **Issue:** Wave 1 (Plan 25-01 executor) already updated eval/runners/eval_runner.py import and save_plots branch as a Rule 3 blocking fix. The worker left 4 stale point_cloud.pdf references in tests/test_eval_runner.py.
- **Resolution:** Task 1 acceptance criteria verified green (grep confirms 0 stale symbols in runner, imports OK). Task 2 executed normally.
- **Impact:** None — Wave 1 work was correct and complete for Task 1.

None for Task 2 — executed exactly as written.

## Phase 25 Gate Checks

- `from eval.viz import plot_trajectory, plot_metrics` — OK
- `from eval.runners.eval_runner import EvaluationRunner` — OK
- grep -c "plot_point_cloud|plot_metrics_summary" eval/runners/eval_runner.py — 0
- grep -c "point_cloud.pdf" tests/test_eval_runner.py — 0
- pytest tests/test_viz.py -q — 9 passed
- pytest tests/test_eval_runner.py::TestEvaluationRunnerSavePlots -q — 5 passed
- pytest -q — 789 passed, 17 skipped

## EXT-02 Success Criteria

- SC1 (updated): plot_trajectory produces alignment_trajectory.pdf/png and/or label_trajectory.pdf/png — verified by TestEvaluationRunnerSavePlots
- SC2 (updated): each file is a 1x3 figure (not 2x3) — verified by TestPlotTrajectory in test_viz.py
- SC3: plot_point_cloud no longer importable from eval.viz — verified (removed in Plan 25-01)
- SC4: all four stage-combination cases produce valid output — verified by TestPlotTrajectory 6-test suite

## Self-Check: PASSED

- FOUND: tests/test_eval_runner.py (modified)
- FOUND commit: c44a165 (feat(25-02): update TestEvaluationRunnerSavePlots)
- grep -c "point_cloud.pdf" tests/test_eval_runner.py = 0 (VERIFIED)
- grep -c "alignment_trajectory.pdf" tests/test_eval_runner.py = 7 (VERIFIED >= 2)
- grep -c "def test_run_alignment_false_omits_alignment_trajectory_pdf" tests/test_eval_runner.py = 1 (VERIFIED)
- grep -c "def test_save_plots_true_creates_alignment_trajectory_pdf" tests/test_eval_runner.py = 1 (VERIFIED)
- pytest TestEvaluationRunnerSavePlots: 5 passed (VERIFIED)
- pytest full suite: 789 passed, 17 skipped (VERIFIED)
