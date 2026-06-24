---
phase: 38
plan: 38-02
subsystem: tests
tags: [rename, refactor, CLN-01, CLN-02, test-suite]
dependency_graph:
  requires: [38-01]
  provides: [green-test-suite-after-color-label-rename]
  affects:
    - tests/conftest.py
    - tests/test_dataset.py
    - tests/test_dtw.py
    - tests/test_data_factory.py
    - tests/test_generators.py
    - tests/test_optimizer.py
    - tests/test_trajectory_export.py
    - tests/test_pairwise_distance_matrix.py
    - tests/test_label_transfer_stage.py
    - tests/test_viz.py
    - tests/test_color_transfer.py
    - tests/test_cpd.py
    - tests/test_downsampling.py
    - tests/test_eval_runner.py
    - tests/test_transforms.py
tech_stack:
  added: []
  patterns: [pure-rename, CLN-02-regression-tests]
key_files:
  created: []
  modified:
    - tests/conftest.py
    - tests/test_dataset.py
    - tests/test_dtw.py
    - tests/test_data_factory.py
    - tests/test_generators.py
    - tests/test_optimizer.py
    - tests/test_trajectory_export.py
    - tests/test_pairwise_distance_matrix.py
    - tests/test_label_transfer_stage.py
    - tests/test_viz.py
    - tests/test_color_transfer.py
    - tests/test_cpd.py
    - tests/test_downsampling.py
    - tests/test_eval_runner.py
    - tests/test_transforms.py
decisions:
  - "Replaced TestLabelTransferStageCoverageGaps with CLN-02-correct tests; old heuristic tests removed"
  - "External .mat 'color' key in test_dataset.py mock data left unchanged (external MATLAB API)"
  - "Added TestGetSourceLabels to test_viz.py for _get_source_labels CLN-02 regression"
  - "Fixed 6 additional test files not in plan (Rule 2: blocking failures)"
metrics:
  duration: 1445s
  completed: "2026-06-24T07:31:34Z"
  tasks_completed: 9
  files_modified: 15
---

# Phase 38 Plan 02: color→label field rename in test suite — Summary

Updated all test files to use the renamed `label` field after Plan 38-01 renamed
`zRegPointCloud["color"]` to `zRegPointCloud["label"]`. Added CLN-02 regression
tests confirming the id-fallback heuristic is gone.

## What Was Built

Every test file that constructed `zRegPointCloud` with `color=` or read `pc["color"]`
was updated. For `LabelTransferStage`, the old heuristic coverage gap tests
(testing `id` fallback and 2-D color to `id` paths) were replaced with CLN-02 tests
that confirm the new direct-`label`-read + `ValueError` behavior.

Two new CLN-02 regression tests in `TestGetSourceLabels` verify that
`_get_source_labels` now returns `None` for `label=None` (no `id` fallback).

## Commits

| Task | File | Commit | Change |
|------|------|--------|--------|
| 1 | tests/conftest.py | 10088b3 | 2 constructor args: color= to label= |
| 2 | tests/test_dataset.py | b79bceb | ~12 field accesses + constructor args |
| 3 | tests/test_dtw.py | 2482d47 | 10 constructor args + field accesses |
| 4 | tests/test_data_factory.py | 69e1b91 | ~25 constructor args + field accesses + comments |
| 5 | tests/test_generators.py | c686d22 | ~23 field accesses + comments |
| 6 | tests/test_optimizer.py | a06569a | ~22 field accesses + constructor args + comments |
| 7 | tests/test_trajectory_export.py | f4c1ea6 | 5 field accesses |
| 8 | tests/test_pairwise_distance_matrix.py | a46abed | 8 constructor args |
| 9 | (new CLN-02 + additional files) | 7d19eaf | CLN-02 tests + 7 additional files fixed |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing fix] 6 additional test files not listed in plan**

- **Found during:** Task 9 — running the full test suite
- **Issue:** `test_color_transfer.py`, `test_cpd.py`, `test_downsampling.py`,
  `test_eval_runner.py`, `test_transforms.py` all used `color=` in `zRegPointCloud`
  constructors or accessed `pc["color"]`. After Plan 38-01, these caused
  `TypeError: 'NoneType' object is not subscriptable` or `KeyError: 'color'`.
- **Fix:** Renamed all `color=` to `label=` and `["color"]` to `["label"]` in these files.
- **Files modified:** tests/test_color_transfer.py, tests/test_cpd.py,
  tests/test_downsampling.py, tests/test_eval_runner.py, tests/test_transforms.py
- **Commit:** 7d19eaf

**2. [Rule 1 - Bug] TestLabelTransferStageCoverageGaps tests tested old heuristic behavior**

- **Found during:** Task 9
- **Issue:** `test_label_key_is_id_when_color_is_none`, `test_none_labels_tensor_raises`,
  and `test_2d_color_uses_id_label_key` all tested the pre-CLN-02 heuristic.
  After Plan 38-01, these tests would fail because the heuristic is gone.
- **Fix:** Replaced all three with CLN-02-correct tests that confirm `label=None`
  raises `ValueError("has no 'label' field")`. Added `test_run_raises_when_label_is_none`.
- **Commit:** 7d19eaf

## Known Stubs

None.

## Threat Flags

None. Pure test-file rename with no new network endpoints or security surfaces.

## Self-Check: PASSED

All 15 modified test files confirmed present. All 9 task commits confirmed in git log.
Full test suite: 976 tests pass, 18 skipped (pytest exits 0).
Test count increased from ~969 to 994 collected (25 new tests, including CLN-02 regressions).
No `["color"]` or `color=` as zRegPointCloud field remaining in any test file.
