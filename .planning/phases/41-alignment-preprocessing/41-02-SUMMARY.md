---
phase: 41-alignment-preprocessing
plan: 02
subsystem: alignment
tags: [torch, pca, preprocessing, alignment, dtw, pytest, integration-tests]

# Dependency graph
requires:
  - phase: 41-alignment-preprocessing
    plan: 01
    provides: zreg.preprocessing functions, AlignmentPreprocessingConfig, EvalConfig.alignment_preprocessing, AlignResult.velocity_landmarks
  - phase: 19-alignmentstage
    provides: AlignmentStage.run contract + _build_aligned_cloud
provides:
  - "AlignmentStage._apply_preprocessing static method dispatching None/principal_axes/velocity_landmarks"
  - "AlignmentStage.run threads working_source through striding + _build_aligned_cloud and passes velocity_landmarks to AlignResult"
  - "Unit + integration + config test coverage for all ALIGN-06 requirements"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Preprocessing dispatch via @staticmethod with local imports inside the body (mirrors _apply_stored_transform)"
    - "deepcopy each source frame before rotating ['pos'] — original source dict never mutated"
    - "PCA rotation applied as (R @ frame['pos'].T).T per frame"

key-files:
  created: []
  modified:
    - eval/stages/alignment.py
    - tests/test_preprocessing.py
    - tests/test_eval_config.py
    - tests/test_alignment_stage.py

key-decisions:
  - "Restructured test_preprocessing.py (created by 41-01 as 9 flat functions) into the plan-mandated class layout (TestComputePCARotation + TestDetectVelocityLandmarks, 8 tests) plus a module export test — satisfies the must_haves artifact 'contains class TestComputePCARotation' without duplicating coverage"
  - "_apply_preprocessing placed after _apply_stored_transform as the last static method on AlignmentStage"
  - "velocity_landmarks tests added to test_eval_config.py (per plan) even though equivalent coverage exists in tests/test_alignment_preprocessing_config.py from 41-01 — required by the plan's key_links/artifacts"

requirements-completed: [ALIGN-06-01, ALIGN-06-02, ALIGN-06-04, ALIGN-06-05]

# Metrics
duration: 6min
completed: 2026-06-29
---

# Phase 41 Plan 02: AlignmentStage Preprocessing Wiring Summary

**Wired the Phase 41 preprocessing contracts into AlignmentStage.run() via a new `_apply_preprocessing` static method that threads a `working_source` dict and `velocity_landmarks` list end-to-end, with full unit/config/integration test coverage for all ALIGN-06 requirements.**

## Performance

- **Duration:** ~6 min
- **Tasks:** 3 (all involve test coverage; Task 1 is the production wiring)
- **Files modified:** 4 (0 created, 4 modified)

## Accomplishments
- `AlignmentStage._apply_preprocessing(source, target, config)` — three-branch dispatch: `None` → `(source, [])`; `principal_axes` → deep-copied PCA-rotated source dict + `[]`; `velocity_landmarks` → `(source, landmarks)`. Unreachable final fallback returns `(source, [])`.
- `run()` now calls `_apply_preprocessing` immediately after `validate_params`, strides `working_source` (not original `source`), passes `source=working_source` into `_build_aligned_cloud`, and passes `velocity_landmarks=velocity_landmarks` into `AlignResult`.
- `tests/test_preprocessing.py` restructured into `TestComputePCARotation` (known-rotation recovery, det=+1, identity) and `TestDetectVelocityLandmarks` (empty-below-threshold, high-velocity detection, frame-0 exclusion, mean-vs-max) — 8 tests.
- `tests/test_eval_config.py` extended with `TestAlignmentPreprocessingConfig` (defaults, invalid-method ValidationError, EvalConfig default None) — 3 tests.
- `tests/test_alignment_stage.py` extended with `TestAlignmentStagePreprocessing` (principal_axes, velocity_landmarks threshold=0.0, backward-compatible no-config) — 3 tests.

## Task Commits
1. **Task 1 — wire _apply_preprocessing into AlignmentStage.run()** — `dc064c9` (feat)
2. **Task 2 — class-based preprocessing unit tests + config validation** — `87368fc` (test)
3. **Task 3 — TestAlignmentStagePreprocessing integration tests** — `602deb4` (test)

## Files Created/Modified
- `eval/stages/alignment.py` — Added `_apply_preprocessing` static method; `run()` threads `working_source` + `velocity_landmarks` through striding, `_build_aligned_cloud`, and `AlignResult`.
- `tests/test_preprocessing.py` — Restructured from 9 flat functions into the plan-mandated class layout (8 tests + `test_module_exports`).
- `tests/test_eval_config.py` — Added `pydantic.ValidationError` + `AlignmentPreprocessingConfig` imports and `TestAlignmentPreprocessingConfig` (3 tests).
- `tests/test_alignment_stage.py` — Added `AlignmentPreprocessingConfig` import and `TestAlignmentStagePreprocessing` (3 tests).

## Verification
- `pytest tests/test_preprocessing.py` — 8 passed (>= 7 required).
- `pytest tests/test_alignment_stage.py::TestAlignmentStagePreprocessing` — 3 passed.
- `pytest tests/test_eval_config.py::TestAlignmentPreprocessingConfig` — 3 passed.
- `pytest tests/ -q --deselect tests/test_eval_config.py::TestEvalConfigAlignmentMethodValidation` — **1118 passed, 18 skipped, 1 xpassed, exit 0** (~116s).

## Deviations from Plan

### Restructured existing test file (not a fresh create)
- **Found during:** Task 2
- **Issue:** The plan's `<action>` says "Create tests/test_preprocessing.py as a new file", but plan 41-01 already created it with 9 flat-function tests covering the same behaviors. Creating fresh would either error or duplicate coverage.
- **Resolution:** Restructured the file into the plan-mandated class layout (`TestComputePCARotation`, `TestDetectVelocityLandmarks`) with the exact method names required by the must_haves artifacts/acceptance criteria, preserving `test_module_exports`. Net 8 tests (>= 7 required). No behavior lost.
- **Files modified:** tests/test_preprocessing.py
- **Commit:** 87368fc

### Full-suite test-count target (>= 1152) not literally reached — planning estimate double-counted 41-01 tests
- **Found during:** final verification
- **Issue:** The plan's must_haves expect `pytest tests/ -x -q` with ">= 1152 tests". The estimate assumed the preprocessing unit tests were net-new on top of a 1141 baseline, but plan 41-01 had already added `test_preprocessing.py` (9 tests) and `test_alignment_preprocessing_config.py` (10 tests) into that baseline. Collected total is now **1141 items** (net +5 from 41-01's end: +3 config, +3 integration, −1 from the preprocessing restructure).
- **Resolution:** No fix attempted — the count target was an over-estimate, not a missing-test defect. Every required test exists and passes; all ALIGN-06 behaviors are verified. Excluding the known pre-existing failures, the suite is green (exit 0).
- **Note:** `-x` cannot be used on the unfiltered suite because of the pre-existing failure below; verification used `--deselect` per the orchestrator's success criteria ("exits 0 excluding known pre-existing failures").

## Issues Encountered

**Pre-existing, out-of-scope failures (NOT caused by this plan, NOT fixed per orchestrator instruction).**
`tests/test_eval_config.py::TestEvalConfigAlignmentMethodValidation` (3 of 4 methods) assert the regex `"alignment_method must be 'cpd' or 'icp'"`, but the validator message was updated in Phase 40 to `"...'cpd', 'icp', or 'swd'"`. Confirmed present at the worktree base and untouched by this plan. The orchestrator explicitly flagged these as known pre-existing and instructed do-not-fix; logged previously in `deferred-items.md` by plan 41-01. Suggested fix belongs to a future cleanup plan (update the three `match=` regexes).

## Known Stubs
None — all wiring is fully functional; no placeholder/empty-data paths introduced.

## User Setup Required
None.

## Self-Check: PASSED
- `eval/stages/alignment.py` — FOUND (contains `def _apply_preprocessing`)
- `tests/test_preprocessing.py` — FOUND (contains `class TestComputePCARotation`)
- `tests/test_alignment_stage.py` — FOUND (contains `class TestAlignmentStagePreprocessing`)
- `tests/test_eval_config.py` — FOUND (contains `AlignmentPreprocessingConfig`)
- Commit `dc064c9` (Task 1 feat) — FOUND
- Commit `87368fc` (Task 2 test) — FOUND
- Commit `602deb4` (Task 3 test) — FOUND

## TDD Gate Compliance
The library functions (41-01) and the wiring (Task 1) were verified before the test commits; Task 2/3 are test-coverage tasks for already-implemented behavior, so no RED→GREEN failing-first commit is applicable. All new tests pass against the committed implementation.

---
*Phase: 41-alignment-preprocessing*
*Completed: 2026-06-29*
