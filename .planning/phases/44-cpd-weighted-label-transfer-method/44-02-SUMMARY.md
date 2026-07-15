---
phase: 44-cpd-weighted-label-transfer-method
plan: 02
subsystem: eval-alignment
tags: [cpd, alignment-stage, posterior, e-step, estep-results]

# Dependency graph
requires:
  - phase: 44-01
    provides: "AlignResult.estep_results field + EvalConfig.label_transfer_method (schema/config contracts)"
provides:
  - "AlignmentStage._build_aligned_cloud returning (aligned, estep_results) tuple"
  - "AlignResult.estep_results populated with per-frame CPD posterior (EstepResult) for CPD-registered runs"
affects: [44-03-labeltransferstage-cpd-weighted]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fork-free posterior capture: single insertion point shared by both StoredTransform-reuse and fresh-CPD-fallback sub-paths (D-01)"
    - "Throwaway side-effect-free RigidCPD construction + public expectation_step() call to reuse zreg's own E-step math without reimplementation (D-01/D-02)"
    - "Tuple-return-then-caller-destructures convention for @staticmethod helpers, mirroring _apply_preprocessing (Phase 41)"

key-files:
  created: []
  modified:
    - eval/stages/alignment.py
    - tests/test_alignment_stage.py

key-decisions:
  - "Posterior-capture block inserted immediately before aligned[tk] = matched_source_frame inside the CPD elif branch — the single point shared by both the StoredTransform reuse sub-path and the fresh-CPD fallback sub-path, per D-01"
  - "sigma2 for the posterior-only E-step estimated via zreg.utils.squared_kernel_sum on the already-registered source vs. target positions — mirrors RigidCPD._initialize's own sigma2-seeding formula"
  - "estep_results left untouched (stays empty for that tk) in the cpd_penalty=None branch, the icp branch, the swd branch, and the final else fallback — D-03 scope guard enforced by 2 new empty-dict assertions plus 2 dedicated TestEstepResultsCapture tests"

requirements-completed: [D-01, D-02, D-03]

# Metrics
duration: 25min
completed: 2026-07-10
---

# Phase 44 Plan 02: AlignmentStage CPD Posterior Capture Summary

**`_build_aligned_cloud` now returns `(aligned, estep_results)`, computing a throwaway `RigidCPD.expectation_step()` posterior at the single fork-free point shared by the reuse and fallback CPD sub-paths, and `run()` threads it into `AlignResult.estep_results` — empty for icp/swd/no-cpd runs.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-10T11:36:13Z (session start)
- **Completed:** 2026-07-10T11:47:16Z
- **Tasks:** 2 completed
- **Files modified:** 2 (eval/stages/alignment.py, tests/test_alignment_stage.py)

## Accomplishments

- `_build_aligned_cloud`'s return type changed from `dict[int, zRegPointCloud]` to `tuple[dict[int, zRegPointCloud], dict[int, EstepResult]]`; docstring `Returns` section updated.
- Posterior-capture block (`sigma2_est` via `utils.squared_kernel_sum`, throwaway `RigidCPD(source=..., use_color=False)`, `.expectation_step(...)`) inserted at the single insertion point shared by both the `StoredTransform` reuse sub-path and the fresh-CPD fallback sub-path — no forked logic (D-01).
- `estep_results[tk] = estep_result` appears exactly once in the file, strictly inside the `elif cpd_penalty is not None and alignment_method == "cpd":` branch — the `cpd_penalty is None`, `icp`, `swd`, and final `else` branches leave `estep_results` untouched for their `tk` (D-03).
- `run()` unpacks the new tuple (`aligned_cloud, estep_results = self._build_aligned_cloud(...)`) and passes `estep_results=estep_results` into the `AlignResult(...)` construction, following the same threading pattern used for `velocity_landmarks` in Phase 41.
- All 8 pre-existing direct callers of `_build_aligned_cloud` in `tests/test_alignment_stage.py` converted from `result = AlignmentStage._build_aligned_cloud(...)` to `result, estep_results = AlignmentStage._build_aligned_cloud(...)`, with all downstream assertions on `result` unchanged.
- Two of those 8 call sites (`test_no_cpd_penalty_ignores_stored_transforms`, `test_build_aligned_cloud_unknown_method_falls_through`) gained an extra `assert estep_results == {}` confirming the D-03 scope guard directly at the unit level.
- New `TestEstepResultsCapture` class (4 tests) added at the end of `tests/test_alignment_stage.py`, calling `AlignmentStage(...).run(...)` end-to-end and asserting: CPD-rigid runs produce `estep_results` keyed identically to `aligned_cloud` with non-empty `.pmat` attributes; no-cpd-penalty, icp, and swd runs all produce `estep_results == {}`.
- Full `tests/test_alignment_stage.py` suite: 63 passed. Full repo suite: 1199 passed, 18 skipped, 1 xpassed, 100% coverage — zero regressions.

## Task Commits

Each task was committed atomically:

1. **Task 1: Compute and thread estep_results through _build_aligned_cloud and run()** - `07b43e1` (feat)
2. **Task 2: Fix existing _build_aligned_cloud call sites for tuple return + add estep_results coverage tests** - `d8fff09` (test)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified

- `eval/stages/alignment.py` — Added `EstepResult` to the `zreg.cpd` import line; changed `_build_aligned_cloud`'s return annotation to a tuple and its final `return aligned` to `return aligned, estep_results`; initialised `estep_results: dict[int, EstepResult] = {}` alongside `aligned`; inserted the D-01 posterior-capture block inside the CPD `elif` branch, immediately before `aligned[tk] = matched_source_frame`; updated `run()`'s `_build_aligned_cloud` call site to unpack the tuple and thread `estep_results=estep_results` into `AlignResult(...)`; updated both docstrings' `Returns` sections.
- `tests/test_alignment_stage.py` — Converted all 8 pre-existing direct `_build_aligned_cloud` call sites to tuple-unpack assignments; added D-03 scope-guard assertions to 2 of them; added `TestEstepResultsCapture` (4 tests: `test_cpd_rigid_populates_estep_results`, `test_no_cpd_penalty_estep_results_empty`, `test_icp_estep_results_empty`, `test_swd_estep_results_empty`).

## Decisions Made

- Followed PATTERNS.md's exact insertion point and posterior-capture code (sigma2 via `squared_kernel_sum`, throwaway `RigidCPD`, `.expectation_step(sigma2_c=0.0, w=0.0)`) verbatim — no deviation needed.
- Placed the new `TestEstepResultsCapture` class at the physical end of `tests/test_alignment_stage.py` (after `TestAlignmentStageCoverageGaps`, which is itself the last class in the current file — the plan's stated "line 483" reference for that class's end was stale relative to the file's current 1171-line length; the class is still the correct anchor, just at a later line number).
- Did not transpose `EstepResult.pmat` in `AlignmentStage` — per D-04, that transpose is `LabelTransferStage`'s responsibility (Plan 44-03), not this plan's.

## Deviations from Plan

None — plan executed exactly as written. Both tasks matched the plan's `<action>` and `<acceptance_criteria>` blocks; the only note is the file being longer than the plan's read_first line references (pre-existing drift from `TestAlignmentStageCoverageGaps` growing across prior phases), which did not require any Rule 1/2/3/4 deviation — the class was still findable and correctly targeted by content search.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `AlignResult.estep_results` is now populated end-to-end for CPD-registered `AlignmentStage.run()` calls and available for Plan 44-03 (`LabelTransferStage` `cpd_weighted` branch, D-04/D-05/D-07/D-08).
- Plan 44-03 must transpose `estep_result.pmat` (currently `(n_source, n_target)`) to `(n_target, n_source)` before calling `transfer_colors(method=CPD_WEIGHTED, ...)`, per D-04 — not done in this plan by design.
- No blockers. All acceptance criteria from the plan verified directly (return-type annotation, exact-one-occurrence greps, full test suite pass).

---
*Phase: 44-cpd-weighted-label-transfer-method*
*Completed: 2026-07-10*

## Self-Check: PASSED

All modified files verified present on disk (eval/stages/alignment.py, tests/test_alignment_stage.py, 44-02-SUMMARY.md). Both task commits (07b43e1, d8fff09) confirmed present in git log.
