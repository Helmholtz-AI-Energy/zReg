---
phase: 44-cpd-weighted-label-transfer-method
plan: 01
subsystem: api
tags: [pydantic, eval-config, cpd, label-transfer, schema]

# Dependency graph
requires:
  - phase: 41-alignment-preprocessing
    provides: "AlignResult frozen model + velocity_landmarks field precedent"
  - phase: 39-icp-registration
    provides: "alignment_method field + validate_alignment_method validator precedent to mirror"
provides:
  - "AlignResult.estep_results: dict[int, EstepResult] field (CPD posterior container, empty by default)"
  - "EvalConfig.label_transfer_method field + validate_label_transfer_method validator ('knn_voting' | 'cpd_weighted')"
affects: [44-02-alignmentstage-posterior-capture, 44-03-labeltransferstage-cpd-weighted]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "field + field_validator pair mirrored from alignment_method/validate_alignment_method"
    - "dict[int, ArbitraryType] frozen-model field (arbitrary_types_allowed=True) mirrored from aligned_cloud"

key-files:
  created: []
  modified:
    - eval/types.py
    - eval/config.py
    - tests/test_eval_config.py

key-decisions:
  - "Imported EstepResult from zreg.cpd (public export), not zreg.cpd._types, per D-09"
  - "Placed EstepResult import between zreg.dataset and torch imports, preserving the documented zreg-before-torch macOS-ARM ordering rule"
  - "label_transfer_method validator has no cross-field dependency (unlike swd_variant's conditional check) since it has no dependent sub-param yet"

patterns-established:
  - "Config-level selectable-method fields (label_transfer_method) mirror the alignment_method field+validator shape exactly for consistency across EvalConfig"

requirements-completed: [D-06, D-09]

# Metrics
duration: 33min
completed: 2026-07-10
---

# Phase 44 Plan 01: Schema/Config Contracts Summary

**`AlignResult.estep_results` (CPD posterior container) and `EvalConfig.label_transfer_method` (selectable label-transfer method with validation) added as pure additive fields — no downstream code touched or broken.**

## Performance

- **Duration:** 33 min
- **Started:** 2026-07-10T13:03:00+02:00 (approx, after prior plan commit)
- **Completed:** 2026-07-10T13:33:00+02:00
- **Tasks:** 2 completed
- **Files modified:** 3 (eval/types.py, eval/config.py, tests/test_eval_config.py)

## Accomplishments
- `AlignResult` gains `estep_results: dict[int, EstepResult] = Field(default_factory=dict)`, keyed the same way as `aligned_cloud` (by target frame key), documented in both docstring `Parameters` and `Attributes` sections.
- `EvalConfig` gains `label_transfer_method: str = Field(default="knn_voting", ...)` plus `validate_label_transfer_method` field_validator restricting values to `{"knn_voting", "cpd_weighted"}`, mirroring `alignment_method`/`validate_alignment_method` exactly.
- 5 new tests in `TestEvalConfigLabelTransferMethodValidation` (invalid value, case sensitivity, empty string, default value, explicit `cpd_weighted`) all pass.
- Full repo test suite (1195 tests) passes with zero regressions after the change — confirms no existing field was retyped/removed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add AlignResult.estep_results field and EvalConfig.label_transfer_method field+validator** - `d642821` (feat)
2. **Task 2: Add TestEvalConfigLabelTransferMethodValidation tests** - `f5420fb` (test)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `eval/types.py` - Added `from zreg.cpd import EstepResult` import; added `estep_results: dict[int, EstepResult] = Field(default_factory=dict)` field to `AlignResult`, with docstring updates in both `Parameters` and `Attributes` sections.
- `eval/config.py` - Added `label_transfer_method: str = Field(default="knn_voting", ...)` field to `EvalConfig` (placed after `swd_variant`, before `alignment_preprocessing`); added `validate_label_transfer_method` classmethod validator (placed after `validate_swd_variant`); added docstring `Attributes` entry.
- `tests/test_eval_config.py` - Added `TestEvalConfigLabelTransferMethodValidation` class with 5 tests, placed immediately after `TestEvalConfigAlignmentMethodValidation`.

## Decisions Made
- Mirrored `alignment_method`/`validate_alignment_method` exactly for `label_transfer_method`/`validate_label_transfer_method` (field declaration style, validator docstring shape, error message wording) — no invention needed, per PATTERNS.md's exact-match analog.
- `EstepResult` import placed between the `zreg.dataset` import and `import torch` in `eval/types.py`, preserving the file's documented zreg-before-torch macOS-ARM import-ordering rule (libomp SIGABRT lesson from Phase 12).
- `estep_results` field placed immediately after `velocity_landmarks` in `AlignResult`'s class body, matching the plan's exact placement instruction.

## Deviations from Plan

None - plan executed exactly as written. Both tasks matched the plan's `<action>` and `<acceptance_criteria>` blocks verbatim; no Rule 1/2/3/4 triggers encountered.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `AlignResult.estep_results` and `EvalConfig.label_transfer_method` are now available for Plan 44-02 (`AlignmentStage` posterior capture, D-01/D-02/D-03) and Plan 44-03 (`LabelTransferStage` cpd_weighted branch, D-04/D-05/D-07/D-08), which can now be planned/executed independently (wave 2) since the shared typed contracts are established.
- No blockers. All acceptance criteria from the plan verified directly (field presence, default value, validator rejection, import source).

---
*Phase: 44-cpd-weighted-label-transfer-method*
*Completed: 2026-07-10*

## Self-Check: PASSED

All created/modified files verified present on disk (eval/types.py, eval/config.py, tests/test_eval_config.py, SUMMARY.md). Both task commits (d642821, f5420fb) confirmed present in git log.
