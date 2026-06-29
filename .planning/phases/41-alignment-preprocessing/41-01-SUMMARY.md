---
phase: 41-alignment-preprocessing
plan: 01
subsystem: api
tags: [pydantic, torch, pca, svd, preprocessing, alignment, dtw]

# Dependency graph
requires:
  - phase: 19-alignmentstage
    provides: AlignResult model and AlignmentStage.run contract
  - phase: 40-alignment-method-selection
    provides: EvalConfig.alignment_method / swd_variant precedent for adding alignment options
provides:
  - "zreg.preprocessing module with compute_pca_rotation (proper-rotation PCA) and detect_velocity_landmarks"
  - "AlignmentPreprocessingConfig Pydantic model (method Literal, velocity_threshold/metric defaults, extra=forbid)"
  - "EvalConfig.alignment_preprocessing field (default None, backward compatible)"
  - "AlignResult.velocity_landmarks field (default_factory=list, backward compatible)"
affects: [41-02, alignment-stage-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PCA rotation via torch.linalg.svd with single determinant-sign fix to guarantee det=+1 (no reflection)"
    - "Pydantic v2 Literal annotations for config validation (no @field_validator needed)"
    - "Optional nested config via `X | None = None` with automatic dict coercion (backward compatible)"

key-files:
  created:
    - src/zreg/preprocessing.py
    - tests/test_preprocessing.py
    - tests/test_alignment_preprocessing_config.py
  modified:
    - src/zreg/__init__.py
    - eval/config.py
    - eval/types.py

key-decisions:
  - "Single det-sign fix (flip least-significant source axis) rather than axis-by-axis sign flipping — sufficient to guarantee proper rotation"
  - "Literal annotations carry all method/metric validation; no explicit @field_validator added"
  - "velocity_landmarks uses Field(default_factory=list) not default=[] to avoid the shared-mutable-default anti-pattern"

patterns-established:
  - "Pattern: zreg preprocessing primitives are stateless functions taking torch.Tensor / dict[int, zRegPointCloud]"
  - "Pattern: new alignment options are added as optional, default-None nested Pydantic models for backward compatibility"

requirements-completed: [ALIGN-06-01, ALIGN-06-02, ALIGN-06-03]

# Metrics
duration: 3min
completed: 2026-06-29
---

# Phase 41 Plan 01: Alignment Preprocessing Contracts Summary

**PCA principal-axes rotation + velocity-landmark detection in a new zreg.preprocessing module, plus the AlignmentPreprocessingConfig model and AlignResult.velocity_landmarks field that plan 41-02 will wire into AlignmentStage.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-29T15:24:24+02:00
- **Completed:** 2026-06-29T15:26:37+02:00
- **Tasks:** 2 (both TDD)
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments
- `src/zreg/preprocessing.py` with `compute_pca_rotation` (SVD-based, det-sign-fixed proper rotation) and `detect_velocity_landmarks` (per-frame mean/max displacement thresholding, frame 0 always excluded)
- `preprocessing` sub-module registered in `zreg/__init__.py` — `from zreg import preprocessing` works
- `AlignmentPreprocessingConfig` Pydantic model with `method` Literal, `velocity_threshold=0.5`, `velocity_metric="mean"`, `extra="forbid"`
- `EvalConfig.alignment_preprocessing` field (default `None`) — existing configs remain valid; nested dict coerces automatically
- `AlignResult.velocity_landmarks: list[int] = Field(default_factory=list)` — all existing AlignResult construction sites unaffected

## Task Commits

Each task followed the TDD RED → GREEN cycle:

1. **Task 1 (RED): failing preprocessing tests** - `313ca73` (test)
2. **Task 1 (GREEN): preprocessing module + __init__ export** - `32ef83c` (feat)
3. **Task 2 (RED): failing config/types tests** - `7cc4e10` (test)
4. **Task 2 (GREEN): AlignmentPreprocessingConfig + velocity_landmarks** - `a6a99a5` (feat)

No REFACTOR commits were needed (implementations were minimal and clean).

## Files Created/Modified
- `src/zreg/preprocessing.py` - New: `compute_pca_rotation`, `detect_velocity_landmarks`, `__all__`
- `src/zreg/__init__.py` - Added `from . import preprocessing as preprocessing`
- `eval/config.py` - Added `AlignmentPreprocessingConfig` class, updated `__all__`, added `EvalConfig.alignment_preprocessing` field
- `eval/types.py` - Added `velocity_landmarks` field to `AlignResult`
- `tests/test_preprocessing.py` - New: 9 tests for the two library functions
- `tests/test_alignment_preprocessing_config.py` - New: 10 tests for config/types contracts

## Decisions Made
- **Single determinant-sign fix** for PCA rotation: when `det(R) < 0`, flip only the least-significant source axis and recompute. This guarantees `det = +1` without axis-by-axis sign flipping (the documented anti-pattern). Verified to recover a 90° Z-rotation with mean L2 residual < 0.1.
- **No explicit `@field_validator`** on the new config — Pydantic v2 validates `Literal` annotations and rejects unknown values at parse time; `extra="forbid"` rejects unknown keys.
- **`default_factory=list`** for `velocity_landmarks` (not `default=[]`) to avoid a shared mutable default across instances.

## Deviations from Plan

None - plan executed exactly as written. The plan referenced 41-PATTERNS.md / 41-RESEARCH.md / 41-CONTEXT.md which are not present in the worktree, but each task's `<action>` block contained the full verified specification (function bodies, field declarations, validation rules), so no information was missing.

## Issues Encountered

**Pre-existing, out-of-scope test failures (NOT caused by this plan).**
Running the full suite surfaced 3 failures in `tests/test_eval_config.py::TestEvalConfigAlignmentMethodValidation`. These assert the validator message `"alignment_method must be 'cpd' or 'icp'"`, but the validator was changed in Phase 40 (SWD support) to emit `"...'cpd', 'icp', or 'swd'"`. Confirmed pre-existing at base commit `5853e6e` and untouched by this plan's diff. Logged to `deferred-items.md` per the scope boundary; not fixed here. Suggested fix: update the three `match=` regexes in a future cleanup plan.

All 19 new tests pass. Aside from the 3 pre-existing failures, the full suite is green (1114 passed, 18 skipped, 1 xpassed).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All contracts plan 41-02 consumes are in place: `zreg.preprocessing.compute_pca_rotation` / `detect_velocity_landmarks`, `AlignmentPreprocessingConfig`, `EvalConfig.alignment_preprocessing`, `AlignResult.velocity_landmarks`.
- Backward compatibility verified: `EvalConfig(data_path='x.mat').alignment_preprocessing is None` and `AlignResult(...).velocity_landmarks == []` without passing the new fields.
- No blockers for plan 41-02 (AlignmentStage wiring).

## Self-Check: PASSED
- `src/zreg/preprocessing.py` — FOUND
- `tests/test_preprocessing.py` — FOUND
- `tests/test_alignment_preprocessing_config.py` — FOUND
- Commit `313ca73` (RED task 1) — FOUND
- Commit `32ef83c` (GREEN task 1) — FOUND
- Commit `7cc4e10` (RED task 2) — FOUND
- Commit `a6a99a5` (GREEN task 2) — FOUND

## TDD Gate Compliance
Both tasks completed a `test(...)` (RED) → `feat(...)` (GREEN) commit sequence. RED commits confirmed failing before implementation; GREEN commits confirmed all tests passing. No unexpected passes during RED.

---
*Phase: 41-alignment-preprocessing*
*Completed: 2026-06-29*
