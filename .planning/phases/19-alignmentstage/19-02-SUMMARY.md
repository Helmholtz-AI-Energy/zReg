---
phase: 19-alignmentstage
plan: "02"
subsystem: eval-framework
tags: [python, pydantic, dtw, cpd, alignment, eval-stages, pytest]

requires:
  - phase: 19-alignmentstage
    plan: "01"
    provides: eval/stages/__init__.py (PipelineStage ABC), tests/test_alignment_stage.py (scaffold)

provides:
  - eval/stages/alignment.py AlignmentStage(PipelineStage) concrete class
  - eval/stages/__init__.py updated to export AlignmentStage
  - tests/test_alignment_stage.py all 4 stubbed classes replaced with real tests (24 tests total)

affects:
  - 20-labeltransferstage (LabelTransferStage will follow same PipelineStage pattern)
  - 21-evaluationrunner (EvaluationRunner calls AlignmentStage via PipelineStage interface)

tech-stack:
  added: []
  patterns:
    - "Thin DTW wrapper: DynamicTimeWarping constructor + .compute() — no reimplementation (FRAME-05)"
    - "validate_params first-line guard in run() (D-09) — TypeError/KeyError never surfaces"
    - "Static _count_jumps heuristic: diagonal-to-non-diagonal transitions in warping_path"
    - "downsample_method=None in DynamicTimeWarping call (works around pre-existing downsampling.py color=None bug)"

key-files:
  created:
    - eval/stages/alignment.py
    - .planning/phases/19-alignmentstage/19-02-SUMMARY.md
  modified:
    - eval/stages/__init__.py
    - tests/test_alignment_stage.py

key-decisions:
  - "downsample_method=None passed to DynamicTimeWarping — generate_trajectory produces clouds with color=None; downsampling.py:374 crashes on None color; disable downsampling at AlignmentStage level (pre-existing bug, out of scope to fix)"
  - "result.aligned_cloud == synthetic_dataset_a (equality) not 'is' (identity) — pydantic v2 validates dict[int, zRegPointCloud] into a new dict; identity check always fails; equality preserves pass-through intent"
  - "REQUIRED_PARAMS and VALID_CPD as class-level tuple attributes (not instance) so test parametrize decorators can reference them without instantiation"
  - "Module docstring documents RigidCPD hardcoding bug (Pitfall 3) with date 2026-05-28 as required by plan"

requirements-completed:
  - FRAME-05

duration: ~10min
completed: 2026-05-28
---

# Phase 19 Plan 02: AlignmentStage Implementation Summary

**AlignmentStage concrete class wrapping DynamicTimeWarping for DTW+CPD alignment with full FRAME-05 gate coverage in tests**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-28T12:43:23Z
- **Completed:** 2026-05-28T12:52:43Z
- **Tasks:** 2
- **Files modified:** 4 (1 created new, 1 updated, 1 updated test file, 1 SUMMARY)

## Accomplishments

- Created `eval/stages/alignment.py` with `AlignmentStage(PipelineStage)`:
  - `REQUIRED_PARAMS = ("window_size", "step", "cpd_penalty", "dtw_dist_fn", "n_breakpoints")`
  - `VALID_CPD = (None, "rigid", "affine", "nonrigid")`
  - `validate_params` raises `ValueError` on missing keys + 5 type/range checks (D-08/D-10)
  - `run()` calls `validate_params` first (D-09), delegates to `DynamicTimeWarping.compute()`
  - `_count_jumps` staticmethod counts diagonal-to-non-diagonal path transitions (D-04/D-05)
  - No `RigidCPD()`, `_backtrace`, or `_compute_accumulated_cost` in implementation
  - Module docstring documents the upstream pairwise_distance_matrix.py hardcoding bug (Pitfall 3)
- Updated `eval/stages/__init__.py`: added `AlignmentStage` import and `__all__` export (D-12)
- Replaced 4 stubbed test classes with real tests (24 total active tests):
  - `TestAlignmentStageRunStandalone`: 1 test — run returns AlignResult with correct fields
  - `TestAlignmentStageValidateParams`: 5+1+6 = 12 parametrized test cases
  - `TestAlignmentDistanceImproves`: 2 tests (seed=0, seed=42)
  - `TestCountJumps`: 7 tests covering all path shapes

## Task Commits

1. **Task 1: Create eval/stages/alignment.py and update __init__.py** - `d770e04` (feat)
2. **Task 2: Populate 4 stubbed test classes** - `f6a68cc` (feat)

## Files Created/Modified

- `eval/stages/alignment.py` — NEW: AlignmentStage(PipelineStage) with validate_params, run, _count_jumps
- `eval/stages/__init__.py` — MODIFIED: added `from .alignment import AlignmentStage` and updated `__all__`
- `tests/test_alignment_stage.py` — MODIFIED: added AlignmentStage import; replaced 4 stubbed classes with 22 new test methods (24 total active)

## Decisions Made

- `downsample_method=None` in `DynamicTimeWarping` call — pre-existing bug in `downsampling.py:374` crashes on `color=None` when both point clouds are the same size (the `else` branch always runs even when `xshape == yshape`). `generate_trajectory` produces clouds with `color=None`. Disabling downsampling at the AlignmentStage level is the correct scope fix (FRAME-05: "wrap, do not reimplement" — the zreg bug is out of scope).
- `result.aligned_cloud == synthetic_dataset_a` (equality) not `is` — pydantic v2 validates `dict[int, zRegPointCloud]` typed fields into a new dict; identity is not preserved. The plan specified identity but pydantic's behavior makes that impossible without changing `AlignResult.aligned_cloud` to `Any` (architectural change). Equality preserves the pass-through intent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] downsample_method=None added to DynamicTimeWarping call**
- **Found during:** Task 1 verification / Task 2 test run
- **Issue:** `downsampling.py:374` throws `TypeError: 'NoneType' object is not subscriptable` when `generate_trajectory` fixtures (which produce `color=None` point clouds) pass through the random downsampler. The `else` branch in `random_down_sample` always executes `target["color"] = target["color"][keep]` even when `xshape == yshape`, crashing on None color.
- **Fix:** Added `downsample_method=None` to `DynamicTimeWarping(...)` constructor in `alignment.py`. This disables downsampling entirely, which is correct for the AlignmentStage use case (same trajectory self-aligned, all frames same size).
- **Files modified:** `eval/stages/alignment.py`
- **Commit:** f6a68cc

**2. [Rule 1 - Bug] test assertion changed from `is` to `==` for aligned_cloud**
- **Found during:** Task 2 test run
- **Issue:** Plan specified `assert result.aligned_cloud is synthetic_dataset_a` but pydantic v2 always creates a new dict when validating `dict[int, zRegPointCloud]` typed fields. The `is` identity check always fails.
- **Fix:** Changed to `assert result.aligned_cloud == synthetic_dataset_a` which verifies pass-through semantics (equal contents) without requiring object identity.
- **Files modified:** `tests/test_alignment_stage.py`
- **Commit:** f6a68cc

## Deferred Items

- **pre-existing bug:** `src/zreg/downsampling.py:374` always executes `target["color"] = target["color"][keep]` even when `xshape == yshape`, crashing when `color=None`. Workaround applied (disable downsampling); root fix requires modifying `downsampling.py` — out of scope for Phase 19 (zreg package).

## Known Stubs

None — all 4 previously stubbed test classes are now fully populated. All 24 tests pass.

## Threat Flags

No new threat surface introduced. All STRIDE mitigations from the plan's threat register are in place:
- **T-19-04 mitigated:** `validate_params` for-loop over REQUIRED_PARAMS raises `ValueError("Missing required param: {key}")`
- **T-19-05 mitigated:** `cpd_penalty not in self.VALID_CPD` whitelist check
- **T-19-06 mitigated:** `window_size > 0` validation rejects <= 0 values
- **T-19-07 mitigated:** `dict(params)` shallow copy passed to AlignResult
- **T-19-08 accepted:** `result.rotations` not referenced in AlignmentStage

## Self-Check: PASSED

- eval/stages/alignment.py: FOUND
- eval/stages/__init__.py: FOUND (updated)
- tests/test_alignment_stage.py: FOUND (updated)
- 19-02-SUMMARY.md: FOUND
- Commit d770e04: FOUND
- Commit f6a68cc: FOUND
- Test result: 24 passed, 0 skipped (628 total, 17 skipped full suite)
- grep -cE "RigidCPD\(|_backtrace|_compute_accumulated_cost" alignment.py: 0

---
*Phase: 19-alignmentstage*
*Completed: 2026-05-28*
