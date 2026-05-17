---
phase: 13-core-metrics-library
plan: 02
subsystem: metrics
tags: [sklearn, f1-score, label-transfer, sentinel-masking, torch, tdd]

# Dependency graph
requires:
  - phase: 13-01
    provides: "src/zreg/metrics/__init__.py with alignment re-exports; src/zreg/metrics/alignment.py"
provides:
  - "src/zreg/metrics/label_transfer.py — production compute_f1 with sentinel masking and weighted default"
  - "src/zreg/metrics/__init__.py updated — compute_f1 re-exported, __all__ now 6 names"
  - "from zreg.metrics import compute_f1 import path fully resolved"
affects: [13-03, downstream eval scripts, HPO pipelines using F1 metric]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sklearn CPU entry point: .detach().cpu().numpy() before sklearn call"
    - "sentinel masking: y_true != -1 drops unlabelled points before metric computation"
    - "TDD RED/GREEN cycle for pure metric functions"

key-files:
  created:
    - src/zreg/metrics/label_transfer.py
    - tests/test_label_transfer_metrics.py
  modified:
    - src/zreg/metrics/__init__.py

key-decisions:
  - "average defaults to 'weighted' (not 'macro') — fixes proto stub bug; 'weighted' is HPO default per REQUIREMENTS §EVAL-02"
  - "Empty-after-masking returns 0.0 (no exception) — contractual per CONTEXT.md and proto stub intent"
  - "No _validate_tensors import — integer label tensors don't need finite-value check (per PATTERNS.md)"
  - "src/zreg/metrics/label_transfer_metrics.py left untouched (per D-03)"

patterns-established:
  - "Label metric sentinel masking: mask = y_true != -1 applied before any sklearn call"
  - "CPU detach pattern: y_true[mask].detach().cpu().numpy() for both masked arrays"
  - "float() cast on sklearn output ensures Python float return, not numpy.float64"

requirements-completed: [EVAL-02]

# Metrics
duration: 7min
completed: 2026-05-17
---

# Phase 13 Plan 02: Label Transfer F1 Metric Summary

**compute_f1 with sentinel masking (y_true != -1), weighted-average default fixing the proto stub bug, and sklearn CPU detach entry point**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-17T17:35:01Z
- **Completed:** 2026-05-17T17:42:56Z
- **Tasks:** 1 (TDD: RED + GREEN commits)
- **Files modified:** 3

## Accomplishments

- Created `src/zreg/metrics/label_transfer.py` with `compute_f1` implementing full REQUIREMENTS §EVAL-02 spec
- Fixed proto stub bug: `average="macro"` hardcoded in `label_transfer_metrics.py` → parametric `average="weighted"` default
- Updated `src/zreg/metrics/__init__.py` to re-export `compute_f1`, completing the `from zreg.metrics import compute_f1` import path
- 17 unit tests covering signature defaults, perfect/wrong predictions, sentinel masking, return type, average variants, shape validation, package import

## Task Commits

TDD execution (RED then GREEN):

1. **RED — failing tests for compute_f1** - `bd69a76` (test)
2. **GREEN — implement compute_f1 + update __init__.py** - `80c69d6` (feat)

## Files Created/Modified

- `src/zreg/metrics/label_transfer.py` — New module; `compute_f1` function with sentinel masking, shape guards, CPU entry, `average="weighted"` default, `float()` return cast
- `src/zreg/metrics/__init__.py` — Added `from .label_transfer import compute_f1`; `__all__` extended to 6 names
- `tests/test_label_transfer_metrics.py` — 17 tests in 8 classes covering all behavior, shape validation, and package import

## Decisions Made

- `average` defaults to `"weighted"` (not `"macro"`): this is the explicit proto stub bug fix; `"weighted"` is the HPO default per REQUIREMENTS §EVAL-02 and gives a score that weights each class by its support, appropriate for imbalanced label distributions common in point cloud evaluation
- Empty-after-masking returns `0.0` with no exception: contractual behaviour chosen in CONTEXT.md — undefined F1 on zero samples maps to 0.0 rather than NaN/exception
- No `_validate_tensors` import: integer label tensors don't carry the finite-value concern that floating-point tensors do; shape guards added explicitly instead (per PATTERNS.md)
- `float(score)` cast ensures Python `float` return type, not `numpy.float64` — required by plan contract

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `from zreg.metrics import compute_f1` is now fully importable
- All six metric functions (5 alignment + 1 label transfer) available via `zreg.metrics`
- Plan 03 (unit tests for the complete metrics surface) can now run against the live implementation
- `src/zreg/metrics/label_transfer_metrics.py` proto stub is preserved and untouched per D-03

## Known Stubs

None — `compute_f1` is fully wired to sklearn with real sentinel masking and no placeholder values.

---
*Phase: 13-core-metrics-library*
*Completed: 2026-05-17*
