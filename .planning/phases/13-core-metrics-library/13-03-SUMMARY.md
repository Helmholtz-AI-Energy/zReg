---
phase: 13-core-metrics-library
plan: 03
subsystem: metrics-tests
tags: [pytest, torch, sklearn, chamfer, hausdorff, knn, temporal-stability, path-smoothness, compute-f1, test-suite]

# Dependency graph
requires:
  - phase: 13-01
    provides: Five alignment metric functions in src/zreg/metrics/alignment.py
  - phase: 13-02
    provides: compute_f1 in src/zreg/metrics/label_transfer.py

provides:
  - Full unit test suite for zreg.metrics at tests/test_eval_metrics.py
  - Regression guards for EVAL-01 (alignment metrics) and EVAL-02 (weighted F1 default)
  - 40 CPU-always-run tests; 7 CUDA-gated tests (skip cleanly on CPU-only machines)

affects:
  - Future refactors to src/zreg/metrics/ — any regression caught immediately
  - 14+ (eval/ framework) — test file is the canonical correctness contract

# Tech tracking
tech-stack:
  added: []
  patterns:
    - One test class per public metric function (D-06)
    - All GPU tests gated by @pytest.mark.skipif(not torch.cuda.is_available(), ...) (D-07)
    - known-value inputs with tight atol assertions
    - inspect.signature for parameter-default contract tests (EVAL-02 lock-in)

# Key files
key-files:
  created:
    - tests/test_eval_metrics.py
  modified: []

# Decisions
decisions:
  - "Used tempdir-based test execution (not worktree pytest) to work around a macOS OpenMP+coverage segfault in worktree context. Tests confirmed passing: 40 passed, 7 skipped on CPU-only machine."
  - "Added a second CUDA test to TestHausdorff (device-mismatch) to meet the minimum-7-CUDA-skipif requirement."

# Metrics
metrics:
  duration_minutes: 158
  completed_date: "2026-05-17"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 0
---

# Phase 13 Plan 03: Create Full zreg.metrics Test Suite — Summary

Full unit test suite for `zreg.metrics` written to `tests/test_eval_metrics.py`. All six public metric functions are covered by one test class each (D-06). All 40 CPU tests pass; 7 CUDA-gated tests skip cleanly on CPU-only machines (D-07). Imports exclusively use `zreg.metrics` paths — zero references to the rejected `zreg.eval.metrics` path.

## Test File Structure

File: `tests/test_eval_metrics.py` (363 lines)

| Class | Tests | CPU Tests | CUDA Tests |
|-------|-------|-----------|------------|
| TestChamfer | 8 | 6 | 2 (chamfer_on_cuda, device_mismatch) |
| TestHausdorff | 7 | 5 | 2 (hausdorff_on_cuda, device_mismatch) |
| TestPathSmoothness | 6 | 6 | 0 (no tensor inputs) |
| TestKnnConsistency | 8 | 7 | 1 (knn_on_cuda) |
| TestTemporalStability | 7 | 6 | 1 (temporal_on_cuda) |
| TestComputeF1 | 11 | 10 | 1 (compute_f1_on_cuda) |
| **Total** | **47** | **40** | **7** |

## Verification Results

- `grep -c '^class Test' tests/test_eval_metrics.py` = **6** (required: 6)
- `grep -c 'skipif(not torch.cuda.is_available' tests/test_eval_metrics.py` = **7** (required: >=7)
- `grep -c 'zreg.eval.metrics' tests/test_eval_metrics.py` = **0** (required: 0)
- Total lines: **363** (required: >=250)
- CPU test result: **40 passed, 7 skipped** (all CUDA tests skip cleanly)

## Import Correctness

All imports use `zreg.metrics` (architecturally correct path):

```python
from zreg.metrics.alignment import (
    chamfer, hausdorff, path_smoothness, knn_consistency, temporal_stability,
)
from zreg.metrics.label_transfer import compute_f1
```

Zero occurrences of `zreg.eval.metrics` anywhere in the file.

## Key Test Cases

- **EVAL-02 Contract Lock:** `test_default_average_is_weighted` uses `inspect.signature` to assert `compute_f1`'s `average` parameter defaults to `"weighted"` — prevents regression to the `"macro"` bug from the proto stub.
- **Sentinel masking:** `test_sentinel_masking_skips_minus_one` and `test_all_masked_returns_zero` verify the `-1` label exclusion path.
- **No-NumPy on GPU:** `test_no_numpy_used` asserts Hausdorff returns `torch.Tensor` (not NumPy scalar) on CPU input, confirming `torch.quantile` is used.
- **Unit translation:** `test_unit_translation_returns_one` pins the Frobenius-norm=1.0 result for a single-step translation of magnitude 1.
- **knn alternating labels:** deterministic 1-D grid arrangement ensures neighbour ordering is reliable; score < 0.5 confirmed.

## Deviations from Plan

### Auto-noted Environment Issue

**[Deviation - Environment] pytest-cov segfault in worktree context**
- **Found during:** Task 1 verification
- **Issue:** Running `pytest` from the worktree directory causes a SIGSEGV (exit -11) due to a macOS OpenMP duplicate-library conflict when `pytest-cov` instruments `zreg` (installed from the main repo, not the worktree). The `setup.cfg` `addopts = --cov zreg --cov-report term-missing` triggers this.
- **Fix:** Tests verified by running pytest from a temporary directory (no `setup.cfg` coverage config picked up). Tests pass with RC=0. This is an environment issue, not a code issue. Running `pytest tests/test_eval_metrics.py` from the main repo (once the file is merged) will work correctly.
- **Impact:** None on test correctness. The test file is correct and all 40 CPU tests pass.

### Auto-added Test

**[Rule 2 - Coverage] Added second CUDA skipif to TestHausdorff**
- **Found during:** Task 2 acceptance-criteria check
- **Issue:** Plan required `>=7` CUDA skipif occurrences; initial write had only 6.
- **Fix:** Added `test_hausdorff_device_mismatch_raises` to TestHausdorff (mirrors the chamfer device-mismatch test pattern).
- **Files modified:** `tests/test_eval_metrics.py`

## Known Stubs

None. The test file has no placeholder content.

## Threat Flags

None. Test files do not introduce network endpoints, auth paths, or schema changes.

## Self-Check

- tests/test_eval_metrics.py: FOUND
- commit 95a6581: FOUND
- class count == 6: PASSED
- CUDA skipif count >= 7: PASSED (7)
- zreg.eval.metrics refs == 0: PASSED
- line count >= 250: PASSED (363)
- pytest 40 passed, 7 skipped: PASSED

## Self-Check: PASSED
