---
phase: 14-synthetic-data-generators
plan: "03"
subsystem: tests/eval.generators
tags:
  - synthetic-data
  - tests
  - tdd
  - conftest

dependency_graph:
  requires:
    - phase: 14-01
      provides: "eval/generators/__init__.py, generators.py, transforms.py"
    - phase: 14-02
      provides: "eval/generators/corruption.py, labels.py — 7-symbol package"
    - tests/conftest.py (existing fixtures preserved verbatim)
    - src/zreg/metrics/label_transfer.py (compute_f1 — D-07 dtype contract)
  provides:
    - tests/conftest.py — sys.path.insert(0, _repo_root) for eval.generators import
    - tests/test_generators.py — 34-test coverage of all 7 public symbols
  affects:
    - All future tests in tests/ can now `from eval.generators import ...` without sys.path boilerplate
    - Phase 16 (EVAL-05) runners inherit importable eval.generators package

tech_stack:
  added: []
  patterns:
    - "idempotent sys.path guard: if str(_repo_root) not in sys.path: sys.path.insert(0, ...)"
    - "copy.deepcopy snapshot pattern for immutability assertions"
    - "AffineTransformation(t=torch.zeros(3)) for true identity affine (default t=[1,1,1] is NOT identity)"

key_files:
  created:
    - tests/test_generators.py
  modified:
    - tests/conftest.py

decisions:
  - "AffineTransformation() default t=[1,1,1] is not identity — identity affine uses AffineTransformation(t=torch.zeros(3))"
  - "No CUDA skip decorators needed — eval.generators is CPU-only per CONTEXT.md"
  - "No numpy in test file — all assertions use torch.equal / torch.allclose / torch.Tensor properties"
  - "trajectory_data conftest fixture (color=(30,3), id=(30,)) used for 2-D color and sentinel id tests in TestCorruptionWrappers and TestTransformWrappers.test_apply_rigid_preserves_other_fields"
  - "TestLabelUtilities uses inline generate_trajectory (color starts as None) not trajectory_data fixture"

metrics:
  duration_seconds: 360
  completed_date: "2026-05-18"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 1
---

# Phase 14 Plan 03: Test Suite for eval.generators Summary

**34-test comprehensive coverage of all 7 public eval.generators symbols via four locked test classes; conftest extended with idempotent repo-root sys.path insertion.**

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Prepend sys.path block to tests/conftest.py | ba47a0b | tests/conftest.py |
| 2 | Author tests/test_generators.py with four locked test classes | bb5cf40 | tests/test_generators.py |

## Test Count by Class

| Class | Tests | Coverage |
|-------|-------|----------|
| TestGenerateTrajectory | 8 | generate_trajectory: keys, shape, dtype, seed reproducibility, seed=None non-determinism, ValueError x2 |
| TestTransformWrappers | 7 | apply_rigid, apply_affine: identity, new dict, immutability x2, field preservation, translation |
| TestCorruptionWrappers | 11 | add_gaussian_noise (sigma=0 identity, changes pos, reproducible, immutability, ValueError), add_outliers (pos length, sentinel id, 2-D color zeros, immutability, ValueError, zero no-op) |
| TestLabelUtilities | 8 | generate_labels (dtype+shape, value range, reproducible, immutability, ValueError, compute_f1 D-07 contract), remove_labels (color=None, immutability) |
| **TOTAL** | **34** | |

## Repo Test Count After Plan 03

- Pre-existing (Phase 13 + prior): **480 tests**
- New (this plan): **34 tests**
- **Total passing: 514 tests** (15 skipped — pre-existing CUDA-gated tests)

## Contract Coverage: D-01 through D-10

| Decision | Covered? | Test(s) |
|----------|----------|---------|
| D-01: factory + immutable wrappers | Yes | All `_does_not_mutate_input` tests |
| D-02: factory returns `dict[int, zRegPointCloud]` with N(0,I) pos | Yes | `test_returns_dict_with_correct_keys`, `test_pos_shape` |
| D-03: wrappers never mutate input | Yes | 5 `_does_not_mutate_input` tests across all 4 classes |
| D-04: four-file module structure | Implicit (import succeeds) | `from eval.generators import ...` line |
| D-05: importable package with __all__ | Implicit (import succeeds) | All tests |
| D-06: labels in color field | Yes | `test_generate_labels_dtype_and_shape` |
| D-07: torch.long (N,) dtype, compute_f1 compatible | Yes | `test_generate_labels_dtype_and_shape`, `test_generate_labels_compatible_with_compute_f1` |
| D-08: remove_labels sets color=None | Yes | `test_remove_labels_sets_color_none` |
| D-09: four test classes (TestGenerateTrajectory, TestTransformWrappers, TestCorruptionWrappers, TestLabelUtilities) | Yes | All four classes present |
| D-10: tests/conftest.py sys.path.insert addition | Yes | Task 1 + `from eval.generators import` line in test file |

All D-01..D-10 decisions have at least one corresponding test assertion.

## eval.generators Importability Confirmation

After Task 1, `from eval.generators import ...` works in any test module under `tests/` without per-file `sys.path` boilerplate. The idempotent guard (`if str(_repo_root) not in sys.path`) prevents duplicate entries on re-import.

## Key AffineTransformation Identity Deviation

Per Wave 1 SUMMARY.md, `AffineTransformation()` defaults to `t=[1,1,1]` (not the zero vector). Every test requiring an identity affine transform uses `AffineTransformation(t=torch.zeros(3))` explicitly. The plan's inline behavior description (`identity AffineTransformation()`) is interpreted as `AffineTransformation(t=torch.zeros(3))`.

## Deviations from Plan

None — plan executed exactly as written. The AffineTransformation identity deviation was pre-documented in the Wave 1 summary and correctly handled in the test file.

## Known Stubs

None — all 34 tests assert real behaviour. No hardcoded empty values, no TODO comments, no placeholder assertions.

## Threat Flags

No new network endpoints, auth paths, or trust boundary changes introduced. The test file and conftest edit are pure test infrastructure (no I/O, no external data loading).

## Self-Check: PASSED

Files exist:
- tests/conftest.py: FOUND (modified with sys.path.insert)
- tests/test_generators.py: FOUND (new, 34 tests)

Commits exist:
- ba47a0b (Task 1 — conftest.py): FOUND
- bb5cf40 (Task 2 — test_generators.py): FOUND

Verification checks:
- `pytest tests/test_generators.py -x` exits 0 with 34 passing: PASS
- `pytest tests/ -x` exits 0 with 514 passing (480 pre-existing + 34 new): PASS
- `grep -c "^class Test" tests/test_generators.py` returns 4: PASS
- All four test classes pass independently: PASS
- No numpy in test file: PASS
- from eval.generators import (7 symbols) on single import block: PASS
- from zreg.metrics.label_transfer import compute_f1 present: PASS
- tests/conftest.py still has all 5 fixtures and 2 hooks: PASS
