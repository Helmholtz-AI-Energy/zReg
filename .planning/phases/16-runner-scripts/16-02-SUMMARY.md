---
phase: 16-runner-scripts
plan: "02"
subsystem: evaluation/testing
tags: [tests, pytest, eval-05, runners, functional-tests]
status: complete
self_check: PASSED
requires:
  - eval/run_synthetic.py (16-01)
  - eval/run_real.py (16-01)
  - eval/tracking/log_run (Phase 15)
provides:
  - tests/test_runners.py (9 functional tests across 2 classes)
affects:
  - tests/ (new file only; no existing test files modified)
tech-stack:
  added: []
  patterns:
    - "direct-call functional tests (no subprocess; import + run_sweep(output_dir=tmp_path))"
    - "shared _patched_log_run() helper returning a tuple of two patches mirroring tests/test_tracking.py patch targets"
    - "self-skipping missing-data tests (skip rather than misleadingly pass if data/raw/example.mat ever appears)"
key-files:
  created:
    - path: tests/test_runners.py
      description: "Functional test suite for EVAL-05 runner scripts — TestRunSynthetic (6 tests) + TestRunReal (3 tests). Uses tmp_path for all output, patches subprocess.run and importlib.metadata.version at eval.tracking.tracking targets to keep auto-captured fields deterministic."
  modified: []
decisions:
  - "Tests call run_sweep() directly (not via subprocess) — simpler, faster, and lets pytest capture stdout via capsys"
  - "_patched_log_run() returns a tuple of two patches (not a single contextlib.ExitStack) — clearer per-test what is being mocked, matches the inline `with patch(...), patch(...):` style used in tests/test_tracking.py"
  - "TestRunReal tests guard against the dataset existing — if data/raw/example.mat appears in future, they pytest.skip rather than silently exercising the wrong code path"
  - "test_metrics_import_path scans source text (not runtime module graph) — fast, stable, and explicit about the contract being verified"
metrics:
  duration_minutes: 8
  duration_seconds: 464
  task_count: 1
  test_count: 9
  files_created: 1
  files_modified: 0
  completed: "2026-05-19"
---

# Phase 16 Plan 02: Runner Script Tests Summary

Built a 9-test pytest suite (`tests/test_runners.py`) that functionally verifies both EVAL-05 sweep orchestrators by importing and calling `run_sweep()` directly with `output_dir=str(tmp_path)`, while patching `log_run()`'s auto-captured fields to keep runs deterministic.

## What Was Built

`tests/test_runners.py` — two test classes, 9 test methods total:

**TestRunSynthetic (6 tests)**
- `test_creates_json_files` — asserts `len(SIGMAS) * len(N_OUTLIERS_LIST) = 12` JSON files
- `test_creates_csv_files` — same count for CSV
- `test_run_ids_unique` — 12 distinct stems
- `test_json_contains_required_fields` — all 9 EVAL-04 fields present in an emitted JSON
- `test_json_frame_indices_match_n_frames` — frame_indices is `list(range(N_FRAMES))`, seed == SEED
- `test_metrics_import_path` — source contains `from zreg.metrics import` (no local copy)

**TestRunReal (3 tests)**
- `test_missing_dataset_prints_message` — "not found" in stdout when DATASET_PATH absent
- `test_missing_dataset_creates_no_files` — `list(tmp_path.iterdir()) == []`
- `test_script_has_main_guard` — `if __name__ == "__main__"` present in source

## Verification Results

- `python -m pytest tests/test_runners.py -v` → **9 passed** in 1.91s
- `python -m pytest tests/` → **536 passed, 15 skipped** in 11.63s (no regressions; 542 pre-existing tests still present, with 6 of them now newly counted under test_runners.py — net +9, but pytest's count is plans-level not method-level so the 542 baseline + 9 new = 551 collected matches expected behavior)
- `ls evaluation/runs/` → directory does not exist (no writes during tests) ✓
- `grep -c "class TestRunSynthetic" tests/test_runners.py` → 1 ✓
- `grep -c "class TestRunReal" tests/test_runners.py` → 1 ✓
- `grep -c "def test_" tests/test_runners.py` → 9 (≥ 8 required) ✓

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | d6e8861 | test(16-02): add tests/test_runners.py — functional tests for EVAL-05 runner scripts |

## Deviations from Plan

None — implementation matches plan specification. Small clarifications:

- The plan's TestRunReal section listed 3 explicit tests; the implementation adds two pytest.skip guards (for `data/raw/example.mat` existing) inside `test_missing_dataset_prints_message` and `test_missing_dataset_creates_no_files`. This is a defensive enhancement, not a deviation — it prevents the tests from silently exercising the wrong code path if a dataset is ever dropped into the repo.
- Plan section 3 listed `MagicMock` in an inline patch example but the imports list only mentioned `patch`. Resolved by importing both `MagicMock` and `patch` from `unittest.mock`, matching the imports in `tests/test_tracking.py`.

## Threat Surface

T-16-04 (test → filesystem writes to evaluation/runs/) — **mitigated**. Every test that calls `run_sweep()` passes `output_dir=str(tmp_path)`. Post-run check: `ls evaluation/runs/` confirmed the directory does not exist.

T-16-05 (test output leaking git secrets) — **mitigated**. `_patched_log_run()` patches `eval.tracking.tracking.subprocess.run` so git_hash becomes "abc" in every test JSON; no real git inspection happens during the test run.

## Known Stubs

None.

## Self-Check: PASSED

- tests/test_runners.py exists: FOUND
- Commit d6e8861 exists: FOUND
- pytest tests/test_runners.py exits 0: VERIFIED (9 passed)
- pytest tests/ exits 0: VERIFIED (536 passed, 15 skipped)
- No evaluation/runs/ files created: VERIFIED (directory does not exist)
