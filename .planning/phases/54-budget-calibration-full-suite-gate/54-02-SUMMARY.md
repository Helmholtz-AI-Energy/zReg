---
phase: 54-budget-calibration-full-suite-gate
plan: "02"
subsystem: baseline_experiments/scripts
tags: [calibration, testing, budget-gate, extract-calibration]
dependency_graph:
  requires: ["54-01"]
  provides: ["BUDG-02", "BUDG-03"]
  affects: ["baseline_experiments/scripts/aggregate_cost.py", "baseline_experiments/scripts/extract_calibration.py"]
tech_stack:
  added: []
  patterns:
    - "CLI script with argparse + json.dumps to stdout (extract_calibration.py)"
    - "Linear interpolation/extrapolation for missing calibration window sizes"
    - "pytest with importlib.reload() to reset mutable module-level state between tests"
    - "conftest.py sys.path insertion for scripts-dir imports without package install"
key_files:
  created:
    - baseline_experiments/scripts/extract_calibration.py
    - baseline_experiments/tests/__init__.py
    - baseline_experiments/tests/conftest.py
    - baseline_experiments/tests/test_aggregate_cost.py
    - baseline_experiments/tests/test_extract_calibration.py
  modified:
    - baseline_experiments/scripts/aggregate_cost.py
decisions:
  - "extract_calibration.py exposes extract(files) callable in addition to main(argv) so tests can call directly without subprocess overhead"
  - "Linear least-squares fit through all observed (window_size, mean_seconds) pairs for interpolation/extrapolation; flat extrapolation when only one point observed"
  - "importlib.reload(aggregate_cost) between tests to reset CPD_TRIAL_SECONDS/NO_CPD_TRIAL_SECONDS in-place mutations"
  - "conftest.py in baseline_experiments/tests/ inserts scripts/ dir onto sys.path for import without package install"
metrics:
  duration: "~15min"
  completed: "2026-07-23T14:25:30Z"
  tasks: 2
  files: 6
---

# Phase 54 Plan 02: Extract Calibration Script and Test Suite Summary

Extract CPD calibration timing from crashed HoreKa job search_history.json files and add automated test coverage for both aggregate_cost.py flags and extract_calibration.py behavior.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create extract_calibration.py | b3deab2 | baseline_experiments/scripts/extract_calibration.py |
| 2 | Write tests for aggregate_cost.py and extract_calibration.py | 6516878 | baseline_experiments/tests/__init__.py, conftest.py, test_aggregate_cost.py, test_extract_calibration.py; aggregate_cost.py (bug fix) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed dict-form calibration loading in aggregate_cost.py**
- **Found during:** Task 2 (writing tests for --calibration dict-form output from extract_calibration.py)
- **Issue:** aggregate_cost.py's calibration loader (line 400) stored raw `v` from JSON without extracting `v["seconds"]` when the value is a dict. extract_calibration.py emits `{"seconds": float, "extrapolated": bool}` dict-form values. When loaded by aggregate_cost.py, `_nearest_cpd_cost()` would fail with `TypeError: unsupported operand type(s) for +: 'dict' and 'int'` because it expects float values.
- **Fix:** Changed loader to `(v["seconds"] if isinstance(v, dict) else v)` — accepts both bare-float form (laptop.json) and dict-form (extract_calibration.py output)
- **Files modified:** baseline_experiments/scripts/aggregate_cost.py
- **Commit:** 6516878

## Tests Added (21 new tests)

### test_aggregate_cost.py (8 tests)
- `test_calibration_loads_cpd_override` — --calibration bare-float form overrides CPD_TRIAL_SECONDS[5]
- `test_calibration_missing_key_uses_default` — partial JSON leaves unspecified defaults intact
- `test_calibration_dict_form_accepted` — dict-form values (from extract_calibration.py) loaded correctly
- `test_configs_dir_skip_missing` — empty --configs-dir prints skip lines, no crash
- `test_budget_gate_exit_0` — --budget-hours 999 returns 0
- `test_budget_gate_exit_1` — --budget-hours 0.0001 raises SystemExit(1)
- `test_budget_gate_writes_markdown_before_exit` — compute_cost.md written before exit 1
- `test_verdict_line_at_top` — first line of compute_cost.md starts with WITHIN or EXCEEDS

### test_extract_calibration.py (13 tests)
- `test_two_window_sizes_extrapolates_third` — 5+10 observed → 20 extrapolated: true
- `test_all_window_sizes_no_extrapolation` — all three observed → none extrapolated
- `test_mean_computation` — three 5-window trials (100, 120, 140) → mean 120.0
- `test_no_cpd_trials_excluded_from_cpd_timing` — null-penalty trials don't contaminate CPD mean
- `test_no_cpd_mean` — two no-CPD trials (3.0, 5.0) → no_cpd_trial_seconds ≈ 4.0
- `test_no_cpd_null_when_no_no_cpd_trials` — no_cpd_trial_seconds is null when absent
- `test_two_files_merged` — split across two files yields same result as merged
- `test_elapsed_seconds_field_accepted` — elapsed_seconds fallback field accepted
- `test_missing_duration_field_skipped` — T-54-06: malformed trial skipped without crash
- `test_output_json_schema` — all required keys present; cpd values always dict-form
- `test_empty_no_readable_trials_exits_1` — empty JSON list → exit 1
- `test_nonexistent_file_exits_1` — missing file → exit 1 with stderr error
- `test_stdout_is_valid_json_and_pasteable` — stdout is parseable JSON with cpd_trial_seconds

## Verification Results

- `python -m pytest baseline_experiments/tests/ -v` — 21 passed, 0 failed
- Smoke test: extract_calibration.py with window_size 5+10 history → 20 extrapolated, 5 mean=130.0 ✓
- Exit 1 on nonexistent file with readable stderr error ✓
- Dict-form calibration loading in aggregate_cost.py works without TypeError ✓

## Known Stubs

None. extract_calibration.py emits null for `label_transfer_overhead_seconds` by design (D-06 contract: not computable from search_history.json). This is documented in the output JSON via `_note`.

## Threat Flags

None — no new network endpoints or auth paths introduced. T-54-06 mitigation applied: malformed trial records (missing both duration fields) are skipped with stderr warning rather than raising uncaught exception.

## Self-Check: PASSED

- baseline_experiments/scripts/extract_calibration.py: FOUND
- baseline_experiments/tests/__init__.py: FOUND
- baseline_experiments/tests/conftest.py: FOUND
- baseline_experiments/tests/test_aggregate_cost.py: FOUND
- baseline_experiments/tests/test_extract_calibration.py: FOUND
- Commit b3deab2 (Task 1): FOUND
- Commit 6516878 (Task 2): FOUND
- 21 tests pass: VERIFIED
