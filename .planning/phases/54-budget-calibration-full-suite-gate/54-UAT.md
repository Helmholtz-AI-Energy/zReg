---
status: complete
phase: 54-budget-calibration-full-suite-gate
source:
  - .planning/phases/54-budget-calibration-full-suite-gate/54-01-SUMMARY.md
  - .planning/phases/54-budget-calibration-full-suite-gate/54-02-SUMMARY.md
started: 2026-07-23T00:00:00Z
updated: 2026-07-23T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. No-arg backward compatibility
expected: Running `python baseline_experiments/scripts/aggregate_cost.py` with no arguments exits 0 or 1 (budget-dependent), writes compute_cost.md, and the first line of that file starts with "WITHIN" or "EXCEEDS".
result: pass

### 2. --calibration flag loads JSON constants
expected: Running `python baseline_experiments/scripts/aggregate_cost.py --calibration baseline_experiments/calibrations/laptop.json` produces the same projection as the no-arg invocation (laptop.json contains the same constants that are hardcoded as defaults). No error or traceback.
result: pass

### 3. --calibration with missing file gives clean error
expected: Running `python baseline_experiments/scripts/aggregate_cost.py --calibration /nonexistent/file.json` prints a readable error to stderr (something like "ERROR: calibration file not found") and exits with a non-zero code — no raw Python traceback.
result: pass

### 4. --configs-dir skip line for missing configs
expected: Running `python baseline_experiments/scripts/aggregate_cost.py --configs-dir /tmp/empty_dir/` (a directory that has no YAML files) prints skip lines for each config that can't be found, produces a projection over zero or very few runs, and exits without a crash or FileNotFoundError traceback.
result: pass

### 5. --budget-hours gate exits 1 when exceeded
expected: Running `python baseline_experiments/scripts/aggregate_cost.py --budget-hours 0.0001` exits with code 1 AND writes compute_cost.md first. Verify: `echo $?` after the command prints `1`, and the file exists with "EXCEEDS" as the first line.
result: pass

### 6. --budget-hours gate exits 0 when within cap
expected: Running `python baseline_experiments/scripts/aggregate_cost.py --budget-hours 999` exits with code 0. First line of compute_cost.md starts with "WITHIN 999h CAP".
result: pass

### 7. extract_calibration.py produces calibration JSON
expected: Running `python baseline_experiments/scripts/extract_calibration.py` with a search_history.json that has window_size 5 and 10 trials prints valid JSON to stdout with keys "5", "10", "20" in cpd_trial_seconds. The "20" entry has `"extrapolated": true`. No file is written — output goes to stdout only.
result: pass

### 8. extract_calibration.py exits 1 on bad input
expected: Running `python baseline_experiments/scripts/extract_calibration.py /nonexistent.json` exits with code 1 and prints a readable error to stderr (not a raw traceback).
result: pass

### 9. Test suite passes
expected: Running `python -m pytest baseline_experiments/tests/ -q` from the repo root exits 0 with all tests passing (at minimum 23 tests — 8 for aggregate_cost.py, 13 for extract_calibration.py, plus 2 regression tests added by code review). No failures or errors.
result: pass

## Summary

total: 9
passed: 9
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
