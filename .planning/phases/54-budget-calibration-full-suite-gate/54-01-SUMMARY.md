---
phase: 54-budget-calibration-full-suite-gate
plan: "01"
subsystem: infra
tags: [argparse, calibration, budget-gate, cost-projection, json]

# Dependency graph
requires:
  - phase: baseline_experiments
    provides: aggregate_cost.py with static cost projection logic
provides:
  - aggregate_cost.py with --calibration, --configs-dir, --budget-hours flags
  - baseline_experiments/calibrations/laptop.json (measured timing constants)
  - baseline_experiments/calibrations/horeka.json (operator placeholder)
  - Per-environment calibration loading (BUDG-03)
  - Go/no-go budget gate via exit code (BUDG-02)
affects: [54-02, horeka-submission, full-suite-run]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Calibration JSON schema: cpd_trial_seconds (string-keyed dict), no_cpd_trial_seconds, label_transfer_overhead_seconds, _source"
    - "Budget gate: write_markdown() always completes before sys.exit(1); verdict line at position 0 of output"
    - "Dynamic CONFIG_PATHS: rebuilt from --configs-dir each invocation; missing configs print skip line and are excluded"

key-files:
  created:
    - baseline_experiments/calibrations/laptop.json
    - baseline_experiments/calibrations/horeka.json
  modified:
    - baseline_experiments/scripts/aggregate_cost.py

key-decisions:
  - "horeka.json uses empty cpd_trial_seconds dict (not null values) so --calibration horeka.json does not overwrite valid defaults with None before operator fills in real measurements"
  - "write_markdown() always called before sys.exit(1) to ensure compute_cost.md is written even when budget is exceeded"
  - "Verdict line inserted at position 0 of lines list (before the header) so head -1 compute_cost.md always shows pass/fail status"
  - "NO_CPD_TRIAL_SECONDS and LABEL_TRANSFER_OVERHEAD_SECONDS declared global inside main() to allow overwrite from calibration JSON"

patterns-established:
  - "Calibration loading: CPD_TRIAL_SECONDS.update({int(k): v for k, v in data.get('cpd_trial_seconds', {}).items()}) — int() key conversion, .get() with empty dict default for safety"
  - "Two-job gate pattern (BUDG-02): operator runs once with --configs-dir configs/ and once with --configs-dir configs_horeka/ before allocation submission"

requirements-completed: [BUDG-02, BUDG-03]

# Metrics
duration: 25min
completed: 2026-07-23
---

# Phase 54 Plan 01: Budget Calibration & Full-Suite Gate Summary

**argparse gate added to aggregate_cost.py with per-environment calibration JSON loading (laptop.json measured, horeka.json placeholder) and verdict-first budget gate that exits 1 when projected worst case exceeds --budget-hours cap**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-23T00:00:00Z
- **Completed:** 2026-07-23
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Extended `aggregate_cost.py` with three new argparse flags: `--calibration`, `--configs-dir`, `--budget-hours`
- `main()` now accepts `argv=None`; `build_rows()` accepts optional `runs` parameter; `write_markdown()` accepts `worst_total` and `budget_hours` parameters
- Calibration loading merges JSON timing constants into module-level dicts via `CPD_TRIAL_SECONDS.update()`; missing keys leave defaults intact (no KeyError)
- Dynamic CONFIG_PATHS rebuilt from `--configs-dir` each invocation; configs not found on disk print a skip line and are excluded from projection
- Budget gate computes worst_total from projected_range[1] (pending/in-progress) or actual_seconds (done); verdict inserted as first line of compute_cost.md and printed to stdout; `sys.exit(1)` only after `write_markdown()` completes
- `baseline_experiments/calibrations/laptop.json` contains exact measured constants (130.5/176.5/278.6s at ws=5/10/20, 4.0s no-CPD, 30.0s label transfer overhead) with `_source` field documenting the Kobitski ew_06 measurement
- `baseline_experiments/calibrations/horeka.json` is a safe operator placeholder with empty `cpd_trial_seconds` dict and `_instructions` field; does not break the script when passed via `--calibration` before operator fills in values

## Task Commits

Each task was committed atomically:

1. **Task 1: Add argparse + calibration loading to aggregate_cost.py** - `87e35e3` (feat)
2. **Task 2: Create calibration JSON files** - `3c2bf76` (feat)

**Plan metadata:** (to be added by final docs commit)

## Files Created/Modified
- `baseline_experiments/scripts/aggregate_cost.py` - argparse, calibration loading, dynamic CONFIG_PATHS, budget gate, verdict line
- `baseline_experiments/calibrations/laptop.json` - measured timing constants for laptop CPU environment (2026-07-15, Kobitski ew_06)
- `baseline_experiments/calibrations/horeka.json` - operator placeholder with instructions; safe to pass via --calibration before values are filled

## Decisions Made
- `horeka.json` uses empty `cpd_trial_seconds: {}` (not null values) so defaults are untouched when passed via `--calibration` before operator fills in real HoreKa measurements
- `write_markdown()` always called before `sys.exit(1)` — compute_cost.md is written even when budget is exceeded, so the operator has the table to review
- Verdict line at position 0 of `lines` list ensures `head -1 compute_cost.md` always returns the pass/fail status line

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- The verify script in the plan was written to run from the main repo (`sys.path.insert(0, '.')`) but the worktree has the modified file, not the main repo. Ran verification from within the worktree instead — all checks passed identically.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BUDG-02 delivered: two-job gate operational; operator runs `python aggregate_cost.py --configs-dir configs/ --budget-hours 3.0` and `python aggregate_cost.py --configs-dir configs_horeka/ --budget-hours 3.0` to confirm full suite fits allocation before submitting
- BUDG-03 delivered: calibration constants are per-environment via JSON file; `horeka.json` is ready to be filled after running `extract_calibration.py` on HoreKa
- Phase 54-02 can proceed: full-suite gate tooling is ready for integration

---
*Phase: 54-budget-calibration-full-suite-gate*
*Completed: 2026-07-23*
