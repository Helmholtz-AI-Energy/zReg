---
phase: 24-trajectory-export
plan: "02"
subsystem: eval/types, eval/runners, tests
tags: [trajectory-export, eval-report, eval-runner, tdd, ext-01]
dependency_graph:
  requires:
    - eval/tracking/trajectory.py (export_trajectory — built in Plan 24-01)
    - eval/types.py (EvalReport model)
    - eval/runners/eval_runner.py (EvaluationRunner.run())
  provides:
    - eval/types.py (trajectory_paths field on EvalReport)
    - eval/runners/eval_runner.py (export_trajectory wired into run())
    - tests/test_trajectory_export.py (EXT-01 gate test suite)
  affects:
    - eval_report.json (now contains trajectory_paths key)
tech_stack:
  added: []
  patterns:
    - TDD RED/GREEN cycle per task
    - frozen pydantic model_copy pattern (single call for both plot_paths + trajectory_paths)
    - unconditional trajectory export (D-13: no save_trajectories flag)
key_files:
  created:
    - tests/test_trajectory_export.py
  modified:
    - eval/types.py
    - eval/runners/eval_runner.py
    - tests/test_metrics.py
decisions:
  - "trajectory_paths defaults to [] on EvalReport — backward-compatible with all existing frozen model usage"
  - "Single model_copy call updates both plot_paths and trajectory_paths (D-14) — never two chained calls"
  - "trajectory_paths=[] added to preliminary_report construction so frozen model has field from creation"
  - "export_trajectory import added at module level not inline in run()"
metrics:
  duration: "~7 minutes"
  completed: "2026-06-04"
  tasks_completed: 2
  files_created: 1
  files_modified: 3
  tests_added: 24
  tests_total: 787
---

# Phase 24 Plan 02: Trajectory Export — EvalReport Wiring + EXT-01 Tests Summary

**One-liner:** trajectory_paths field added to frozen EvalReport, export_trajectory wired unconditionally into EvaluationRunner.run() via single model_copy, and 21-test EXT-01 suite passes end-to-end.

## What Was Built

### eval/types.py — trajectory_paths field

Added `trajectory_paths: list[str] = Field(default_factory=list)` to `EvalReport`, positioned between `plot_paths` and `sanity_flags` in field declaration order. Both Parameters and Attributes docstring sections updated. Defaults to `[]` for backward compatibility — all existing frozen model constructions continue to work without providing the new field.

### eval/runners/eval_runner.py — export_trajectory wiring

Three targeted changes:

1. **Import** — `from eval.tracking import export_trajectory` added to top-level import block alongside `from eval.viz import ...`
2. **preliminary_report** — `trajectory_paths=[]` added as explicit keyword arg in the `EvalReport(...)` construction so the frozen model has the field set from creation
3. **Step 2b + model_copy** — after the `save_plots` branch (Step 2), new Step 2b calls `export_trajectory(result, dataset, self.config, output_dir_path)`. The existing single-field `model_copy(update={"plot_paths": plot_paths})` is replaced with a two-field call: `model_copy(update={"plot_paths": plot_paths, "trajectory_paths": trajectory_paths})` — one atomic update for both fields (D-14).

### tests/test_trajectory_export.py — EXT-01 gate test suite

New file with 5 test classes and 21 tests:

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestExportTrajectoryAlignOnly` | 5 | align CSV created, label absent, header, row count, return length 2 |
| `TestExportTrajectoryLabelOnly` | 5 | label CSV created, align absent, header, row count, D-08 raw positions |
| `TestExportTrajectoryCombined` | 4 | both CSVs, return length 4, same run_id across metadata, D-07 aligned positions |
| `TestExportTrajectoryMetadata` | 5 | metadata file created, all 11 fields present, UUID valid, git_hash and zreg_version captured |
| `TestExportTrajectoryIntegration` | 2 | EvaluationRunner.run() returns >=2 trajectory_paths; eval_report.json contains trajectory_paths key |

### tests/test_metrics.py — Task 1 TDD gate

Added `TestEvalReportTrajectoryPaths` (3 tests) to satisfy TDD RED/GREEN cycle for the `eval/types.py` field addition. Tests assert: field defaults to `[]`, appears immediately after `plot_paths`, and appears before `sanity_flags` in `model_fields` order.

## TDD Cycle

**Task 1 — trajectory_paths field:**
- **RED** commit `72f8e94`: 3 failing tests in `TestEvalReportTrajectoryPaths` (field does not exist)
- **GREEN** commit `0411018`: field declaration + docstring edits in `eval/types.py`, all 3 pass

**Task 2 — eval_runner wiring:**
- **RED** commit `7d42091`: 21-test file created; tests 1-20 pass (export_trajectory already works), test 21 fails (`trajectory_paths=[]` confirms wiring absent)
- **GREEN** commit `0fd6756`: 3 edits to `eval_runner.py`; all 21 tests pass

## TDD Gate Compliance

- RED gate commit (test): `72f8e94`, `7d42091`
- GREEN gate commit (feat): `0411018`, `0fd6756`
- Both gates present and in correct order.

## Deviations from Plan

None — plan executed exactly as written.

The RED phase for Task 2 revealed that `export_trajectory` tests (tests 1-20) passed immediately because the function was built in Wave 1 (Plan 24-01). Only the `eval_runner.py` integration test was genuinely RED. This is correct behavior — the RED gate requirement is satisfied because at least one test in the new file failed before implementation.

## Known Stubs

None — all trajectory_paths entries are live file-path strings written to disk; no placeholder values.

## Threat Flags

No new threat surface beyond what was documented in the plan's threat model.

- T-24-04 (Tampering — trajectory_paths injection): mitigated — `Field(default_factory=list)` with frozen model enforces type validation; `model_copy` is the only mutation path.
- T-24-05 (Information Disclosure): accepted per plan — local researcher output only.
- T-24-06 (DoS — export_trajectory raises): accepted per plan — same risk profile as `save_report`.

## Verification Results

1. `python -m pytest tests/test_trajectory_export.py -v` — 21/21 passed
2. `python -m pytest tests/ -q` — 787 passed, 17 skipped (no regressions)
3. `grep "trajectory_paths" eval/runners/eval_runner.py` — 5 lines: comment, field init, call assignment, comment, model_copy key
4. `grep "trajectory_paths" eval/types.py` — 3 lines: Parameters docstring, Attributes docstring, field declaration
5. `python -m pytest tests/test_metrics.py::TestEvalReportTrajectoryPaths` — 3/3 passed; trajectory_paths is between plot_paths and sanity_flags

## Self-Check: PASSED
