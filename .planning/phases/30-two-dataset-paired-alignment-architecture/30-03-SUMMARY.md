---
phase: 30-two-dataset-paired-alignment-architecture
plan: "03"
subsystem: eval-framework
tags: [paired-alignment, scenario-config, test-mocks, tdd, mode-01]
dependency_graph:
  requires: [30-01, 30-02]
  provides: [configs/paired_alignment.yaml, MODE-01-complete]
  affects: [tests/test_trajectory_export.py, tests/test_cli.py]
tech_stack:
  added: []
  patterns: [tdd-red-green, yaml-eval-config, kobitski-self-alignment-smoke-test]
key_files:
  created:
    - configs/paired_alignment.yaml
  modified:
    - tests/test_trajectory_export.py
    - tests/test_cli.py
decisions:
  - "paired_alignment.yaml uses Kobitski tracklets as both data_path and target_data_path (D-03 smoke-test: validates mechanics before true heterogeneous paired data)"
  - "n_synthetic=0 chosen for paired_alignment.yaml — no synthetic data needed when real target dataset is provided"
  - "test_paired_alignment_yaml_loads_and_declares_paired_mode added as dedicated method in TestScenarioConfigs (not added to _SCENARIO_TABLE parametrize — different field set than D-04 table)"
metrics:
  duration: "~8 minutes"
  completed: "2026-06-12"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 2
  tests_before: 851
  tests_after: 839
  tests_skipped: 18
  note: "851 was Plan 30-02 count; 839 passed + 18 skipped = 857 total collected — net +6 (1 new CLI test + 2 restored trajectory integration tests that were failing)"
---

# Phase 30 Plan 03: paired_alignment.yaml + Test Mock Fixes Summary

Phase 30 D-03 scenario config shipped and remaining test-mock gap closed: `configs/paired_alignment.yaml` validated via `EvalConfig.from_yaml`; two `TestExportTrajectoryIntegration` mock blocks in `test_trajectory_export.py` receive `load_target.return_value`; `TestScenarioConfigs` gains a Phase-30-specific smoke-test in `test_cli.py`. Full suite exits 0.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Add failing test for paired_alignment.yaml smoke | 95bc2e9 | tests/test_cli.py |
| GREEN | Create paired_alignment.yaml; fix test mocks | 175819a | configs/paired_alignment.yaml, tests/test_trajectory_export.py, tests/test_cli.py |

## Files Created / Modified

| File | Key Changes |
|------|-------------|
| configs/paired_alignment.yaml | NEW — 27 lines; 15 declared EvalConfig keys; pipeline_mode=paired; Kobitski tracklets as both data_path and target_data_path; n_synthetic=0; run_alignment=true; run_label_transfer=false |
| tests/test_trajectory_export.py | +2 lines — `mock_factory.load_target.return_value = synthetic_dataset` added after each `load_real.return_value` line in both @patch DataFactory integration blocks |
| tests/test_cli.py | +11 lines — `test_paired_alignment_yaml_loads_and_declares_paired_mode` method added to TestScenarioConfigs |

## paired_alignment.yaml Keys Declared (15 total)

```
data_path, target_data_path, data_format, pipeline_mode,
tier, n_trials, n_synthetic, run_alignment, run_label_transfer,
default_params (window_size, step, cpd_penalty, dtw_dist_fn, n_breakpoints),
output_dir
```

`target_data_format` intentionally absent (deferred per CONTEXT).

## Mock Block Update Count

| File | @patch DataFactory blocks updated | load_real count | load_target count |
|------|-----------------------------------|-----------------|-------------------|
| tests/test_trajectory_export.py | 2 | 2 | 2 |

Every `load_real.return_value` now has a paired `load_target.return_value` (D-04 Kobitski-vs-Kobitski smoke semantics).

## Phase 30 Final Test Count

| Scope | Count |
|-------|-------|
| Passed | 839 |
| Skipped | 18 |
| Failed | 0 |
| Total collected | 857 |

Pre-Phase-30 baseline was 826. Net new tests across all three plans: +31 (Plan 30-01: +9 active, -1 deleted; Plan 30-02: 0; Plan 30-03: +1 CLI + 2 trajectory integration restored = +3).

## Verification Commands Passed

```
pytest tests/ -x -q  → 839 passed, 18 skipped, exit 0
python -c "from eval.config import EvalConfig; cfg = EvalConfig.from_yaml('configs/paired_alignment.yaml'); assert cfg.pipeline_mode == 'paired'; assert cfg.target_data_path == cfg.data_path; print('OK')"  → OK
grep -c "mock_factory.load_target.return_value" tests/test_trajectory_export.py  → 2
grep -n "paired_alignment" tests/test_cli.py  → 3 lines
```

## MODE-01 Requirement Closure

All 5 MODE-01 success criteria are now satisfied:

| Criterion | Status |
|-----------|--------|
| 1. AlignmentStage.run(source, target, params) signature | Complete (Plan 30-01) |
| 2. DataFactory.load_target() loads target_data_path | Complete (Plan 30-01) |
| 3. EvaluationRunner.run() loads both source and target | Complete (Plan 30-02) |
| 4. HyperparamOptimizer._objective dispatches source/target by tier | Complete (Plan 30-02) |
| 5. paired_alignment.yaml scenario config runs end-to-end without errors | Complete (Plan 30-03) |

MODE-01 in `.planning/REQUIREMENTS.md` traceability table can be marked **Complete**.

## Deviations from Plan

**1. [Rule 2 - TDD] test_trajectory_export.py mock blocks were blocking integration tests, not a new deviation**

The plan correctly identified these as existing failures from Plan 30-02 wiring (not new bugs introduced by Plan 30-03). The fix (adding `load_target.return_value`) was the direct plan-specified action, not an auto-fix deviation.

No unplanned deviations. Plan executed exactly as written with standard TDD RED/GREEN cycle.

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (test commit) | 95bc2e9 | PASSED — test_paired_alignment_yaml_loads_and_declares_paired_mode failed with EvalConfigError: file not found |
| GREEN (feat commit) | 175819a | PASSED — 839 tests pass, 0 failed |

## Known Stubs

None — `configs/paired_alignment.yaml` references a real file path (Kobitski tracklets confirmed present per RESEARCH Environment Availability). All production code paths are functional.

## Threat Flags

No new threat surface beyond the plan's threat model:
- T-30-07 (Tampering/paired_alignment.yaml): mitigated — EvalConfig extra="forbid" + Literal validation inherited; YAML contains no executable content
- T-30-08 (Information Disclosure/data_path): accepted — same posture as alignment_sanity.yaml (same file path already committed)

## Self-Check: PASSED

- configs/paired_alignment.yaml: FOUND — 27 lines, pipeline_mode: paired at line 13
- grep pipeline_mode: paired → line 13 (exactly 1 match)
- grep target_data_path → line 11 (exactly 1 match)
- grep kobitski tracklets → 2 lines (data_path + target_data_path)
- grep target_data_format → 0 lines (correctly absent)
- python EvalConfig.from_yaml smoke → OK
- mock_factory.load_target.return_value count in test_trajectory_export.py → 2
- load_real count (2) == load_target count (2) → equal=YES
- grep paired_alignment in test_cli.py → 3 lines
- pytest tests/ -x -q → 839 passed, 18 skipped, exit 0
- Commit 95bc2e9: FOUND
- Commit 175819a: FOUND
