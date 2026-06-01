---
phase: 23-cli-entrypoint-scenario-configs
plan: "02"
subsystem: testing
tags: [yaml, pydantic, pytest, parametrize, eval-config, frame-12]

requires:
  - phase: 23-01
    provides: run_eval.py CLI entrypoint + 8 FRAME-11 tests in tests/test_cli.py (TestScenarioConfigs placeholder)
  - phase: 17
    provides: EvalConfig (pydantic v2, extra=forbid, from_yaml) and EvalConfigError

provides:
  - 5 scenario YAML configs in configs/ with D-04 tier/stage-flag matrix
  - TestScenarioConfigs class with 10 parametrised FRAME-12 tests (5 load-without-error + 5 D-04-value assertions)
  - FRAME-12 closed — all 5 scenario configs load cleanly via EvalConfig.from_yaml

affects:
  - Phase 23 verification (23-VALIDATION.md manual smoke check references these configs)
  - Future CLI users (configs/ is the primary entry point for running experiments)

tech-stack:
  added: []
  patterns:
    - "pytest.mark.parametrize over a module-level scenario table for config load tests"
    - "7-key minimal YAML (data_path, data_format, tier, n_trials, n_synthetic, run_alignment, run_label_transfer) — all omitted fields use EvalConfig defaults"

key-files:
  created:
    - configs/alignment_sanity.yaml
    - configs/alignment_dev.yaml
    - configs/label_transfer_sanity.yaml
    - configs/label_transfer_dev.yaml
    - configs/combined_full.yaml
  modified:
    - tests/test_cli.py

key-decisions:
  - "D-03: all 5 configs point to the same kobitski tracklets file; data_format=tracklets"
  - "D-04: config differentiation by tier (sanity/dev/full) and stage flags only; no search_space in scenario configs"
  - "TestScenarioConfigs uses _SCENARIO_TABLE list-of-tuples with filename stems as parametrize ids for human-readable test output"
  - "is True / is False for bool assertions in test_config_has_d04_values to avoid silent int/bool comparison"

patterns-established:
  - "Parametrized config-load test: _SCENARIO_TABLE at module scope, ids from filename stem, Path(__file__).parent.parent for repo root resolution"

requirements-completed:
  - FRAME-12

duration: 8min
completed: 2026-06-01
---

# Phase 23 Plan 02: Scenario YAML Configs & FRAME-12 Tests Summary

**5 scenario YAML configs (alignment/label-transfer/combined, sanity/dev/full tiers) for EvalConfig.from_yaml, plus 10 parametrised FRAME-12 tests that verify D-04 field values — FRAME-12 closed, milestone v1.2 complete**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-01T13:00:00Z
- **Completed:** 2026-06-01T13:08:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created 5 scenario YAML configs in `configs/` with exactly 7 keys each (data_path, data_format, tier, n_trials, n_synthetic, run_alignment, run_label_transfer); all load via `EvalConfig.from_yaml` without raising `EvalConfigError`
- Replaced `TestScenarioConfigs.test_placeholder` skip with 2 parametrised test methods covering all 5 scenarios (10 tests total); 0 skips remaining in that class
- Full test suite: 741 passed, 17 skipped — up from 731 pre-plan baseline (+10 new tests, -1 skip placeholder)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write all 5 scenario YAML configs in configs/** - `5f4e749` (feat)
2. **Task 2: Populate TestScenarioConfigs with parametrised load + field-value assertions** - `5fa2777` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `configs/alignment_sanity.yaml` — sanity tier, 3 trials, 20 synthetic, alignment-only
- `configs/alignment_dev.yaml` — dev tier, 10 trials, 50 synthetic, alignment-only
- `configs/label_transfer_sanity.yaml` — sanity tier, 3 trials, 20 synthetic, label-transfer-only
- `configs/label_transfer_dev.yaml` — dev tier, 10 trials, 50 synthetic, label-transfer-only
- `configs/combined_full.yaml` — full tier, 20 trials, 100 synthetic, alignment+label-transfer
- `tests/test_cli.py` — TestScenarioConfigs replaced: test_placeholder removed, _SCENARIO_TABLE + test_config_loads_without_error + test_config_has_d04_values added

## Decisions Made

- D-03 executed: all 5 configs share `data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets` and `data_format: tracklets`
- D-04 executed: exactly 7 YAML keys per file; omit output_dir, search_space, verbose, and 9 other fields (defaults apply); tier/n_trials/n_synthetic/run_alignment/run_label_transfer follow differentiation table verbatim
- `is True` / `is False` bool assertions in test to prevent silent int==bool pass-through (per plan spec)
- _SCENARIO_TABLE defined at module scope (above class) so it can be reused by both parametrize decorators

## Test Count Delta

| State | passed | skipped | total |
|---|---|---|---|
| Phase 22 baseline (post-22-02) | 723 | 17 | 740 |
| After Plan 23-01 | 731 | 18 | 749 |
| After Plan 23-02 (this plan) | 741 | 17 | 758 |

- +10 new passing tests (5 load × 2 assertion methods)
- -1 skip (test_placeholder removed)
- 18 tests now pass in test_cli.py (8 Plan 01 + 10 Plan 02)

## Milestone v1.2 Status

**All FRAME requirements closed:**

| Req | Phase | Status |
|---|---|---|
| FRAME-01 (EvalConfig) | 17 | Complete |
| FRAME-02 (DataFactory) | 17 | Complete |
| FRAME-03 (MetricsEngine) | 18 | Complete |
| FRAME-04 (Result types) | 18 | Complete |
| FRAME-05 (AlignmentStage) | 19 | Complete |
| FRAME-06 (LabelTransferStage) | 20 | Complete |
| FRAME-07 (EvaluationRunner) | 21 | Complete |
| FRAME-08 (viz.py) | 21 | Complete |
| FRAME-09 (HyperparamOptimizer) | 22 | Complete |
| FRAME-10 (SearchStrategies) | 22 | Complete |
| FRAME-11 (run_eval.py CLI) | 23-01 | Complete |
| FRAME-12 (5 scenario configs) | 23-02 | **Complete** |

Milestone v1.2 (Evaluation Framework & Debt Resolution): **COMPLETE**

## Manual Smoke Check (Deferred)

Per 23-VALIDATION.md: `python run_eval.py --config configs/alignment_sanity.yaml --mode eval` — deferred to post-merge (requires real tracklets data at the relative path from CWD; not part of automated CI).

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- FRAME-12 closed; Phase 23 (and milestone v1.2) complete
- Manual smoke check deferred: run `python run_eval.py --config configs/alignment_sanity.yaml --mode eval` from repo root against real tracklets data

## Self-Check: PASSED

Files exist:
- configs/alignment_sanity.yaml: FOUND
- configs/alignment_dev.yaml: FOUND
- configs/label_transfer_sanity.yaml: FOUND
- configs/label_transfer_dev.yaml: FOUND
- configs/combined_full.yaml: FOUND
- tests/test_cli.py: FOUND (modified)

Commits:
- 5f4e749: FOUND (feat(23-02): add 5 scenario YAML configs)
- 5fa2777: FOUND (feat(23-02): populate TestScenarioConfigs)

Test results: 741 passed, 17 skipped (pytest tests/ -x -q)

---
*Phase: 23-cli-entrypoint-scenario-configs*
*Completed: 2026-06-01*
