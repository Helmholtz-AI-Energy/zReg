---
phase: 32-heterogeneous-paired-evaluation-target-data-format
plan: "02"
subsystem: eval
tags: [config, scenario-configs, heterogeneous-data, smoke-tests, format-dispatch]

# Dependency graph
requires:
  - phase: 32-01
    provides: EvalConfig.target_data_format field, DataFactory.load_target() format dispatch

provides:
  - "configs/kobitski_vs_shah.yaml — Kobitski ew06 tracklets source + Shah sample-1 CSV target"
  - "configs/shah_vs_kobitski.yaml — Shah sample-1 CSV source + Kobitski ew06 tracklets target"
  - "configs/kobitski_vs_kobitski_cross.yaml — Kobitski ew06 vs ew08 cross-embryo, same format"
  - "TestScenarioConfigs: 3 new smoke tests for heterogeneous paired configs"

affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pytest.skip() guard pattern: check config file exists then check data file exists before asserting"
    - "no-target_data_format fallback: omitting key causes EvalConfig.target_data_format to be None"

key-files:
  created:
    - configs/kobitski_vs_shah.yaml
    - configs/shah_vs_kobitski.yaml
    - configs/kobitski_vs_kobitski_cross.yaml
  modified:
    - tests/test_cli.py

key-decisions:
  - "pytest.skip() called inline (not decorator) to allow multiple path guards per test — consistent with plan spec"
  - "kobitski_vs_kobitski_cross.yaml omits target_data_format key to exercise None fallback path in load_target()"

# Metrics
duration: 2min
completed: 2026-06-14
---

# Phase 32 Plan 02: Heterogeneous Scenario Configs and Smoke Tests Summary

**Three heterogeneous paired scenario configs created (kobitski_vs_shah.yaml, shah_vs_kobitski.yaml, kobitski_vs_kobitski_cross.yaml) and three new TestScenarioConfigs smoke tests added to tests/test_cli.py, exercising the target_data_format field added in plan 32-01**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-06-14T09:50:04Z
- **Completed:** 2026-06-14T09:52:47Z
- **Tasks:** 6 (Tasks 1-3 config files; Task 4 smoke tests; Task 5 backward compat verify; Task 6 full test suite)
- **Files modified/created:** 4

## Accomplishments

- Created `configs/kobitski_vs_shah.yaml` with `data_format: tracklets` / `target_data_format: csv` (Kobitski ew06 source, Shah sample-1 target)
- Created `configs/shah_vs_kobitski.yaml` with `data_format: csv` / `target_data_format: tracklets` (reverse direction)
- Created `configs/kobitski_vs_kobitski_cross.yaml` with no `target_data_format` key (same-format cross-embryo sanity check; fallback to `data_format`)
- Added three smoke tests to `TestScenarioConfigs` in `tests/test_cli.py` — one per config; each uses `pytest.skip()` guards for missing config and data files
- Verified `configs/paired_alignment.yaml` backward compat: `target_data_format is None` — OK
- All 879 tests pass (up from 876 after plan 32-01 — exactly 3 new tests), 18 skipped

## Task Commits

1. **Task 1: Create configs/kobitski_vs_shah.yaml** - `29007d1` (feat)
2. **Task 2: Create configs/shah_vs_kobitski.yaml** - `a8031df` (feat)
3. **Task 3: Create configs/kobitski_vs_kobitski_cross.yaml** - `f598168` (feat)
4. **Task 4: Add TestScenarioConfigs smoke tests** - `73bb7cf` (test)
5. **Task 5: Backward compat verify** — verified inline (no separate commit needed)
6. **Task 6: Full test suite** — verified inline (no separate commit needed)

## Files Created/Modified

- `configs/kobitski_vs_shah.yaml` — `data_format: tracklets`, `target_data_format: csv`, paired mode, tier: sanity
- `configs/shah_vs_kobitski.yaml` — `data_format: csv`, `target_data_format: tracklets`, paired mode, tier: sanity
- `configs/kobitski_vs_kobitski_cross.yaml` — `data_format: tracklets`, no `target_data_format` key, cross-embryo ew06 vs ew08
- `tests/test_cli.py` — Added 3 methods to `TestScenarioConfigs`: `test_kobitski_vs_shah_yaml_loads_and_declares_heterogeneous_mode`, `test_shah_vs_kobitski_yaml_loads_and_declares_heterogeneous_mode`, `test_kobitski_vs_kobitski_cross_yaml_loads_and_declares_cross_embryo`

## Decisions Made

- `pytest.skip()` called inline (not as decorator) to allow multiple path checks per test method — consistent with plan spec and allows guarding both the config file and data file existence independently
- `kobitski_vs_kobitski_cross.yaml` deliberately omits `target_data_format` key to exercise the `None` fallback path in `DataFactory.load_target()`, confirming that same-format paired evaluation still works after the 32-01 change

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

The worktree branch was created from an older base commit (b76a576 — before Phase 31 and 32-01 work). A `git merge feature/evaluation_framework` was performed at the start to bring the worktree up-to-date before beginning implementation. This is normal worktree initialization behavior, not a plan deviation.

## Known Stubs

None — all three configs are fully wired with real data paths. Smoke tests guard data availability via `pytest.skip()`.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. Changes are confined to YAML configs and test additions.

## Self-Check

- [x] `configs/kobitski_vs_shah.yaml` exists and `EvalConfig.from_yaml()` parses it: target_data_format == 'csv'
- [x] `configs/shah_vs_kobitski.yaml` exists and `EvalConfig.from_yaml()` parses it: target_data_format == 'tracklets'
- [x] `configs/kobitski_vs_kobitski_cross.yaml` exists and `EvalConfig.from_yaml()` parses it: target_data_format is None
- [x] `tests/test_cli.py` contains `test_kobitski_vs_shah_yaml_loads_and_declares_heterogeneous_mode`
- [x] `paired_alignment.yaml` backward compat: target_data_format is None — OK
- [x] 879 tests pass (876 + 3 new), 18 skipped
- [x] Commits 29007d1, a8031df, f598168, 73bb7cf exist in git log

## Self-Check: PASSED

---
*Phase: 32-heterogeneous-paired-evaluation-target-data-format*
*Completed: 2026-06-14*
