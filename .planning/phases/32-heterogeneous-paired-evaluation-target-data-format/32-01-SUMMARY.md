---
phase: 32-heterogeneous-paired-evaluation-target-data-format
plan: "01"
subsystem: eval
tags: [pydantic, data-factory, config, heterogeneous-data, format-dispatch]

# Dependency graph
requires:
  - phase: 30-two-dataset-paired-alignment-architecture
    provides: DataFactory.load_target(), EvalConfig.target_data_path, MODE-01 paired mode
  - phase: 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
    provides: EvalConfig.transform_spec, generate_target, synthetic pipeline
provides:
  - "EvalConfig.target_data_format: str | None = None field (backward-compatible, after transform_spec)"
  - "DataFactory.load_target() format dispatch via fmt = target_data_format or data_format"
  - "TestEvalConfigTargetDataFormat (3 tests) — field acceptance and default"
  - "TestLoadTargetFormatDispatch (4 tests) — format override and fallback dispatch"
affects:
  - 32-02  # scenario configs for kobitski_vs_shah, shah_vs_kobitski, kobitski_vs_kobitski_cross

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "error-at-use-time: target_data_format validated at load_target() call, not at EvalConfig construction"
    - "fallback pattern: fmt = target_data_format or data_format"

key-files:
  created: []
  modified:
    - eval/config.py
    - eval/data_factory.py
    - tests/test_data_factory.py

key-decisions:
  - "target_data_format is str | None (not Literal) to stay consistent with data_format; validation at use-time"
  - "None default inherits data_format — all existing YAMLs without the key continue to work"
  - "fmt resolved as target_data_format or data_format — single lookup replaces direct config.data_format references"

patterns-established:
  - "Format-override pattern: fmt = self.config.target_data_format or self.config.data_format before dispatch if/elif block"

requirements-completed:
  - HETERO-01

# Metrics
duration: 4min
completed: 2026-06-14
---

# Phase 32 Plan 01: Heterogeneous Paired Evaluation — EvalConfig + DataFactory Summary

**`target_data_format: str | None = None` added to EvalConfig; DataFactory.load_target() now dispatches via `fmt = target_data_format or data_format`, enabling heterogeneous paired evaluation (e.g. Kobitski tracklets as source, Shah CSV as target)**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-14T09:41:53Z
- **Completed:** 2026-06-14T09:45:16Z
- **Tasks:** 5 (Tasks 1-4 with code/test changes; Task 5 test verification)
- **Files modified:** 3

## Accomplishments

- Added `target_data_format: str | None = None` field to `EvalConfig` after `transform_spec`; field docstring added to Attributes section
- Updated `DataFactory.load_target()` to resolve `fmt = self.config.target_data_format or self.config.data_format` before the tracklets/csv dispatch — no other logic changes
- Added `TestEvalConfigTargetDataFormat` (3 tests) and `TestLoadTargetFormatDispatch` (4 tests) to `tests/test_data_factory.py`
- All 876 tests pass (up from 869), 18 skipped — all existing tests pass without modification

## Task Commits

1. **Task 1: Add target_data_format field to EvalConfig** - `611e3be` (feat)
2. **Task 2: Update DataFactory.load_target() format dispatch** - `3b5edbd` (feat)
3. **Task 3: Add TestEvalConfigTargetDataFormat test class** - `8b7e414` (test)
4. **Task 4: Add TestLoadTargetFormatDispatch test class** - `b4119c6` (test)
5. **Task 5: Run test suite** - verified inline (no separate commit needed)

## Files Created/Modified

- `eval/config.py` — Added `target_data_format: str | None = None` field after `transform_spec`; updated Attributes docstring
- `eval/data_factory.py` — `load_target()` now uses `fmt = self.config.target_data_format or self.config.data_format`; docstring updated to reflect fallback behaviour
- `tests/test_data_factory.py` — Added `TestEvalConfigTargetDataFormat` (3 tests) and `TestLoadTargetFormatDispatch` (4 tests)

## Decisions Made

- `target_data_format` is `str | None` (not `Literal["tracklets", "csv"]`) to stay consistent with `data_format: str`; invalid values are caught at `load_target()` call time via the existing `unknown data_format` ValueError path
- Default `None` means "inherit from `data_format`" — backward compatible; all existing YAMLs without the key continue to work unchanged
- Position: after `transform_spec` (last field in the model) per plan spec

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None — `target_data_format` is fully wired from EvalConfig through DataFactory.load_target().

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The change is internal to the config model and data loader dispatch.

## Next Phase Readiness

- HETERO-01 core logic complete: `target_data_format` field declared and dispatch wired
- Plan 32-02 can now add the three heterogeneous scenario configs (`kobitski_vs_shah.yaml`, `shah_vs_kobitski.yaml`, `kobitski_vs_kobitski_cross.yaml`) and their smoke tests in `TestScenarioConfigs`

## Self-Check

- [x] `eval/config.py` contains `target_data_format: str | None`
- [x] `eval/data_factory.py` contains `target_data_format or self.config.data_format`
- [x] `tests/test_data_factory.py` contains `class TestEvalConfigTargetDataFormat`
- [x] Commits 611e3be, 3b5edbd, 8b7e414, b4119c6 exist in git log
- [x] 876 tests pass, 18 skipped

## Self-Check: PASSED

---
*Phase: 32-heterogeneous-paired-evaluation-target-data-format*
*Completed: 2026-06-14*
