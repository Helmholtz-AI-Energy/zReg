---
phase: 17-framework-config-datafactory
plan: "01"
subsystem: eval-framework
tags: [config, pydantic, yaml, evalconfig, frame-01]
dependency_graph:
  requires: []
  provides: [EvalConfig, EvalConfigError, eval/config.py, tests/test_data_factory.py]
  affects: [17-02, 18, 19, 20, 21, 22, 23]
tech_stack:
  added: [pydantic>=2.10, pyyaml>=6.0]
  patterns: [pydantic-v2-BaseModel, ConfigDict-extra-forbid, ValidationError-wrapping, yaml-safe-load]
key_files:
  created:
    - eval/config.py
    - tests/test_data_factory.py
  modified:
    - setup.cfg
decisions:
  - D-01: EvalConfig is pydantic BaseModel (not plain dataclass) for type coercion/validation
  - D-02: Only data_path required; 15 other fields have sensible defaults
  - D-03: EvalConfigError(ValueError) wraps all pydantic.ValidationError; no raw errors leak to CLI
  - D-04: from_yaml classmethod uses yaml.safe_load + ordered exception handlers (FileNotFoundError, YAMLError, ValidationError)
  - D-06: val_split field (float=0.2) added beyond FRAME-01 for FRAME-02 prepare_split
metrics:
  duration: ~15 minutes
  completed: 2026-05-27
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 1
---

# Phase 17 Plan 01: Framework Config (EvalConfig) Summary

## One-Liner

EvalConfig pydantic v2 BaseModel with 16 typed fields, YAML loading via yaml.safe_load, and EvalConfigError wrapping all pydantic.ValidationError for clean Phase 23 CLI display.

## What Was Built

**Task 1 — setup.cfg:** Appended `pydantic>=2.10,<3` and `pyyaml>=6.0,<7` to `install_requires`. Both packages confirmed importable after `pip install -e .` (pydantic 2.12.2, pyyaml 6.0.3).

**Task 2 — eval/config.py:** New standalone module at the `eval/` namespace root (no `__init__.py` created). Contains:
- `EvalConfigError(ValueError)`: one-line subclass, wraps all pydantic + file errors
- `EvalConfig(BaseModel)`: 16 typed fields (`data_path` required, 15 defaulted), `model_config = ConfigDict(extra="forbid")` catches typos
- `EvalConfig.from_yaml(path)`: classmethod using `yaml.safe_load` with empty-file guard (`or {}`), exception handlers in correct MRO order (FileNotFoundError → YAMLError → ValidationError)

**Task 3 — tests/test_data_factory.py:** New test file with:
- `TestEvalConfigFromYAML` (6 tests): happy path, missing data_path, unknown field, bad type, empty YAML, missing file — all green
- Stub classes for Plan 17-02: `TestDataFactoryConstruction`, `TestLoadReal`, `TestGenerateSynthetic`, `TestAugment`, `TestPrepareSplit`, `TestGetGroundTruth` (empty `pass` bodies)

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 | a298fb1 | chore(17-01): declare pydantic>=2.10,<3 and pyyaml>=6.0,<7 in install_requires |
| 2 | a317c01 | feat(17-01): create eval/config.py with EvalConfig pydantic BaseModel and EvalConfigError |
| 3 | 17d8029 | test(17-01): create tests/test_data_factory.py with TestEvalConfigFromYAML (6 tests) and stub classes |

## Verification

All plan verification checks passed:
1. `python -c "import yaml; import pydantic"` — ok (2.12.2, 6.0.3)
2. `EvalConfig(data_path='x').val_split` — 0.2
3. `pytest tests/test_data_factory.py::TestEvalConfigFromYAML` — 6 passed
4. `pytest tests/ -x` — 564 passed, 17 skipped (no regressions; 6 new tests added)
5. `eval/__init__.py` does NOT exist
6. `yaml.load(` not present in eval/config.py (only `yaml.safe_load`)
7. setup.cfg declares both pydantic and pyyaml

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced beyond what the plan's threat model covers. T-17-01 (yaml.safe_load only) and T-17-02 (ValidationError wrapped) mitigations are implemented and verified by acceptance criteria.

## Known Stubs

- Stub test classes (`TestDataFactoryConstruction`, `TestLoadReal`, `TestGenerateSynthetic`, `TestAugment`, `TestPrepareSplit`, `TestGetGroundTruth`) in `tests/test_data_factory.py` have empty `pass` bodies. These are intentional placeholders — Plan 17-02 (DataFactory) will populate them.

## Self-Check: PASSED

- eval/config.py: FOUND
- tests/test_data_factory.py: FOUND
- setup.cfg (modified): FOUND
- a298fb1: FOUND
- a317c01: FOUND
- 17d8029: FOUND
