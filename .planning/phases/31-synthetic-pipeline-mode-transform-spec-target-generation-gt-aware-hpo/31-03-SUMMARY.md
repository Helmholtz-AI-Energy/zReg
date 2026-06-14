---
phase: 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
plan: 03
subsystem: configs/, tests/test_cli.py
tags: [synthetic-mode, scenario-config, regression]
key-files:
  - configs/synthetic_mode.yaml
  - tests/test_cli.py
metrics:
  tests_before: 868
  tests_after: 869
  tests_added: 1
  tests_skipped: 18
---

# Plan 31-03 Summary: synthetic_mode.yaml + TestScenarioConfigs smoke test

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | d8e079f | feat(31-03): add configs/synthetic_mode.yaml scenario config for synthetic pipeline mode |
| Task 2 | e0f7ae2 | feat(31-03): add TestScenarioConfigs smoke test for synthetic_mode.yaml and run full regression sweep |

## Deviations

None.

## Self-Check: PASSED

All 2 tasks completed and committed on `feature/evaluation_framework`.

Verification:
- `configs/synthetic_mode.yaml` exists with `pipeline_mode: synthetic`, rigid `transform_spec`, no `target_data_path`
- `EvalConfig.from_yaml('configs/synthetic_mode.yaml')` loads cleanly — pydantic validation passes, `extra="forbid"` satisfied
- `tests/test_cli.py::TestScenarioConfigs::test_synthetic_mode_yaml_loads_and_declares_synthetic_mode` asserts all structural invariants
- Full suite: 869 passed, 18 skipped (baseline 868 + 1 new test; no regressions)
