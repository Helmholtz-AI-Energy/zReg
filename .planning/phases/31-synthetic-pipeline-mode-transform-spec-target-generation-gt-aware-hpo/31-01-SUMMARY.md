---
phase: 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
plan: 01
subsystem: eval/config.py, eval/data_factory.py
tags: [synthetic-mode, datafactory, evalconfig, transform-spec, ground-truth]
key-files:
  - eval/config.py
  - eval/data_factory.py
  - tests/test_data_factory.py
metrics:
  tests_before: 839
  tests_after: 860
  tests_added: 21
  tests_skipped: 18
---

# Plan 31-01 Summary: EvalConfig.transform_spec + DataFactory.generate_target() + get_synthetic_ground_truth()

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | 15b7329 | feat(31-01): declare EvalConfig.transform_spec field and TestEvalConfigTransformSpec |
| Task 2 | f2f30e9 | feat(31-01): add DataFactory.generate_target() with config save/restore and instance state |
| Task 3 | f66d106 | feat(31-01): add DataFactory.get_synthetic_ground_truth() with id-or-ordinal fallback |

## Deviations

None.

## Self-Check: PASSED

All 3 tasks completed and committed on `feature/evaluation_framework`.

Verification:
- `eval/config.py` contains `transform_spec: dict | None = None` (after `target_data_path`)
- `eval/data_factory.py` contains `def generate_target(` and `def get_synthetic_ground_truth(` (2 new methods)
- `eval/data_factory.py` `__init__` initialises `_synthetic_target`, `_source_dataset`, `_transform_spec` to `None`
- `generate_target()` wraps `self.augment(dataset)` in `try/finally` to restore `self.config.augmentation_params`
- `generate_target()` strips `"type"` key before dispatch; raises `ValueError` for empty/type-only specs
- `get_synthetic_ground_truth()` raises `RuntimeError` before `generate_target()` is called; returns `torch.long` tensors via `pc["id"].to(torch.long)` or `torch.arange(n_points, dtype=torch.long)` fallback
- 21 new tests across 3 classes (TestEvalConfigTransformSpec × 5, TestGenerateTarget × 10, TestGetSyntheticGroundTruth × 6)
- Full suite: 860 passed, 18 skipped (no regressions)
