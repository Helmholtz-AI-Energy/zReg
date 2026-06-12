---
phase: 30-two-dataset-paired-alignment-architecture
plan: "02"
subsystem: eval-framework
tags: [paired-alignment, eval-runner, hyperparamoptimizer, source-target-propagation, pitfall4, pitfall7]
dependency_graph:
  requires: [30-01]
  provides: [EvaluationRunner.run(source+target), EvaluationRunner._run_single(source,target,params), HyperparamOptimizer._objective(tier_target)]
  affects: [eval/runners/eval_runner.py, eval/runners/optimizer.py, tests/test_eval_runner.py, tests/test_optimizer.py]
tech_stack:
  added: []
  patterns: [two-input-stage-propagation, tier-aware-target-dispatch, pitfall4-variable-rename]
key_files:
  created: []
  modified:
    - eval/runners/eval_runner.py
    - eval/runners/optimizer.py
    - tests/test_eval_runner.py
    - tests/test_optimizer.py
decisions:
  - "Pitfall 4 resolved: source_pos/target_pos and source_sorted_keys/target_sorted_keys replace shadowed source/target locals in both eval_runner._run_single and optimizer._objective"
  - "Pitfall 7 implemented per recommendation (a)+(b): sanity tier reuses tier_dataset as both source and target; dev/full tiers call load_target()"
  - "Pitfall 6 atomic update: eval_runner and optimizer updated in the same plan — no intermediate state where one file is updated and the other is not"
  - "D-04 mock pattern: load_target.return_value = same fixture as load_real.return_value in all test blocks (Kobitski-vs-Kobitski smoke semantics)"
metrics:
  duration: "~4 minutes"
  completed: "2026-06-12"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 4
  tests_before: 851
  tests_after: 851
  tests_added: 0
  tests_deleted: 0
---

# Phase 30 Plan 02: EvaluationRunner + HyperparamOptimizer Source/Target Propagation Summary

Two-input stage signatures propagated through the orchestration layer: `EvaluationRunner.run()` loads both source (via `load_real`) and target (via `load_target`) and passes them through `_run_single`; `HyperparamOptimizer._objective` receives tier-aware `tier_target` (sanity reuses `tier_dataset`; dev/full call `load_target()`); Pitfall 4 variable shadowing resolved in both files via `source_pos`/`target_pos` and `source_sorted_keys`/`target_sorted_keys`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Update EvaluationRunner.run() and _run_single to load and propagate source/target; resolve variable shadowing | 8a252a0 | eval/runners/eval_runner.py, tests/test_eval_runner.py |
| 2 | Update HyperparamOptimizer._objective with tier-aware source/target dispatch; update test_optimizer.py mocks | 85b6829 | eval/runners/optimizer.py, tests/test_optimizer.py |

## Files Modified

| File | Lines | Key Changes |
|------|-------|-------------|
| eval/runners/eval_runner.py | 391 | `run()`: load_real()->source, load_target()->target; `_run_single(source, target, params)`; AlignmentStage.run(source,target,params); LabelTransferStage.run(stage_input,target,params); source_sorted_keys/target_sorted_keys; source_pos/target_pos; get_ground_truth(source); y_pred from target_sorted_keys[-1]; module docstring updated |
| eval/runners/optimizer.py | 514 | `_objective`: tier_target dispatch block (sanity reuses/dev+full call load_target); AlignmentStage.run(tier_dataset,tier_target,merged); LabelTransferStage.run(stage_input,tier_target,merged); source_sorted_keys/target_sorted_keys; source_pos/target_pos; y_true from source_sorted_keys[-1]; y_pred from target_sorted_keys[-1]; module docstring updated |
| tests/test_eval_runner.py | 569 | 18 @patch DataFactory blocks: all now set mock_factory.load_target.return_value = same fixture as load_real.return_value |
| tests/test_optimizer.py | 283 | 3 @patch DataFactory blocks: all now set mock_factory.load_target.return_value = synthetic_dataset |

## Mock-Block Update Counts

| File | @patch DataFactory blocks updated | load_real count | load_target count |
|------|-----------------------------------|-----------------|-------------------|
| tests/test_eval_runner.py | 18 | 18 | 18 |
| tests/test_optimizer.py | 3 | 3 | 3 |

All `load_real.return_value` occurrences now have a paired `load_target.return_value` set to the same fixture variable (D-04 Kobitski-vs-Kobitski smoke semantics).

## Pitfall 6 Atomic Update

eval_runner.py and optimizer.py were updated in the same plan (Tasks 1 and 2 in the same execution wave). At no point was one file updated without the other — the Pitfall 6 risk of ~30 simultaneous test failures due to signature mismatch was avoided by atomic treatment.

**Final test count:** 26 passed (21 test_eval_runner + 5 test_optimizer); 142 stage-suite tests (test_alignment_stage, test_label_transfer_stage, test_data_factory) unaffected.

## Pitfall 7 Dispatch — No Deviation

Plan recommendation (a)+(b) combined was implemented exactly as specified:
- `tier_name == "sanity"` → `tier_target = tier_dataset` (same dataset, D-03 smoke-test parity)
- `tier_name in {"dev", "full"}` → `tier_target = self._factory.load_target()` (target from config.target_data_path)

No alternative sanity/dev/full split was chosen.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all production code paths are functional. `target_data_path=None` raises `EvalConfigError` at `load_target()` call time (D-05 from Plan 30-01).

## Threat Flags

No new threat surface introduced beyond the plan's threat model:
- T-30-04 (Tampering/tier dispatch): mitigated — `if tier_name == "sanity": ... else:` has no input-validation surface; tier_name is already validated upstream
- T-30-06 (Information Disclosure/variable shadowing): mitigated — source_pos/target_pos rename makes Tensor-vs-dict type distinction syntactically unambiguous

## Self-Check: PASSED

- eval/runners/eval_runner.py: FOUND — `source = self.factory.load_real()` at line 187, `target = self.factory.load_target()` at line 188, `_run_single(source, target, self.params)` at line 190
- eval/runners/optimizer.py: FOUND — `tier_target = tier_dataset` at line 335, `tier_target = self._factory.load_target()` at line 337
- tests/test_eval_runner.py: FOUND — load_real count 18 == load_target count 18
- tests/test_optimizer.py: FOUND — load_real count 3 == load_target count 3
- Commit 8a252a0: FOUND
- Commit 85b6829: FOUND
- pytest tests/test_eval_runner.py tests/test_optimizer.py: 26 passed, 0 failed
- pytest tests/test_alignment_stage.py tests/test_label_transfer_stage.py tests/test_data_factory.py: 142 passed, 0 failed
- Smoke test (EvaluationRunner with mocked DataFactory): OK: 5
