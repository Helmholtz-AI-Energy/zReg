---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Minor Adjustments
status: executing
stopped_at: Phase 58 complete (VERIFICATION.md passed, 6/6 must-haves) — v1.7 milestone complete; merged into feature/evaluation_framework
last_updated: "2026-08-05T00:00:00.000Z"
last_activity: 2026-08-05 -- v1.7 (Phases 57-58) merged into feature/evaluation_framework, renumbered from 56-57 to resolve collision with that branch's own Phase 56
progress:
  total_phases: 50
  completed_phases: 25
  total_plans: 69
  completed_plans: 97
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-15 after v1.5/v1.6 roadmap creation)

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** v1.7, v1.6, and v1.5 are all complete. No active phase — next step is `/gsd:complete-milestone` (v1.5/v1.6/v1.7 are shippable) or starting a new milestone.

## Current Position

Phase: 58 of 58 — v1.7 milestone complete (6/6 requirements satisfied), now merged into feature/evaluation_framework
Plan: 58-01, 58-02, 58-03, 58-04 (all 4 executed, VERIFICATION.md passed)
Status: v1.7 Minor Adjustments complete and merged. v1.6 (Phases 51-54) and v1.5 (Phases 44-50) also confirmed complete on reconciliation (2026-08-05) - their completion had drifted out of tracking. Next: /gsd:complete-milestone for any/all of v1.5/v1.6/v1.7, or start a new milestone.
Tests (this branch, pre-merge): 1410 passed, 22 skipped, 1 xpassed, 1 failed (tests/test_icp_registration.py::test_icp_translation_recovery — pre-existing full-suite-order flake, confirmed unrelated and passes in isolation)
Last activity: 2026-08-05 -- merged into feature/evaluation_framework; renumbered 56-57 -> 57-58 to resolve a Phase 56 collision with that branch's own concurrent work

## Shipped Milestones

| Milestone | Phases | Shipped |
|-----------|--------|---------|
| v1.0 Consolidation | 1–5 | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 | 2026-07-08 |

Full history: .planning/MILESTONES.md · Retrospective: .planning/RETROSPECTIVE.md

## v1.7 Phases — Minor Adjustments (Phases 57–58, current milestone)

| Phase | Name | Requirements | Status |
|-------|------|---------------|--------|
| 57 | Ground-Truth Field Consistency | GT-01, GT-02, GT-03 | ✅ Complete (2026-08-04) |
| 58 | Synthetic Labeled Subsample-Pair Generation | GT-04, GT-05, GT-06 | ✅ Complete (2026-08-04) |

Note: v1.7 was originally planned as Phases 56–57. Phase 55 was already consumed by an ad-hoc out-of-band phase (spherical-cap/Gaussian label generators), completed 2026-07-30 — shared history on both branches. Phase 56 was independently consumed by a *different* ad-hoc phase developed concurrently on `feature/evaluation_framework` (Configurable multi-label region-based labeling, completed 2026-07-31, see `.planning/phases/56-configurable-multi-label-region-based-labeling-rework-genera/`) — discovered only when this branch merged back in on 2026-08-05. v1.7 was renumbered to Phases 57–58 to resolve the collision. v1.6 (Phases 51-54) is fully complete (UAT 9/9 passed 2026-07-23) - its completion had drifted out of STATE.md/ROADMAP.md tracking after the completing branch diverged from feature/evaluation_framework before merging back; reconciled 2026-08-05.

## v1.5 Phases — Learned Label Transfer Methods (Complete)

| Phase | Name | Plans | Status |
|-------|------|-------|--------|
| 44 | CPD-Weighted Label Transfer Method | 4 | ✅ Completed |
| 45 | eGNN & PointNet++ Framework Selection & Evaluation Strategy | 1 | ✅ Completed |
| 46 | Training Data Pipeline for Learned Label Transfer | 3 | ✅ Completed |
| 47 | eGNN and PointNet++ Model Implementation & Training Infrastructure | 5 | ✅ Completed |
| 48 | LabelTransferStage Integration for Learned Methods | 2 | ✅ Completed |
| 49 | Evaluation & Benchmarking of Learned Label-Transfer Methods | 3 | ✅ Completed |
| 50 | GPU-Native Geometry Ops for Cluster Deployment | 2/2 | ✅ Completed (2026-07-31) |

## v1.6 Phases — HoreKa Cluster Execution (Complete)

| Phase | Name | Requirements | Status |
|-------|------|---------------|--------|
| 51 | Environment & Access | ENV-01, ENV-02, ENV-03, OUT-01 | ✅ Completed |
| 52 | Multi-Rank Parallelism & Validation | PARA-01, PARA-02, PARA-03, BUDG-01, BUDG-04 | ✅ Completed (HoreKa smoke + multirank tests passed) |
| 53 | GPU Acceleration | GPU-01, GPU-02, GPU-03 | ✅ Completed (EvalConfig.device + cluster configs device: cuda) |
| 54 | Budget Calibration & Full-Suite Gate | BUDG-02, BUDG-03 | ✅ Completed (UAT 9/9 passed 2026-07-23) |

## Performance Metrics (v1.5 ML track)

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 44 | 01 | 33min | 2 | 3 |
| 44 | 02 | 25min | 2 | 2 |
| 44 | 03 | 24min | 2 | 2 |
| 44 | 04 | 12min | 1 | 2 |
| 45 | 01 | 12min | 2 | 1 |
| 46 | 01 | 10min | 2 | 3 |
| 46 | 02 | 10min | 1 | 2 |
| 46 | 03 | 35min | 3 | 2 |
| 47 | 01 | 35min | 3 | 4 |
| 47 | 02 | 20min | 2 | 2 |
| 47 | 03 | 8min | 2 | 2 |
| 47 | 04 | 25min | 3 | 4 |
| 47 | 05 | 12min | 1 | 1 |
| 48 | 01 | ~15min | 2 | 2 |
| 48 | 02 | ~25min | 2 | 2 |
| 49 | 01 | ~20min | 2 | 2 |
| 49 | 02 | ~15min | 2 | 3 |
| 49 | 03 | ~20min | 2 | 2 |
| 56 | 01 | ~15min | 2 | 1 |
| 56 | 02 | ~17min | 2 | 11 |
| 56 | 03 | ~20min | 3 | 5 |
| 56 | 04 | ~25min | 2 | 2 |
| 56 | 05 | ~15min | 2 | 4 |

## Accumulated Context

### Roadmap Evolution

- Phase 56 added: Configurable multi-label region-based labeling — rework generate_labels() to support arbitrary n_labels, voronoi/gaussian-blob/gaussian-cone region shapes, deterministic and probabilistic assignment modes, and config-driven specification via EvalConfig
- Phase 56 completed (5/5 plans, verified) and merged into feature/evaluation_framework on 2026-07-31, alongside this branch's own concurrent Phase 50/51 progress
- v1.7 Minor Adjustments (Phases 57-58, originally planned as 56-57 on a separate branch before this merge — renumbered to resolve the Phase 56 collision with the entry above) added and completed: Phase 57 Ground-Truth Field Consistency (GT-01/02/03) and Phase 58 Synthetic Labeled Subsample-Pair Generation (GT-04/05/06), merged into feature/evaluation_framework on 2026-08-05

### v1.6 Design Decisions

- **Phase sequencing**: ENV first (nothing runs on cluster without it) → PARA next, validated via a short test job (BUDG-04) before GPU work lands → GPU device threading → final budget gate combining environment-aware calibration with real GPU+multi-rank timing data
- **Budget requirements split across phases**: BUDG-01 and BUDG-04 land with PARA (Phase 52); BUDG-02 and BUDG-03 land last (Phase 54) since the full-suite estimate needs both real multi-rank timing (Phase 52) and real GPU timing (Phase 53)
- **OUT-01 grouped with ENV-03**: output directory convention is defined by the job script itself

### v1.5 Design Decisions

- Phase 44: CPD_WEIGHTED wired into LabelTransferStage following Phase 39's alignment_method optional-param precedent
- Phase 45: PointNet++ hand-rolled on Open3D; eGNN hand-rolled MessagePassing on torch_geometric 2.8.0; zreg must be imported before torch_geometric on macOS ARM (libomp SIGABRT)
- Phase 46: sample_ball()/sample_bowl() ported from scripts/ as numpy-native geometry (np.random.default_rng, intentional deviation from generate_trajectory's torch.manual_seed contract)
- Phase 47: joint-cloud logits[n_source:] slicing contract single-sourced in test_zreg_models_joint_cloud.py; D-03 (Open3D ops benchmark) resolved sub-50ms
- Phase 48: VALID_METHODS now ("knn_voting", "cpd_weighted", "pointnet2", "egnn"); checkpoint loaded once per run() call
- Phase 49: cpd_weighted proven to work on raw non-CPD-aligned input (Assumption A1); leakage guard requires explicit held_out_seeds arg (Assumption A2)
- Phase 56 Plan 01: `weight` resolved as a relative log-space multiplier (`+log(weight)` before `logsumexp`), not a normalized/softmax prior; voronoi deterministic score is unscaled `-dist_sq` (argmax-preserving, no division)
- Phase 56 Plan 02: `generate_labels()` rewritten as config-driven orchestrator (`n_labels`/`label_specs` paths, `mode` switch); D-07 fixed by resolving all region/component centers once before the per-frame loop; `assign_cap_labels`/`assign_gaussian_labels` deleted (D-06, dead-clean, no shims); the `n_classes`→`n_labels` rename broke 8 additional call sites beyond `56-CONTEXT.md`'s documented 2 (eval/data_factory.py, eval/runners/optimizer.py, and 6 test files) — fixed as a Rule-1 deviation, which makes Plan 56-05's Task 1 (5 of those same files) a no-op when it runs later
- Phase 56 Plan 03: `EvalConfig.label_generation` (`LabelGenerationConfig` model) added, defaulting to `None`; `DataFactory.generate_training_triple`/`generate_training_set` renamed `n_classes`→`n_labels` with a sentinel `int | None = None` three-way precedence (explicit caller arg > `self.config.label_generation` > hardcoded `n_labels=6` fallback); `optimizer.py`'s sanity tier consults `self.config.label_generation` with a hardcoded `n_labels=4` fallback; `benchmark_runner.py`/`train_label_transfer.py`'s own public `n_classes`-named surfaces left unchanged (out of D-04 scope), only their internal forwarding calls updated; `configs/label_generation_example.yaml` added. Full suite: 1335 passed/22 skipped/1 xpassed/17 failed — all 17 failures are pre-existing and out of this plan's scope (14 are `n_classes=` kwarg TypeErrors in test files deferred to Plan 56-05's Task 2; 2 are the already-logged `test_icp_registration.py` full-suite-order flake)
- Phase 56 Plan 04: added `TestLabelRegionShapes`/`TestLabelAssignmentModes`/`TestLabelGenerationD07Regression` (15 tests) to `tests/test_generators.py` and a new `tests/test_label_generation_config.py` (10 tests) mirroring `test_alignment_preprocessing_config.py`'s style; the D-07 regression tests prove `torch.equal` labels across frames with an identical point position, for both the `n_labels` and `label_specs` paths; the end-to-end `EvalConfig.label_generation`-vs-explicit-`n_labels` precedence test uses `seed=2` (not the plan's illustrative `seed=0`) because `seed=0` leaves a Voronoi label with zero points for this ball-shape/point-count combination, which would make the "exactly N unique values" assertion fail on a correct implementation. Full suite: 1359 passed/22 skipped/1 xpassed/17 failed — same 17 pre-existing failures as Plan 56-03's baseline, unchanged (24 new tests added, zero new failures)
- Phase 56 Plan 05 (final plan, Phase 56 complete): closed the `n_classes`->`n_labels` rename cascade's last real call sites — `tests/test_benchmark_runner.py`'s `generate_training_set`, plus `tests/test_data_factory_training_triples.py`/`tests/test_zreg_models_pointnet2.py`/`tests/test_train_label_transfer.py`'s `generate_training_triple`/`generate_training_set` calls (5 sites, 4 files); Task 1's 5 target files (`test_optimizer.py`, `test_viz.py`, `test_trajectory_export.py`, `test_label_transfer_stage.py`, `test_eval_runner.py`) needed no edits, confirmed already fixed by Plan 56-02's Rule-1 deviation. Full suite: 1374 passed/22 skipped/1 xpassed/2 failed — the 2 failures are the pre-existing `test_icp_registration.py` full-suite-order flake (unrelated, already logged); all 14 `n_classes=` `TypeError`s are resolved.

### Requirements

See: `.planning/REQUIREMENTS.md`

- GT-01/02/03: ground-truth field consistency — `pc["label"]` not `pc["id"]`, correspondence-preserving y_true/y_pred pairing, config audit (Phase 57)
- GT-04/05/06: synthetic labeled subsample-pair generation — YAML-configurable, DataFactory method, EvaluationRunner/HyperparamOptimizer support (Phase 58)
- ENV-01/02/03: environment setup, data transfer, SLURM job script (Phase 51)
- PARA-01/02/03: propulate multi-rank HPO, run_all.py rank-awareness, separate cluster configs (Phase 52)
- GPU-01/02/03: EvalConfig device field, DataFactory device loading, verified GPU execution (Phase 53)
- BUDG-01/02/03/04: subsampling default, 3-hour cap projection, per-environment calibration, short-job validation (split across Phases 52 & 54)
- OUT-01: output directory convention (Phase 51)

### Pending Todos

None open. v1.5, v1.6, and v1.7 are all complete; none are formally closed via /gsd:complete-milestone yet.

### Blockers/Concerns

None — test_search_strategies mock fixed (FakeIndividual k=int→str) to align with Phase 52 categorical string encoding.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 20260717 | Create evaluation framework tutorial | 2026-07-17 | 30c18e2 | [20260717-evaluation-framework-tutorial](./quick/20260717-evaluation-framework-tutorial/) |
| 20260717-02 | Revise tutorial: cross-platform (MPS/CPU/cluster) status | 2026-07-17 | 24fce77 | [20260717-02-revise-tutorial-cross-platform](./quick/20260717-02-revise-tutorial-cross-platform/) |
| 20260718 | Convert tutorial to interactive Jupyter notebook | 2026-07-18 | 30c89ab | [20260718-tutorial-to-notebook](./quick/20260718-tutorial-to-notebook/) |
| 20260718-02 | Create docs/tutorials/ with eval framework + revised notebook tutorials | 2026-07-18 | beec1d8 | [20260718-02-tutorials-directory-restructure](./quick/20260718-02-tutorials-directory-restructure/) |
| 20260718-03 | Remove notebooks/ directory (content in docs/tutorials/) | 2026-07-18 | f4e8f30 | [20260718-03-remove-notebooks-dir](./quick/20260718-03-remove-notebooks-dir/) |
| 20260722 | Add --clear-checkpoints to run_all.py; split window_size=20 into separate HorEKA job | 2026-07-22 | cff2a81 | [20260722-baseline-checkpoint-and-job-split](./quick/20260722-baseline-checkpoint-and-job-split/) |
| 260728-q01 | Add EGNN and POINTNET2 to LabelTransferMethod with model dispatch | 2026-07-28 | 0fea346 | [260728-q01-model-dispatch-label-transfer](./quick/260728-q01-model-dispatch-label-transfer/) |

## Deferred Items (from v1.4 close, still open)

| Category | Item | Status |
|----------|------|--------|
| verification | Phase 26 SC-4 — live `mpirun -n 2` Propulate integration | Resolved on HoreKa — smoke + multirank tests passed in Phase 52 |
| code-review | Open CR/WR items (matplotlib Agg backend leak; `subprocess.run` timeout in `trajectory.py`; `eval/` excluded from `--cov`) | Non-blocking; carried forward |
| test-flakiness | `tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery`/`test_icp_rotation_recovery` fail only in full-suite runs, pass in isolation (found during Phase 56 Plan 02) | Non-blocking; unrelated to Phase 56; logged to `.planning/phases/56-.../deferred-items.md` |

## Session Continuity

Last session: 2026-08-05T00:00:00.000Z
Stopped at: v1.7 Minor Adjustments (Phases 57-58) merged into feature/evaluation_framework, renumbered to resolve Phase 56 collision
Next action: `/gsd:plan-phase 54` — Budget Calibration & Full-Suite Gate (BUDG-02, BUDG-03); requires real HoreKa timing data from Phase 52/53 full run. Or `/gsd:complete-milestone` to formally close out v1.7 first.
