---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Minor Adjustments
status: planning
last_updated: "2026-08-01T12:36:21.477Z"
last_activity: 2026-08-01
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-15 after v1.5/v1.6 roadmap creation)

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** Phase 50 — gpu-native-geometry-ops

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-08-01 — Milestone v1.7 started

## Shipped Milestones

| Milestone | Phases | Shipped |
|-----------|--------|---------|
| v1.0 Consolidation | 1–5 | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 | 2026-07-08 |

Full history: .planning/MILESTONES.md · Retrospective: .planning/RETROSPECTIVE.md

## v1.5 Phases — Learned Label Transfer Methods (Phases 44–49 complete, 50 pending)

| Phase | Name | Plans | Status |
|-------|------|-------|--------|
| 44 | CPD-Weighted Label Transfer Method | 4 | ✅ Completed |
| 45 | eGNN & PointNet++ Framework Selection & Evaluation Strategy | 1 | ✅ Completed |
| 46 | Training Data Pipeline for Learned Label Transfer | 3 | ✅ Completed |
| 47 | eGNN and PointNet++ Model Implementation & Training Infrastructure | 5 | ✅ Completed |
| 48 | LabelTransferStage Integration for Learned Methods | 2 | ✅ Completed |
| 49 | Evaluation & Benchmarking of Learned Label-Transfer Methods | 3 | ✅ Completed |
| 50 | GPU-Native Geometry Ops for Cluster Deployment | 0/TBD | Pending |

## v1.6 Phases — HoreKa Cluster Execution

| Phase | Name | Requirements | Status |
|-------|------|---------------|--------|
| 51 | Environment & Access | ENV-01, ENV-02, ENV-03, OUT-01 | ✅ Completed |
| 52 | Multi-Rank Parallelism & Validation | PARA-01, PARA-02, PARA-03, BUDG-01, BUDG-04 | ✅ Completed (HoreKa smoke + multirank tests passed) |
| 53 | GPU Acceleration | GPU-01, GPU-02, GPU-03 | ✅ Completed (EvalConfig.device + cluster configs device: cuda) |
| 54 | Budget Calibration & Full-Suite Gate | BUDG-02, BUDG-03 | Not started |

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

## Accumulated Context

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

### Requirements

See: `.planning/REQUIREMENTS.md`

- ENV-01/02/03: environment setup, data transfer, SLURM job script (Phase 51)
- PARA-01/02/03: propulate multi-rank HPO, run_all.py rank-awareness, separate cluster configs (Phase 52)
- GPU-01/02/03: EvalConfig device field, DataFactory device loading, verified GPU execution (Phase 53)
- BUDG-01/02/03/04: subsampling default, 3-hour cap projection, per-environment calibration, short-job validation (split across Phases 52 & 54)
- OUT-01: output directory convention (Phase 51)

### Pending Todos

Phase 54 planning not yet started — requires real HoreKa timing data from Phase 52/53 runs. Phase 50 (GPU-native geometry ops) still pending.

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

## Session Continuity

Last session: 2026-07-27T07:40:45.602Z
Stopped at: context exhaustion at 76% (2026-07-27)
Next action: `/gsd:plan-phase 54` — Budget Calibration & Full-Suite Gate (BUDG-02, BUDG-03); requires real HoreKa timing data from Phase 52/53 full run
