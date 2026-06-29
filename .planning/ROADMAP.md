# Roadmap: zReg

## Milestones

- ✅ **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–38 (shipped 2026-06-26) — [archive](.planning/milestones/v1.2-ROADMAP.md)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1–5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

<details open>
<summary>🚀 v1.4 Trajectory Alignment & Optimization Enhancements (Phases 39–43) — IN PROGRESS</summary>

- [x] Phase 39: ICP Registration as CPD Alternative (2/2 plans) — completed 2026-06-29
- [x] Phase 40: Sliced Wasserstein Variants as Alignment Method (4/4 plans) — completed 2026-06-29
- [x] Phase 41: Alignment Preprocessing — Principal Axes + Velocity Landmarks (2/2 plans) — planned (completed 2026-06-29)
  - [x] 41-01-PLAN.md — preprocessing.py library + config/types contracts (Wave 1)
  - [x] 41-02-PLAN.md — alignment.py stage wiring + full test suite (Wave 2)
- [ ] Phase 42: Sobol Quasi-Random Search as Default (1/1 plans) — queued
- [ ] Phase 43: Per-Trajectory Data Standardization (1/1 plans) — queued

**Status**: Phase 39–40 complete (6/10 plans). Phase 41 ready to execute.

Full details: [.planning/REQUIREMENTS-v1.4.md](.planning/REQUIREMENTS-v1.4.md)

</details>

<details>
<summary>✅ v1.2 Evaluation Framework & Debt Resolution (Phases 12–38) — SHIPPED 2026-06-26</summary>

- [x] Phase 12: Carry-Forward Debt Closure (3/3 plans) — completed 2026-05-14
- [x] Phase 13: Core Metrics Library (3/3 plans) — completed 2026-05-15
- [x] Phase 14: Synthetic Data Generators (3/3 plans) — completed 2026-05-18
- [x] Phase 15: Experiment Tracking & Run Management (2/2 plans) — completed 2026-05-18
- [x] Phase 16: Runner Scripts (2/2 plans) — completed 2026-05-19
- [x] Phase 17: Framework Config & DataFactory (2/2 plans) — completed 2026-05-27
- [x] Phase 18: MetricsEngine & Result Types (2/2 plans) — completed 2026-05-28
- [x] Phase 19: AlignmentStage (2/2 plans) — completed 2026-05-28
- [x] Phase 20: LabelTransferStage (2/2 plans) — completed 2026-05-29
- [x] Phase 21: EvaluationRunner & Visualisation (2/2 plans) — completed 2026-05-29
- [x] Phase 22: HyperparamOptimizer & Search Strategies (2/2 plans) — completed 2026-05-29
- [x] Phase 23: CLI Entrypoint & Scenario Configs (2/2 plans) — completed 2026-06-02
- [x] Phase 24: Trajectory Export (2/2 plans) — completed 2026-06-04
- [x] Phase 25: Visualisation Refactor (2/2 plans) — completed 2026-06-04
- [x] Phase 26: Propulate Optimizer (2/2 plans) — completed 2026-06-06
- [x] Phase 27: DataFactory Geometric Augmentation Methods (2/2 plans) — completed 2026-06-11
- [x] Phase 28: Script Integration — generate_datasets uses DataFactory (1/1 plans) — completed 2026-06-11
- [x] Phase 29: Viz Unification (2/2 plans) — completed 2026-06-12
- [x] Phase 30: Two-Dataset Paired Alignment Architecture (3/3 plans) — completed 2026-06-12
- [x] Phase 31: Synthetic Pipeline Mode — Transform-Spec Target & GT-Aware HPO (3/3 plans) — completed 2026-06-14
- [x] Phase 32: Heterogeneous Paired Evaluation — target_data_format (2/2 plans) — completed 2026-06-14
- [x] Phase 33: CPD-Aligned Trajectory Output from AlignmentStage (2/2 plans) — completed 2026-06-16
- [x] Phase 34: Alignment Quality Guard in LabelTransferStage (1/1 plans) — completed 2026-06-19
- [x] Phase 35: Reuse Step-1 CPD Transforms in Aligned-Cloud Construction (2/2 plans) — completed 2026-06-22
- [x] Phase 36: plot_trajectory Alignment Figure Refactor (1/1 plans) — completed 2026-06-22
- [x] Phase 37: plot_trajectory Label Figure Refactor (1/1 plans) — completed 2026-06-23
- [x] Phase 38: zRegPointCloud color→label Field Rename (2/2 plans) — completed 2026-06-24

Full details: [.planning/milestones/v1.2-ROADMAP.md](.planning/milestones/v1.2-ROADMAP.md)

</details>

<details>
<summary>✅ v1.1 Code Quality & Refactoring (Phases 6–11.1) — SHIPPED 2026-05-13</summary>

- [x] Phase 6: Python 3.12 Migration (2/2 plans) — completed 2026-04-13
- [x] Phase 7: CPD Deep Restructure (3/3 plans) — completed 2026-04-20
- [x] Phase 8: DTW Deep Restructure (2/2 plans) — completed 2026-04-23
- [x] Phase 9: Distance & Transform Restructure (2/2 plans) — completed 2026-04-27
- [x] Phase 10: Code Quality & Verification (3/3 plans) — completed 2026-04-29
- [x] Phase 11: Validate Refactoring and Fix Coverage (1/1 plan) — completed 2026-05-12
- [x] Phase 11.1: Close DTW-02 — Consistent Metric Variant Interface (1/1 plan, INSERTED) — completed 2026-05-13

Full details: [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1.0 Consolidation (Phases 1–5) — SHIPPED 2026-04-09</summary>

- [x] Phase 1: Validation Foundation & Quick Wins (2/2 plans) — completed 2026-04-09
- [x] Phase 2: Distance Metric & CPD Bug Fixes (3/3 plans) — completed 2026-04-09
- [x] Phase 3: DTW, Transform & CPD Enhancements (3/3 plans) — completed 2026-04-09
- [x] Phase 4: Infrastructure & Color Transfer Quality (2/2 plans) — completed 2026-04-09
- [x] Phase 5: Test Coverage (3/3 plans) — completed 2026-04-09

Full details: [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

</details>

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 Consolidation | 1–5 (5) | 13 | ✅ Complete | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 (7) | 14 | ✅ Complete | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 (27) | 55 | ✅ Complete | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 (5) | 10 | 🚀 IN PROGRESS (6/10 complete) | (target 2026-08-09) |

_After phases 39-43: config-driven ICP + SWD alignment, preprocessing, Sobol search, data standardization. All parallel execution. Ready for `/gsd:plan-phase 39` to start._
