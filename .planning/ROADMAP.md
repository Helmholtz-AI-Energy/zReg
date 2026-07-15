# Roadmap: zReg

## Milestones

- 🚧 **v1.5 HoreKa Cluster Execution** — Phases 44–47 (in progress)
- ✅ **v1.4 Trajectory Alignment & Optimization Enhancements** — Phases 39–43 (shipped 2026-07-08) — [archive](.planning/milestones/v1.4-ROADMAP.md)
- ✅ **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–38 (shipped 2026-06-26) — [archive](.planning/milestones/v1.2-ROADMAP.md)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1–5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

**Phase Numbering:**
- Integer phases (44, 45, 46, 47): Planned milestone work
- Decimal phases (44.1, 44.2): Urgent insertions (marked with INSERTED)

### 🚧 v1.5 HoreKa Cluster Execution (In Progress)

**Milestone Goal:** Run the `baseline_experiments` evaluation suite on the HoreKa HPC cluster with GPU support, replacing the current laptop-constrained (CPU-only, subsampled) execution, within a 3-hour GPU time budget.

- [ ] **Phase 44: Environment & Access** - Operator can activate the HoreKa environment, transfer real datasets, and submit a working end-to-end job script *(1 plan — ready to execute)*
- [ ] **Phase 45: Multi-Rank Parallelism & Validation** - HPO trials run concurrently across MPI ranks via propulate, orchestration stays single-writer, validated on a short test job
- [ ] **Phase 46: GPU Acceleration** - Real per-operation GPU acceleration threaded through EvalConfig, DataFactory, and AlignmentStage
- [ ] **Phase 47: Budget Calibration & Full-Suite Gate** - Full 7-run suite is calibrated and verified to fit the 3-hour GPU cap before the full allocation is submitted

## Phase Details

### Phase 44: Environment & Access
**Goal**: Operator can stand up a working HoreKa environment — Python + MPI-built dependencies, real data transferred, and a SLURM job script that launches the suite end-to-end with outputs landing in the correct directory convention.
**Depends on**: Nothing (first phase of v1.5)
**Requirements**: ENV-01, ENV-02, ENV-03, OUT-01
**Success Criteria** (what must be TRUE):
  1. Operator can activate a Python environment on HoreKa with zReg plus the `propulate`/`mpi4py` extras installed, and `mpi4py` is built against HoreKa's system MPI (verified via a successful import on a compute node)
  2. Operator can transfer the real datasets (Kobitski tracklets, Shah CSV) to HoreKa, and every existing eval config's `data_path`/`target_data_path` resolves without a file-not-found error
  3. Operator can submit a SLURM job script that requests GPU nodes/ranks and launches the experiment suite end-to-end (job runs to completion or a defined stopping point, not a script/config error)
  4. Suite outputs land in the same `baseline_experiments/experiments/<phase>/<name>/` directory convention as local runs, hosted on HoreKa's workspace filesystem
**Plans**: 1 plan
Plans:
- [ ] 44-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

### Phase 45: Multi-Rank Parallelism & Validation
**Goal**: HPO trials execute concurrently across MPI ranks via the existing propulate backend, run_all.py orchestration is single-writer regardless of world size, cluster configs exist independently of laptop configs, and correct multi-rank behavior is proven on a short test job before committing to a full allocation.
**Depends on**: Phase 44
**Requirements**: PARA-01, PARA-02, PARA-03, BUDG-01, BUDG-04
**Success Criteria** (what must be TRUE):
  1. HPO trials for an optimize run execute concurrently across MPI ranks (not sequentially) when a config specifies `search_strategy: propulate`
  2. `run_all.py`'s orchestration (file writes, `EvaluationRunner` calls, per-run bookkeeping) executes exactly once per run regardless of MPI world size, while `HyperparamOptimizer.run()` remains collective across ranks
  3. Cluster-targeted configs (`search_strategy: propulate`/`auto`) exist as separate files from the local laptop configs (`search_strategy: sobol`), and both remain independently runnable without interfering
  4. Cluster configs default to the same `max_points_per_frame`/`step` subsampling already calibrated on the laptop, not full point density
  5. A short test job on HoreKa validates correct multi-rank behavior (no duplicated or racing output writes) and produces real per-trial timing data, completed before any full 3-hour allocation is submitted
**Plans**: 1 plan
Plans:
- [ ] 44-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

### Phase 46: GPU Acceleration
**Goal**: Real per-operation GPU acceleration is threaded through the pipeline — configurable device, data loaded onto that device, and registration provably executing on GPU tensors with no silent CPU fallback.
**Depends on**: Phase 45
**Requirements**: GPU-01, GPU-02, GPU-03
**Success Criteria** (what must be TRUE):
  1. `EvalConfig` exposes a `device` field that controls where tensors are loaded and computed
  2. `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU
  3. Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback anywhere in the path
**Plans**: 1 plan
Plans:
- [ ] 44-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

### Phase 47: Budget Calibration & Full-Suite Gate
**Goal**: Calibration constants are environment-aware, and the full 7-run suite's projected runtime is verified to fit the 3-hour GPU cap before the full allocation is submitted.
**Depends on**: Phase 46
**Requirements**: BUDG-02, BUDG-03
**Success Criteria** (what must be TRUE):
  1. `aggregate_cost.py`'s calibration constants are parameterized per environment (laptop vs. HoreKa), so cluster timing data doesn't silently mix with or overwrite laptop calibration
  2. Operator can run an `aggregate_cost.py`-style estimation using HoreKa GPU timing data (from Phase 45's short test job plus Phase 46's GPU path) to project the full 7-run suite's total runtime
  3. The projected total runtime for the full suite is confirmed to fit within the 3-hour GPU time cap before the full allocation is submitted (or the plan is revised if it doesn't fit)
**Plans**: 1 plan
Plans:
- [ ] 44-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

<details>
<summary>✅ v1.4 Trajectory Alignment & Optimization Enhancements (Phases 39–43) — SHIPPED 2026-07-08</summary>

- [x] Phase 39: ICP Registration as CPD Alternative (2/2 plans) — completed 2026-06-29
- [x] Phase 40: Sliced Wasserstein Variants as Alignment Method (4/4 plans) — completed 2026-06-29
- [x] Phase 41: Alignment Preprocessing — Principal Axes + Velocity Landmarks (2/2 plans) — completed 2026-06-29
- [x] Phase 42: Sobol Quasi-Random Search as Default (2/2 plans) — completed 2026-06-30
- [x] Phase 43: Per-Trajectory Data Standardization (1/1 plans) — completed 2026-06-30

Full details: [.planning/milestones/v1.4-ROADMAP.md](.planning/milestones/v1.4-ROADMAP.md)

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

**Execution Order:**
Phases execute in numeric order: 44 → 45 → 46 → 47

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 44. Environment & Access | v1.5 | 0/1 | Planned | - |
| 45. Multi-Rank Parallelism & Validation | v1.5 | 0/TBD | Not started | - |
| 46. GPU Acceleration | v1.5 | 0/TBD | Not started | - |
| 47. Budget Calibration & Full-Suite Gate | v1.5 | 0/TBD | Not started | - |

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 Consolidation | 1–5 (5) | 13 | ✅ Complete | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 (7) | 14 | ✅ Complete | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 (27) | 55 | ✅ Complete | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 (5) | 11 | ✅ Complete | 2026-07-08 |
| v1.5 HoreKa Cluster Execution | 44–47 (4) | TBD | 🚧 In progress | - |

_Next: `/gsd:plan-phase 44` to plan Phase 44 (Environment & Access)._
</content>
