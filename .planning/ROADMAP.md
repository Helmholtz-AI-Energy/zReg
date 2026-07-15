# Roadmap: zReg

## Milestones

- 🚧 **v1.6 HoreKa Cluster Execution** — Phases 51–54 (in progress)
- 🚧 **v1.5 Learned Label Transfer Methods** — Phases 44–50 (in progress — Phase 50 pending)
- ✅ **v1.4 Trajectory Alignment & Optimization Enhancements** — Phases 39–43 (shipped 2026-07-08) — [archive](.planning/milestones/v1.4-ROADMAP.md)
- ✅ **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–38 (shipped 2026-06-26) — [archive](.planning/milestones/v1.2-ROADMAP.md)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1–5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

**Phase Numbering:**

- Integer phases (44–54): Active or planned work
- Decimal phases (e.g. 44.1): Urgent insertions (marked with INSERTED)

### 🚧 v1.6 HoreKa Cluster Execution (In Progress)

**Milestone Goal:** Run the `baseline_experiments` evaluation suite on the HoreKa HPC cluster with GPU support, replacing the current laptop-constrained (CPU-only, subsampled) execution, within a 3-hour GPU time budget.

- [ ] **Phase 51: Environment & Access** - Operator can activate the HoreKa environment, transfer real datasets, and submit a working end-to-end job script *(1 plan — ready to execute)*
- [ ] **Phase 52: Multi-Rank Parallelism & Validation** - HPO trials run concurrently across MPI ranks via propulate, orchestration stays single-writer, validated on a short test job
- [ ] **Phase 53: GPU Acceleration** - Real per-operation GPU acceleration threaded through EvalConfig, DataFactory, and AlignmentStage
- [ ] **Phase 54: Budget Calibration & Full-Suite Gate** - Full 7-run suite is calibrated and verified to fit the 3-hour GPU cap before the full allocation is submitted

## Phase Details

### Phase 51: Environment & Access

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

- [x] 51-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

### Phase 52: Multi-Rank Parallelism & Validation

**Goal**: HPO trials execute concurrently across MPI ranks via the existing propulate backend, run_all.py orchestration is single-writer regardless of world size, cluster configs exist independently of laptop configs, and correct multi-rank behavior is proven on a short test job before committing to a full allocation.
**Depends on**: Phase 51
**Requirements**: PARA-01, PARA-02, PARA-03, BUDG-01, BUDG-04
**Success Criteria** (what must be TRUE):

  1. HPO trials for an optimize run execute concurrently across MPI ranks (not sequentially) when a config specifies `search_strategy: propulate`
  2. `run_all.py`'s orchestration (file writes, `EvaluationRunner` calls, per-run bookkeeping) executes exactly once per run regardless of MPI world size, while `HyperparamOptimizer.run()` remains collective across ranks
  3. Cluster-targeted configs (`search_strategy: propulate`/`auto`) exist as separate files from the local laptop configs (`search_strategy: sobol`), and both remain independently runnable without interfering
  4. Cluster configs default to the same `max_points_per_frame`/`step` subsampling already calibrated on the laptop, not full point density
  5. A short test job on HoreKa validates correct multi-rank behavior (no duplicated or racing output writes) and produces real per-trial timing data, completed before any full 3-hour allocation is submitted

**Plans**: TBD

### Phase 53: GPU Acceleration

**Goal**: Real per-operation GPU acceleration is threaded through the pipeline — configurable device, data loaded onto that device, and registration provably executing on GPU tensors with no silent CPU fallback.
**Depends on**: Phase 52
**Requirements**: GPU-01, GPU-02, GPU-03
**Success Criteria** (what must be TRUE):

  1. `EvalConfig` exposes a `device` field that controls where tensors are loaded and computed
  2. `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU
  3. Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback anywhere in the path

**Plans**: TBD

### Phase 54: Budget Calibration & Full-Suite Gate

**Goal**: Calibration constants are environment-aware, and the full 7-run suite's projected runtime is verified to fit the 3-hour GPU cap before the full allocation is submitted.
**Depends on**: Phase 53
**Requirements**: BUDG-02, BUDG-03
**Success Criteria** (what must be TRUE):

  1. `aggregate_cost.py`'s calibration constants are parameterized per environment (laptop vs. HoreKa), so cluster timing data doesn't silently mix with or overwrite laptop calibration
  2. Operator can run an `aggregate_cost.py`-style estimation using HoreKa GPU timing data (from Phase 52's short test job plus Phase 53's GPU path) to project the full 7-run suite's total runtime
  3. The projected total runtime for the full suite is confirmed to fit within the 3-hour GPU time cap before the full allocation is submitted (or the plan is revised if it doesn't fit)

**Plans**: TBD

### 🚧 v1.5 Learned Label Transfer Methods (Phases 44–49 complete, Phase 50 pending)

**Milestone Goal:** Extend LabelTransferStage with learned point-cloud methods (CPD-weighted, eGNN, PointNet++) — full stack from architecture selection through training infrastructure, inference integration, and evaluation/benchmarking.

- [x] **Phase 44: CPD-Weighted Label Transfer** — completed 2026-07-15
- [x] **Phase 45: eGNN/PointNet++ Framework Selection & Evaluation Strategy** — completed 2026-07-15
- [x] **Phase 46: Training Data Pipeline** — completed 2026-07-15
- [x] **Phase 47: eGNN/PointNet++ Model Implementation & Training Infrastructure** — completed 2026-07-15
- [x] **Phase 48: LabelTransferStage Integration for Learned Methods** — completed 2026-07-15
- [x] **Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods** — completed 2026-07-15
- [ ] **Phase 50: GPU-Native Geometry Ops** — pending (blocked on cluster verification)

---

### Phase 50: GPU-Native Geometry Ops for Cluster Deployment

**Goal:** Investigate replacing Phase 47's Open3D-CPU-based FPS/ball-query ops
(`src/zreg/models/_ops.py`) with `torch_cluster`'s GPU-native equivalents for PointNet++, if the
user's target cluster (Linux/CUDA) can actually install `torch_cluster` — a build failure that
was only ever reproduced on this dev machine (macOS ARM). `zRegPointCloud` is unaffected either
way (the ops already take/return plain `torch.Tensor`, not Open3D objects). eGNN's hand-rolled
equivariant conv layer is out of scope — that choice was architecture-fit-driven (no suitable
packaged implementation exists at any point-cloud scale), not platform-driven, and stays
unchanged regardless of this phase's outcome.
**Requirements**: TBD (see 50-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 49
**Plans:** 2 plans

Plans:

- [ ] 50-01-PLAN.md — HoreKa torch_cluster install verification gate (autonomous: false, human-executed) (D-01, D-02)
- [ ] 50-02-PLAN.md — torch_cluster dual-path in _ops.py for all three ops + setup.cfg cluster extra + smoke tests (D-03–D-08)

### Phase 44: CPD-Weighted Label Transfer Method

**Goal:** Wire zreg.color_transfer's existing CPD_WEIGHTED method into LabelTransferStage as a
selectable alternative to the hardcoded KNN_VOTING, mirroring the Phase 39 alignment_method
optional-param precedent — fixing the pmat-orientation and categorical-averaging bugs found during
investigation, without touching zreg.color_transfer itself.
**Requirements**: D-01 through D-09 (see 44-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 43
**Plans:** 4/4 plans complete

Plans:

- [x] 44-01-PLAN.md — AlignResult.estep_results field + EvalConfig.label_transfer_method field/validator (D-06, D-09)
- [x] 44-02-PLAN.md — AlignmentStage CPD posterior capture (_build_aligned_cloud + run()) (D-01, D-02, D-03)
- [x] 44-03-PLAN.md — LabelTransferStage cpd_weighted wiring (transpose + one-hot/argmax + align_result kwarg) (D-04, D-05, D-06, D-07, D-08)
- [x] 44-04-PLAN.md — EvaluationRunner align_result threading (D-08)

**Status:** ✅ Complete — all 4 plans executed, Phase 44 fully wired end-to-end.

### Phase 45: eGNN & PointNet++ Label Transfer — Framework Selection & Evaluation Strategy

**Goal:** Decide the model architecture/library approach for eGNN and PointNet++ as new
LabelTransferStage methods (off-the-shelf library vs. hand-rolled, against this project's current
lean torch/numpy/open3d dependency footprint), research each architecture's implementation
requirements, and design an evaluation strategy for learned label-transfer methods. Produces a
design document (45-DESIGN.md) synthesizing the locked decisions for Phases 46–49 — no production
code in this phase. (Note: /gsd:ai-integration-phase was evaluated and rejected as a category error —
eGNN/PointNet++ are classical supervised point-cloud networks, not LLM/agent frameworks; see
45-CONTEXT.md.)
**Requirements**: D-01 through D-04 (see 45-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 44
**Plans:** 1/1 plans complete

Plans:

- [x] 45-01-PLAN.md — Verify torch_geometric core install + write 45-DESIGN.md locking per-model library, module structure, joint-cloud adaptation, train/infer split, and eval-strategy pointers for Phases 46–49 (D-01, D-02, D-03, D-04)

**Status:** ✅ Complete — torch_geometric core verified installable, 45-DESIGN.md locks all Phase 46-49 architecture/library decisions.

### Phase 46: Training Data Pipeline for Learned Label Transfer

**Goal:** Extend DataFactory to emit seed-driven (source cloud + labels, target cloud, target
labels) training triples for Phase 47's eGNN/PointNet++ training loop — porting a single-frame
ball/bowl geometry sampler into `zreg.generators`, wiring the existing Voronoi `generate_labels`,
reusing Phase 30's transform-based exact-correspondence `generate_target`, and adding a seed-level
train/val split. Small point-count regime only (100-300 pts/frame), on-the-fly seeded generation,
no persisted dataset.
**Requirements**: D-01 through D-04 (see 46-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 45
**Plans:** 3/3 plans complete

Plans:

- [x] 46-01-PLAN.md — Port single-frame sample_ball()/sample_bowl() geometry samplers into zreg/generators (D-04)
- [x] 46-02-PLAN.md — Add frozen TrainingTriple result model to eval/types.py (D-02)
- [x] 46-03-PLAN.md — split_seeds() + DataFactory.generate_training_triple()/generate_training_set() + augment() per-seed RNG threading + Wave-0 tests (D-01, D-02, D-03, D-04)

**Status:** ✅ Complete — 1248 tests pass (was 1229 at 46-02 close), 100% coverage on zreg/eval.

### Phase 47: eGNN and PointNet++ Model Implementation & Training Infrastructure

**Goal:** Implement the eGNN and PointNet++ model architectures selected in Phase 45, plus a
training loop and checkpoint management (none of which exists in this repo today).
**Requirements**: MODEL-01 through MODEL-07 (derived in 47-RESEARCH.md — phase added ad hoc, no formal REQUIREMENTS.md IDs); binds D-01 through D-03 (see 47-CONTEXT.md)
**Depends on:** Phase 46
**Plans:** 5/5 plans complete

Plans:

- [x] 47-01-PLAN.md — Foundation: declare torch_geometric in setup.cfg + src/zreg/models/_ops.py (Open3D FPS/ball-query/radius-graph, isolated-point guard) + ops tests incl. D-03 benchmark (MODEL-01, MODEL-04)
- [x] 47-02-PLAN.md — PointNet++ joint-cloud model: SetAbstraction/FeaturePropagation/PointNet2LabelTransfer + tests (MODEL-02, MODEL-04)
- [x] 47-03-PLAN.md — eGNN: verified single-propagate EGNNConv (MessagePassing subclass) + EGNNLabelTransfer + E(3) equivariance test (MODEL-03, MODEL-04)
- [x] 47-04-PLAN.md — Training entry point (train_label_transfer.py, MPS-aware) + models __init__ exports + smoke test (forward/backward/loss-decrease/checkpoint round-trip + MPS run) (MODEL-04, MODEL-05, MODEL-06, MODEL-07)
- [x] 47-05-PLAN.md — Joint-cloud output-shape==target-point-count contract test for both models (MODEL-02, MODEL-03)

**Status:** ✅ Complete — all 5 plans executed. 1285 tests pass (was 1248 at Phase 46 close), zero regressions.

### Phase 48: LabelTransferStage Integration for Learned Methods

**Goal:** Wire trained eGNN/PointNet++ models into LabelTransferStage as new selectable method
values (checkpoint loading, inference-time dispatch), following the OPTIONAL_PARAMS +
EvalConfig.label_transfer_method precedent established in Phase 44 for cpd_weighted.
**Requirements**: D-01 through D-03 (see 48-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 47
**Plans:** 2/2 plans complete

Plans:

- [x] 48-01-PLAN.md — EvalConfig egnn_checkpoint_path/pointnet2_checkpoint_path fields + 4-value label_transfer_method validator (D-02)
- [x] 48-02-PLAN.md — LabelTransferStage learned-method wiring: VALID_METHODS + _load_learned_model + train_step-parity joint-cloud branch + smoke-test verification (D-01, D-02, D-03)

**Status:** ✅ Complete — both plans executed. 1300 tests pass (was 1285 at Phase 47 close), zero regressions.

### Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods

**Goal:** Apply the existing F1/knn_consistency metrics plus any additional evaluation dimensions
from Phase 45's strategy to benchmark eGNN/PointNet++ against the knn_voting/cpd_weighted
baselines across this project's dataset sources.
**Requirements**: D-01 through D-03, D-02.1 through D-02.4 (see 49-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 48
**Plans:** 3/3 plans complete

Plans:

- [x] 49-01-PLAN.md — MethodBenchmarkResult/BenchmarkReport types + Wave-0 verification (D-01, D-02.2 fixtures)
- [x] 49-02-PLAN.md — LabelTransferBenchmark orchestration class (D-01, D-02.1–D-02.4, D-03)
- [x] 49-03-PLAN.md — benchmark_label_transfer.py CLI entry point (D-01)

**Status:** ✅ Complete — all 3 plans executed. 1312 tests pass (was 1300 at Phase 48 close), zero regressions. eGNN/PointNet++ track (Phases 44–49) fully complete.

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

**Execution Order (v1.6):**
Phases execute in numeric order: 51 → 52 → 53 → 54

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 44. CPD-Weighted Label Transfer | v1.5 | 4/4 | ✅ Complete | 2026-07-15 |
| 45. eGNN/PointNet++ Framework Selection | v1.5 | 1/1 | ✅ Complete | 2026-07-15 |
| 46. Training Data Pipeline | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 47. eGNN/PointNet++ Models & Training | v1.5 | 5/5 | ✅ Complete | 2026-07-15 |
| 48. LabelTransferStage Integration | v1.5 | 2/2 | ✅ Complete | 2026-07-15 |
| 49. Benchmarking of Learned Methods | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 50. GPU-Native Geometry Ops | v1.5 | 0/TBD | Pending | - |
| 51. Environment & Access | v1.6 | 1/1 | Complete   | 2026-07-15 |
| 52. Multi-Rank Parallelism & Validation | v1.6 | 0/TBD | Not started | - |
| 53. GPU Acceleration | v1.6 | 0/TBD | Not started | - |
| 54. Budget Calibration & Full-Suite Gate | v1.6 | 0/TBD | Not started | - |

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 Consolidation | 1–5 (5) | 13 | ✅ Complete | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 (7) | 14 | ✅ Complete | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 (27) | 55 | ✅ Complete | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 (5) | 11 | ✅ Complete | 2026-07-08 |
| v1.5 Learned Label Transfer Methods | 44–50 (7) | 18 | 🚧 In progress (50 pending) | - |
| v1.6 HoreKa Cluster Execution | 51–54 (4) | TBD | 🚧 In progress | - |

_Next: `/gsd:execute-phase 51` to run Phase 51 (Environment & Access)._
