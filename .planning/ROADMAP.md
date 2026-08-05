# Roadmap: zReg

## Milestones

- ✅ **v1.7 Minor Adjustments** — Phases 57–58 (shipped 2026-08-04)
- ⏸️ **v1.6 HoreKa Cluster Execution** — Phases 51–54 (paused — Phase 54 remaining, resumes after v1.7)
- 🚧 **v1.5 Learned Label Transfer Methods** — Phases 44–50 (in progress — Phase 50 pending)
- ✅ **v1.4 Trajectory Alignment & Optimization Enhancements** — Phases 39–43 (shipped 2026-07-08) — [archive](.planning/milestones/v1.4-ROADMAP.md)
- ✅ **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–38 (shipped 2026-06-26) — [archive](.planning/milestones/v1.2-ROADMAP.md)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1–5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

**Phase Numbering:**

- Integer phases (44–58): Active or planned work
- Decimal phases (e.g. 44.1): Urgent insertions (marked with INSERTED)
- Note: Phase 55 was already consumed by an ad-hoc out-of-band phase (spherical-cap/Gaussian label generators for `zreg.data_generation.labels`, unrelated to any numbered milestone), completed 2026-07-30. Phase 56 was also already consumed — by a separate ad-hoc phase developed concurrently on `feature/evaluation_framework` (Configurable multi-label region-based labeling, completed 2026-07-31) before this milestone's branch merged back in. v1.7 was originally planned as Phases 56–57, but was renumbered to 57–58 on merge to resolve that collision — see Phase Details below for the corresponding `.planning/phases/` directory renames.

### ✅ v1.7 Minor Adjustments (Complete)

**Milestone Goal:** General-purpose catch-all milestone for small, unrelated fixes and additions that don't warrant their own themed milestone — fixing the ground-truth field mismatch in the evaluation framework and adding a synthetic labeled subsample-pair generation mechanism. (Paused v1.6 HoreKa work resumes at Phase 54 after this milestone ships.)

- [x] **Phase 57: Ground-Truth Field Consistency** - Label-transfer F1 ground truth is drawn from the same field `LabelTransferStage` actually transfers, and stays correctly paired even when `transform_spec` changes point counts (completed 2026-08-04)
- [x] **Phase 58: Synthetic Labeled Subsample-Pair Generation** - Label-transfer HPO can run against a config-driven synthetic subsample-pair ground truth mechanism, alongside the existing `transform_spec` mechanism (completed 2026-08-04)

### 🚧 v1.6 HoreKa Cluster Execution (In Progress)

**Milestone Goal:** Run the `baseline_experiments` evaluation suite on the HoreKa HPC cluster with GPU support, replacing the current laptop-constrained (CPU-only, subsampled) execution, within a 3-hour GPU time budget.

- [x] **Phase 51: Environment & Access** - Operator can activate the HoreKa environment, transfer real datasets, and submit a working end-to-end job script *(1 plan — ready to execute)* (completed 2026-07-15)
- [ ] **Phase 52: Multi-Rank Parallelism & Validation** - HPO trials run concurrently across MPI ranks via propulate, orchestration stays single-writer, validated on a short test job
- [ ] **Phase 53: GPU Acceleration** - Real per-operation GPU acceleration threaded through EvalConfig, DataFactory, and AlignmentStage
- [ ] **Phase 54: Budget Calibration & Full-Suite Gate** - Full 7-run suite is calibrated and verified to fit the 3-hour GPU cap before the full allocation is submitted

## Phase Details

### Phase 57: Ground-Truth Field Consistency

**Goal**: Label-transfer F1 scoring is computed against the correct ground-truth field (`pc["label"]`, the field `LabelTransferStage` actually transfers) with correct per-point correspondence preserved even when `transform_spec`'s `dropout_fraction`/`n_new_points` change point counts between source and target, and existing ground-truth configs are audited/updated to the corrected convention.
**Depends on**: Nothing (first phase of v1.7)
**Requirements**: GT-01, GT-02, GT-03
**Success Criteria** (what must be TRUE):

  1. `DataFactory.get_ground_truth()` and `get_synthetic_ground_truth()` read `pc["label"]` (not `pc["id"]`) for both `pipeline_mode: paired` and `pipeline_mode: synthetic`, matching the field `LabelTransferStage` actually transfers
  2. When `transform_spec`'s `dropout_fraction`/`n_new_points` cause source and target point counts to diverge, `eval_runner._run_single`'s `y_true`/`y_pred` pairing preserves correct per-point correspondence instead of truncating both arrays to `min(len(y_true), len(y_pred))` by position
  3. Existing ground-truth configs (`baseline_experiments/configs/ground_truth/shah_sample1.yaml`, `kobitski_ew06.yaml`, and `configs/experiments/stage2_label_transfer/*/synthetic.yaml`) are audited and updated so label-transfer F1 remains meaningful under the corrected GT-field convention
  4. A label-transfer evaluation run against a `transform_spec` config with nonzero `dropout_fraction`/`n_new_points` produces a non-degenerate F1 score (not silently near-zero or spuriously perfect from misaligned arrays)

**Plans**: 2 plans

Plans:

- [x] 57-01-PLAN.md — EvalConfig.ground_truth_field + DataFactory correspondence-tracking (drop_points/sample_new_points) + GT extraction rewrite + eval_runner WR-01 re-scoping (GT-01, GT-02)
- [x] 57-02-PLAN.md — Ground-truth config audit: Shah/Kobitski comment fixes + degenerate-label documentation on stage2_label_transfer synthetic.yaml configs (GT-03)

### Phase 58: Synthetic Labeled Subsample-Pair Generation

**Goal**: Users can configure a synthetic evaluation dataset pair generated by subsampling a larger labeled point cloud into two views (source/target) with known per-point correspondence, and run label-transfer HPO against it — a sibling mechanism to the existing known-transform (`transform_spec: rigid`/`noise`) path.
**Depends on**: Phase 57
**Requirements**: GT-04, GT-05, GT-06
**Success Criteria** (what must be TRUE):

  1. User can specify, via YAML config, a synthetic dataset pair generated by subsampling a larger labeled point cloud into two views (source/target) with known per-point correspondence
  2. `DataFactory` exposes a method that generates/retrieves such labeled subsample pairs, reusing `generate_training_triple()`'s geometry -> `generate_labels()` -> `generate_target()` composition pattern
  3. `EvaluationRunner` and `HyperparamOptimizer` can run a label-transfer HPO sweep end-to-end against subsample-pair-generated ground truth (not just the training pipeline that originally consumed `generate_training_triple()`)
  4. The subsample-pair mechanism coexists with the existing `transform_spec` known-transform mechanism — both remain independently selectable via config without one breaking the other

**Plans**: 4 plans

Plans:

**Wave 1**

- [x] 58-01-PLAN.md — DataFactory.generate_subsample_pair() + _subsample_source_view attribute + config docstring (GT-04, GT-05)

**Wave 2** *(both depend on 57-01, no file overlap between them)*

- [x] 58-02-PLAN.md — EvaluationRunner.run() subsample_pair dispatch branch (GT-04, GT-06)
- [x] 58-03-PLAN.md — HyperparamOptimizer single-seed subsample_pair wiring across run()/_tier_dataset()/_objective() (GT-06)

**Wave 3** *(depends on 57-03, same file)*

- [x] 58-04-PLAN.md — D-07 opt-in multi-seed averaging, gated to the full tier (GT-04, GT-06)

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

**Plans**: 3 plans
Plans:

- [x] 52-01-PLAN.md — run_all.py rank-awareness + --configs-dir arg (PARA-01, PARA-02)
- [x] 52-02-PLAN.md — Cluster configs (configs_horeka/ — 7 mirrored YAMLs) (PARA-03, BUDG-01)
- [x] 52-03-PLAN.md — Multi-rank test job (smoke config + launch_horeka_multirank_test.sbatch) (BUDG-04)

### Phase 53: GPU Acceleration

**Goal**: Real per-operation GPU acceleration is threaded through the pipeline — configurable device, data loaded onto that device, and registration provably executing on GPU tensors with no silent CPU fallback.
**Depends on**: Phase 52
**Requirements**: GPU-01, GPU-02, GPU-03
**Success Criteria** (what must be TRUE):

  1. `EvalConfig` exposes a `device` field that controls where tensors are loaded and computed
  2. `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU
  3. Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback anywhere in the path

**Plans**: 2 plans
Plans:

- [ ] 53-01-PLAN.md — EvalConfig device field + DataFactory device threading + tests (GPU-01, GPU-02)
- [ ] 53-02-PLAN.md — 8 cluster YAML configs + sbatch GPU-03 annotation (GPU-03)

### Phase 54: Budget Calibration & Full-Suite Gate

**Goal**: Calibration constants are environment-aware, and the full 7-run suite's projected runtime is verified to fit the 3-hour GPU cap before the full allocation is submitted.
**Depends on**: Phase 53
**Requirements**: BUDG-02, BUDG-03
**Success Criteria** (what must be TRUE):

  1. `aggregate_cost.py`'s calibration constants are parameterized per environment (laptop vs. HoreKa), so cluster timing data doesn't silently mix with or overwrite laptop calibration
  2. Operator can run an `aggregate_cost.py`-style estimation using HoreKa GPU timing data (from Phase 52's short test job plus Phase 53's GPU path) to project the full 7-run suite's total runtime
  3. The projected total runtime for the full suite is confirmed to fit within the 3-hour GPU time cap before the full allocation is submitted (or the plan is revised if it doesn't fit)

**Plans**: 2 plans

Plans:

**Wave 1**

- [x] 54-01-PLAN.md — aggregate_cost.py argparse (--calibration, --configs-dir, --budget-hours) + calibration JSON files (BUDG-02, BUDG-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 54-02-PLAN.md — extract_calibration.py new script + tests for aggregate_cost.py and extract_calibration.py (BUDG-02, BUDG-03)

### 🚧 v1.5 Learned Label Transfer Methods (Phases 44–49 complete, Phase 50 pending)

**Milestone Goal:** Extend LabelTransferStage with learned point-cloud methods (CPD-weighted, eGNN, PointNet++) — full stack from architecture selection through training infrastructure, inference integration, and evaluation/benchmarking.

- [x] **Phase 44: CPD-Weighted Label Transfer** — completed 2026-07-15
- [x] **Phase 45: eGNN/PointNet++ Framework Selection & Evaluation Strategy** — completed 2026-07-15
- [x] **Phase 46: Training Data Pipeline** — completed 2026-07-15
- [x] **Phase 47: eGNN/PointNet++ Model Implementation & Training Infrastructure** — completed 2026-07-15
- [x] **Phase 48: LabelTransferStage Integration for Learned Methods** — completed 2026-07-15
- [x] **Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods** — completed 2026-07-15
- [x] **Phase 50: GPU-Native Geometry Ops** — pending (blocked on cluster verification) (completed 2026-07-31)

### Phase 55: Spherical-cap and Gaussian label generators for zreg.data_generation.labels ✅ 2026-07-30

**Goal:** zreg.data_generation.labels exposes assign_cap_labels (hard spherical-cap boundary) and assign_gaussian_labels (angle-dependent Bernoulli labels), both following the immutable deep-copy contract and exported from the package.
**Requirements**: none mapped
**Depends on:** Phase 54
**Plans:** 4/4 plans complete

Plans:

- [x] 55-01-PLAN.md — Implement assign_cap_labels + assign_gaussian_labels, export them, and add tests

### Phase 56: Configurable multi-label region-based labeling: rework generate_labels() to support arbitrary n_labels, voronoi/gaussian-blob/gaussian-cone region shapes, deterministic and probabilistic assignment modes, and config-driven specification via EvalConfig ✅ 2026-07-31

**Goal:** zreg.data_generation.labels.generate_labels() becomes the single, config-driven entry point for labeling a point cloud trajectory — any number of labels, each defined by one or more region components (voronoi/gaussian blob/gaussian cone), assigned deterministically or probabilistically, with region centers fixed once per trajectory (not redrawn per frame) so label change is spatially traceable. assign_cap_labels/assign_gaussian_labels (Phase 55) are deleted entirely, absorbed as the cone shape's special case. EvalConfig.label_generation lets scenario YAML declare a label spec instead of hardcoded Python literals.
**Requirements**: none mapped (see 56-CONTEXT.md D-01 through D-13 — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 55
**Plans:** 5/5 plans complete

Plans:

**Wave 1**

- [x] 56-01-PLAN.md — LabelComponentSpec/LabelSpec pydantic models + per-shape scoring (_component_score) + mixture aggregation (_label_scores) (D-08, D-09, D-10, D-11)

**Wave 2** *(depends on Wave 1)*

- [x] 56-02-PLAN.md — Assignment modes (_assign_deterministic/_assign_probabilistic) + reworked generate_labels() orchestrator with D-07 center-once fix + deletion of assign_cap_labels/assign_gaussian_labels and their 13 tests (D-01, D-04, D-05, D-06, D-07, D-12)

**Wave 3** *(depends on Wave 2)*

- [x] 56-03-PLAN.md — EvalConfig.label_generation field + n_classes->n_labels rename cascade through DataFactory/optimizer.py/benchmark_runner.py/train_label_transfer.py + example scenario YAML (D-04, D-13)

**Wave 4** *(depends on Waves 2+3, parallel)*

- [x] 56-04-PLAN.md — Comprehensive new tests: per-shape correctness, assignment-mode behavior, D-07 regression, LabelGenerationConfig coverage (D-07, D-08, D-09, D-10, D-11, D-12, D-13)
- [x] 56-05-PLAN.md — Fix rename-cascade breakage across the remaining existing test suite (9 files) (D-04)

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
**Plans:** 2/2 plans complete

Plans:

- [x] 50-01-PLAN.md — HoreKa torch_cluster install verification gate (autonomous: false, human-executed) (D-01, D-02)
- [x] 50-02-PLAN.md — torch_cluster dual-path in _ops.py for all three ops + setup.cfg cluster extra + smoke tests (D-03–D-08)

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

**Execution Order (v1.7):**
Phases execute in numeric order: 57 → 58 (v1.6 paused after Phase 51; resumes at Phase 54 once v1.7 ships)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 44. CPD-Weighted Label Transfer | v1.5 | 4/4 | ✅ Complete | 2026-07-15 |
| 45. eGNN/PointNet++ Framework Selection | v1.5 | 1/1 | ✅ Complete | 2026-07-15 |
| 46. Training Data Pipeline | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 47. eGNN/PointNet++ Models & Training | v1.5 | 5/5 | ✅ Complete | 2026-07-15 |
| 48. LabelTransferStage Integration | v1.5 | 2/2 | ✅ Complete | 2026-07-15 |
| 49. Benchmarking of Learned Methods | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 50. GPU-Native Geometry Ops | v1.5 | 2/2 | Complete    | 2026-07-31 |
| 51. Environment & Access | v1.6 | 1/1 | Complete    | 2026-07-15 |
| 52. Multi-Rank Parallelism & Validation | v1.6 | 0/TBD | Not started | - |
| 53. GPU Acceleration | v1.6 | 0/TBD | Not started | - |
| 54. Budget Calibration & Full-Suite Gate | v1.6 | 0/TBD | Not started | - |
| 57. Ground-Truth Field Consistency | v1.7 | 2/2 | Complete    | 2026-08-04 |
| 58. Synthetic Labeled Subsample-Pair Generation | v1.7 | 4/4 | Complete    | 2026-08-04 |

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 Consolidation | 1–5 (5) | 13 | ✅ Complete | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 (7) | 14 | ✅ Complete | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 (27) | 55 | ✅ Complete | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 (5) | 11 | ✅ Complete | 2026-07-08 |
| v1.5 Learned Label Transfer Methods | 44–50 (7) | 18 | 🚧 In progress (50 pending) | - |
| v1.6 HoreKa Cluster Execution | 51–54 (4) | TBD | 🚧 In progress | - |
| v1.7 Minor Adjustments | 57–58 (2) | 6 | ✅ Complete | 2026-08-04 |

_v1.7 Minor Adjustments complete (Phases 57–58, both verified; renumbered from the originally-planned 56–57 on merge into `feature/evaluation_framework` to resolve a Phase 56 collision with that branch's own concurrent work — see the Phase Numbering note above). v1.6 HoreKa Cluster Execution resumes next: `/gsd:plan-phase 54` (Budget Calibration & Full-Suite Gate). Or run `/gsd:complete-milestone` to formally close out v1.7 first._
