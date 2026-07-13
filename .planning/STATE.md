---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: milestone
status: Phase 47 Complete
stopped_at: Completed 47-05-PLAN.md (5 of 5 plans in Phase 47 — Phase 47 fully complete)
last_updated: "2026-07-13T11:50:00.000Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 13
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-08 after v1.4 milestone)

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** Phase 47 — egnn-and-pointnet-model-implementation-training-infrastructu

## Current Position

Phase: 47 (egnn-and-pointnet-model-implementation-training-infrastructu) — ✅ COMPLETE (5/5 plans)
Plan: 5 of 5 complete (47-01, 47-02, 47-03, 47-04, 47-05 all complete)
Milestone: v1.4 Trajectory Alignment & Optimization Enhancements — **ARCHIVED** 2026-07-08
Tests: 1285 passed, 18 skipped, 1 xpassed (was 1281 at 47-04 close; +4 new tests from 47-05's test_zreg_models_joint_cloud.py)
Next action: Plan Phase 48 (LabelTransferStage integration for learned methods) via /gsd:plan-phase 48

## Shipped Milestones

| Milestone | Phases | Shipped |
|-----------|--------|---------|
| v1.0 Consolidation | 1–5 | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 | 2026-06-26 |

Full history: .planning/MILESTONES.md · Retrospective: .planning/RETROSPECTIVE.md
v1.2 archives: .planning/milestones/v1.2-ROADMAP.md · v1.2-REQUIREMENTS.md · v1.2-MILESTONE-AUDIT.md

## v1.4 Milestones (In Progress)

| Phase | Name | Plans | Status |
|-------|------|-------|--------|
| 39 | ICP Registration as CPD Alternative | 2 | ✅ Completed |
| 40 | Sliced Wasserstein Variants as Alignment | 4 | ✅ Completed |
| 41 | Alignment Preprocessing (PCA + Velocity) | 2 | ✅ Completed |
| 42 | Sobol Quasi-Random Search as Default | 2 | ✅ Completed |
| 43 | Per-Trajectory Data Standardization | 1 | ✅ Completed |
| 44 | CPD-Weighted Label Transfer Method (ad hoc, post-v1.4) | 4 | ✅ Completed |
| 45 | eGNN & PointNet++ Framework Selection & Evaluation Strategy (ad hoc, post-v1.4) | 1 | ✅ Completed |
| 46 | Training Data Pipeline for Learned Label Transfer (ad hoc, post-v1.4) | 3 | ✅ Completed |
| 47 | eGNN and PointNet++ Model Implementation & Training Infrastructure (ad hoc, post-v1.4) | 5 | ✅ Completed |

## Accumulated Context

### Roadmap Evolution

- Phase 44 added: CPD-Weighted Label Transfer Method — wire zreg.color_transfer's existing CPD_WEIGHTED method into LabelTransferStage as an alternative to the hardcoded KNN_VOTING, mirroring the alignment_method optional-param precedent from Phase 39. Added outside a formal milestone (v1.4 archived, v1.5 not yet started) at user request.
- Phases 45–49 added: eGNN & PointNet++ as further LabelTransferStage methods — split into (45) framework selection & eval strategy design (AI-SPEC.md, no code), (46) training data pipeline, (47) model + training infra, (48) inference integration mirroring Phase 44's OPTIONAL_PARAMS precedent, (49) evaluation/benchmarking. Linear dependency chain 45→46→47→48→49. Added outside a formal milestone at user request; unlike Phase 44 (pure plumbing over existing zreg code), this is genuinely new ML system scope — no GNN/PointNet/equivariant-net code or training infra exists anywhere in this repo yet.
- Phase 45 complete: `45-DESIGN.md` locks the per-model library decision (PointNet++ hand-rolled on Open3D; eGNN hand-rolled `MessagePassing` subclass on `torch_geometric` 2.8.0, verified installable), target `src/zreg/models/` module layout, joint-cloud conditioning adaptation, label-vs-id discipline, GPU-train/CPU-infer split, and downstream phase ownership map for Phases 46–49. New finding: `zreg` (or any `scipy`-importing module) must be imported before `torch_geometric` to avoid a libomp SIGABRT — extends the existing macOS-ARM zreg-before-torch convention.
- Phase 46 plan 01 complete: research found `DataFactory.generate_synthetic()` does NOT implement the bowl/ball growth model assumed by 45-CONTEXT.md/45-DESIGN.md (D-04) — it produces unlabeled Gaussian-blob frames only. Ported a simplified, single-frame (non-growth) `sample_ball()`/`sample_bowl()` pair from `scripts/generate_datasets.py`'s standalone rejection-sampling math into `zreg/generators/generators.py`, exported from `zreg.generators`. Both samplers use `np.random.default_rng(seed)` (numpy-native geometry math) rather than `torch.manual_seed`, documented as an explicit, intentional deviation from `generate_trajectory`'s seed contract.
- Phase 46 plan 02 complete: added `TrainingTriple` frozen pydantic model to `eval/types.py` (D-02) — source_cloud/target_cloud/seed fields, `source_labels`/`target_labels` properties reading exclusively from `pc["label"]` (never `pc["id"]`, per 45-DESIGN.md's label-vs-id discipline). Matches `AlignResult`/`LabelResult`'s exact `ConfigDict(frozen=True, arbitrary_types_allowed=True)` convention — seventh frozen result model in `eval/types.py`, zero raw tuple/dict public return types anywhere in `eval/`. TDD plan-level gate followed (RED test commit `8fbc8be` → GREEN implementation commit `c750cf4`). 1229 tests pass (was 1218; +11 new), 100% coverage on `eval`/`zreg`.
- Phase 46 plan 03 complete (Phase 46 now fully complete, 3/3): `augment()` now threads a caller-supplied `"augment_seed"` (default 42) to its four internal stochastic sub-calls (Pitfall 3 resolved); added module-level `split_seeds(n_train, n_val, base_seed=0)` (disjoint-by-construction seed ranges, D-03); added `DataFactory.generate_training_triple(seed, n_classes=6, shape=None, n_points=None) -> TrainingTriple` composing `sample_ball`/`sample_bowl` + `generate_labels` + `generate_target` (100-300 pts/frame per D-01, ball/bowl alternates by seed parity per D-04, seed-reproducible/seed-varied per D-02, deliberately bypasses the `_synthetic_dataset` singleton cache per Pitfall 2); added `DataFactory.generate_training_set(seeds, n_classes=6)` batching companion for `split_seeds()`. `tests/test_data_factory_training_triples.py` is the Wave-0 test file mandated by 45-DESIGN.md (7 classes, 22 tests). 1248 tests pass (was 1229; +19 new), 100% coverage on `eval`/`zreg`. Phase 46's full output (`split_seeds`, `generate_training_triple`, `generate_training_set`) is ready for Phase 47's training loop.
- Phase 47 plan 01 complete (1/5): `torch_geometric` declared in `setup.cfg install_requires` (no compiled PyG siblings). Added `src/zreg/models/_ops.py` — `farthest_point_sample`, `ball_query` (isolated-point self-fallback guard, Pitfall 2), `build_radius_graph` (directed `[2, E]` edge_index, self-loops excluded by design since `EGNNConv`'s residual update already covers the self term, isolated points fall back to a self-loop) — all device-agnostic (`pos.device`, zero bare CUDA calls). D-03 (Open3D wrapper benchmark-first) resolved: `tests/test_zreg_models_ops.py`'s benchmark test confirms combined FPS+ball-query wall-clock stays under 50ms at 100/200/300 points (1.9-4.7ms measured) — no vectorization needed. Tasks 2+3 (both `tdd="true"`) executed as a single RED (test file, `056ff84`) → GREEN (implementation, `f3aad21`) cycle. 1259 tests pass (was 1248; +11 new), zero regressions.
- Phase 47 plan 02 complete (2/5): Added `src/zreg/models/pointnet2.py` — `SetAbstraction` (FPS → `_ops.ball_query` → per-group `[relative_pos|neighbour_feat]` MLP → max-pool, ratio-based `n_samples=max(1,int(n*ratio))`), `FeaturePropagation` (3-NN inverse-distance interpolation via `open3d.geometry.KDTreeFlann.search_knn_vector_3d(p,3)` + U-Net skip concat + MLP), and `PointNet2LabelTransfer` (2 SA hidden 32→64 + 2 symmetric FP + per-point head → `n_classes` logits over the full joint cloud). The model has no notion of `n_source` — callers slice `logits[n_source:]` for the target-only supervised subset, documented in the class docstring and exercised by tests. `_ops.ball_query`'s isolated-point self-fallback guard (from plan 01) is consumed as-is, keeping `sample_bowl`'s sparse rim regions from crashing max-pool (T-47-03). Task 1 (`e2165aa`, feat) then Task 2 (`20f9de9`, test) — Task 2's test suite passed immediately against the already-correct Task 1 implementation, no RED failures needed since Task 1's own inline verify command already gated correctness. 1265 tests pass (was 1259; +6 new), zero regressions, 100% coverage on `pointnet2.py`.
- Phase 47 plan 03 complete (3/5): Added `src/zreg/models/egnn.py` — `EGNNConv(MessagePassing)` copied unmodified from 47-RESEARCH.md Pattern 4's session-verified code (`super().__init__(aggr=None, flow="source_to_target")`, single `propagate()` call, `message()` returns `cat([m_ij, coord_msg])`, custom `aggregate()` splits `[hidden_dim, 3]` and independently scatters sum (features) / mean (coordinates) via `torch_geometric.utils.scatter`) and `EGNNLabelTransfer` (Linear embed `n_classes+1 -> hidden_dim`, `_ops.build_radius_graph` over the joint cloud built once and reused across all layers, 4 stacked `EGNNConv`, Linear readout to `n_classes` logits — same `logits[n_source:]` slicing contract as `PointNet2LabelTransfer`). `tests/test_egnn_equivariance.py` reproduces the session's manual numerical equivariance check as an automated `atol=1e-4` regression (coordinate equivariance + feature invariance under a fixed z-rotation), plus forward/backward gradient and end-to-end joint-cloud tests. Task 1 (`cca5b1f`, feat) then Task 2 (`777e158`, test) — same no-RED-needed pattern as 47-02 since Task 1's inline verify command already gated correctness. 1269 tests pass (was 1265; +4 new), zero regressions, 100% coverage on `egnn.py`.
- Phase 47 plan 04 complete (4/5): `src/zreg/models/__init__.py` now exports the full public API (`PointNet2LabelTransfer`, `EGNNLabelTransfer` alongside the existing `_ops` functions). Added repo-root `train_label_transfer.py` mirroring `run_eval.py`'s argparse/sys.path convention (47-RESEARCH.md Open Question 3): `resolve_device` (MPS-aware, cuda->mps->cpu, D-01), shared model-agnostic `train_step` (Pattern 5 — joint-cloud build, one-hot source labels + unknown-flag target rows, masked cross-entropy over `logits[n_source:]`, optimizer step), a per-triple training loop over `DataFactory.generate_training_set`, and `save_checkpoint` (Pattern 6 — plain dict of `state_dict`/`model_class`/`hyperparams`/`epoch`, `weights_only=True`-compatible). `tests/test_train_label_transfer.py` (12 tests): forward/backward gradient flow, loss-decrease over a handful of steps, checkpoint round-trip via `torch.equal`, and an explicit MPS iteration test guarded by a local (file-scoped) MPS-aware device fixture — `tests/conftest.py`'s shared CUDA-only fixture was left untouched. Notably, the MPS iteration test **passed** (not skipped) for both models on this dev machine — empirically resolves 47-RESEARCH.md Open Question 2 for this dev machine's op coverage. Task 1 (`0e7f386`, feat) → Task 2 (`d057b25`, feat) → Task 3 (`7f8f941`, test) — same no-RED-needed pattern as 47-02/47-03 since Task 2's own inline verify command already gated correctness. 1281 tests pass (was 1269; +12 new), zero regressions. `checkpoints/` added to `.gitignore` (training artifacts, not source).
- Phase 47 plan 05 complete (5/5, Phase 47 now fully complete): added `tests/test_zreg_models_joint_cloud.py`, the mandated 45-DESIGN.md Wave-0 test single-sourcing the joint-cloud slicing contract for BOTH `PointNet2LabelTransfer` and `EGNNLabelTransfer` — parametrized over both model classes and two (n_source, n_target) size regimes ((100,100), (300,200)), asserting `logits[n_source:].shape == (n_target, n_classes)` in every case (4 collected tests, all green immediately since both models' contracts were already correct from 47-02/47-03). Both models imported directly from their submodules, independent of 47-04's `__init__.py` rewrite. Single task (`6a8d7e1`, test) — no RED phase needed, same pattern as 47-02/47-03/47-04. 1285 tests pass (was 1281; +4 new), zero regressions. Full-suite run (all of Phase 47's tests) confirms no regressions across the whole phase.

### v1.4 Design Decisions

- **ICP**: Wrap Open3D (point-to-point) as registration method
- **SWD**: Leverage existing zReg Sliced Wasserstein variants (SWD, ASWD, OSWD, GSWD, PSWD, MaxSWD)
- **Preprocessing**: Principal axes alignment + velocity landmark detection
- **Sobol**: Default search strategy (grid search as fallback)
- **Standardization**: Default data preprocessing (normalization as optional)
- **Execution**: All 5 phases independent → can run in parallel

### Requirements

See: `.planning/REQUIREMENTS-v1.4.md`

- ALIGN-04: ICP integration (5 requirements)
- ALIGN-05: SWD variants (5 requirements)
- ALIGN-06: Preprocessing (5 requirements)
- OPT-04: Sobol search (6 requirements)
- DATA-02: Standardization (7 requirements)

### Open Blockers

None — Phase 47 complete (5/5 plans). No blockers for Phase 48 planning.

### Performance Metrics

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

### Deferred Items (acknowledged at v1.2 close)

| Category | Item | Status |
|----------|------|--------|
| verification | Phase 26 SC-4 — live `mpirun -n 2` Propulate integration | Unverified (missing GPy in dev env; Optuna path unaffected) |
| nyquist | 14 of 27 phases without VALIDATION.md | Backfill optional via `/gsd:validate-phase N` |
| metadata | Empty `requirements-completed` SUMMARY frontmatter on most phases | Coverage confirmed via VERIFICATION evidence tables |
| code-review | Open CR/WR items (matplotlib Agg backend leak; `subprocess.run` timeout in `trajectory.py`; `eval/` excluded from `--cov`) | Non-blocking; track in next milestone |

## Session Continuity

Last session: 2026-07-13T11:50:00+02:00
Stopped at: Completed 47-05-PLAN.md (5 of 5 plans in Phase 47 — Phase 47 fully complete)
Next action: Plan Phase 48 (LabelTransferStage integration for learned methods) via /gsd:plan-phase 48
