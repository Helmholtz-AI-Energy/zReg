---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: milestone
status: Executing Phase 46
stopped_at: Completed 46-02-PLAN.md (2 of 3 plans in Phase 46)
last_updated: "2026-07-11T10:20:09Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 8
  completed_plans: 7
  percent: 44
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-08 after v1.4 milestone)

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** Phase 46 — training-data-pipeline-for-learned-label-transfer

## Current Position

Phase: 46 (training-data-pipeline-for-learned-label-transfer) — EXECUTING
Plan: 3 of 3 (46-01, 46-02 complete)
Milestone: v1.4 Trajectory Alignment & Optimization Enhancements — **ARCHIVED** 2026-07-08
Tests: 1229 passed, 18 skipped, 1 xpassed (was 1218 at 46-01 close; +11 new tests from 46-02's TrainingTriple model)
Next action: Execute 46-03-PLAN.md (split_seeds() + DataFactory.generate_training_triple()/generate_training_set() + augment() per-seed RNG threading + Wave-0 tests)

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
| 46 | Training Data Pipeline for Learned Label Transfer (ad hoc, post-v1.4) | 3 | 🔄 In Progress (1/3) |

## Accumulated Context

### Roadmap Evolution

- Phase 44 added: CPD-Weighted Label Transfer Method — wire zreg.color_transfer's existing CPD_WEIGHTED method into LabelTransferStage as an alternative to the hardcoded KNN_VOTING, mirroring the alignment_method optional-param precedent from Phase 39. Added outside a formal milestone (v1.4 archived, v1.5 not yet started) at user request.
- Phases 45–49 added: eGNN & PointNet++ as further LabelTransferStage methods — split into (45) framework selection & eval strategy design (AI-SPEC.md, no code), (46) training data pipeline, (47) model + training infra, (48) inference integration mirroring Phase 44's OPTIONAL_PARAMS precedent, (49) evaluation/benchmarking. Linear dependency chain 45→46→47→48→49. Added outside a formal milestone at user request; unlike Phase 44 (pure plumbing over existing zreg code), this is genuinely new ML system scope — no GNN/PointNet/equivariant-net code or training infra exists anywhere in this repo yet.
- Phase 45 complete: `45-DESIGN.md` locks the per-model library decision (PointNet++ hand-rolled on Open3D; eGNN hand-rolled `MessagePassing` subclass on `torch_geometric` 2.8.0, verified installable), target `src/zreg/models/` module layout, joint-cloud conditioning adaptation, label-vs-id discipline, GPU-train/CPU-infer split, and downstream phase ownership map for Phases 46–49. New finding: `zreg` (or any `scipy`-importing module) must be imported before `torch_geometric` to avoid a libomp SIGABRT — extends the existing macOS-ARM zreg-before-torch convention.
- Phase 46 plan 01 complete: research found `DataFactory.generate_synthetic()` does NOT implement the bowl/ball growth model assumed by 45-CONTEXT.md/45-DESIGN.md (D-04) — it produces unlabeled Gaussian-blob frames only. Ported a simplified, single-frame (non-growth) `sample_ball()`/`sample_bowl()` pair from `scripts/generate_datasets.py`'s standalone rejection-sampling math into `zreg/generators/generators.py`, exported from `zreg.generators`. Both samplers use `np.random.default_rng(seed)` (numpy-native geometry math) rather than `torch.manual_seed`, documented as an explicit, intentional deviation from `generate_trajectory`'s seed contract.
- Phase 46 plan 02 complete: added `TrainingTriple` frozen pydantic model to `eval/types.py` (D-02) — source_cloud/target_cloud/seed fields, `source_labels`/`target_labels` properties reading exclusively from `pc["label"]` (never `pc["id"]`, per 45-DESIGN.md's label-vs-id discipline). Matches `AlignResult`/`LabelResult`'s exact `ConfigDict(frozen=True, arbitrary_types_allowed=True)` convention — seventh frozen result model in `eval/types.py`, zero raw tuple/dict public return types anywhere in `eval/`. TDD plan-level gate followed (RED test commit `8fbc8be` → GREEN implementation commit `c750cf4`). 1229 tests pass (was 1218; +11 new), 100% coverage on `eval`/`zreg`.

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

None — Phase 46 plan 01 complete.

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

### Deferred Items (acknowledged at v1.2 close)

| Category | Item | Status |
|----------|------|--------|
| verification | Phase 26 SC-4 — live `mpirun -n 2` Propulate integration | Unverified (missing GPy in dev env; Optuna path unaffected) |
| nyquist | 14 of 27 phases without VALIDATION.md | Backfill optional via `/gsd:validate-phase N` |
| metadata | Empty `requirements-completed` SUMMARY frontmatter on most phases | Coverage confirmed via VERIFICATION evidence tables |
| code-review | Open CR/WR items (matplotlib Agg backend leak; `subprocess.run` timeout in `trajectory.py`; `eval/` excluded from `--cov`) | Non-blocking; track in next milestone |

## Session Continuity

Last session: 2026-07-11T10:20:09Z
Stopped at: Completed 46-02-PLAN.md (2 of 3 plans in Phase 46)
Next action: Execute 46-03-PLAN.md (split_seeds() + DataFactory.generate_training_triple()/generate_training_set() + augment() per-seed RNG threading + Wave-0 tests)
