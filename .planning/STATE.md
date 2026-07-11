---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: milestone
status: Ready to plan
stopped_at: Completed 45-01-PLAN.md (Phase 45 complete)
last_updated: "2026-07-11T09:09:06.124Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-08 after v1.4 milestone)

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** Phase 46 — training-data-pipeline-for-learned-label-transfer

## Current Position

Phase: 46
Plan: Not started
Milestone: v1.4 Trajectory Alignment & Optimization Enhancements — **ARCHIVED** 2026-07-08
Tests: 1205 passed, 18 skipped, 1 xpassed, 100% coverage (Phase 45 was design-only — no test suite changes)
Next action: `/gsd:plan-phase 46` to research and break Phase 46 (training data pipeline) down into plans.

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

## Accumulated Context

### Roadmap Evolution

- Phase 44 added: CPD-Weighted Label Transfer Method — wire zreg.color_transfer's existing CPD_WEIGHTED method into LabelTransferStage as an alternative to the hardcoded KNN_VOTING, mirroring the alignment_method optional-param precedent from Phase 39. Added outside a formal milestone (v1.4 archived, v1.5 not yet started) at user request.
- Phases 45–49 added: eGNN & PointNet++ as further LabelTransferStage methods — split into (45) framework selection & eval strategy design (AI-SPEC.md, no code), (46) training data pipeline, (47) model + training infra, (48) inference integration mirroring Phase 44's OPTIONAL_PARAMS precedent, (49) evaluation/benchmarking. Linear dependency chain 45→46→47→48→49. Added outside a formal milestone at user request; unlike Phase 44 (pure plumbing over existing zreg code), this is genuinely new ML system scope — no GNN/PointNet/equivariant-net code or training infra exists anywhere in this repo yet.
- Phase 45 complete: `45-DESIGN.md` locks the per-model library decision (PointNet++ hand-rolled on Open3D; eGNN hand-rolled `MessagePassing` subclass on `torch_geometric` 2.8.0, verified installable), target `src/zreg/models/` module layout, joint-cloud conditioning adaptation, label-vs-id discipline, GPU-train/CPU-infer split, and downstream phase ownership map for Phases 46–49. New finding: `zreg` (or any `scipy`-importing module) must be imported before `torch_geometric` to avoid a libomp SIGABRT — extends the existing macOS-ARM zreg-before-torch convention.

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

None — Phase 45 complete (design-only, 1 plan executed).

### Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 44 | 01 | 33min | 2 | 3 |
| 44 | 02 | 25min | 2 | 2 |
| 44 | 03 | 24min | 2 | 2 |
| 44 | 04 | 12min | 1 | 2 |
| 45 | 01 | 12min | 2 | 1 |

### Deferred Items (acknowledged at v1.2 close)

| Category | Item | Status |
|----------|------|--------|
| verification | Phase 26 SC-4 — live `mpirun -n 2` Propulate integration | Unverified (missing GPy in dev env; Optuna path unaffected) |
| nyquist | 14 of 27 phases without VALIDATION.md | Backfill optional via `/gsd:validate-phase N` |
| metadata | Empty `requirements-completed` SUMMARY frontmatter on most phases | Coverage confirmed via VERIFICATION evidence tables |
| code-review | Open CR/WR items (matplotlib Agg backend leak; `subprocess.run` timeout in `trajectory.py`; `eval/` excluded from `--cov`) | Non-blocking; track in next milestone |

## Session Continuity

Last session: 2026-07-11T09:02:00Z
Stopped at: Completed 45-01-PLAN.md (Phase 45 complete)
Next action: Run `/gsd:plan-phase 46` to research and break down Phase 46 (training data pipeline for learned label transfer)
