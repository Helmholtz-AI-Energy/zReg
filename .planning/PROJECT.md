# zReg

## Current State

**Shipped:** v1.0 Consolidation — 2026-04-09 | v1.1 Code Quality & Refactoring — 2026-05-13 | **v1.2 Evaluation Framework & Debt Resolution — 2026-06-26**

zReg is a Python library for GPU-accelerated 3D point cloud registration, temporal alignment, and label (celltype) transfer using PyTorch — now paired with a complete, config-driven **evaluation framework** at the repo root (`eval/`). v1.2 (27 phases, 55 plans, 2026-05-13 → 2026-06-26) delivered: `EvalConfig` + `DataFactory`; `MetricsEngine` + frozen result types; isolated `AlignmentStage`/`LabelTransferStage`; `EvaluationRunner` + `viz.py`; a 3-tier `HyperparamOptimizer` (Optuna + MPI-parallel Propulate); a `run_eval.py` CLI with scenario configs; dual-mode evaluation (paired source↔target and synthetic transform-spec target generation with GT-aware HPO); heterogeneous cross-format (tracklets/CSV) paired evaluation; CPD-aligned trajectory output with stored-transform reuse (fixed the 8× scale convergence failure); trajectory export for LaTeX/pgfplots; a per-frame visualisation refactor; and full carry-forward debt closure incl. the codebase-wide `color`→`label` field rename. **976 tests pass, 18 skipped.** Closed via the v1.2 milestone audit (`.planning/v1.2-MILESTONE-AUDIT.md`).

## Next Milestone Goals

**v1.4 (2026-06-29 onwards): Trajectory Alignment & Optimization Enhancements**

Expand alignment and optimization pipeline with alternative algorithms and preprocessing. Currently CPD + grid search only; add SWD-based alignment, preprocessing (principal axes + velocity landmarks), Sobol quasi-random search as default, and per-trajectory data standardization. All config-driven with zero breaking changes.

**Goals:**
1. ICP as CPD alternative (wrap Open3D library)
2. Sliced Wasserstein variants (SWD, ASWD, OSWD, GSWD, PSWD) as OT alignment
3. Alignment preprocessing (principal axes alignment + velocity landmark detection)
4. Sobol quasi-random search as default (grid search remains as optional fallback)
5. Per-trajectory data standardization as default (with optional normalization)

**Success Criteria:**
- All 5 methods integrate into `EvalConfig`
- Config-driven selection via YAML (e.g., `alignment_method: swd`, `search_strategy: grid` to opt out of Sobol)
- 100% backward compatible (existing CPD + grid search workflows unchanged)
- 80%+ test coverage for new code paths
- <5% performance regression on existing pipelines

## What This Is

A Python library for GPU-accelerated 3D point cloud registration, temporal alignment, and color (celltype) transfer using PyTorch. Supports CPD registration (rigid, affine, non-rigid), sliced Wasserstein distance variants, dynamic time warping, MPI-distributed pairwise distance computation, and geometric transformations. Built for scientific computing with Open3D and torch_cluster interoperability.

## Core Value

Every existing capability works correctly, fails informatively, and is covered by tests.

## Requirements

### Validated

- ✓ Coherent Point Drift (CPD) registration (rigid, affine, non-rigid) — existing
- ✓ Sliced Wasserstein Distance variants (SWD, ASWD, OSWD, GSWD, PSWD) — existing
- ✓ Dynamic Time Warping for temporal alignment — existing
- ✓ Pairwise distance matrix computation with optional MPI — existing
- ✓ Color (celltype) transfer between aligned point clouds — existing
- ✓ Point cloud downsampling (FPS, random, uniform) — existing
- ✓ Geometric transformations (rigid, affine, non-rigid, TPS, combined) — existing
- ✓ Open3D and torch_cluster interoperability — existing
- ✓ Data loading from MATLAB/CSV formats — existing
- ✓ Input validation (NaN/inf, device mismatch) at all public API entry points — v1.0
- ✓ MaxSWD, RigidCPD scale, GSWD degree — all fixed and regression-tested — v1.0
- ✓ DTW path reconstruction — boundary and windowed edge cases fixed — v1.0
- ✓ Homogeneous coordinate stability — w-clamping + composition validation — v1.0
- ✓ CPD convergence diagnostics (n_iters, sigma2_history) and sigma2 clamping — v1.0
- ✓ ASWD tempfile auto-cleanup, MPI row-only communication — v1.0
- ✓ Color transfer enum dispatch and strict pmat shape validation — v1.0
- ✓ Configurable logging (ZREG_LOG_LEVEL env var, set_log_level() API) — v1.0
- ✓ Python 3.8 compat code removed (3.9+ minimum enforced) — v1.0
- ✓ RBF kernel normalization cached once per set_source — v1.0
- ✓ 275 regression tests covering all fixed subsystems — v1.0

### Active

(None — v1.2 shipped all planned requirements. Define the next set via `/gsd:new-milestone`.)

### Validated in v1.2 (2026-06-26)

- ✓ CARRY-01…05: carry-forward debt closure (Callable, DistanceMetric annotation, relative import, top-level config, VALIDATION backfill) — Phase 12
- ✓ EVAL-01/02: core alignment + label-transfer metrics at `src/zreg/metrics/` — Phase 13
- ✓ EVAL-03: synthetic data generators at `src/zreg/generators/` — Phase 14
- ✓ EVAL-04: experiment tracking at `eval/tracking/` (`log_run()` stdlib-only) — Phase 15
- ✓ EVAL-05: runner scripts (`run_synthetic.py`, `run_real.py`) — Phase 16
- ✓ FRAME-01/02: `EvalConfig` (pydantic v2 + `from_yaml`) + `DataFactory` — Phase 17
- ✓ FRAME-03/04: `MetricsEngine` + 6 frozen result types — Phase 18
- ✓ FRAME-05: `AlignmentStage` (PipelineStage ABC, DTW + CPD) — Phase 19
- ✓ FRAME-06: `LabelTransferStage` (kNN, raw or aligned clouds) — Phase 20
- ✓ FRAME-07/08: `EvaluationRunner` + `viz.py` (Agg backend, PDF+PNG) — Phase 21
- ✓ FRAME-09/10: `HyperparamOptimizer` (3-tier) + Grid/Random/Optuna strategies — Phase 22
- ✓ FRAME-11/12: `run_eval.py` CLI + scenario configs — Phase 23
- ✓ EXT-01: trajectory export (LaTeX/pgfplots CSVs + metadata) — Phase 24
- ✓ EXT-02: visualisation refactor (`plot_trajectory`, `plot_metrics`, `label_names`) — Phase 25
- ✓ EXT-03: Propulate MPI optimizer backend (auto-select on SLURM/MPI) — Phase 26
- ✓ DF-01/02: `DataFactory` geometric augmentation + `generate_datasets.py` integration — Phases 27–28
- ✓ VIZ-01/02/03: viz unification + alignment/label figure refactors — Phases 29, 36, 37
- ✓ MODE-01/02/03: paired + synthetic dual-mode evaluation with GT-aware HPO — Phases 30–31
- ✓ HETERO-01: heterogeneous cross-format paired evaluation — Phase 32
- ✓ ALIGN-01/02/03: CPD-aligned trajectory, pre-transfer guard, stored-transform reuse (8× scale fix) — Phases 33–35
- ✓ CLN-01/02: codebase-wide `color`→`label` rename + loud label-source logic — Phase 38

### Validated in v1.1 (2026-05-13)

- ✓ Python 3.12 migration (type hints, syntax, stdlib updates) — Phase 6
- ✓ CPD module deep restructure (base class, public API) — Phase 7
- ✓ DTW module deep restructure (algorithm/metric separation) — Phase 8
- ✓ Distance & Transform module restructure (consistent interfaces) — Phase 9
- ✓ Silent failure detection and fixes (lazy imports, swallowed exceptions) — Phase 10
- ✓ All-refactoring validated with 391 passing tests — Phase 11
- ✓ DTW-02: DistanceMetric Protocol public; callable pass-through in sanitize — Phase 11.1

### Future (v1.2+)

- [ ] Property-based testing with Hypothesis for transform and distance metric invariants (QOL-01)
- [ ] Performance regression tests with pytest-benchmark (QOL-02)
- [ ] py.typed marker for mypy/pyright downstream support (QOL-03)
- [ ] Structured result objects for all registration return values (QOL-04)

### Out of Scope

- Alignment quality metrics (TRE, chamfer distance) — new feature, not consolidation
- Uncertainty quantification for transformations — new feature, not consolidation
- Color transfer occlusion handling — new feature, not consolidation
- Full MPI refactoring (hierarchical communication) — beyond quick wins
- Sparse matrix storage for large distance matrices — beyond quick wins
- Replacing Open3D/torch_cluster with pure PyTorch — major architectural change
- Batch/vectorized pairwise distance computation — beyond quick wins

## Context

- Brownfield Python library for scientific computing (3D point cloud analysis)
- Built on PyTorch with Open3D interoperability
- Supports optional MPI for distributed computation
- Python 3.12+ (upgraded from 3.9+ in v1.1)
- Shipped v1.0 with ~10,270 net insertions across 48 files; v1.1 added +11,953 / −1,890 lines across 85 files
- ~6,195 LOC Python in src/; 391 passing regression tests at 94% coverage after v1.1
- CPD, DTW, distances, transforms all restructured as packages with explicit `__all__` exports
- Known tech debt: `typing.Callable` retained in cpd/ (valid 3.12, minor inconsistency); `DistanceMetric` Protocol documentary only; `color_transfer.py` uses absolute intra-package import; `config` not top-level; no VALIDATION.md for any v1.1 phase

## Constraints

- **Backwards compatibility**: Public API signatures must not break without deprecation
- **Dependencies**: Keep Open3D and torch_cluster as-is; improve error handling around them
- **Performance scope**: Quick wins only (caching, removing redundancy)
- **Python version**: 3.12+ (upgrading from 3.9+)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix MaxSWD rather than remove | User wants full distance metric coverage | ✓ Good — in-place projection update resolved gradient flow |
| Quick-win performance only | Avoid scope creep into major refactoring | ✓ Good — RBF caching was sufficient improvement |
| Keep Open3D/torch_cluster | Improve error handling, not replace dependencies | ✓ Good — error handling improved without breakage |
| Skip new features | Focus on fixing what exists, not adding capabilities | ✓ Good — 23 requirements satisfied cleanly |
| All tests deferred to Phase 5 | Tests verify actual fixes, not pre-fix behavior | ✓ Good — clean TDD cycle per phase then full regression |
| Phases 2 and 4 ran in parallel | Both depend only on Phase 1 | ✓ Good — no conflicts, reduced elapsed time |
| sigma2 clamped to dtype.eps | Avoids log(0) without arbitrary floor | ✓ Good — torch.finfo(dtype).eps is dtype-adaptive |
| w-clamped to torch.finfo(dtype).eps | Prevents inf/NaN in homogeneous division | ✓ Good — mirrors sigma2 clamping pattern |
| ValueError (not warning) in __mul__ | Silent corruption worse than loud failure | ✓ Good — user sees clear det/cond values in error |
| pmat strict validation (no auto-transpose) | 'looks transposed' hint is safer than silent correction | ✓ Good — preserves data integrity |
| MPI allgather+sum (not allreduce) | Each rank zeros non-computed entries | ✓ Good — correct merge with minimal message size |
| `eval/` as repo-root namespace dir (no top-level `__init__.py`) | Keep eval framework separate from installed `zreg` package; test discovery via conftest sys.path | ✓ Good — clean separation, stages/runners are regular sub-packages |
| Frozen pydantic v2 result types | Immutability for reproducible reports; mutate via `model_copy(update=...)` | ✓ Good — but shallow-frozen hazard documented (nested dicts mutable) |
| Two-input stage signatures `run(source, target, params)` | DTW must never self-align (`x=y`) | ✓ Good — enabled paired + heterogeneous modes |
| Stored CPD transform reuse (normalise→apply→denormalise) | Re-running CPD from identity fails at 8× scale difference | ✓ Good — fixed Shah/Kobitski convergence (ALIGN-03) |
| `color`→`label` rename with loud failure (no `id` fallback) | Silent fallback masked missing labels | ✓ Good — errors are now informative (CLN-02) |
| Propulate as optional lazy-imported MPI extra | Heavy MPI dep; Optuna is the default backend | ⚠️ Revisit — live mpirun path unverified (missing GPy in dev env) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-30 after Phase 42 — Sobol quasi-random search added as default HPO strategy (`search_strategy="sobol"`); SobolSearch class with SOBOL_MIN_TRIALS=8 fallback to RandomSearch; sobol_seed/sobol_randomize config fields; OPT-04 complete; 1132 tests pass*
