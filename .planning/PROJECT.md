# zReg

## Current State

**Shipped:** v1.0 Consolidation — 2026-04-09 | v1.1 Code Quality & Refactoring — 2026-05-13 | v1.2 Evaluation Framework & Debt Resolution — 2026-06-26 | **v1.4 Trajectory Alignment & Optimization Enhancements — 2026-07-08**

zReg is a Python library for GPU-accelerated 3D point cloud registration, temporal alignment, and label (celltype) transfer using PyTorch — paired with a complete, config-driven **evaluation framework** at the repo root (`eval/`). v1.4 (5 phases, 11 plans, 2026-06-29 → 2026-07-08) expanded the alignment and optimization pipeline: ICP (Open3D point-to-point) and SWD variants (SWD/ASWD/OSWD/GSWD/PSWD) as registration alternatives; preprocessing via `compute_pca_rotation` + `detect_velocity_landmarks`; Sobol quasi-random search as default HPO (`SOBOL_MIN_TRIALS=8` fallback); and per-trajectory z-score standardization as default data preprocessing — all config-driven and backward compatible. **1,156 tests pass, 18 skipped.**

## Current Milestone: v1.5 HoreKa Cluster Execution

**Goal:** Run the `baseline_experiments` evaluation suite on the HoreKa HPC cluster with GPU support, replacing the current laptop-constrained (CPU-only, subsampled) execution.

**Target features:**
- Multi-trial parallelism via the already-built `propulate`/MPI search backend (`eval/runners/optimizer.py`, `eval/search_strategies.py`)
- `baseline_experiments/scripts/run_all.py` made MPI-rank-aware
- HoreKa job script + environment setup + data transfer
- Real per-operation GPU acceleration (`device` field on `EvalConfig`, threaded through `DataFactory`)
- Subsampling stays the default on the cluster (not full point density) — suite is budgeted to a **3-hour GPU time cap**, validated on a short test job before any full allocation

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

v1.5 HoreKa Cluster Execution — requirements being defined, see `.planning/REQUIREMENTS.md`.

### Validated in v1.4 (2026-07-08)

- ✓ ALIGN-04-01…05: ICP registration (Open3D point-to-point) + StoredTransform + AlignmentStage dispatcher — Phase 39
- ✓ ALIGN-05-01…05: SlicedWassersteinAligner (SWD/ASWD/OSWD/GSWD/PSWD) + SO(3) projection + AlignmentStage dispatcher — Phase 40
- ✓ ALIGN-06-01…05: `compute_pca_rotation` + `detect_velocity_landmarks` + `AlignmentPreprocessingConfig` — Phase 41
- ✓ OPT-04-01…06: `SobolSearch` as default HPO + `SOBOL_MIN_TRIALS=8` fallback + `sobol_seed`/`sobol_randomize` config — Phase 42
- ✓ DATA-02-01…07: `DataPreprocessingConfig` + `DataFactory._standardize()` per-trajectory z-score as default — Phase 43

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
*Last updated: 2026-07-31 — Phase 50 complete (torch_cluster FAIL on HoreKa: no wheel for torch 2.13.0+cu130; Open3D fallback unchanged)*
