# zReg

## Current Milestone: v1.1 Code Quality & Refactoring

**Goal:** Deep refactoring for maintainability, Python 3.12 update, and eliminating silent failures

**Target features:**
- Python 3.12 migration (type hints, syntax, stdlib updates)
- CPD module deep restructure (primary focus)
- DTW module deep restructure (primary focus)
- All-files code quality review
- Silent failure detection (lazy imports, swallowed exceptions)
- Module reorganization and abstraction improvements

## Current State

**Shipped:** v1.0 Consolidation — 2026-04-09 | v1.1 Code Quality & Refactoring — 2026-05-13
**Active:** v1.2 Evaluation Framework & Debt Resolution — Phase 14 complete 2026-05-18

zReg is a Python library for GPU-accelerated 3D point cloud registration, temporal alignment, and color (celltype) transfer using PyTorch. Phase 14 delivered the `eval/generators/` package at repo root: 7 public symbols (`generate_trajectory`, `apply_rigid`, `apply_affine`, `add_gaussian_noise`, `add_outliers`, `generate_labels`, `remove_labels`) — all seed-deterministic, immutable (`copy.deepcopy`), and `dict[int, zRegPointCloud]`-shaped. 514 tests pass, 15 skipped. Phase 15 (Experiment Tracking at `eval/tracking/`) is next.

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

None.

### Validated in v1.2 (in progress)

- ✓ Core alignment metrics (chamfer, hausdorff, path_smoothness, knn_consistency, temporal_stability) + label transfer (compute_f1) at `src/zreg/metrics/` — Phase 13
- ✓ Synthetic data generators at `eval/generators/` (7 symbols, seed-deterministic, immutable, dict[int, zRegPointCloud]-shaped) — Phase 14

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
*Last updated: 2026-05-18 after Phase 14 (Synthetic Data Generators) complete*
