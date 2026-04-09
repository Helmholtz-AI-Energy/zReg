# Roadmap: zReg Consolidation v1.0

## Overview

This roadmap consolidates 23 requirements across bug fixes, validation, code quality, CPD enhancements, and test coverage into 5 phases. The approach is foundation-first: input validation utilities and quick wins land early so that subsequent bug fixes and enhancements can use them, and all tests come last to verify the fixes they depend on.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Validation Foundation & Quick Wins** - Input validation utilities, Python 3.9 baseline, configurable logging
- [x] **Phase 2: Distance Metric & CPD Bug Fixes** - Fix SWD variants and RigidCPD scale, cache RBF normalization
- [x] **Phase 3: DTW, Transform & CPD Enhancements** - Fix DTW paths, stabilize homogeneous transforms, add CPD diagnostics and transform validation (completed 2026-04-09)
- [x] **Phase 4: Infrastructure & Color Transfer Quality** - Tempfile cleanup, MPI fix, color transfer dispatch and validation, downsampling logging (completed 2026-04-09)
- [ ] **Phase 5: Test Coverage** - Comprehensive tests for all fixed and enhanced subsystems

## Phase Details

### Phase 1: Validation Foundation & Quick Wins
**Goal**: Public API entry points have consistent input validation, and zero-risk modernization is complete
**Depends on**: Nothing (first phase)
**Requirements**: VALID-01, VALID-02, QUALITY-02, QUALITY-03
**Success Criteria** (what must be TRUE):
  1. Passing a NaN or inf tensor to any public API function raises a clear ValueError naming the offending parameter
  2. Passing tensors on different devices to any multi-tensor API function raises a clear error before computation begins
  3. No conditional imports or version checks for Python 3.8 remain in the codebase
  4. Setting `ZREG_LOG_LEVEL=WARNING` suppresses all INFO-level output from zReg
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Create validation module and wire into Tier 1 entry points (VALID-01, VALID-02)
- [x] 01-02-PLAN.md — Remove Python 3.8 compat and add configurable logging (QUALITY-02, QUALITY-03)

### Phase 2: Distance Metric & CPD Bug Fixes
**Goal**: All sliced Wasserstein distance variants and RigidCPD produce correct results
**Depends on**: Phase 1
**Requirements**: BUGFIX-01, BUGFIX-02, BUGFIX-03, QUALITY-01
**Success Criteria** (what must be TRUE):
  1. MaxSlicedWassersteinDistance returns finite, non-zero distances for non-identical point clouds
  2. RigidCPD with `scale=True` converges and the returned scale factor reflects the actual size ratio of the inputs
  3. GeneralisedSlicedWassersteinDistance produces valid distances with default parameters (no manual degree override needed)
  4. RBF kernel computation on repeated calls with the same point clouds does not re-normalize each time
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Fix MaxSWD gradient flow and GSWD default degree parameter (BUGFIX-01, BUGFIX-03)
- [x] 02-02-PLAN.md — Fix RigidCPD scale formula with division-by-zero guard (BUGFIX-02)
- [x] 02-03-PLAN.md — Cache RBF kernel normalization in CPD set_source (QUALITY-01)

### Phase 3: DTW, Transform & CPD Enhancements
**Goal**: DTW path reconstruction is correct at boundaries, transforms are numerically stable, and CPD exposes convergence diagnostics
**Depends on**: Phase 2
**Requirements**: BUGFIX-04, BUGFIX-05, CPD-01, CPD-02, CPD-03
**Success Criteria** (what must be TRUE):
  1. DTW path reconstruction returns valid paths for windowed variants and boundary conditions without index errors
  2. Composing multiple homogeneous transforms does not produce NaN or inf values in the result matrix
  3. CPD registration returns iteration count and sigma2 trajectory alongside the transformation result
  4. CPD sigma2 is clamped to safe bounds and a warning is logged when bounds are hit
  5. Composing transforms triggers automatic validation (determinant, condition number) with clear errors on invalid results
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Fix DTW _backtrace dead-end handling for boundary and windowed variants (BUGFIX-04)
- [x] 03-02-PLAN.md — Add CPD convergence diagnostics and sigma2 safety clamping (CPD-01, CPD-02)
- [x] 03-03-PLAN.md — Stabilize homogeneous transforms and add composition validation (BUGFIX-05, CPD-03)

### Phase 4: Infrastructure & Color Transfer Quality
**Goal**: File handling, MPI communication, color transfer dispatch, and downsampling logging are clean and robust
**Depends on**: Phase 1
**Requirements**: VALID-03, VALID-04, QUALITY-04, QUALITY-05, QUALITY-06
**Success Criteria** (what must be TRUE):
  1. AdaptiveSlicedWassersteinDistance does not leave temporary files after computation completes or raises an exception
  2. MPI distance matrix computation sends only the rows each rank computed, not the full matrix
  3. Color transfer method can be specified by enum or string, and invalid methods raise a clear error listing valid options
  4. Color transfer probability matrix shape mismatches raise errors that name the expected vs actual shapes
  5. Downsampling logs which backend was selected and why (e.g., "Using torch_cluster FPS: N points")
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — ASWD tempfile auto-cleanup, MPI row-only allgather, downsampling backend logging (VALID-03, VALID-04, QUALITY-06)
- [x] 04-02-PLAN.md — Color transfer enum dispatch and pmat strict shape validation (QUALITY-04, QUALITY-05)

### Phase 5: Test Coverage
**Goal**: All fixed and enhanced subsystems have targeted regression tests covering edge cases and error paths
**Depends on**: Phase 3, Phase 4
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05
**Success Criteria** (what must be TRUE):
  1. CPD tests cover extreme scale ratios, near-degenerate point configurations, and verify convergence diagnostics output
  2. DTW tests cover multiple distance metrics, windowed variants, and boundary condition paths
  3. Transform composition tests verify matrix property invariants (determinant, orthogonality) across 3+ chained transforms
  4. Device handling tests confirm correct behavior on CPU and verify clear errors for mixed-device inputs
  5. Color transfer tests cover empty point clouds, single-point clouds, and dimension-mismatched inputs
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — CPD numerical stability, transform composition depth, and device handling tests (TEST-01, TEST-03, TEST-04)
- [ ] 05-02-PLAN.md — DTW metric variants and boundary condition tests (TEST-02)
- [x] 05-03-PLAN.md — Color transfer empty-source guard and edge case tests (TEST-05)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5
(Phases 2 and 4 both depend on Phase 1; Phase 5 depends on both 3 and 4)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Validation Foundation & Quick Wins | 2/2 | Complete | 2026-04-09 |
| 2. Distance Metric & CPD Bug Fixes | 3/3 | Complete | 2026-04-09 |
| 3. DTW, Transform & CPD Enhancements | 3/3 | Complete | 2026-04-09 |
| 4. Infrastructure & Color Transfer Quality | 2/2 | Complete   | 2026-04-09 |
| 5. Test Coverage | 0/3 | Not started | - |

## Coverage Matrix

| Requirement | Phase |
|-------------|-------|
| BUGFIX-01 | Phase 2 |
| BUGFIX-02 | Phase 2 |
| BUGFIX-03 | Phase 2 |
| BUGFIX-04 | Phase 3 |
| BUGFIX-05 | Phase 3 |
| VALID-01 | Phase 1 |
| VALID-02 | Phase 1 |
| VALID-03 | Phase 4 |
| VALID-04 | Phase 4 |
| QUALITY-01 | Phase 2 |
| QUALITY-02 | Phase 1 |
| QUALITY-03 | Phase 1 |
| QUALITY-04 | Phase 4 |
| QUALITY-05 | Phase 4 |
| QUALITY-06 | Phase 4 |
| CPD-01 | Phase 3 |
| CPD-02 | Phase 3 |
| CPD-03 | Phase 3 |
| TEST-01 | Phase 5 |
| TEST-02 | Phase 5 |
| TEST-03 | Phase 5 |
| TEST-04 | Phase 5 |
| TEST-05 | Phase 5 |

**Coverage: 23/23 requirements mapped**
