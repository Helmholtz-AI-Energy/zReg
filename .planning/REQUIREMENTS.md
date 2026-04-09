# Requirements: zReg Consolidation

**Defined:** 2026-04-09
**Core Value:** Every existing capability works correctly, fails informatively, and is covered by tests.

## v1 Requirements

Requirements for milestone v1.0.

### Bug Fixes

- [x] **BUGFIX-01**: MaxSlicedWassersteinDistance computes correct distances (currently marked broken)
- [x] **BUGFIX-02**: RigidCPD scale is correctly updated in the maximization step
- [x] **BUGFIX-03**: GeneralisedSlicedWassersteinDistance uses correct default degree parameter
- [x] **BUGFIX-04**: DTW path reconstruction handles edge cases correctly (boundaries, windowed variants)
- [x] **BUGFIX-05**: Homogeneous coordinate transforms are numerically stable

### Validation & Robustness

- [x] **VALID-01**: All public API entry points reject NaN/inf tensor inputs with informative errors
- [x] **VALID-02**: All public API entry points raise clear errors when tensors are on different devices
- [x] **VALID-03**: AdaptiveSlicedWassersteinDistance uses tempfiles with auto-cleanup (no leaked files)
- [x] **VALID-04**: MPI communication sends only computed rows (not full matrix)

### Code Quality

- [x] **QUALITY-01**: RBF kernel normalizes point clouds once and caches the result
- [x] **QUALITY-02**: Python 3.8 conditional imports removed (minimum version is 3.9)
- [x] **QUALITY-03**: Logging verbosity is configurable via `ZREG_LOG_LEVEL` env var or init parameter
- [x] **QUALITY-04**: Color transfer probability matrix shape validation is clean and consistent
- [x] **QUALITY-05**: Color transfer method dispatch uses enum/strategy pattern (strings still accepted)
- [x] **QUALITY-06**: Downsampling backend selection is logged (which backend and why)

### CPD Enhancements

- [x] **CPD-01**: CPD registration loop exposes convergence diagnostics (iteration count, sigma2 trajectory)
- [x] **CPD-02**: CPD registration loop clamps sigma2 to safe bounds with warning when bounds hit
- [x] **CPD-03**: Transform validation runs after composition (determinant, condition number, orthogonality)

### Test Coverage

- [x] **TEST-01**: CPD numerical stability tested (edge cases, extreme scales, near-degenerate configs)
- [x] **TEST-02**: DTW path reconstruction tested (multiple metrics, windowed, boundary conditions)
- [x] **TEST-03**: Transform composition tested (3+ transforms, matrix property invariants)
- [x] **TEST-04**: Device handling tested (GPU/CPU transfers, mixed-device errors)
- [x] **TEST-05**: Color transfer edge cases tested (empty clouds, single point, mismatched dims)

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Quality of Life

- **QOL-01**: Property-based testing with Hypothesis for transform and distance metric properties
- **QOL-02**: Performance regression tests with pytest-benchmark on core operations
- **QOL-03**: py.typed marker for mypy/pyright downstream support
- **QOL-04**: Structured result objects for all registration return values

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Alignment quality metrics (TRE, chamfer distance) | New feature, not consolidation |
| Uncertainty quantification for transformations | New feature, not consolidation |
| Color transfer occlusion handling | New feature, not consolidation |
| Full MPI refactoring (hierarchical communication) | Beyond quick wins — only fixing row communication |
| Sparse matrix storage for large distance matrices | Beyond quick wins |
| Replacing Open3D/torch_cluster with pure PyTorch | Major architectural change, different behavior profile |
| Batch/vectorized pairwise distance computation | Beyond quick wins, architectural changes required |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUGFIX-01 | Phase 2 | Complete |
| BUGFIX-02 | Phase 2 | Complete |
| BUGFIX-03 | Phase 2 | Complete |
| BUGFIX-04 | Phase 3 | Complete |
| BUGFIX-05 | Phase 3 | Complete |
| VALID-01 | Phase 1 | Complete |
| VALID-02 | Phase 1 | Complete |
| VALID-03 | Phase 4 | Complete |
| VALID-04 | Phase 4 | Complete |
| QUALITY-01 | Phase 2 | Complete |
| QUALITY-02 | Phase 1 | Complete |
| QUALITY-03 | Phase 1 | Complete |
| QUALITY-04 | Phase 4 | Complete |
| QUALITY-05 | Phase 4 | Complete |
| QUALITY-06 | Phase 4 | Complete |
| CPD-01 | Phase 3 | Complete |
| CPD-02 | Phase 3 | Complete |
| CPD-03 | Phase 3 | Complete |
| TEST-01 | Phase 5 | Complete |
| TEST-02 | Phase 5 | Complete |
| TEST-03 | Phase 5 | Complete |
| TEST-04 | Phase 5 | Complete |
| TEST-05 | Phase 5 | Complete |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after roadmap creation*
