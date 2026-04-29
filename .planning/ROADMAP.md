# Roadmap: zReg

## Milestones

- **v1.1 Code Quality & Refactoring** — Phases 6-10 (active)
- ✅ **v1.0 Consolidation** — Phases 1-5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

### v1.1 Code Quality & Refactoring (Phases 6-10)

- [ ] **Phase 6: Python 3.12 Migration** - Modernize syntax and type hints for 3.12+
- [ ] **Phase 7: CPD Deep Restructure** - Extract base class and clean module architecture
- [ ] **Phase 8: DTW Deep Restructure** - Separate algorithm from metrics, clean interfaces
- [ ] **Phase 9: Distance & Transform Restructure** - Consistent interfaces and shared patterns
- [ ] **Phase 10: Code Quality & Verification** - Silent failure fixes, consistency, test validation

<details>
<summary>✅ v1.0 Consolidation (Phases 1-5) — SHIPPED 2026-04-09</summary>

- [x] Phase 1: Validation Foundation & Quick Wins (2/2 plans) — completed 2026-04-09
- [x] Phase 2: Distance Metric & CPD Bug Fixes (3/3 plans) — completed 2026-04-09
- [x] Phase 3: DTW, Transform & CPD Enhancements (3/3 plans) — completed 2026-04-09
- [x] Phase 4: Infrastructure & Color Transfer Quality (2/2 plans) — completed 2026-04-09
- [x] Phase 5: Test Coverage (3/3 plans) — completed 2026-04-09

Full details: [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

</details>

## Phase Details

### Phase 6: Python 3.12 Migration
**Goal**: Codebase uses modern Python 3.12+ syntax and type hints throughout
**Depends on**: Nothing (can begin immediately)
**Requirements**: PY312-01, PY312-02, PY312-03, PY312-04, PY312-05
**Success Criteria** (what must be TRUE):
  1. All Union[X, Y] type hints replaced with X | Y syntax
  2. All self-returning methods use Self type annotation
  3. pyproject.toml specifies python_requires >= 3.12
  4. No deprecated stdlib patterns remain (typing module generics replaced with built-ins)
**Plans**: 2 plans
- [x] 06-01-PLAN.md — Version constraints + Union/Optional to PEP 604 syntax
- [x] 06-02-PLAN.md — Built-in generics + Self type + typing import cleanup

### Phase 7: CPD Deep Restructure
**Goal**: CPD module has clean inheritance hierarchy with shared base and explicit public API
**Depends on**: Phase 6 (modern syntax available for new code)
**Requirements**: CPD-01, CPD-02, CPD-03, CPD-04, CPD-05
**Success Criteria** (what must be TRUE):
  1. Base CPD class exists with M-step/E-step interface that variants override
  2. Rigid, Affine, and NonRigid variants inherit from base without code duplication
  3. RBF kernel computation lives in dedicated utility function/class
  4. Convergence diagnostics are reusable across registration types
  5. Module __all__ exports only public API; internal helpers are underscore-prefixed
**Plans**: 3 plans
- [x] 07-01-PLAN.md — Package foundation: types, kernels, abstract base class
- [x] 07-02-PLAN.md — Registration variants: Rigid, Affine, NonRigid, ConstrainedNonRigid
- [x] 07-03-PLAN.md — Registration functions and public API finalization

### Phase 8: DTW Deep Restructure
**Goal**: DTW module cleanly separates algorithm core from metric computation with composable components
**Depends on**: Phase 6 (modern syntax available for new code)
**Requirements**: DTW-01, DTW-03, DTW-04, DTW-05 (DTW-02 deferred to Phase 9)
**Success Criteria** (what must be TRUE):
  1. DTW algorithm core is metric-agnostic (delegates to create_pairwise_distance_matrix)
  2. Path reconstruction is focused private method (_backtrace) with single responsibility
  3. Windowing constraints are composable (compose_constraints + set_cost_matrix pattern)
  4. Module __all__ exports only public API; internal helpers are underscore-prefixed
**Plans**: 2 plans
- [x] 08-01-PLAN.md — Package foundation: DTWResult dataclass, compose_constraints utility
- [x] 08-02-PLAN.md — Core class migration and public API finalization

### Phase 9: Distance & Transform Restructure
**Goal**: Distance metrics and transforms have consistent interfaces with shared abstraction patterns
**Depends on**: Phase 6 (modern syntax available for new code)
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04, XFORM-01, XFORM-02, XFORM-03, XFORM-04
**Success Criteria** (what must be TRUE):
  1. Base Wasserstein class provides clean interface for all SWD variants
  2. SWD variants share common projection/slicing implementation
  3. Pairwise distance computation uses clear abstraction (metric, data) -> matrix
  4. Transform composition is well-abstracted with clear pre/post conditions
  5. Homogeneous coordinate handling is consolidated in one location
  6. TPS and RBF transforms share base patterns where appropriate
  7. Both distance and transform modules have explicit __all__ exports
**Plans**: 2 plans
- [x] 09-01-PLAN.md — Distance module API cleanup and documentation
- [x] 09-02-PLAN.md — Transform module API and documentation

### Phase 10: Code Quality & Verification
**Goal**: Codebase has no silent failures, consistent patterns, and all tests pass
**Depends on**: Phases 7, 8, 9 (all restructuring complete)
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-06
**Success Criteria** (what must be TRUE):
  1. All lazy imports are explicit or properly guarded with clear error messages
  2. No swallowed exceptions remain; all error paths log or raise informatively
  3. Logging follows consistent pattern across all modules (same format, levels)
  4. Identified code duplications are extracted to shared utilities
  5. Naming conventions are consistent (same style for similar constructs)
  6. All 275+ existing tests pass after restructuring (no regressions)
**Plans**: 3 plans
- [ ] 10-01-PLAN.md — Silent failure fixes (minkowski_distance, normalize_point_cloud, CPD source_colors)
- [ ] 10-02-PLAN.md — Logging consistency (print->log.debug, configure_pytorch function)
- [ ] 10-03-PLAN.md — Code cleanup (mutable defaults, TODO markers, TODO inventory, test verification)

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 6. Python 3.12 Migration | v1.1 | 2/2 | Complete | - |
| 7. CPD Deep Restructure | v1.1 | 3/3 | Complete | - |
| 8. DTW Deep Restructure | v1.1 | 2/2 | Complete   | 2026-04-23 |
| 9. Distance & Transform Restructure | v1.1 | 2/2 | Complete | 2026-04-27 |
| 10. Code Quality & Verification | v1.1 | 0/3 | Not started | - |
| 1. Validation Foundation & Quick Wins | v1.0 | 2/2 | Complete | 2026-04-09 |
| 2. Distance Metric & CPD Bug Fixes | v1.0 | 3/3 | Complete | 2026-04-09 |
| 3. DTW, Transform & CPD Enhancements | v1.0 | 3/3 | Complete | 2026-04-09 |
| 4. Infrastructure & Color Transfer Quality | v1.0 | 2/2 | Complete | 2026-04-09 |
| 5. Test Coverage | v1.0 | 3/3 | Complete | 2026-04-09 |
