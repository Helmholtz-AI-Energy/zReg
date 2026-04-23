# Requirements: zReg

**Defined:** 2026-04-12
**Core Value:** Every existing capability works correctly, fails informatively, and is covered by tests.

## v1.1 Requirements

Requirements for v1.1 Code Quality & Refactoring. Each maps to roadmap phases.

### Python 3.12 Migration

- [ ] **PY312-01**: All type hints use PEP 604 union syntax (X | Y instead of Union[X, Y])
- [ ] **PY312-02**: All self-returning methods use Self type (PEP 673)
- [ ] **PY312-03**: pyproject.toml updated with python_requires >= 3.12
- [ ] **PY312-04**: Deprecated stdlib patterns replaced with 3.12+ equivalents
- [ ] **PY312-05**: All typing imports modernized (use built-in generics where available)

### CPD Restructure

- [ ] **CPD-01**: Base CPD class extracted with shared M-step/E-step interface
- [ ] **CPD-02**: Registration variants (Rigid, Affine, NonRigid) inherit cleanly from base
- [ ] **CPD-03**: RBF kernel computation extracted into focused utility
- [ ] **CPD-04**: Convergence diagnostics factored into reusable component
- [ ] **CPD-05**: CPD module has clear public API with internal helpers marked private

### DTW Restructure

- [ ] **DTW-01**: DTW core algorithm separated from metric computation
- [ ] **DTW-02**: Metric variants (manhattan, cpd, minkowski, etc.) use consistent interface
- [ ] **DTW-03**: Path reconstruction logic extracted into focused function
- [x] **DTW-04**: Windowing/constraint logic factored into composable components
- [x] **DTW-05**: DTW module has clear public API with internal helpers marked private

### Distance Metrics Restructure

- [ ] **DIST-01**: Base Wasserstein distance class cleaned up with consistent interface
- [ ] **DIST-02**: SWD variants share common projection/slicing logic
- [ ] **DIST-03**: Pairwise distance matrix computation uses clear abstraction layers
- [ ] **DIST-04**: Distance module public API is explicit and documented

### Transforms Restructure

- [ ] **XFORM-01**: Transform composition logic cleaned up and well-abstracted
- [ ] **XFORM-02**: Homogeneous coordinate handling consolidated
- [ ] **XFORM-03**: TPS and RBF transforms share appropriate base patterns
- [ ] **XFORM-04**: Transform module public API is explicit and documented

### Code Quality

- [ ] **QUAL-01**: All lazy imports audited and made explicit or properly guarded
- [ ] **QUAL-02**: All swallowed exceptions replaced with informative error handling
- [ ] **QUAL-03**: Consistent logging patterns applied across all modules
- [ ] **QUAL-04**: Code duplication identified and extracted into shared utilities
- [ ] **QUAL-05**: Naming conventions consistent across codebase
- [ ] **QUAL-06**: All existing tests pass after restructuring

## Future Requirements (v1.2+)

Deferred to future release. Tracked but not in current roadmap.

### Quality of Life

- **QOL-01**: Property-based testing with Hypothesis for transform and distance metric invariants
- **QOL-02**: Performance regression tests with pytest-benchmark
- **QOL-03**: py.typed marker for mypy/pyright downstream support
- **QOL-04**: Structured result objects for all registration return values

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New algorithms or distance metrics | v1.1 is refactoring, not new features |
| MPI architecture overhaul | Deferred to dedicated milestone |
| Sparse matrix storage | Deferred to dedicated milestone |
| Batch/vectorized computation | Deferred to dedicated milestone |
| Breaking API changes | Must maintain backwards compatibility |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PY312-01 | Phase 6 | Pending |
| PY312-02 | Phase 6 | Pending |
| PY312-03 | Phase 6 | Pending |
| PY312-04 | Phase 6 | Pending |
| PY312-05 | Phase 6 | Pending |
| CPD-01 | Phase 7 | Pending |
| CPD-02 | Phase 7 | Pending |
| CPD-03 | Phase 7 | Pending |
| CPD-04 | Phase 7 | Pending |
| CPD-05 | Phase 7 | Pending |
| DTW-01 | Phase 8 | Pending |
| DTW-02 | Phase 8 | Pending |
| DTW-03 | Phase 8 | Pending |
| DTW-04 | Phase 8 | Complete |
| DTW-05 | Phase 8 | Complete |
| DIST-01 | Phase 9 | Pending |
| DIST-02 | Phase 9 | Pending |
| DIST-03 | Phase 9 | Pending |
| DIST-04 | Phase 9 | Pending |
| XFORM-01 | Phase 9 | Pending |
| XFORM-02 | Phase 9 | Pending |
| XFORM-03 | Phase 9 | Pending |
| XFORM-04 | Phase 9 | Pending |
| QUAL-01 | Phase 10 | Pending |
| QUAL-02 | Phase 10 | Pending |
| QUAL-03 | Phase 10 | Pending |
| QUAL-04 | Phase 10 | Pending |
| QUAL-05 | Phase 10 | Pending |
| QUAL-06 | Phase 10 | Pending |

**Coverage:**
- v1.1 requirements: 29 total
- Mapped to phases: 29
- Unmapped: 0

---
*Requirements defined: 2026-04-12*
*Last updated: 2026-04-13 after roadmap creation*
