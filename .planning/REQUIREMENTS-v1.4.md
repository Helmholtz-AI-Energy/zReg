# v1.4 Requirements: Trajectory Alignment & Optimization Enhancements

**Milestone**: Trajectory Alignment & Optimization Enhancements (2026-06-29)

**Goal**: Expand alignment and optimization pipeline with 5 new methods (ICP, Sliced Wasserstein variants, preprocessing, Sobol search, data standardization) — all config-driven, backward compatible.

---

## Functional Requirements

### ALIGN-04: ICP Registration as CPD Alternative

| Requirement | Details |
|---|---|
| **ALIGN-04-01** | Wrap Open3D ICP (point-to-point) as standalone registration method |
| **ALIGN-04-02** | Implement `StoredTransform` pattern for ICP results (registration matrix + denorm context) |
| **ALIGN-04-03** | Integrate into `AlignmentStage` dispatcher: `alignment_method: icp` in YAML |
| **ALIGN-04-04** | Unit tests: ICP on synthetic + real data; convergence on known transforms |
| **ALIGN-04-05** | Integration tests: ICP in full paired evaluation pipeline |

### ALIGN-05: Sliced Wasserstein Distance Variants as Alignment Method

| Requirement | Details |
|---|---|
| **ALIGN-05-01** | Expose zReg SWD variants (SWD, ASWD, OSWD, GSWD, PSWD, MaxSWD) as alignment distance metrics |
| **ALIGN-05-02** | Implement SWD-based alignment wrapper (`SlicedWassersteinAligner`) using existing distance classes |
| **ALIGN-05-03** | Support distance variant selection: `alignment_method: swd` + `swd_variant: aswd` in YAML |
| **ALIGN-05-04** | Unit tests: SWD alignment convergence on synthetic data |
| **ALIGN-05-05** | Integration tests: SWD alignment in paired evaluation (comparison vs CPD quality) |

### ALIGN-06: Alignment Preprocessing (Principal Axes + Velocity Landmarks)

| Requirement | Details |
|---|---|
| **ALIGN-06-01** | Principal axes alignment: rotate clouds to align principal axes before CPD/ICP |
| **ALIGN-06-02** | Velocity landmark detection: identify high-motion frames and lock them during alignment |
| **ALIGN-06-03** | Config-driven preprocessing: `alignment_preprocessing: {method: principal_axes, velocity_threshold: 0.5}` |
| **ALIGN-06-04** | Unit tests: PCA alignment correctness, landmark detection sensitivity analysis |
| **ALIGN-06-05** | Integration tests: preprocessing + alignment impact on quality metrics |

### OPT-04: Sobol Quasi-Random Search as Default

| Requirement | Details |
|---|---|
| **OPT-04-01** | Integrate Sobol sequence sampler (via scipy.stats) into `HyperparamOptimizer` |
| **OPT-04-02** | Sobol as default: `search_strategy` defaults to `sobol` (not `grid`) |
| **OPT-04-03** | Grid search as fallback: `search_strategy: grid` to opt out |
| **OPT-04-04** | Config: `sobol_seed`, `sobol_randomize` for reproducibility |
| **OPT-04-05** | Unit tests: Sobol sequence quality (determinism, coverage), convergence vs grid |
| **OPT-04-06** | Backward compat: existing configs without `search_strategy` default to Sobol; explicit `grid` flag works |

### DATA-02: Per-Trajectory Data Standardization as Default

| Requirement | Details |
|---|---|
| **DATA-02-01** | Standardization (z-score) as default data preprocessing |
| **DATA-02-02** | Optional normalization (min-max, robust scaler) support via config |
| **DATA-02-03** | Scope: per-trajectory (not global or per-dataset) |
| **DATA-02-04** | Config: `data_preprocessing: {method: standardize, robust_outlier_threshold: 3}` |
| **DATA-02-05** | Integration into `DataFactory.load_*` methods |
| **DATA-02-06** | Unit tests: standardization correctness, numeric stability |
| **DATA-02-07** | Integration tests: impact on alignment quality metrics |

---

## Design Constraints

| Constraint | Rationale |
|---|---|
| **No Breaking Changes** | `AlignmentStage.run(source, target, params)` signature unchanged; new methods added via dispatcher |
| **Config-Driven** | All method selection via YAML; no code changes for method swapping |
| **StoredTransform Threading** | ICP + SWD alignment must follow existing pattern (normalise→compute→denormalise→cache) |
| **Backward Compatibility** | Existing CPD + grid search workflows must work unchanged; Sobol default requires fallback to grid |
| **80%+ Coverage** | All new code paths unit-tested + integration-tested |
| **<5% Perf Regression** | Benchmark before/after on existing pipelines (CPD + grid) |

---

## Non-Functional Requirements

| Requirement | Details |
|---|---|
| **Performance** | Sobol search <5% slower than grid on typical hyperparameter grids |
| **Compatibility** | Existing `EvalConfig` YAML must load unchanged (defaults apply) |
| **Documentation** | Config examples for each method; integration guide in README |
| **Test Coverage** | 35+ new tests (unit + integration); maintain 98%+ coverage on existing code |
| **Execution Time** | Phase parallelization: all 5 phases independent, can run in parallel |

---

## Acceptance Criteria

- [ ] All ALIGN-04/05/06, OPT-04, DATA-02 requirements implemented
- [ ] 976+ tests passing (existing + new)
- [ ] 80%+ coverage on new code paths
- [ ] No performance regression on existing CPD + grid pipelines (<5% tolerance)
- [ ] Config examples (YAML) for each new method
- [ ] Backward compatibility verified: existing configs load and run unchanged
- [ ] Integration tests pass: new methods in full paired evaluation pipeline

---

## Timeline & Phases

**Phases 39-43 (parallel execution):**
- Phase 39: ALIGN-04 (ICP integration)
- Phase 40: ALIGN-05 (SWD variants)
- Phase 41: ALIGN-06 (Preprocessing)
- Phase 42: OPT-04 (Sobol search)
- Phase 43: DATA-02 (Standardization)

**Estimated Duration**: 4-6 weeks (parallel phases)

---

## Out of Scope (v1.5+)

- Advanced ICP variants (point-to-plane, colored ICP)
- Custom Sobol seed control per-user
- GPU-accelerated Sobol sampling
- Per-dataset or global standardization scopes
- Uncertainty quantification for preprocessing choices
