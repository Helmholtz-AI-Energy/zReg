# Requirements: zReg v1.5 HoreKa Cluster Execution

**Defined:** 2026-07-15
**Core Value:** Every existing capability works correctly, fails informatively, and is covered by tests.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Environment & Access

- [ ] **ENV-01**: Operator can activate a working Python environment on HoreKa with all zReg dependencies plus the `propulate`/`mpi4py` extras installed, with `mpi4py` built against HoreKa's system MPI
- [ ] **ENV-02**: Operator can transfer the real datasets (Kobitski tracklets, Shah CSV — currently gitignored, laptop-only) to HoreKa and have every config's `data_path`/`target_data_path` resolve correctly
- [ ] **ENV-03**: Operator can submit a SLURM job script that requests GPU nodes/ranks and launches the experiment suite end-to-end

### Multi-Trial Parallelism

- [ ] **PARA-01**: HPO trials for a given optimize run execute concurrently across MPI ranks instead of sequentially, via the existing `propulate` search backend
- [ ] **PARA-02**: `run_all.py`'s orchestration (file writes, `EvaluationRunner` calls, per-run bookkeeping) executes exactly once per run regardless of MPI world size, while `HyperparamOptimizer.run()` calls remain collective across all ranks
- [ ] **PARA-03**: Cluster-targeted configs (`search_strategy: propulate`/`auto`) exist independently of the local laptop configs (`search_strategy: sobol`), so both stay runnable without interfering

### GPU Acceleration

- [ ] **GPU-01**: `EvalConfig` exposes a `device` field controlling where tensors are loaded and computed
- [ ] **GPU-02**: `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU
- [ ] **GPU-03**: Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback anywhere in the path

### Budget-Bounded Execution

- [ ] **BUDG-01**: Cluster runs default to the same `max_points_per_frame`/`step` subsampling approach already calibrated on the laptop, not full point density — subsampling is the safe starting point, not a fallback
- [ ] **BUDG-02**: The full 7-run suite is calibrated/projected to complete within a **3-hour GPU time cap**, verified via an `aggregate_cost.py`-style estimation before submitting a full allocation
- [ ] **BUDG-03**: `aggregate_cost.py`'s calibration constants are parameterized per environment (not hardcoded to laptop measurements), so cluster timing data doesn't silently mix with or overwrite laptop calibration
- [ ] **BUDG-04**: Operator validates correct multi-rank behavior (no duplicated/racing output writes) and real per-trial timing on a short test job before committing the full 3-hour allocation

### Output & Verification

- [ ] **OUT-01**: Suite outputs land in the same `baseline_experiments/experiments/<phase>/<name>/` directory convention as local runs, hosted on HoreKa's workspace filesystem

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Full Point Density

- **CALIB-01**: Re-run the `max_points_per_frame`/`step` calibration on a HoreKa GPU node to determine whether full point density (no subsampling at all) becomes tractable within budget — explicitly deferred until BUDG-01/02 are proven at the current subsampled setup first

### Broader Experiment Scope

- **SCOPE-01**: Expand selfcal calibration beyond the single Kobitski embryo (ew06) to 2+ embryos — explicitly declined for v1.5 per user confirmation ("if only one embryo is used for selfcal that works")

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full point density by default | Budget risk — subsampling stays the default until proven safe (BUDG-01); revisit as CALIB-01 in a future milestone |
| Expanding selfcal to more embryos | User explicitly confirmed 1 embryo (ew06) is sufficient for this milestone |
| Domain/HPC research phase | Scope is already concretely grounded in this codebase (existing propulate/MPI support, existing JUWELS job script precedent, confirmed CUDA-capable zreg library) — research would mostly rediscover `baseline_experiments/HOREKA_PLAN.md` |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 44 | Mapped |
| ENV-02 | Phase 44 | Mapped |
| ENV-03 | Phase 44 | Mapped |
| PARA-01 | Phase 45 | Mapped |
| PARA-02 | Phase 45 | Mapped |
| PARA-03 | Phase 45 | Mapped |
| GPU-01 | Phase 46 | Mapped |
| GPU-02 | Phase 46 | Mapped |
| GPU-03 | Phase 46 | Mapped |
| BUDG-01 | Phase 45 | Mapped |
| BUDG-02 | Phase 47 | Mapped |
| BUDG-03 | Phase 47 | Mapped |
| BUDG-04 | Phase 45 | Mapped |
| OUT-01 | Phase 44 | Mapped |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14 (Phases 44–47)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-15*
*Last updated: 2026-07-15 after ROADMAP.md creation (Phases 44-47, 14/14 mapped)*
