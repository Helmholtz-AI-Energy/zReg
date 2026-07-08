# Milestones

## v1.0 Consolidation (Shipped: 2026-04-09)

**Phases completed:** 5 phases, 13 plans, 18 tasks

**Key accomplishments:**

- _validate_tensors() with NaN/inf and device-mismatch detection wired into 6 entry points (CPD, BaseWD, minkowski_distance, pairwise_distance_matrix) with 13 tests covering unit and integration scenarios
- ZREG_LOG_LEVEL env var and zreg.set_log_level() added; Python 3.8 conditional importlib block and dead importlib-metadata dependency removed
- Fixed MaxSWD gradient flow via in-place projection update and GSWD default degree propagation through pairwise_distance_matrix
- RigidCPD scale computation guarded against division by zero with torch.clamp, formula verified against Myronenko & Song 2010, and 3 regression tests added
- Removed redundant normalize_point_cloud from rbf_kernel; normalization now cached once per set_source in NonRigidCPD/ConstrainedNonRigidCPD
- One-liner:
- Extended MstepResult with 5-field namedtuple (n_iters, sigma2_history) and added torch.clamp sigma2 safety after each M-step with one-time WARNING log
- w-coordinate clamping with torch.finfo eps in transform_points_homogeneous, plus det/cond post-composition validation raising ValueError in RigidTransformation.__mul__
- Task 1 — ASWD tempfile auto-cleanup (`sw_varients.py`):
- One-liner:
- CPD numerical stability tests for extreme/degenerate inputs, 3-chain transform composition invariant tests, and CPU/GPU device handling tests
- DTW regression tests covering manhattan/cpd/minkowski metrics, windowed asymmetric trajectories, and single-timepoint boundary paths
- Empty-source guard in transfer_colors plus 8 edge case tests covering empty, single-point, and dimension-mismatched inputs

---

## v1.1 Code Quality & Refactoring (Shipped: 2026-05-13)

**Phases completed:** 7 phases (6–11.1, including 1 inserted), 14 plans
**Timeline:** 2026-04-13 → 2026-05-13 (30 days)
**Stats:** 70 commits, 85 files changed, +11,953 / −1,890 lines, ~6,195 LOC Python

**Key accomplishments:**

- Full Python 3.12 migration — PEP 604/673/585 applied across all modules; typing imports minimized; pyproject.toml updated to python_requires >= 3.12
- CPD deep restructure — abstract `BaseCPD` with M/E-step interface; Rigid, Affine, NonRigid, ConstrainedNonRigid variants; RBF kernel utility; 10-export public API
- DTW deep restructure — `DynamicTimeWarping`, `DTWResult`, `compose_constraints` as structured package; `_backtrace` private; variadic AND-semantics constraint composition
- Distance & Transform restructure — explicit `__all__` in both modules; consistent interfaces; comprehensive numpy-style docstrings; `DistanceMetric` Protocol; `TransformBase` exported
- Silent failure fixes + logging consistency — all `print` → `log.debug`; lazy import guards; swallowed exceptions surfaced; mutable defaults fixed in CPD package
- 391 passing tests at 94% coverage — dead code removed from downsampling; FPS/return_o3d paths fully exercised; zero regressions from any refactoring
- DTW-02 closed (Phase 11.1) — `DistanceMetric` Protocol promoted to public API from `zreg.distances`; callable pass-through added to `_sanitize_pairwise_distance_matrix`

**Requirements:** 29/29 satisfied (100%)

---

## v1.4 Trajectory Alignment & Optimization Enhancements (Shipped: 2026-07-08)

**Phases completed:** 5 phases (39–43), 11 plans
**Timeline:** 2026-06-29 → 2026-06-30 (2 days execution); gaps closed 2026-07-08
**Stats:** 69 commits since v1.2 tag; ~13,665 LOC Python (src/ + eval/)
**Tests:** 1,156 passed, 18 skipped (all pre-existing failures resolved at close)

**Key accomplishments:**

- `ICPRegistration` wrapping Open3D point-to-point ICP integrated into `AlignmentStage` dispatcher as `alignment_method: icp`; `StoredTransform` normalise→register→denormalise pattern; backward-compatible default (CPD unchanged)
- `SlicedWassersteinAligner` with 5 SWD variants (SWD/ASWD/OSWD/GSWD/PSWD) via gradient descent + SO(3) SVD projection; `alignment_method: swd` + `swd_variant` YAML selection; MaxSWD deferred to v1.5
- `src/zreg/preprocessing.py` with `compute_pca_rotation()` (det=+1 guarantee) + `detect_velocity_landmarks()`; `AlignmentPreprocessingConfig` pydantic model; backward-compatible `EvalConfig.alignment_preprocessing` field
- `SobolSearch` via `scipy.stats.qmc.Sobol` as new default HPO (`search_strategy` defaults to `"sobol"`); `SOBOL_MIN_TRIALS=8` fallback to `RandomSearch`; `sobol_seed`/`sobol_randomize` config fields
- `DataPreprocessingConfig` pydantic model; `DataFactory._standardize()` per-trajectory z-score; `EvalConfig.data_preprocessing` defaults to live instance (standardization active by default)

**Requirements:** 28/28 satisfied

**Gaps closed at milestone close (2026-07-08):**
- IN-01 (43): dtype inconsistency in stats dict — all_pos cast to float32 before stats block
- IN-04 (43): strengthened `test_load_target_without_prior_load_real_computes_own_stats` with allclose check
- OSWD num_projs constraint: fixed in `test_swd_aligner.py` and `test_distances.py` (num_projs=3 for 3D data)
- Cross-embryo YAML: `kobitski_vs_kobitski_cross.yaml` target_data_path corrected to ew_08

---

## v1.2 Evaluation Framework & Debt Resolution (Shipped: 2026-06-26)

**Phases completed:** 27 phases (12–38), 55 plans
**Timeline:** 2026-05-13 → 2026-06-26 (~44 days)
**Stats:** 353 commits, 82 files changed (src/eval/configs/scripts/tests), +16,570 / −302 lines
**Tests:** 976 passed, 18 skipped (994 collected)
**Git range:** v1.1 (74cf5c0) → v1.2 (tagged)

**Key accomplishments:**

- Complete config-driven evaluation framework at repo-root `eval/` — `EvalConfig` + `DataFactory`, `MetricsEngine` + 6 frozen pydantic result types, isolated `AlignmentStage`/`LabelTransferStage`, `EvaluationRunner` + `viz.py` — all wrapping existing `zreg.metrics`/`zreg.generators` (Phases 13–21)
- 3-tier `HyperparamOptimizer` (sanity/dev/full) with Optuna TPE+SQLite and MPI-parallel Propulate backends, Grid/Random strategies, `run_eval.py` CLI + scenario configs (Phases 22, 23, 26)
- Dual-mode evaluation — paired source↔target alignment and synthetic transform-spec target generation with GT-aware HPO, plus heterogeneous cross-format (tracklets/CSV) paired evaluation (Phases 30, 31, 32)
- CPD-aligned trajectory output with stored-transform reuse (normalise→apply→denormalise) fixing the 8× scale convergence failure, plus pre-transfer alignment quality guard (Phases 33, 34, 35)
- Trajectory export (point-per-row CSVs + metadata for LaTeX/pgfplots), per-frame visualisation refactor (alignment + label branches → independent 1×3 figures), and viz unification (Phases 24, 25, 29, 36, 37)
- Carry-forward debt closure (CARRY-01–05) + codebase-wide `color`→`label` field rename with loud label-source logic + core metrics/generators/tracking/runner-script foundation (Phases 12, 13, 14, 15, 16, 38)

**Requirements:** 30/30 satisfied (27 categorised + EXT-01/02/03 delivered & verified)

**Known deferred items at close:** Phase 26 SC-4 (live `mpirun -n 2` Propulate integration) unverified — `propulate` fails to import in dev env (missing GPy transitive dep); Optuna path unaffected. 14 phases without VALIDATION.md (Nyquist backfill optional). See `.planning/v1.2-MILESTONE-AUDIT.md`.

---
