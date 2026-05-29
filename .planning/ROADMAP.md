# Roadmap: zReg

## Milestones

- 🚧 **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–23 (in progress)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1-5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

### v1.2 Evaluation Framework & Debt Resolution (in progress)

- [x] Phase 12: Carry-Forward Debt Closure (3/3 plans) — completed 2026-05-14
- [x] Phase 13: Core Metrics Library (3/3 plans) — completed 2026-05-15
- [x] Phase 14: Synthetic Data Generators (3/3 plans) — completed 2026-05-18
  **Goal:** Deliver the `src/zreg/generators/` package with a from-scratch trajectory factory and immutable corruption wrappers (rigid/affine transforms, Gaussian noise, outlier injection, Voronoi label generation, label removal) — all seed-deterministic, immutable, and `dict[int, zRegPointCloud]`-shaped per EVAL-03.
  **Requirements:** EVAL-03
  **Plans:** 3 plans
  Plans:
  - [x] 14-01-PLAN.md — Package skeleton + `generate_trajectory` factory + rigid/affine transform wrappers
  - [x] 14-02-PLAN.md — Corruption wrappers (Gaussian noise, outliers) + label utilities (Voronoi generate, remove) + extend `__init__.py`
  - [x] 14-03-PLAN.md — `tests/conftest.py` sys.path extension + `tests/test_generators.py` (4 test classes covering all 7 public symbols)

- [x] Phase 15: Experiment Tracking & Run Management (2/2 plans) — completed 2026-05-18
  **Goal:** Deliver the `eval/tracking/` package at the repo root with a single `log_run()` function that writes local JSON + CSV per run using stdlib only — capturing all 9 EVAL-04 required fields (6 caller-supplied + 3 auto-captured: git_hash, zreg_version, timestamp) — plus unit tests in `tests/test_tracking.py` with all auto-captured fields mocked.
  **Requirements:** EVAL-04
  **Depends on:** Phase 14
  Plans:
  - [x] 15-01-PLAN.md — `eval/tracking/__init__.py` + `eval/tracking/tracking.py` (`log_run()` implementation, stdlib-only)
  - [x] 15-02-PLAN.md — `tests/test_tracking.py` (TestLogRun class: 13 tests covering all 9 fields, fallbacks, file output)

- [x] Phase 16: Runner Scripts (2/2 plans) — completed 2026-05-19
  **Goal:** Deliver `eval/run_synthetic.py` and `eval/run_real.py` as standalone (non-importable) scripts at the repo root that run noise/corruption sweeps and scale/density sweeps respectively — accepting `dict[int, zRegPointCloud]` inputs, importing metrics from `zreg.metrics` (installed package), and writing outputs to `evaluation/runs/` via `log_run()`.
  **Requirements:** EVAL-05
  **Depends on:** Phase 15
  Plans:
  - [x] 16-01-PLAN.md — `eval/run_synthetic.py` (noise + outlier sweep) + `eval/run_real.py` (scale/density sweep with missing-data guard)
  - [x] 16-02-PLAN.md — `tests/test_runners.py` (TestRunSynthetic + TestRunReal: file creation, field presence, missing-data skip, import path verification)

- [x] **Phase 17: Framework Config & DataFactory** (2/2 plans) — completed 2026-05-27
  **Goal:** Deliver `eval/config.py` (`EvalConfig` dataclass with YAML loading/validation and `EvalConfigError`) and `eval/data_factory.py` (`DataFactory` wrapping existing `load_data_from_tracklets`, `load_shah_from_csv`, generators, and corruption functions) — the data foundation all downstream phases depend on.
  **Requirements:** FRAME-01, FRAME-02
  **Depends on:** Phase 16
  Plans:
  - [x] 17-01-PLAN.md — `eval/config.py` (EvalConfig pydantic BaseModel + EvalConfigError + from_yaml) + setup.cfg pydantic/pyyaml deps + tests/test_data_factory.py TestEvalConfigFromYAML class (FRAME-01)
  - [x] 17-02-PLAN.md — `eval/data_factory.py` (DataFactory class: load_real, generate_synthetic, augment, prepare_split, get_ground_truth) + 6 populated test classes in tests/test_data_factory.py (FRAME-02)

- [x] **Phase 18: MetricsEngine & Result Types** (2/2 plans) — Complete 2026-05-28
  **Goal:** Deliver `eval/types.py` (six dataclasses: `AlignResult`, `LabelResult`, `StageMetrics`, `Trial`, `SearchResult`, `EvalReport`) and `eval/metrics.py` (`MetricsEngine` wrapping all existing `zreg.metrics.*` with normalization, aggregation, scoring, and sanity checking).
  **Requirements:** FRAME-03, FRAME-04
  **Depends on:** Phase 17
  Plans:
  - [x] 18-01-PLAN.md — `eval/types.py` (6 frozen pydantic result models per FRAME-04) + `tests/test_metrics.py` scaffold (2 populated + 4 stubbed test classes)
  - [x] 18-02-PLAN.md — `EvalConfig.metric_weights` extension + `eval/metrics.py` `MetricsEngine` + populate remaining 4 test classes (FRAME-03)
  **Success criteria:**
  1. All six dataclasses importable from `eval.types`
  2. `MetricsEngine.normalize()` maps every metric to [0,1] with correct direction (lower/higher is better)
  3. `MetricsEngine.sanity_check()` fires warnings on degenerate inputs (empty cloud, all-same labels)
  4. `MetricsEngine.compute_score()` returns a scalar in [0,1]
  5. `tests/test_metrics.py` passes all gate criteria on handcrafted fixtures

- [x] **Phase 19: AlignmentStage** (2/2 plans) — Complete 2026-05-28
  **Goal:** Deliver `eval/stages/base.py` (`PipelineStage` ABC) and `eval/stages/alignment.py` (`AlignmentStage`) wrapping existing DTW + CPD code — standalone-runnable, `validate_params`-gated, with full test coverage.
  **Requirements:** FRAME-05
  **Depends on:** Phase 18
  **Success criteria:**
  1. `AlignmentStage.run()` completes without `LabelTransferStage` present
  2. DTW distance after alignment is measurably smaller than before on ≥2 synthetic datasets
  3. `validate_params()` raises on missing/invalid params
  4. `tests/test_alignment_stage.py` passes all gate criteria
  5. No reimplementation of DTW or CPD — all calls delegate to `zreg.dtw.*` and `zreg.cpd.*`
  **Plans:** 2 plans
  Plans:
  - [x] 19-01-PLAN.md — `StageResult` TypeAlias in `eval/types.py` + `eval/stages/base.py` (`PipelineStage` ABC) + `eval/stages/__init__.py` + test scaffold (2 populated + 4 stubbed)
  - [x] 19-02-PLAN.md — `eval/stages/alignment.py` (`AlignmentStage`) + update `eval/stages/__init__.py` + populate 4 test stubs

- [x] **Phase 20: LabelTransferStage** (2/2 plans) — completed 2026-05-29
  **Goal:** Deliver `eval/stages/label_transfer.py` (`LabelTransferStage`) wrapping existing `color_transfer` code — accepts raw or aligned clouds, standalone-runnable, with tests proving accuracy beats random baseline and correct chaining with `AlignmentStage`.
  **Requirements:** FRAME-06
  **Depends on:** Phase 19
  **Plans:** 2 plans
  Plans:
  - [x] 20-01-PLAN.md — `eval/stages/label_transfer.py` (`LabelTransferStage`) + update `eval/stages/__init__.py`
  - [x] 20-02-PLAN.md — `tests/test_label_transfer_stage.py` (5 test classes covering all FRAME-06 gate criteria)
  **Success criteria:**
  1. `LabelTransferStage.run()` completes without `AlignmentStage` present (raw cloud input)
  2. Label accuracy on synthetic data beats random baseline
  3. Stage accepts `AlignResult.aligned_cloud` as input (correct chaining)
  4. `tests/test_label_transfer_stage.py` passes all gate criteria
  5. No reimplementation of color transfer — delegates to `zreg.color_transfer`

- [ ] **Phase 21: EvaluationRunner & Visualisation**
  **Goal:** Deliver `eval/runners/eval_runner.py` (`EvaluationRunner`) orchestrating DataFactory → stages → MetricsEngine → aggregation → sanity checks → plots → `eval_report.json`, plus `eval/viz.py` with matplotlib Agg backend point-cloud and metric-summary plots.
  **Requirements:** FRAME-07, FRAME-08
  **Depends on:** Phase 20
  **Success criteria:**
  1. `EvaluationRunner.run()` with fixed params produces complete report (JSON + plots) in `output_dir`
  2. `eval_report.json` contains both per-dataset metrics and aggregated overview
  3. `sanity_flags` list is non-empty when intentionally bad inputs are provided
  4. All figure code runs inside `matplotlib.rc_context`; `plt.close(fig)` called; Agg backend used
  5. `tests/test_eval_runner.py` passes all gate criteria

- [ ] **Phase 22: HyperparamOptimizer & Search Strategies**
  **Goal:** Deliver `eval/search_strategies.py` (GridSearch, RandomSearch, Optuna Bayesian via Optuna 4.x with SQLite storage) and `eval/runners/optimizer.py` (`HyperparamOptimizer`) with sanity/dev/full tier logic, candidate pruning, and JSON output.
  **Requirements:** FRAME-09, FRAME-10
  **Depends on:** Phase 21
  **Success criteria:**
  1. Sanity tier completes on laptop in under 2 minutes
  2. `best_params.json` and `search_history.json` written to `output_dir`
  3. Best params from optimizer improve score vs default params (verified via EvaluationRunner)
  4. `prune_candidates()` demonstrably reduces candidate count between tiers
  5. `tests/test_optimizer.py` passes all gate criteria; Optuna ≥4.0,<5 with TPE sampler

- [ ] **Phase 23: CLI Entrypoint & Scenario Configs**
  **Goal:** Deliver `run_eval.py` CLI (`--config`, `--mode optimize|eval|full`) and 5 scenario YAML configs in `configs/`; `full` mode chains Optimizer → EvaluationRunner; all 5 configs run end-to-end without errors.
  **Requirements:** FRAME-11, FRAME-12
  **Depends on:** Phase 22
  **Success criteria:**
  1. All 5 scenario configs (`alignment_sanity`, `alignment_dev`, `label_transfer_sanity`, `label_transfer_dev`, `combined_full`) run without errors
  2. `--mode optimize`, `--mode eval`, `--mode full` all work correctly
  3. Config validation errors produce readable messages — no raw stacktraces
  4. `run_config.yaml` copy written to `output_dir` for each run (reproducibility)
  5. `run_eval.py --help` shows all flags with descriptions

<details>
<summary>✅ v1.1 Code Quality & Refactoring (Phases 6–11.1) — SHIPPED 2026-05-13</summary>

- [x] Phase 6: Python 3.12 Migration (2/2 plans) — completed 2026-04-13
- [x] Phase 7: CPD Deep Restructure (3/3 plans) — completed 2026-04-20
- [x] Phase 8: DTW Deep Restructure (2/2 plans) — completed 2026-04-23
- [x] Phase 9: Distance & Transform Restructure (2/2 plans) — completed 2026-04-27
- [x] Phase 10: Code Quality & Verification (3/3 plans) — completed 2026-04-29
- [x] Phase 11: Validate Refactoring and Fix Coverage (1/1 plan) — completed 2026-05-12
- [x] Phase 11.1: Close DTW-02 — Consistent Metric Variant Interface (1/1 plan, INSERTED) — completed 2026-05-13

Full details: [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1.0 Consolidation (Phases 1-5) — SHIPPED 2026-04-09</summary>

- [x] Phase 1: Validation Foundation & Quick Wins (2/2 plans) — completed 2026-04-09
- [x] Phase 2: Distance Metric & CPD Bug Fixes (3/3 plans) — completed 2026-04-09
- [x] Phase 3: DTW, Transform & CPD Enhancements (3/3 plans) — completed 2026-04-09
- [x] Phase 4: Infrastructure & Color Transfer Quality (2/2 plans) — completed 2026-04-09
- [x] Phase 5: Test Coverage (3/3 plans) — completed 2026-04-09

Full details: [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Validation Foundation & Quick Wins | v1.0 | 2/2 | Complete | 2026-04-09 |
| 2. Distance Metric & CPD Bug Fixes | v1.0 | 3/3 | Complete | 2026-04-09 |
| 3. DTW, Transform & CPD Enhancements | v1.0 | 3/3 | Complete | 2026-04-09 |
| 4. Infrastructure & Color Transfer Quality | v1.0 | 2/2 | Complete | 2026-04-09 |
| 5. Test Coverage | v1.0 | 3/3 | Complete | 2026-04-09 |
| 6. Python 3.12 Migration | v1.1 | 2/2 | Complete | 2026-04-13 |
| 7. CPD Deep Restructure | v1.1 | 3/3 | Complete | 2026-04-20 |
| 8. DTW Deep Restructure | v1.1 | 2/2 | Complete | 2026-04-23 |
| 9. Distance & Transform Restructure | v1.1 | 2/2 | Complete | 2026-04-27 |
| 10. Code Quality & Verification | v1.1 | 3/3 | Complete | 2026-04-29 |
| 11. Validate Refactoring and Fix Coverage | v1.1 | 1/1 | Complete | 2026-05-12 |
| 11.1. Close DTW-02: consistent metric variant interface | v1.1 | 1/1 | Complete | 2026-05-13 |
| 12. Carry-Forward Debt Closure | v1.2 | 3/3 | Complete | 2026-05-14 |
| 13. Core Metrics Library | v1.2 | 3/3 | Complete | 2026-05-15 |
| 14. Synthetic Data Generators | v1.2 | 3/3 | Complete | 2026-05-18 |
| 15. Experiment Tracking & Run Management | v1.2 | 2/2 | Complete | 2026-05-18 |
| 16. Runner Scripts | v1.2 | 2/2 | Complete | 2026-05-19 |
| 17. Framework Config & DataFactory | v1.2 | 2/2 | Complete | 2026-05-27 |
| 18. MetricsEngine & Result Types | v1.2 | 2/2 | Complete | 2026-05-28 |
| 19. AlignmentStage | v1.2 | 2/2 | Complete | 2026-05-28 |
| 20. LabelTransferStage | v1.2 | 0/2 | Pending | — |
| 21. EvaluationRunner & Visualisation | v1.2 | 0/2 | Pending | — |
| 22. HyperparamOptimizer & Search Strategies | v1.2 | 0/2 | Pending | — |
| 23. CLI Entrypoint & Scenario Configs | v1.2 | 0/2 | Pending | — |
