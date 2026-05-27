# Requirements: zReg v1.2 — Evaluation Framework & Debt Resolution

## Milestone Requirements

### Category 1 — Carry-Forward Debt

- [x] **CARRY-01**: `typing.Callable` → `collections.abc.Callable` in `cpd/base.py` and `cpd/_registration.py`
- [x] **CARRY-02**: `DistanceMetric` Protocol used as type annotation in at least one consumer module
- [x] **CARRY-03**: `color_transfer.py` absolute intra-package import replaced with relative import
- [x] **CARRY-04**: `config` module exposed at top-level (`import zreg; zreg.config` works)
- [x] **CARRY-05**: `VALIDATION.md` backfill written for all v1.1 phases (Phases 6–11.1)

### Category 2 — Core Metrics Library

- [x] **EVAL-01**: Alignment metrics available as pure functions in `src/zreg/metrics/alignment.py`: chamfer distance (`squared: bool = False`), hausdorff distance (`percentile: float = 95`), path smoothness, kNN consistency, temporal stability — all validate tensors, chamfer/hausdorff use `torch.cdist`/`torch.quantile` (no NumPy on GPU), kNN uses explicit `.detach().cpu().numpy()` entry — Complete Phase 13
- [x] **EVAL-02**: Label transfer metric available as pure function in `src/zreg/metrics/label_transfer.py`: F1-score with `y_true != -1` sentinel masking, `average` parameter (`weighted` default for HPO, `macro` for reporting), `zero_division=0` — Complete Phase 13

### Category 3 — Synthetic Data Generators

- [x] **EVAL-03**: Synthetic data generators in `src/zreg/generators/` as installed package: rigid/affine/noise transforms, label removal, synthetic label generation — all accept `seed: int | None = 42`, produce `dict[int, zRegPointCloud]`, support Gaussian noise and outlier injection corruption types — Validated in Phase 14

### Category 4 — Experiment Tracking

- [x] **EVAL-04**: Experiment tracking in `eval/tracking/` at repo root: writes local JSON + CSV per run using stdlib only (`csv.DictWriter`, `json.dump`) — required fields: `run_id`, `dataset_path`, `frame_indices`, `seed`, `n_points_before`, `n_points_after`, `git_hash`, `zreg_version`, `timestamp` — Complete Phase 15

### Category 5 — Evaluation Runners

- [x] **EVAL-05**: Evaluation runner scripts in `eval/` at repo root (not an importable package): `run_synthetic.py`, `run_real.py`, noise/corruption sweep, scale/density sweep — accept `dict[int, zRegPointCloud]` inputs, import metrics from `zreg.metrics` (installed source package), write outputs to `evaluation/runs/` — Complete Phase 16

### Category 6 — Framework Config & Data Layer

- [x] **FRAME-01**: `eval/config.py` — `EvalConfig` dataclass with YAML loading/validation via `EvalConfigError`; fields: `data_path`, `data_format`, `ground_truth_path`, `n_synthetic`, `transform_degree`, `augmentation_params`, `run_alignment`, `run_label_transfer`, `search_space`, `search_strategy`, `tier`, `n_trials`, `output_dir`, `save_plots`, `verbose` — Complete Phase 17
- [x] **FRAME-02**: `eval/data_factory.py` — `DataFactory` class; `load_real()` loads `.tracklets`/CSV via existing `load_data_from_tracklets`/`load_shah_from_csv`; `generate_synthetic()` wraps existing generators; `augment()` wraps existing corruption functions; `prepare_split()` → `(train, val)` tuple; `get_ground_truth()` extracts cell `id` or reads separate GT file per config; `tests/test_data_factory.py` with EvalConfig-from-YAML and split-shape gates — Complete Phase 17

### Category 7 — MetricsEngine & Result Types

- [ ] **FRAME-03**: `eval/metrics.py` — `MetricsEngine` class wrapping all existing `zreg.metrics.*` functions; adds `normalize(dict) → dict` mapping all scores to [0,1] with correct direction (lower/higher is better); `aggregate(list[StageMetrics]) → dict` returning mean/std/min/max; `compute_score(StageMetrics) → float` weighted scalar; `sanity_check(result) → list[str]` warning list for degenerate inputs
- [ ] **FRAME-04**: `eval/types.py` — dataclasses: `AlignResult` (aligned_cloud, warp_path, dtw_distance, n_changepoints, params_used); `LabelResult` (transferred_labels, params_used); `StageMetrics` (alignment + label fields + normalized dict); `Trial` (params, score, metrics, tier); `SearchResult` (best_params, best_score, history, tier); `EvalReport` (params, metrics, aggregated_metrics, per_dataset, plot_paths, sanity_flags); `tests/test_metrics.py` covering normalize direction, sanity_check triggers, compute_score output

### Category 8 — Pipeline Stages

- [ ] **FRAME-05**: `eval/stages/base.py` — `PipelineStage` ABC with `run(dataset, params) → StageResult` abstractmethod and `validate_params(params) → bool`; `eval/stages/alignment.py` — `AlignmentStage(PipelineStage)` using existing DTW + CPD code; hyperparams: `window_size`, `step`, `cpd_penalty`, `dtw_dist_fn`, `n_breakpoints`; runs without LabelTransferStage; `tests/test_alignment_stage.py` with standalone-run and validate_params gates
- [ ] **FRAME-06**: `eval/stages/label_transfer.py` — `LabelTransferStage(PipelineStage)` using existing color_transfer code; hyperparams: `k_neighbours`, `dist_metric`, `smoothing`, `threshold`; accepts raw or aligned clouds (output of AlignmentStage); `tests/test_label_transfer_stage.py` with standalone-run, chained-run, and label-accuracy-beats-random gates

### Category 9 — Runners & Visualisation

- [ ] **FRAME-07**: `eval/runners/eval_runner.py` — `EvaluationRunner(config, params)`; `run() → EvalReport` orchestrating DataFactory → stages → MetricsEngine → aggregation → sanity checks → plots → `eval_report.json`; `_run_single(ds, params) → dict`; `save_report(report, path)`; `tests/test_eval_runner.py` with fixed-params full-report and sanity-flag-on-bad-input gates
- [ ] **FRAME-08**: `eval/viz.py` — `plot_point_cloud(result, path)` and `plot_metrics_summary(report, path)`; all figure code inside `matplotlib.rc_context`; Agg backend; `plt.close(fig)` enforced; PDF output with `bbox_inches="tight"`; mathtext only (no system TeX)

### Category 10 — Hyperparameter Optimisation

- [ ] **FRAME-09**: `eval/runners/optimizer.py` — `HyperparamOptimizer(config)`; `run() → SearchResult` with sanity → dev → full tier logic; `_objective(params) → float`; `_tier_dataset(tier) → Dataset`; `prune_candidates(history, keep_top_k) → list[dict]`; `save_best_params(result, path)` writing `best_params.json` + `search_history.json`
- [ ] **FRAME-10**: `eval/search_strategies.py` — `GridSearch`, `RandomSearch` strategy classes; Optuna 4.x Bayesian search via `optuna.create_study` with TPE sampler (`n_startup_trials >= 2×N_params`) and SQLite storage (`load_if_exists=True`); `tests/test_optimizer.py` with sanity-tier under-2-min, pruning-reduces-candidates, best-params-improve-default gates

### Category 11 — CLI & Scenario Configs

- [ ] **FRAME-11**: `run_eval.py` CLI entrypoint — `--config cfg.yaml --mode optimize|eval|full`; loads and validates EvalConfig; `full` mode: Optimizer → reads `best_params.json` → EvaluationRunner; clean error messages for config errors (`EvalConfigError` → readable message, no stacktrace); `run_config.yaml` copy written to `output_dir` for reproducibility
- [ ] **FRAME-12**: 5 scenario YAML configs in `configs/`: `alignment_sanity.yaml`, `alignment_dev.yaml`, `label_transfer_sanity.yaml`, `label_transfer_dev.yaml`, `combined_full.yaml`; all 5 run without errors end-to-end

---

## Deferred / Superseded

- **EVAL-06** (standalone Optuna HPO scripts): Superseded by FRAME-09/FRAME-10 which implement the same optimization with a richer, modular architecture
- **EVAL-07** (standalone reporting module): Superseded by FRAME-07/FRAME-08 which integrate reporting and visualisation into EvaluationRunner + viz.py

---

## Future Requirements

- Property-based testing with Hypothesis for metric invariants (QOL-01)
- Performance regression tests with pytest-benchmark (QOL-02)
- py.typed marker for mypy/pyright downstream support (QOL-03)
- Structured result objects for all registration return values (QOL-04)
- Optuna MedianPruner (defer until baseline HPO results exist)
- Optuna Dashboard (defer — matplotlib submodule sufficient for now)
- Combined HPO objective with real data (defer — requires pilot run after EVAL-05)

---

## Out of Scope

| Item | Reason |
|------|--------|
| MLflow / W&B / Neptune | Heavyweight MLOps — local JSON+CSV is sufficient for the research use case |
| pandas | Zero benefit over stdlib csv/json; adds ~20MB dependency |
| plotly | Matplotlib covers all visualisation needs; plotly brings browser runtime dep |
| faiss | GPU kNN not needed — sklearn KDTree on CPU is correct for kNN consistency metric |
| scikit-learn as optional dep | Already imported without declaration; must be in `install_requires` |
| Frame/Sequence dataclasses | Duplicate existing `dict[int, zRegPointCloud]` pattern; use canonical type |
| `src/zreg/metrics/` proto stubs | Replaced in-place with correct implementations in Phase 13; proto stubs (`alignment_metrics.py`, `label_transfer_metrics.py`) left as orphaned reference code |
| Monolithic pipeline | Stages must remain independently testable — no single-class pipeline |
| Hardcoded paths | All paths via EvalConfig — no implicit CWD dependencies |

---

## Traceability

| REQ-ID | Phase | Status | Notes |
|--------|-------|--------|-------|
| CARRY-01 | Phase 12 | Complete | `typing.Callable` → `collections.abc.Callable` in cpd/ |
| CARRY-02 | Phase 12 | Complete | DistanceMetric used as annotation in one consumer |
| CARRY-03 | Phase 12 | Complete | color_transfer.py absolute → relative import |
| CARRY-04 | Phase 12 | Complete | config exposed at top-level |
| CARRY-05 | Phase 12 | Complete | VALIDATION.md backfill for Phases 6–11.1 |
| EVAL-01 | Phase 13 | Complete | `src/zreg/metrics/alignment.py` — chamfer, hausdorff, path_smoothness, knn_consistency, temporal_stability |
| EVAL-02 | Phase 13 | Complete | `src/zreg/metrics/label_transfer.py` — compute_f1, average="weighted" default fixed |
| EVAL-03 | Phase 14 | Complete | Validated 2026-05-18 |
| EVAL-04 | Phase 15 | Complete | Validated 2026-05-18 |
| EVAL-05 | Phase 16 | Complete | Validated 2026-05-19 |
| EVAL-06 | — | Superseded | Replaced by FRAME-09 + FRAME-10 |
| EVAL-07 | — | Superseded | Replaced by FRAME-07 + FRAME-08 |
| FRAME-01 | Phase 17 | Complete | EvalConfig + YAML loading — Validated 2026-05-27 |
| FRAME-02 | Phase 17 | Complete | DataFactory — Validated 2026-05-27 |
| FRAME-03 | Phase 18 | Pending | MetricsEngine |
| FRAME-04 | Phase 18 | Pending | Result types |
| FRAME-05 | Phase 19 | Pending | AlignmentStage |
| FRAME-06 | Phase 20 | Pending | LabelTransferStage |
| FRAME-07 | Phase 21 | Pending | EvaluationRunner |
| FRAME-08 | Phase 21 | Pending | viz.py |
| FRAME-09 | Phase 22 | Pending | HyperparamOptimizer |
| FRAME-10 | Phase 22 | Pending | SearchStrategies |
| FRAME-11 | Phase 23 | Pending | CLI entrypoint |
| FRAME-12 | Phase 23 | Pending | 5 scenario YAML configs |
