# Requirements: zReg v1.2 — Evaluation Framework & Debt Resolution

## Milestone Requirements

### Category 1 — Carry-Forward Debt

- [x] **CARRY-01**: `typing.Callable` → `collections.abc.Callable` in `cpd/base.py` and `cpd/_registration.py`
- [x] **CARRY-02**: `DistanceMetric` Protocol used as type annotation in at least one consumer module
- [x] **CARRY-03**: `color_transfer.py` absolute intra-package import replaced with relative import
- [x] **CARRY-04**: `config` module exposed at top-level (`import zreg; zreg.config` works)
- [x] **CARRY-05**: `VALIDATION.md` backfill written for all v1.1 phases (Phases 6–11.1)

### Category 2 — Core Metrics Library

- [ ] **EVAL-01**: Alignment metrics available as pure functions in `src/zreg/metrics/alignment.py`: chamfer distance (`squared: bool = False`), hausdorff distance (`percentile: float = 95`), path smoothness, kNN consistency, temporal stability — all validate tensors, chamfer/hausdorff use `torch.cdist`/`torch.quantile` (no NumPy on GPU), kNN uses explicit `.detach().cpu().numpy()` entry
- [ ] **EVAL-02**: Label transfer metric available as pure function in `src/zreg/metrics/label_transfer.py`: F1-score with `y_true != -1` sentinel masking, `average` parameter (`weighted` default for HPO, `macro` for reporting), `zero_division=0`

### Category 3 — Synthetic Data Generators

- [x] **EVAL-03**: Synthetic data generators in `eval/generators/` at repo root: rigid/affine/noise transforms, label removal, synthetic label generation — all accept `seed: int | None = 42`, produce `dict[int, zRegPointCloud]`, support Gaussian noise and outlier injection corruption types — Validated in Phase 14: Synthetic Data Generators

### Category 4 — Experiment Tracking

- [ ] **EVAL-04**: Experiment tracking in `eval/tracking/` at repo root: writes local JSON + CSV per run using stdlib only (`csv.DictWriter`, `json.dump`) — required fields: `run_id`, `dataset_path`, `frame_indices`, `seed`, `n_points_before`, `n_points_after`, `git_hash`, `zreg_version`, `timestamp`

### Category 5 — Evaluation Runners

- [ ] **EVAL-05**: Evaluation runner scripts in `eval/` at repo root (not an importable package): `run_synthetic.py`, `run_real.py`, noise/corruption sweep, scale/density sweep — all include Open3D import guard, import metrics from `zreg.metrics` (installed source package), write outputs to `evaluation/runs/`

### Category 6 — Hyperparameter Optimisation

- [ ] **EVAL-06**: HPO scripts in `eval/hpo_*.py` at repo root: alignment standalone, label transfer standalone, combined label-agreement objective — Optuna 4.x (`>=4.0,<5`), TPE sampler with `n_startup_trials >= 2×N_params`, SQLite storage with `load_if_exists=True`, combined objective uses `alpha*(1-F1_weighted) + (1-alpha)*chamfer_normalised` where `chamfer_normalised = chamfer(registered, target) / chamfer(source, target)`, HPO data source is disjoint from held-out evaluation data

### Category 7 — Reporting & Visualisation

- [ ] **EVAL-07**: Reporting module in `eval/reporting/` at repo root: per-run CSV + PDF figure + JSON metadata — all figure code inside `matplotlib.rc_context`, Agg backend, `plt.close(fig)` enforced, PDF output uses `bbox_inches="tight"`, LaTeX-ready (mathtext, no system TeX required)

---

## Future Requirements (Deferred)

- Property-based testing with Hypothesis for metric invariants (QOL-01)
- Performance regression tests with pytest-benchmark (QOL-02)
- py.typed marker for mypy/pyright downstream support (QOL-03)
- Structured result objects for all registration return values (QOL-04)
- Optuna MedianPruner (deferred until baseline HPO results exist)
- Optuna Dashboard visualisation (defer — matplotlib submodule sufficient for now)
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
| EVAL-04 | Phase 15 | Pending | Depends on Phase 13; parallel with Phase 14 |
| EVAL-05 | Phase 16 | Pending | Depends on Phases 13+14+15 |
| EVAL-06 | Phase 17 | Pending | Depends on Phase 16; parallel with Phase 18 |
| EVAL-07 | Phase 18 | Pending | Depends on Phase 16; parallel with Phase 17 |
