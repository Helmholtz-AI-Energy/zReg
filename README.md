[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# zReg

> GPU-accelerated 3D point cloud registration, temporal alignment, color transfer, and evaluation using PyTorch.

zReg is a scientific computing library for analyzing 3D point cloud data. It provides Coherent Point Drift (CPD) registration, Dynamic Time Warping (DTW) for trajectory alignment, Sliced Wasserstein Distance variants, and color/celltype transfer between aligned clouds — all built on PyTorch with optional GPU acceleration. A built-in evaluation framework provides YAML-driven experiment configuration, tiered hyperparameter optimization, six quantitative metrics, and automated trajectory export.

**v1.2** — 994 passing regression tests, paired and synthetic evaluation pipelines, CPD-aligned trajectory export, geometric data augmentation, full input validation, CPD convergence diagnostics, and a complete config-driven evaluation framework.

## Features

- **CPD Registration** — rigid, affine, non-rigid, and constrained non-rigid point cloud alignment
- **Sliced Wasserstein Distances** — SWD, MaxSWD, ASWD, OSWD, GSWD, PSWD
- **Dynamic Time Warping** — temporal alignment of point cloud trajectories with Sakoe-Chiba band support
- **Pairwise Distance Matrix** — compute matrices across trajectory sets with optional MPI distribution
- **Color/Celltype Transfer** — propagate labels from source to target via nearest-neighbor, CPD-weighted, KNN voting, or Gaussian kernel
- **Downsampling** — Farthest Point Sampling (GPU-accelerated via torch_cluster), random, uniform
- **Geometric Transformations** — Rigid, Affine, NonRigid, TPS, Combined; composable and invertible
- **Open3D & torch_cluster interoperability** — convert to/from Open3D point clouds; FPS via torch_cluster when available
- **Evaluation Framework** — YAML-configured pipeline with paired and synthetic modes, alignment + label-transfer stages, six metrics, tiered HPO (grid/random/Bayesian/MPI-parallel), CPD-aligned trajectory CSV export, and matplotlib visualizations

## Installation

1. Review `environment.yml` and create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate zReg
   ```
   > The conda environment installs zReg in editable mode. Re-run `pip install -e .` after changes to `setup.cfg`.

2. For pip-only installs:
   ```bash
   pip install -e .
   # With MPI support:
   pip install -e ".[mpi]"
   # With visualization tools:
   pip install -e ".[viz]"
   ```

Optional, run once after `git clone`:

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Install nbstripout to keep notebook outputs out of git history:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

## Quick Start

**Point cloud registration (CPD):**
```python
import zreg

# source, target: torch tensors of shape (N, 3) and (M, 3)
result = zreg.cpd.cpd_registration(source, target, tf_type_name="rigid")
transformed = result.transformation.transform(source)
```

**Temporal alignment (DTW):**
```python
from zreg.dtw import DynamicTimeWarping

# x, y: dicts mapping time index -> point cloud tensor
dtw = DynamicTimeWarping(x=trajectory_x, y=trajectory_y, distance_metric="swd")
result = dtw.compute()
# result.warping_path, result.distance, result.cost_matrix
```

**Pairwise distance matrix:**
```python
from zreg.pairwise_distance_matrix import create_pairwise_distance_matrix

matrix, rotations = create_pairwise_distance_matrix(
    x=x_trajectories, y=y_trajectories,
    distance_metric="swd", cpd_type="rigid"
)
```

**Color/celltype transfer:**
```python
from zreg.color_transfer import transfer_colors, ColorTransferMethod

colors = transfer_colors(
    source, target,
    method=ColorTransferMethod.NEAREST_NEIGHBOR,
    source_colors=source_labels,
)
```

## Modules

| Module | Description |
|--------|-------------|
| `zreg.cpd` | CPD registration — `RigidCPD`, `AffineCPD`, `NonRigidCPD`, `cpd_registration()` |
| `zreg.transforms` | Transformation classes — `RigidTransformation`, `AffineTransformation`, `NonRigidTransformation`, `TPSTransformation`, `CombinedTransformation` |
| `zreg.dtw` | `DynamicTimeWarping` — trajectory alignment with windowed DP and multiple distance metrics |
| `zreg.distances` | Sliced Wasserstein variants — SWD, MaxSWD, ASWD, OSWD, GSWD, PSWD; also Euclidean/Manhattan/Minkowski |
| `zreg.pairwise_distance_matrix` | `create_pairwise_distance_matrix()` — full matrix computation with optional MPI |
| `zreg.color_transfer` | `transfer_colors()` — four label propagation methods |
| `zreg.downsampling` | `farthest_point_down_sample()`, `random_down_sample()`, `uniform_down_sample()` |
| `zreg.dataset` | `zRegPointCloud`, data loading from MATLAB/CSV formats |
| `zreg.metrics.alignment` | Geometric alignment metrics — `chamfer`, `hausdorff`, `path_smoothness` |
| `zreg.metrics.label_transfer` | Label transfer metrics — `compute_f1`, `knn_consistency`, `temporal_stability` |
| `zreg.validation` | Input tensor validation (NaN/inf, device mismatch) used across the public API |

**Evaluation framework** (`eval/` directory, not an installable package):

| Module | Description |
|--------|-------------|
| `eval.config` | `EvalConfig` — pydantic v2 YAML config; `EvalConfigError` for user-facing validation errors |
| `eval.stages.alignment` | `AlignmentStage` — DTW + CPD wrapper; params: `window_size`, `step`, `cpd_penalty`, `dtw_dist_fn`, `n_breakpoints` |
| `eval.stages.label_transfer` | `LabelTransferStage` — k-NN voting label transfer wrapper; params: `k_neighbours`, `dist_metric`, `smoothing`, `threshold` |
| `eval.runners.eval_runner` | `EvaluationRunner` — orchestrates DataFactory → stages → MetricsEngine → EvalReport; writes `eval_report.json` |
| `eval.runners.optimizer` | `HyperparamOptimizer` — tiered HPO (sanity → dev → full) with warm-start pruning; writes `best_params.json` and `search_history.json` |
| `eval.search_strategies` | `GridSearch`, `RandomSearch`, `BayesianSearch` (Optuna TPE + SQLite), `PropulateSearch` (MPI parallel evolutionary) |
| `eval.metrics` | `MetricsEngine` — normalize, compute_score, aggregate, sanity_check, compute_stage_metrics |
| `eval.data_factory` | `DataFactory` — lazy cached loader for real (`.tracklets` / `.csv`) and synthetic trajectories; geometric augmentation (`scale`, `rotate`, `drop_points`, `sample_new_points`, `augment`); paired-mode target loading (`load_target`) and synthetic-mode target generation (`generate_target`, `get_synthetic_ground_truth`); train/val split |
| `eval.viz` | `plot_trajectory` — 3D alignment + label figures (PDF + PNG); `plot_metrics` — normalised scores bar chart (PDF); `render_dataset_triptych` — dataset preview figure |
| `eval.tracking.trajectory` | `export_trajectory` — writes `align_trajectory.csv`, `label_trajectory.csv`, and per-stage metadata JSON |
| `eval.tracking.tracking` | `log_run` — writes `{run_id}.json` + `{run_id}.csv` with auto-captured git hash, package version, and timestamp |
| `eval.types` | Frozen pydantic result types — `AlignResult`, `LabelResult`, `StageMetrics`, `Trial`, `SearchResult`, `EvalReport` |

## Evaluation Framework

The evaluation framework lives in `eval/` and is driven entirely from YAML config files. Use `run_eval.py` at the repo root as the entrypoint.

### Quick start

```bash
# Optimize hyperparameters, then evaluate with best params
python run_eval.py --config configs/combined_full.yaml --mode full --output-dir experiments/runs/exp1

# Hyperparameter search only
python run_eval.py --config configs/alignment_sanity.yaml --mode optimize

# Evaluation only (reads best_params.json from output-dir if present)
python run_eval.py --config configs/label_transfer_dev.yaml --mode eval --verbose
```

Modes:
- **`optimize`** — runs `HyperparamOptimizer`, writes `best_params.json` and `search_history.json`
- **`eval`** — loads `best_params.json` (if present), runs `EvaluationRunner`, writes `eval_report.json`
- **`full`** — chains optimize then eval; the config YAML is copied to `run_config.yaml` before the pipeline runs

### Config reference

All fields except `data_path` are optional:

```yaml
data_path: data/external/sample/...tracklets   # required
data_format: tracklets                          # "tracklets" (default) or "csv"
pipeline_mode: paired                           # "paired" (default) or "synthetic"
target_data_path: null                          # paired mode: source aligned to this target
transform_spec: null                            # synthetic mode: e.g. {type: rigid, degrees: 30, axis: z}
run_alignment: true
run_label_transfer: true
tier: full                                      # "sanity" | "dev" | "full"
n_trials: 20
search_strategy: bayesian                       # "grid" | "random" | "bayesian" | "propulate" | "auto"
output_dir: experiments/runs
save_plots: true
val_split: 0.2
metric_weights:
  chamfer: 0.35
  hausdorff: 0.15
  path_smoothness: 0.10
  temporal_stability: 0.10
  f1: 0.20
  knn_consistency: 0.10
label_names:                                    # optional: map integer IDs to display names
  0: "T cell"
  1: "B cell"
search_space:
  window_size: [3, 5, 10]
  k_neighbours: [3, 5, 10]
default_params:
  window_size: 10
  step: 1
  cpd_penalty: null
  dtw_dist_fn: euclidean
  n_breakpoints: 5
  k_neighbours: 5
  dist_metric: euclidean
  smoothing: 0.0
  threshold: 0.0
```

#### Pipeline modes

The pipeline aligns a **source** trajectory to a **target**. `pipeline_mode`
selects where that target comes from:

- **`paired`** (default) — the target is a second real trajectory loaded from
  `target_data_path` (via `DataFactory.load_target()`). Use this to align two
  recorded datasets to each other.
- **`synthetic`** — the target is generated from the source by applying
  `transform_spec` (e.g. a known rigid rotation), giving a controlled
  ground-truth for HPO and metric validation (`DataFactory.generate_target()` /
  `get_synthetic_ground_truth()`).

Ready-made scenario configs are provided in `configs/`:

| Config | Stages | Tier / Mode |
|--------|--------|-------------|
| `alignment_sanity.yaml` | alignment only | sanity |
| `alignment_dev.yaml` | alignment only | dev |
| `label_transfer_sanity.yaml` | label transfer only | sanity |
| `label_transfer_dev.yaml` | label transfer only | dev |
| `combined_full.yaml` | both | full |
| `paired_alignment.yaml` | alignment only | paired mode |
| `synthetic_mode.yaml` | alignment only | synthetic mode |

Additional dataset-specific comparison configs (e.g. `shah_vs_kobitski*.yaml`,
`kobitski_vs_shah.yaml`, `selfcal_*.yaml`) also live in `configs/` for
real-data registration experiments.

### Pipeline

```
DataFactory
    └─ load_real() / generate_synthetic()
         ↓
AlignmentStage          (DTW + optional CPD; params: window_size, step, cpd_penalty, dtw_dist_fn, n_breakpoints)
         ↓
LabelTransferStage      (k-NN voting; params: k_neighbours, dist_metric, smoothing, threshold)
         ↓
MetricsEngine           (6 metrics → normalize → weighted score)
         ↓
EvalReport              (eval_report.json + optional plots + trajectory CSVs)
```

### Metrics

Six metrics are computed, normalized to `[0, 1]`, and combined into a weighted scalar score:

| Metric | Field | Direction |
|--------|-------|-----------|
| Chamfer distance | `chamfer_distance` | lower is better |
| Hausdorff distance (95th pct) | `hausdorff_distance` | lower is better |
| Path smoothness (DTW warp variance) | `path_smoothness` | lower is better |
| Temporal stability (transform norm) | `temporal_stability` | lower is better |
| Weighted F1 score | `f1_score` | higher is better |
| k-NN label consistency | `knn_consistency` | higher is better |

### Hyperparameter optimization

`HyperparamOptimizer` runs up to three tiers in sequence, warm-starting each tier with the top-3 candidates from the previous:

- **sanity** — small synthetic dataset, 5 trials; fast correctness check
- **dev** — real data (or synthetic fallback), 20 trials
- **full** — full real dataset, `n_trials` from config

Four search strategies are available:

| Strategy | Class | Notes |
|----------|-------|-------|
| `grid` | `GridSearch` | exhaustive Cartesian product; stateless |
| `random` | `RandomSearch` | uniform random sampling; stateless |
| `bayesian` | `BayesianSearch` | Optuna TPE sampler; persists `optuna.db` for resumable runs |
| `propulate` | `PropulateSearch` | MPI-parallel evolutionary search; requires `pip install zreg[propulate]` |
| `auto` | — | selects `propulate` when `mpi4py` world size > 1 or `SLURM_JOB_ID` is set, otherwise `bayesian` |

### Outputs

Each run produces a directory (default `experiments/runs/`) containing:

| File | Description |
|------|-------------|
| `run_config.yaml` | exact copy of the input YAML for reproducibility |
| `eval_report.json` | full `EvalReport` — params, metrics, aggregated stats, paths |
| `best_params.json` | best hyperparameter assignment (optimize / full modes) |
| `search_history.json` | all trials with scores and per-trial metrics |
| `optuna.db` | Optuna SQLite study (Bayesian strategy only; resumable) |
| `alignment_trajectory.pdf/.png` | 3D scatter: source vs aligned cloud for 3 frames |
| `label_trajectory.pdf/.png` | 3D scatter: point cloud coloured by transferred label |
| `metrics_summary.pdf` | horizontal bar chart of 6 normalised metric scores |
| `align_trajectory.csv` | per-point per-frame positions from the alignment stage |
| `align_metadata.json` | run metadata: run_id, git hash, zreg version, timestamps |
| `label_trajectory.csv` | per-point per-frame positions + transferred label IDs |
| `label_metadata.json` | run metadata for the label-transfer stage |

### Programmatic usage

```python
from eval.config import EvalConfig
from eval.runners import EvaluationRunner, HyperparamOptimizer

config = EvalConfig.from_yaml("configs/combined_full.yaml")

# Optimize
optimizer = HyperparamOptimizer(config)
search_result = optimizer.run()   # writes best_params.json

# Evaluate with best params
runner = EvaluationRunner(config, search_result.best_params)
report = runner.run()             # writes eval_report.json

print(report.metrics.f1_score)
print(report.aggregated_metrics)  # {metric: {mean, std, min, max}}
```

## Dependencies

**Core:** `numpy`, `scipy`, `torch`, `open3d`, `colorlog`, `tqdm`

**Optional:**
- `mpi4py` — MPI-distributed pairwise distance computation (`pip install -e ".[mpi]"`)
- `torch_cluster` — GPU-accelerated Farthest Point Sampling
- `matplotlib`, `seaborn` — visualization (`pip install -e ".[viz]"`)
- `optuna` — Bayesian hyperparameter search in the evaluation framework
- `propulate`, `mpi4py` — MPI-parallel evolutionary HPO (`pip install zreg[propulate]`)
- `pydantic` — typed config and result models in the evaluation framework
- `PyYAML` — YAML config loading

**Python:** 3.12+

## Logging

```python
import zreg

zreg.set_log_level("DEBUG")          # programmatic
# or via environment variable:
# ZREG_LOG_LEVEL=DEBUG python ...
```

## Project Organization

```
├── AUTHORS.md
├── CHANGELOG.md
├── COMPARISON.md           <- Side-by-side comparison of registration approaches.
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE.txt
├── README.md
├── VALIDATION.md           <- Validation results and regression summaries.
├── configs                 <- Evaluation scenario YAML configs (scenarios + paired/synthetic modes + dataset comparisons).
│   ├── alignment_sanity.yaml
│   ├── alignment_dev.yaml
│   ├── label_transfer_sanity.yaml
│   ├── label_transfer_dev.yaml
│   ├── combined_full.yaml
│   ├── paired_alignment.yaml      <- Paired-mode: align two real trajectories.
│   ├── synthetic_mode.yaml        <- Synthetic-mode: transform-spec target generation.
│   └── shah_vs_kobitski*.yaml …   <- Dataset-specific registration comparisons.
├── conftest.py             <- Root-level pytest configuration.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate transformed data.
│   ├── processed           <- Final canonical datasets.
│   ├── raw                 <- Original immutable data.
│   └── synthetic           <- Generated synthetic point cloud data.
├── docs                    <- Sphinx documentation.
├── environment.yml         <- Conda environment for reproducibility.
├── eval                    <- Evaluation framework (not an installable package).
│   ├── config.py           <- EvalConfig pydantic model + EvalConfigError.
│   ├── data_factory.py     <- DataFactory: lazy loader for real/synthetic trajectories.
│   ├── metrics.py          <- MetricsEngine: normalize, score, aggregate, sanity_check.
│   ├── search_strategies.py <- GridSearch, RandomSearch, BayesianSearch, PropulateSearch.
│   ├── types.py            <- Frozen pydantic result types (AlignResult, LabelResult, …).
│   ├── viz.py              <- plot_trajectory, plot_metrics, render_dataset_triptych (Agg backend, PDF + PNG).
│   ├── runners
│   │   ├── __init__.py
│   │   ├── eval_runner.py  <- EvaluationRunner: full pipeline orchestration.
│   │   └── optimizer.py    <- HyperparamOptimizer: tiered HPO with warm-start.
│   ├── stages
│   │   ├── __init__.py
│   │   ├── base.py         <- PipelineStage ABC.
│   │   ├── alignment.py    <- AlignmentStage: DTW + CPD wrapper.
│   │   └── label_transfer.py <- LabelTransferStage: k-NN voting wrapper.
│   └── tracking
│       ├── __init__.py
│       ├── tracking.py     <- log_run: per-run JSON + CSV metadata writer.
│       └── trajectory.py   <- export_trajectory: CSV + metadata JSON per stage.
├── experiments             <- Experiment outputs (JSON + CSV logs per sweep cell).
│   ├── datasets            <- Datasets used in experiments.
│   └── runs                <- Per-run metadata files ({run_id}.json, {run_id}.csv).
├── models                  <- Trained models and predictions.
├── notebooks               <- Jupyter notebooks.
│   ├── basics.ipynb
│   ├── cpd.ipynb
│   └── debug.ipynb
├── run_eval.py             <- CLI entrypoint for the evaluation framework (--config, --mode, --output-dir, --verbose).
├── pyproject.toml          <- Build configuration.
├── references              <- Papers, manuals, and reference material.
├── reports
│   └── figures             <- Generated figures and plots.
├── scripts                 <- Standalone analysis and launch scripts.
│   ├── color_transfer_example.py
│   ├── dtw_testing.py
│   ├── example_plots.py
│   ├── launch.sbatch       <- SLURM batch job script.
│   ├── launch_srun.sh      <- SLURM interactive launch script.
│   └── train_model.py
├── setup.cfg               <- Declarative project configuration.
├── setup.py
├── src
│   └── zreg                <- Package source.
│       ├── cpd             <- CPD registration (rigid, affine, non-rigid).
│       │   ├── base.py
│       │   ├── rigid.py
│       │   ├── affine.py
│       │   ├── nonrigid.py
│       │   ├── kernels.py
│       │   ├── _registration.py
│       │   └── _types.py
│       ├── distances       <- Sliced Wasserstein variants and general distances.
│       │   ├── sw_varients.py
│       │   ├── general.py
│       │   └── _protocol.py
│       ├── dtw             <- Dynamic Time Warping (core, constraints, result).
│       │   ├── core.py
│       │   ├── constraints.py
│       │   └── result.py
│       ├── generators      <- Synthetic trajectory, corruption, and label generators.
│       │   ├── generators.py
│       │   ├── corruption.py
│       │   ├── transforms.py
│       │   └── labels.py
│       ├── metrics         <- Evaluation metrics.
│       │   ├── alignment.py      <- Geometric metrics: chamfer, hausdorff, path_smoothness.
│       │   └── label_transfer.py <- Label transfer metrics: compute_f1, knn_consistency, temporal_stability.
│       ├── transforms      <- Transformation classes (rigid, affine, non-rigid, TPS, combined).
│       │   ├── base.py
│       │   ├── rigid.py
│       │   ├── affine.py
│       │   ├── nonrigid.py
│       │   ├── tps.py
│       │   ├── combined.py
│       │   └── homogeneous.py
│       ├── color_transfer.py           <- Label/color propagation (NN, CPD-weighted, KNN, Gaussian).
│       ├── config.py                   <- Package-level configuration.
│       ├── dataset.py                  <- Data loading from MATLAB/CSV formats.
│       ├── downsampling.py             <- FPS, random, and uniform downsampling.
│       ├── pairwise_distance_matrix.py <- Full matrix computation with optional MPI.
│       ├── setup_log.py                <- Logging setup (set_log_level).
│       ├── utils.py                    <- Shared utilities.
│       └── validation.py              <- Input tensor validation used across the public API.
├── tests                   <- Pytest test suite (994 tests).
│   ├── conftest.py
│   ├── test_alignment_metrics.py
│   ├── test_color_transfer.py
│   ├── test_config.py
│   ├── test_cpd.py
│   ├── test_dataset.py
│   ├── test_distances.py
│   ├── test_downsampling.py
│   ├── test_dtw.py
│   ├── test_eval_metrics.py       <- MetricsEngine unit tests.
│   ├── test_eval_runner.py        <- EvaluationRunner + HyperparamOptimizer tests.
│   ├── test_generators.py
│   ├── test_label_transfer_metrics.py
│   ├── test_pairwise_distance_matrix.py
│   ├── test_runners.py
│   ├── test_tracking.py
│   ├── test_transforms.py
│   ├── test_utils.py
│   └── test_validation.py
├── tox.ini                 <- Tox test automation configuration.
└── .pre-commit-config.yaml <- Pre-commit hook configuration.
```

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[nbstripout]: https://github.com/kynan/nbstripout
