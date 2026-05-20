[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# zReg

> GPU-accelerated 3D point cloud registration, temporal alignment, and color transfer using PyTorch.

zReg is a scientific computing library for analyzing 3D point cloud data. It provides Coherent Point Drift (CPD) registration, Dynamic Time Warping (DTW) for trajectory alignment, Sliced Wasserstein Distance variants, and color/celltype transfer between aligned clouds — all built on PyTorch with optional GPU acceleration.

**v1.0** — 275 passing regression tests, full input validation, CPD convergence diagnostics.

## Features

- **CPD Registration** — rigid, affine, non-rigid, and constrained non-rigid point cloud alignment
- **Sliced Wasserstein Distances** — SWD, MaxSWD, ASWD, OSWD, GSWD, PSWD
- **Dynamic Time Warping** — temporal alignment of point cloud trajectories with Sakoe-Chiba band support
- **Pairwise Distance Matrix** — compute matrices across trajectory sets with optional MPI distribution
- **Color/Celltype Transfer** — propagate labels from source to target via nearest-neighbor, CPD-weighted, KNN voting, or Gaussian kernel
- **Downsampling** — Farthest Point Sampling (GPU-accelerated via torch_cluster), random, uniform
- **Geometric Transformations** — Rigid, Affine, NonRigid, TPS, Combined; composable and invertible
- **Open3D & torch_cluster interoperability** — convert to/from Open3D point clouds; FPS via torch_cluster when available

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
| `zreg.validation` | Input tensor validation (NaN/inf, device mismatch) used across the public API |

## Dependencies

**Core:** `numpy`, `scipy`, `torch`, `open3d`, `colorlog`, `tqdm`

**Optional:**
- `mpi4py` — MPI-distributed pairwise distance computation (`pip install -e ".[mpi]"`)
- `torch_cluster` — GPU-accelerated Farthest Point Sampling
- `matplotlib`, `seaborn` — visualization (`pip install -e ".[viz]"`)

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
├── configs                 <- Model and application configurations.
├── conftest.py             <- Root-level pytest configuration.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate transformed data.
│   ├── processed           <- Final canonical datasets.
│   ├── raw                 <- Original immutable data.
│   └── synthetic           <- Generated synthetic point cloud data.
├── docs                    <- Sphinx documentation.
├── environment.yml         <- Conda environment for reproducibility.
├── eval                    <- Sweep orchestrators and run-tracking support.
│   ├── tracking            <- Per-run metadata writer (log_run → JSON + CSV).
│   │   └── tracking.py
│   ├── run_synthetic.py    <- Noise/outlier sweep script.
│   └── run_real.py         <- Real-data scale/density sweep script.
├── experiments             <- Experiment outputs (JSON + CSV logs per sweep cell).
│   ├── datasets            <- Datasets used in experiments.
│   └── runs                <- Per-run metadata files ({run_id}.json, {run_id}.csv).
├── models                  <- Trained models and predictions.
├── notebooks               <- Jupyter notebooks.
│   ├── basics.ipynb
│   ├── cpd.ipynb
│   └── debug.ipynb
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
│       ├── metrics         <- Alignment and label-transfer evaluation metrics.
│       │   ├── alignment.py
│       │   └── label_transfer.py
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
├── tests                   <- Pytest test suite.
│   ├── conftest.py
│   ├── test_alignment_metrics.py
│   ├── test_color_transfer.py
│   ├── test_config.py
│   ├── test_cpd.py
│   ├── test_dataset.py
│   ├── test_distances.py
│   ├── test_downsampling.py
│   ├── test_dtw.py
│   ├── test_eval_metrics.py
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
