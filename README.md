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
├── LICENSE.txt
├── README.md
├── configs                 <- Model and application configurations.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate transformed data.
│   ├── processed           <- Final canonical datasets.
│   └── raw                 <- Original immutable data.
├── docs                    <- Sphinx documentation.
├── environment.yml         <- Conda environment for reproducibility.
├── models                  <- Trained models and predictions.
├── notebooks               <- Jupyter notebooks (basics.ipynb, cpd.ipynb).
├── pyproject.toml          <- Build configuration.
├── scripts                 <- Analysis scripts (example_plots.py, dtw_testing.py, color_transfer_example.py).
├── setup.cfg               <- Declarative project configuration.
├── src
│   └── zreg                <- Package source.
├── tests                   <- Pytest test suite (275 regression tests).
└── .pre-commit-config.yaml <- Pre-commit hook configuration.
```

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[nbstripout]: https://github.com/kynan/nbstripout
