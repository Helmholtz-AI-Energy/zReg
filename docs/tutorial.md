# zReg Evaluation Framework Tutorial

## 1. Introduction

The zReg evaluation framework is a YAML-driven pipeline that runs alignment (DTW + CPD/ICP/SWD),
label transfer, tiered hyperparameter optimization, and six quantitative metrics — all from a
single config file. The entrypoint is `run_eval.py` at the repository root; pass it a YAML config
and a run mode (`optimize`, `eval`, or `full`) and the framework handles everything else.

---

## 2. Hardware requirements

### 2.1 Any laptop (CPU — fully supported)

All three CLI modes (`optimize`, `eval`, `full`) run without a GPU. All shipped configs
default to `device: cpu` — no config change is needed for laptop runs. To use Apple Silicon
GPU acceleration, you must explicitly add `device: mps` to your YAML (see Section 2.2).

**Practical runtime guidance by tier:**

| Tier | Trials | Typical laptop time |
|------|--------|---------------------|
| `sanity` (3–5 trials) | 3–5 | ~1–3 min |
| `dev` (20 trials, bayesian, `max_points_per_frame: 100`) | 20 | ~20–40 min |
| `full` (n_trials from config, full dataset) | configurable | hours |

`max_points_per_frame` is the primary knob for keeping laptop runtimes manageable: set it to
100–500 to subsample each frame, and start with `tier: sanity` or `tier: dev`.

**ICP constraint:** `alignment_method: icp` requires `device: cpu`. Open3D ICP performs
explicit `.cpu().numpy()` round-trips internally; `EvalConfig` raises a `ValueError` at
config-load time if you pair `icp` with any non-CPU device.

**Learned label transfer:** `label_transfer_method: pointnet2` and `label_transfer_method: egnn`
run on CPU when no CUDA device is present, but are significantly faster with a GPU.

### 2.2 Apple Silicon / MPS

MacBooks with an M-series chip can use the Metal GPU backend. Add `device: mps` to your
YAML config to activate it:

```yaml
device: mps
```

**What works on MPS:**

- `DataFactory` checks `torch.backends.mps.is_available()` at startup and raises a
  `RuntimeError` if MPS is not available (macOS < 12.3 or non-Apple hardware). See
  Pitfall 9 below.
- SWD aligner (`alignment_method: swd`) is fully MPS-compatible: all tensor operations
  derive the device from the input tensors and create intermediates on the same device.
- Learned label transfer (`pointnet2`, `egnn`): models are constructed on CPU and moved
  to the device via `model.to(device)` — MPS is supported. See the import-order note below.
- `train_label_transfer.py` auto-detects MPS when `--device` is not passed (priority:
  CUDA → MPS → CPU).

**Known MPS limitations:**

- **ICP is CPU-only regardless of hardware.** `EvalConfig` raises a `ValueError` at
  config-load time if `device: mps` is paired with `alignment_method: icp`. Use
  `alignment_method: cpd` or `alignment_method: swd` to run on MPS.
- **FPS/kNN downsampling:** `torch_cluster` is activated only when `pos.is_cuda` is `True`.
  MPS tensors automatically fall back to the scipy/numpy path. This is functionally
  correct but slightly slower than the CUDA path for very large point clouds.
- **Import order on macOS ARM:** eGNN and PointNet++ models import `torch_geometric`, which
  triggers a SIGABRT from duplicate `libomp` initialisation if no `zreg`/`scipy`-importing
  module has been loaded first. `run_eval.py` and `train_label_transfer.py` already follow
  the correct order. In custom scripts, import `zreg` (or any scipy-importing module) before
  importing `torch_geometric`. See Pitfall 10 below.

### 2.3 CUDA / HoreKa cluster (for large-scale runs)

Set `device: cuda` (or `cuda:0` / `cuda:1` for a specific card) in the config.

**GPU-03 status (Phase 53 — not yet verified):** The code path for running
`AlignmentStage` and CPD fully on GPU tensors is implemented, but end-to-end
verification that no silent CPU fallback occurs anywhere in the path is pending Phase 53.
The CPU and MPS paths work correctly; the CUDA path is implemented but has not yet been
smoke-tested on real GPU hardware.

**Search strategy for dual-environment configs:** Use `search_strategy: auto` in configs
meant to run on both laptop and cluster. `auto` picks `propulate` when `SLURM_JOB_ID` is
set or MPI world_size > 1, and falls back to `bayesian` on a laptop. `search_strategy:
propulate` uses MPI and is designed for multi-rank SLURM jobs; it requires:

```bash
pip install -e ".[propulate]"
```

**HoreKa cluster setup:** Run `scripts/setup_horeka.sh` once on the login node. This script
loads the required CUDA 12.x and OpenMPI modules, creates a virtualenv, and source-rebuilds
`mpi4py` against the cluster's MPI installation.

**Data transfer to HoreKa:** Data must be transferred to the HoreKa workspace before jobs
run. Use the `rsync` command inside `setup_horeka.sh` to copy `data/external/sample/` from
your laptop, then create an absolute symlink from `data/external/sample` inside the repo
checkout to the workspace data directory. Relative symlinks break under `srun` — use
absolute paths as documented in `setup_horeka.sh`.

**Learned label transfer on cluster:** `pointnet2` and `egnn` require a checkpoint `.pt` file
produced by the Phase 47 training pipeline (`scripts/train_model.py`). Provide the path via
`egnn_checkpoint_path` or `pointnet2_checkpoint_path` in the config.

---

## 3. Prerequisites and installation

### Environment setup (one-time)

```bash
conda env create -f environment.yml
conda activate zReg
```

Re-run `pip install -e .` after any changes to `setup.cfg`.

### Optional extras

```bash
pip install -e ".[mpi]"       # MPI-parallel pairwise distances + propulate HPO
pip install -e ".[viz]"       # matplotlib + seaborn for plots
```

### Data files required (before running any config)

- **Source trajectory:** a `.tracklets` file (MATLAB format) or a `.csv` cell-tracks file.
  Set `data_path` and `data_format` in the config.
- **Target trajectory (paired mode only):** a second `.tracklets` or `.csv` file at
  `target_data_path`. Only required when `pipeline_mode: paired` and `run_alignment: true`.
- **Synthetic mode:** no second file needed — the target is generated from the source by
  applying `transform_spec`.
- **Checkpoint file:** required only when `label_transfer_method` is `pointnet2` or `egnn`.
  Set `pointnet2_checkpoint_path` or `egnn_checkpoint_path` to the `.pt` file produced by
  `scripts/train_model.py`.

The `data/external/sample/` directory ships two sample datasets for smoke-testing:

- `kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets`
- `shah_data/sample-1/sample-1-cell-tracks.csv`

Use these to verify your setup before running on your own data.

---

## 4. Config file structure: all EvalConfig fields

Every field except `data_path` is optional — all others carry sensible defaults.

**Note on unknown keys:** EvalConfig uses `extra='forbid'` — any unrecognised key in the YAML
raises `EvalConfigError` at load time. Typos in field names are caught before any computation
runs.

### Data loading

These fields control which files are loaded and how the input data is handled.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_path` | `str` | *(required)* | Path to the source dataset file (`.tracklets` or `.csv`). The only required field. |
| `data_format` | `str` | `"tracklets"` | Loader to use: `"tracklets"` (MATLAB format) or `"csv"` (cell-tracks CSV). |
| `pipeline_mode` | `str` | `"paired"` | Pipeline mode: `"paired"` loads a real target from `target_data_path`; `"synthetic"` generates a target from `transform_spec`. |
| `target_data_path` | `str \| null` | `null` | Path to the second (target) dataset. Required at runtime when `pipeline_mode: paired` and `run_alignment: true`. Error is raised at `DataFactory.load_target()` call time, not at config load. |
| `target_data_format` | `str \| null` | `null` | Format for the target loader: `"tracklets"` or `"csv"`. When `null`, inherits `data_format`. |
| `transform_spec` | `dict \| null` | `null` | Transform dict for synthetic mode. Must contain a `type` key (`"rigid"` or `"noise"`) plus the corresponding parameters (e.g. `rotation_deg`, `rotation_axis`, `sigma`). Error is raised at `DataFactory.generate_target()` call time. |
| `n_synthetic` | `int` | `100` | Number of synthetic frames to generate. |
| `max_points_per_frame` | `int \| null` | `null` | If set, subsample each frame of source and target to at most this many points (random without replacement, seed 42). `null` disables subsampling. Applied immediately after loading. |
| `ground_truth_path` | `str \| null` | `null` | Optional external ground-truth file. When `null`, ground truth is extracted from the dataset's `id` field. |

### Pipeline stages

These fields select which stages run and which algorithm each stage uses.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_alignment` | `bool` | `true` | Whether `AlignmentStage` executes. Set to `false` to skip spatial alignment. |
| `run_label_transfer` | `bool` | `true` | Whether `LabelTransferStage` executes. |
| `alignment_method` | `str` | `"cpd"` | Spatial registration method: `"cpd"` (Coherent Point Drift), `"icp"` (Open3D point-to-point rigid; CPU only), or `"swd"` (Sliced Wasserstein Distance). |
| `swd_variant` | `str` | `"aswd"` | SWD variant when `alignment_method: swd`. One of: `"swd"`, `"aswd"`, `"oswd"`, `"gswd"`, `"pswd"`, `"maxswd"`. Validated only when `alignment_method: swd`. |
| `label_transfer_method` | `str` | `"knn_voting"` | Label transfer algorithm: `"knn_voting"` (k-NN majority vote), `"cpd_weighted"` (CPD E-step posterior-weighted), `"pointnet2"` (requires checkpoint), or `"egnn"` (requires checkpoint). |
| `egnn_checkpoint_path` | `str \| null` | `null` | Path to the eGNN `.pt` checkpoint. Required when `label_transfer_method: egnn`. Validated at `LabelTransferStage.run()` time. |
| `pointnet2_checkpoint_path` | `str \| null` | `null` | Path to the PointNet++ `.pt` checkpoint. Required when `label_transfer_method: pointnet2`. Validated at `LabelTransferStage.run()` time. |

### Preprocessing

These fields control data scaling and alignment-stage preprocessing.

**`data_preprocessing`** is a nested sub-config (or `null` to disable):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_preprocessing` | `DataPreprocessingConfig \| null` | `DataPreprocessingConfig()` | Per-trajectory data scaling applied after loading. `null` disables scaling entirely. Default is z-score standardization. Only the `pos` field is scaled; `label`, `id`, and `fps-idx` pass through unchanged. |
| `data_preprocessing.method` | `str` | `"standardize"` | Scaling strategy: `"standardize"` (z-score, mean ≈ 0, std ≈ 1), `"normalize"` (min-max to [0, 1]), or `"robust"` (median + IQR scaling, then clipping). |
| `data_preprocessing.robust_outlier_threshold` | `float` | `3.0` | Clip threshold after IQR scaling, used only when `method: robust`. Values are clipped to `[-threshold, +threshold]`. Ignored for other methods. |

**`alignment_preprocessing`** is a nested sub-config (or `null` to disable):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alignment_preprocessing` | `AlignmentPreprocessingConfig \| null` | `null` | Pre-DTW preprocessing applied inside `AlignmentStage`. `null` disables it (default). |
| `alignment_preprocessing.method` | `str` | *(required if set)* | `"principal_axes"` applies PCA rotation; `"velocity_landmarks"` flags high-velocity frames as DTW landmarks. |
| `alignment_preprocessing.velocity_threshold` | `float` | `0.5` | Velocity above which a frame is recorded as a landmark. Only consulted when `method: velocity_landmarks`. |
| `alignment_preprocessing.velocity_metric` | `str` | `"mean"` | Reduction applied to per-point displacement norms: `"mean"` or `"max"`. Only consulted when `method: velocity_landmarks`. |

### Hyperparameter search

These fields control the optimizer, search budget, and the parameter space.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_strategy` | `str` | `"sobol"` | Optimizer backend: `"grid"`, `"random"`, `"sobol"`, `"bayesian"`, `"propulate"`, or `"auto"`. See Section 7 for guidance. |
| `tier` | `str` | `"sanity"` | Search tier: `"sanity"` (fast, 5 trials), `"dev"` (20 trials), or `"full"` (`n_trials` trials). |
| `n_trials` | `int` | `10` | Trial count for the `full` tier. Sanity always uses 5 and dev uses 20. |
| `search_space` | `dict` | `{}` | Candidate values for each hyperparameter to search. Keys override `default_params` during each trial. |
| `default_params` | `dict` | `{}` | Fallback hyperparameter values for any key not in `search_space`. Merged with trial params by the optimizer. |
| `sobol_seed` | `int` | `42` | Seed for the scrambled Owen Sobol sequence when `sobol_randomize: true`. Silently ignored when `sobol_randomize: false`. |
| `sobol_randomize` | `bool` | `true` | `true` uses the scrambled Owen sequence (better uniformity, reproducible via `sobol_seed`). `false` uses the classical Van der Corput sequence (seed ignored). |

### Evaluation and output

These fields control metrics, device selection, and output paths.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `val_split` | `float` | `0.2` | Fraction of frames held out as validation. |
| `metric_weights` | `dict[str, float]` | `{chamfer: 0.35, hausdorff: 0.15, path_smoothness: 0.10, temporal_stability: 0.10, f1: 0.20, knn_consistency: 0.10}` | Relative weights for the six metrics. Auto-rescaled by their sum; values need not sum to 1.0. |
| `label_names` | `dict[int, str] \| null` | `null` | Optional map of integer label IDs to display names (e.g. `{0: 'T cell', 1: 'B cell'}`). Used in legend labels of the label-trajectory figure. |
| `output_dir` | `str` | `"experiments/runs"` | Directory for experiment outputs. A timestamp subdirectory is appended automatically at runtime. |
| `save_plots` | `bool` | `true` | Whether to save matplotlib figures (requires `pip install -e ".[viz]"`). |
| `verbose` | `bool` | `false` | Enable INFO-level logging. Equivalent to passing `--verbose` at the CLI. |
| `device` | `str` | `"cpu"` | Compute device: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`, or `"mps"`. |
| `transform_degree` | `float` | `0.1` | Perturbation magnitude for synthetic data generation. |
| `augmentation_params` | `dict` | `{}` | Extra keyword arguments forwarded to augmentation functions in `DataFactory`. |

---

## 5. Pipeline modes

The pipeline aligns a **source** trajectory to a **target**. `pipeline_mode` selects where the
target comes from.

### `paired` (default)

- Aligns a source trajectory to a real second trajectory.
- Requires `target_data_path` in the config. The error is raised at `DataFactory.load_target()`
  call time — not at config construction — so you will not see it until the pipeline actually runs.
- Use `target_data_format` if the target file format differs from the source.
- Use case: cross-dataset registration (e.g. Kobitski `.tracklets` vs. Shah `.csv`).

### `synthetic`

- Generates a synthetic target by applying `transform_spec` to the source trajectory.
- Ground truth correspondence is known (identity mapping before perturbation), enabling metric
  validation without manually labelled data.
- `transform_spec` must contain a `type` key and the corresponding parameters:

```yaml
pipeline_mode: synthetic
transform_spec:
  type: rigid
  rotation_deg: 30.0
  rotation_axis: [0, 0, 1]
```

```yaml
pipeline_mode: synthetic
transform_spec:
  type: noise
  sigma: 0.05
```

---

## 6. Step-by-step walkthrough

### Step 1 — Smoke-test on the sample data (CPU, ~2 min)

Run the shipped sanity config to verify your installation:

```bash
python run_eval.py --config configs/alignment_sanity.yaml --mode eval
```

What this does:

- Loads the Kobitski `.tracklets` file from `data/external/sample/`.
- Runs `AlignmentStage` only — `run_label_transfer: false`.
- Uses `default_params` directly (no HPO in `eval` mode unless `best_params.json` is present).
- Writes output to `experiments/runs/alignment_sanity/<YYYY-MM-DD_HH-MM-SS>/`.

If the run completes and writes `eval_report.json`, your environment is working correctly.

### Step 2 — Write your first config

A minimal working config for a single-dataset alignment-only run in synthetic mode:

```yaml
# my_first_run.yaml
data_path: data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets
data_format: tracklets
pipeline_mode: synthetic
transform_spec:
  type: rigid
  rotation_deg: 15.0
  rotation_axis: [0, 0, 1]
run_alignment: true
run_label_transfer: false
tier: sanity
n_trials: 3
output_dir: experiments/runs/my_first_run
```

Then run:

```bash
python run_eval.py --config my_first_run.yaml --mode full
```

`full` mode chains HPO (optimize) then evaluation (eval). With `tier: sanity` and 3 trials this
completes in under 3 minutes on a laptop.

### Step 3 — Paired alignment (two real datasets)

Full paired config example (based on `configs/shah_vs_kobitski_hpo.yaml`, simplified):

```yaml
data_path: data/external/sample/shah_data/sample-1/sample-1-cell-tracks.csv
data_format: csv
target_data_path: data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets
target_data_format: tracklets
pipeline_mode: paired
run_alignment: true
run_label_transfer: true
max_points_per_frame: 100
tier: dev
search_strategy: bayesian
search_space:
  cpd_penalty: [null, rigid, affine]
  window_size: [10, 30, 60]
default_params:
  window_size: 40
  step: 2
  cpd_penalty: rigid
  dtw_dist_fn: euclidean
  n_breakpoints: 5
  k_neighbours: 5
  dist_metric: euclidean
  smoothing: 0.0
  threshold: 0.5
output_dir: experiments/runs/shah_vs_kobitski
```

`max_points_per_frame: 100` keeps per-trial runtime manageable on a laptop (roughly 2 minutes
per trial at `tier: dev`).

### Step 4 — Understand the three run modes

| Mode | Command | What it does | Writes |
|------|---------|--------------|--------|
| `optimize` | `python run_eval.py --config cfg.yaml --mode optimize` | Runs HPO only, no evaluation | `best_params.json`, `search_history.json` |
| `eval` | `python run_eval.py --config cfg.yaml --mode eval` | Runs evaluation using `best_params.json` if present, else `default_params` | `eval_report.json`, plots, trajectory CSVs |
| `full` | `python run_eval.py --config cfg.yaml --mode full` | Chains optimize then eval | All of the above |

**Runtime overrides:**

- `--output-dir <path>` overrides `output_dir` from the config at runtime.
- `--verbose` enables INFO-level logging (equivalent to `verbose: true` in the YAML).

### Step 5 — Read the outputs

Each run creates a timestamp-stamped directory under your `output_dir`:

```
experiments/runs/<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
  run_config.yaml          # exact byte-for-byte copy of your input YAML (written before pipeline runs)
  eval_report.json         # full EvalReport: best params, all six metrics, per-dataset aggregated stats
  best_params.json         # best hyperparameter assignment found during HPO
  search_history.json      # all trials with scores and per-trial metrics
  alignment_trajectory.pdf # 3D scatter plot: source vs aligned cloud (3 sample frames)
  label_trajectory.pdf     # point cloud coloured by transferred label ID
  metrics_summary.pdf      # horizontal bar chart of 6 normalised metric scores
  align_trajectory.csv     # per-point per-frame positions from the alignment stage
  label_trajectory.csv     # per-point per-frame positions + transferred label IDs
  align_metadata.json      # run_id, git hash, zreg version, timestamps (alignment stage)
  label_metadata.json      # same for the label-transfer stage
```

`run_config.yaml` is written before the pipeline executes, so even a failed run leaves a
reproducibility record.

---

## 7. Hyperparameter search: strategy guide

| Strategy | When to use |
|----------|-------------|
| `grid` | Small discrete search spaces; exhaustive and fully reproducible |
| `random` | Larger spaces where a grid is too slow; stateless, no setup required |
| `sobol` | **Default.** Quasi-random low-discrepancy sequences; better coverage than `random`; reproducible via `sobol_seed`; falls back to `RandomSearch` when `n_trials < 8` |
| `bayesian` | Best for continuous parameters and medium budgets (10–50 trials); persists `optuna.db` for resumable runs |
| `propulate` | Multi-rank MPI jobs on HoreKa or any SLURM cluster; requires `pip install -e ".[propulate]"` |
| `auto` | Shared configs that run on both laptop and cluster: picks `propulate` when `SLURM_JOB_ID` is set or MPI world_size > 1, otherwise picks `bayesian` |

### Tiered search

`HyperparamOptimizer` runs up to three tiers in sequence, warm-starting each tier with the
top-3 candidates from the previous tier:

| Tier | Trials | Dataset |
|------|--------|---------|
| `sanity` | 5 (fixed) | Small synthetic data — fast correctness check |
| `dev` | 20 (fixed) | Real data (or synthetic fallback) |
| `full` | `n_trials` (from config) | Full real dataset |

`tier: sanity` runs only the sanity tier. `tier: dev` runs sanity then dev. `tier: full` runs
all three. The `n_trials` field controls only the full-tier trial count.

---

## 8. `default_params` and `search_space`

`default_params` provides fallback values for every hyperparameter. `search_space` defines the
candidates the optimizer explores. During each trial, keys in `search_space` override the
corresponding `default_params` values. After optimization, `best_params.json` holds the winning
values and is merged with `default_params` at eval time — so any hyperparameter not searched
falls back to its `default_params` value.

### AlignmentStage params

| Key | Type | Description |
|-----|------|-------------|
| `window_size` | `int` | DTW Sakoe-Chiba band half-width (frames) |
| `step` | `int` | Frame step for subsampling the trajectory before DTW |
| `cpd_penalty` | `null`, `"rigid"`, `"affine"` | CPD spatial registration type after DTW; `null` means DTW temporal alignment only (no CPD) |
| `dtw_dist_fn` | `str` | Distance metric for DTW cost: `"euclidean"`, `"cosine"`, or any SWD variant name |
| `n_breakpoints` | `int` | Change-point detection sensitivity (higher = more breakpoints) |

Note: `alignment_method` and `dtw_dist_fn` are **independent** choices. `alignment_method`
selects the spatial registration wrapper (CPD, ICP, or SWD). `dtw_dist_fn` selects the
temporal alignment distance function used inside DTW.

### LabelTransferStage params

| Key | Type | Description |
|-----|------|-------------|
| `k_neighbours` | `int` | Number of neighbours for k-NN voting |
| `dist_metric` | `str` | Distance metric for k-NN: `"euclidean"` or other |
| `smoothing` | `float` | Label smoothing weight in range [0, 1] |
| `threshold` | `float` | Confidence threshold below which labels are suppressed |

---

## 9. Data preprocessing

### `data_preprocessing` (ON by default)

Data preprocessing applies scaling to the `pos` field of each trajectory after loading.
`label`, `id`, and `fps-idx` fields pass through unchanged. Statistics are computed globally
across all frames of the trajectory (concatenated).

Three methods are available:

| Method | Behaviour |
|--------|-----------|
| `standardize` (default) | Z-score: subtract per-dimension mean, divide by per-dimension standard deviation. After scaling, each dimension has mean ≈ 0, std ≈ 1. |
| `normalize` | Min-max scaling: maps each dimension to [0, 1] using `(pos - min) / (max - min + eps)`. |
| `robust` | Subtract per-dimension median, divide by IQR (Q75 - Q25), then clip to `±robust_outlier_threshold` (default 3.0). Resistant to outliers. |

To disable preprocessing entirely, set `data_preprocessing: null` in the YAML. To use a
non-default method:

```yaml
data_preprocessing:
  method: robust
  robust_outlier_threshold: 2.5
```

### `alignment_preprocessing` (OFF by default)

Applied inside `AlignmentStage` before DTW runs. Two strategies are available:

- `method: principal_axes` — applies PCA rotation (`compute_pca_rotation`) to align the
  trajectory's principal axes before temporal alignment.
- `method: velocity_landmarks` — flags high-velocity frames as DTW landmarks before running
  the warp (`detect_velocity_landmarks`; controlled by `velocity_threshold` and `velocity_metric`).

To enable:

```yaml
alignment_preprocessing:
  method: principal_axes
```

```yaml
alignment_preprocessing:
  method: velocity_landmarks
  velocity_threshold: 0.8
  velocity_metric: max
```

---

## 10. Common pitfalls

**1. `EvalConfig: field 'data_path': Field required`**

- **Symptom:** Config load fails immediately.
- **Cause:** `data_path` is the only required field and was omitted from the YAML.
- **Fix:** Add `data_path: path/to/your/data.tracklets` as the first line of your config.

**2. `EvalConfig: field 'some_key': Extra inputs are not permitted`**

- **Symptom:** Config load fails with an "Extra inputs are not permitted" message.
- **Cause:** A key in the YAML is not a recognised EvalConfig field — usually a typo or an
  outdated config from a previous version.
- **Fix:** Check the field name against the reference table in Section 4 and correct or remove
  the unknown key.

**3. `ValueError: device='cuda' is incompatible with alignment_method='icp'`**

- **Symptom:** `EvalConfig.from_yaml` raises a `ValueError` at config load time.
- **Cause:** Open3D ICP requires CPU-resident tensors and performs explicit `.cpu().numpy()`
  round-trips internally. Pairing `icp` with any non-CPU device is always an error.
- **Fix:** Either set `device: cpu`, or switch to `alignment_method: cpd` or
  `alignment_method: swd` if you want GPU computation.

**4. `target_data_path` is missing in `paired` mode**

- **Symptom:** The pipeline crashes at runtime (not at config load) with an error from
  `DataFactory.load_target()`.
- **Cause:** `pipeline_mode: paired` with `run_alignment: true` requires a target dataset, but
  `target_data_path` was not set. EvalConfig does not validate this at construction time —
  the error is raised at use time.
- **Fix:** Add `target_data_path: path/to/target.tracklets` (and optionally
  `target_data_format`) to your config.

**5. Long runtimes on a laptop**

- **Symptom:** Individual trials take many minutes; the full run takes hours.
- **Cause:** Large point clouds per frame and/or many trials.
- **Fix:**
  - Set `max_points_per_frame: 100` to `500` to subsample each frame before alignment.
  - Use `tier: sanity` (5 trials) or `tier: dev` (20 trials) before committing to `tier: full`.
  - Reduce `n_trials` for the full tier.
  - Increase `step` in `default_params` (e.g. `step: 2`) to use every other frame.

**6. `sobol` search produces fewer unique samples than expected**

- **Symptom:** Fewer distinct trial configurations than requested; the log may mention
  `RandomSearch` fallback.
- **Cause:** `n_trials < 8` triggers an automatic fallback from `SobolSearch` to
  `RandomSearch`. The Sobol sequence requires at least 8 trials for meaningful
  low-discrepancy coverage.
- **Fix:** Either increase `n_trials` to 8 or more, or explicitly set
  `search_strategy: random` to make the fallback intentional.

**7. `egnn_checkpoint_path` or `pointnet2_checkpoint_path` not set**

- **Symptom:** `ValueError` raised at `LabelTransferStage.run()` when `label_transfer_method`
  is `egnn` or `pointnet2`.
- **Cause:** Learned label transfer methods require a trained model checkpoint. The path is
  validated at stage-run time, not at config construction.
- **Fix:** Point `egnn_checkpoint_path` or `pointnet2_checkpoint_path` to the `.pt` checkpoint
  file produced by `scripts/train_model.py`.

**8. Plots missing**

- **Symptom:** No `.pdf` or `.png` files in the output directory, or an import error mentioning
  `matplotlib`.
- **Cause:** The `viz` optional extra is not installed.
- **Fix:** Run `pip install -e ".[viz]"` to install `matplotlib` and `seaborn`. Also verify
  that `save_plots: true` is set in your config (it is the default, but double-check if you
  copied a minimal config that omits it).

**9. `RuntimeError: device='mps' requested but torch.backends.mps.is_available() is False`**

- **Symptom:** `DataFactory` raises a `RuntimeError` at startup when `device: mps` is set.
- **Cause:** MPS requires macOS 12.3 or later and an Apple Silicon chip (M1/M2/M3/M4).
  The error fires at `DataFactory.__init__` time, before any data is loaded.
- **Fix:** Either upgrade to macOS 12.3+ on Apple Silicon hardware, or fall back to
  `device: cpu` in your config.

**10. SIGABRT on macOS ARM when using eGNN or PointNet++**

- **Symptom:** The process aborts with `SIGABRT` immediately after importing the model,
  with a message about duplicate library initialisation or `libomp`.
- **Cause:** `torch_geometric` initialises `libomp` when it is imported. If `torch_geometric`
  is imported before any `zreg`/`scipy`-importing module in the same process, a second
  `libomp` instance is loaded and macOS kills the process.
- **Fix:** Do not reorder imports relative to `run_eval.py` or `train_label_transfer.py`,
  which already follow the correct order. In custom scripts, import `zreg` (or any
  scipy-importing module such as `scipy` or `numpy`) before importing `torch_geometric`.
  The guard is documented in `src/zreg/models/egnn.py` (lines 38–39).
