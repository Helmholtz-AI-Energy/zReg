# Plan: running baseline_experiments on HoreKa with GPUs

Phased per your choice: Phase 1 gets multi-trial parallelism working with
minimal code change (propulate is already built into `eval/runners/optimizer.py`
and `eval/search_strategies.py` — this repo has run on a GPU SLURM cluster
before, see `scripts/launch.sbatch`/`scripts/launch_srun.sh`, JUWELS Booster,
not HoreKa, but same paradigm). Phase 2 adds real per-operation GPU
acceleration and revisits the `max_points_per_frame`/`step` subsampling now
that more compute is available.

Everything marked **[VERIFY ON HOREKA]** is an assumption about HoreKa's
specific setup (partition names, module names, filesystem conventions) that
I can't confirm without cluster access — check against HoreKa's docs / your
project allocation before running.

---

## Phase 0 — Access & allocation

1. Confirm you have a HoreKa account and an active project allocation
   (compute hours + storage quota) — **[VERIFY ON HOREKA]** exact process,
   typically via bwHPC/NHR entitlement.
2. Allocate a workspace for this project's data + outputs (HoreKa/bwHPC
   convention is `ws_allocate <name> <days>`, giving a scratch path under
   `/hkfs/work/workspace/...` — **[VERIFY ON HOREKA]** exact command and
   path prefix). Don't put large data/outputs in `$HOME` — HPC home
   directories are typically small-quota and not tuned for I/O throughput.
3. Note the GPU partition name (JUWELS used `--partition=booster`; HoreKa's
   equivalent is likely something like `accelerated` or `dev_accelerated`
   for short test jobs — **[VERIFY ON HOREKA]**) and your account/project
   string for `#SBATCH --account=`.

## Phase 1 — Multi-trial parallelism (propulate + MPI), minimal code change

### 1.1 Environment setup

1. Load HoreKa's Python, CUDA, and MPI modules — **[VERIFY ON HOREKA]** exact
   module names, but expect something like:
   ```bash
   module load devel/cuda/12.4
   module load compiler/gnu
   module load mpi/openmpi
   # python3.12 is available by default — no module load needed
   ```
2. Create a venv (matches the existing JUWELS convention —
   `scripts/launch_srun.sh` activates `regvenv311`):
   ```bash
   python3 -m venv ~/regvenv_horeka
   source ~/regvenv_horeka/bin/activate
   ```
3. Install the project with the `propulate` extra (already defined in
   `setup.cfg:81-86`):
   ```bash
   cd <repo-on-horeka>
   pip install -e ".[propulate]"
   ```
4. **Critical**: `mpi4py` must be built against HoreKa's system MPI, not a
   generic pip wheel — a mismatched MPI implementation causes silent
   hangs or crashes under `srun`. After loading the MPI module:
   ```bash
   pip install mpi4py --no-binary mpi4py --force-reinstall
   ```
   Verify: `python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`
   under `srun -n 2` should print two different ranks (0 and 1).
5. Verify GPU visibility on a compute node (not the login node):
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.

### 1.2 Data transfer

1. `rsync` the real datasets from your local machine to the HoreKa
   workspace (they're gitignored, never in git — see
   `baseline_experiments/README.md` "Data dependency"):
   ```bash
   rsync -avz data/external/sample/ \
     <user>@horeka.scc.kit.edu:<workspace>/zReg-data/sample/
   ```
2. On HoreKa, either symlink it back into the repo layout (mirroring what
   this worktree does locally) or point every config's `data_path`/
   `target_data_path` at the workspace path directly. Symlinking is less
   config-churn:
   ```bash
   ln -s <workspace>/zReg-data/sample data/external/sample
   ```

### 1.3 Code change: make `run_all.py` rank-aware

This is the one real fix Phase 1 needs. Right now every MPI rank would
independently run the *entire* `run_all.py` orchestration loop (all 7
runs, file writes, `EvaluationRunner` calls) — only the propulate search
*inside* `HyperparamOptimizer.run()` is rank-coordinated
(`eval/search_strategies.py:436-490`: non-rank-0 returns an empty result
list). Left as-is, every rank races on the same `run_config.yaml`,
`best_params.json`, `eval_report.json` paths.

Fix in `baseline_experiments/scripts/run_all.py`:

```python
try:
    from mpi4py import MPI
    RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    RANK = 0  # single-process / no-MPI fallback, unchanged local behaviour
```

- **`HyperparamOptimizer(config).run()` calls**: every rank must call this
  (it's a collective MPI operation — propulate needs all ranks
  participating, you cannot have only rank 0 call it or it deadlocks on
  the internal `comm.Barrier()`).
- **Everything else** (`_write_run_config`, `mkdir`, reading/writing
  `best_params.json` outside the optimizer, `EvaluationRunner(...).run()`
  calls, `merge_params`, logging, the outer loop over the 7 runs' bookkeeping):
  gate behind `if RANK == 0:`. Non-zero ranks just participate in the
  collective `.run()` calls and otherwise idle/return.
- The 2 eval-only runs (`baseline_no_hpo`, `baseline_with_selfcal`) use
  `EvaluationRunner` directly with no MPI awareness at all — skip them
  entirely on non-zero ranks (`if RANK == 0:` around the whole call).

### 1.4 Config changes: switch search_strategy for cluster runs

Current configs hardcode `search_strategy: sobol` (sequential, single
process — correct for the laptop). For propulate to actually parallelize
trials across ranks, the 5 optimize configs need
`search_strategy: propulate` (or `auto`, which resolves to `propulate`
automatically when `SLURM_JOB_ID` is set — see
`eval/runners/optimizer.py:581-615`).

Rather than editing the existing configs (which need to keep working
locally), copy them into a `baseline_experiments/configs_horeka/` directory
with `search_strategy: auto` swapped in, so local (`sobol`) and cluster
(`propulate`) runs stay independently reproducible. Point
`baseline_experiments/scripts/run_all.py`'s `CONFIGS` constant at the
right directory via an env var or `--configs-dir` CLI flag (small addition
to the existing `argparse`-free script, or promote it to use `argparse`
like `run_eval.py` already does).

### 1.5 Job script

Adapting the existing JUWELS template (`scripts/launch.sbatch` +
`scripts/launch_srun.sh`) to HoreKa and to `run_all.py` instead of the old
`dtw_testing.py` prototype:

```bash
#!/bin/bash
#SBATCH --job-name=baseline-experiments
#SBATCH --partition=accelerated        # [VERIFY ON HOREKA]
#SBATCH --account=<your-project>       # [VERIFY ON HOREKA]
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4            # 4 GPUs/node -> 4 ranks/node, 16 total
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00                # generous vs. Phase 1's projected wall-clock
#SBATCH --output=<workspace>/logs/slurm-%j.out

module load devel/cuda/12.4 mpi/openmpi
source ~/regvenv_horeka/bin/activate

cd <workspace>/zReg
export KMP_DUPLICATE_LIB_OK=TRUE

srun --mpi=pmix --label \
  python -u baseline_experiments/scripts/run_all.py --phase all --verbose \
  > <workspace>/zReg/baseline_experiments/logs/run_all_horeka_${SLURM_JOB_ID}.log 2>&1
```

Submit with `sbatch launch_horeka.sbatch`.

### 1.6 Output directory

Same convention as always — `baseline_experiments/experiments/<phase>/<name>/`
— but on HoreKa this should resolve inside the workspace, not `$HOME`.
Either run the whole repo checkout from the workspace path (simplest — no
config changes, `output_dir` in each YAML is already a relative path
resolved from cwd), or override `output_dir` at the CLI level if
`run_all.py` grows a `--output-root` flag. Simplest: just `cd` into the
workspace-hosted repo checkout before `srun`, as shown above.

`aggregate_cost.py` and `aggregate_results.py` work unchanged against
whatever `experiments/` tree results — run them after the job completes to
get the same `summary/compute_cost.md` / `summary/summary.md` outputs
you're used to locally. The calibration constants in `aggregate_cost.py`
(`CPD_TRIAL_SECONDS`, measured on this laptop's CPU) will not be
meaningful for a GPU/multi-rank run — flag or recompute them separately
once Phase 1 timing data exists (see Phase 1.7).

### 1.7 Validate before trusting

Given how many times the local runtime estimate turned out wrong (see
`baseline_experiments/README.md` "Runtime" section — three separate
revisions), don't launch the full 7-run suite on HoreKa blind:

1. First submit a short job (`--time=00:15:00`, 1 node, `search_strategy:
   propulate` but low `n_trials`) running just `selfcal/kobitski_ew06_alignment`
   to confirm ranks actually coordinate correctly (check the log for
   distinct rank IDs, confirm only rank 0 writes `best_params.json`, no
   duplicate/racing writes).
2. Then scale to the full allocation and full suite.

## Phase 2 — Real GPU acceleration + revisit subsampling

Only pursue this once Phase 1 is confirmed working — it's a real code
change with more moving parts.

### 2.1 Add a `device` field

`eval/config.py`'s `EvalConfig` has no `device` field today. Add one
(default `"cpu"`, matching current behaviour so nothing local breaks):

```python
device: str = "cpu"
```

### 2.2 Thread it through DataFactory

`eval/data_factory.py` hardcodes `device="cpu"` in 6 places (`load_real`,
`load_target`, `get_ground_truth`, lines ~111/113/177/179/463/465).
Replace each with `device=self.config.device`. The underlying
`zreg.dataset.load_data_from_tracklets`/`load_shah_from_csv` already
accept and honor a `device` argument (confirmed — this is exactly the
pattern `scripts/dtw_testing.py:23-33` used directly against JUWELS,
`device="cuda:0"`), so this is a low-risk, mechanical change.

### 2.3 Verify downstream stages are device-consistent

`AlignmentStage`/`create_pairwise_distance_matrix`
(`src/zreg/pairwise_distance_matrix.py`) and the CPD/transform classes
already follow whatever device their input tensors are on (`.to(device=...)`
patterns throughout `src/zreg/transforms/*.py`, `src/zreg/cpd/*.py`) — this
should mostly just work once frames load onto CUDA, but needs a real
end-to-end smoke test on a HoreKa GPU node before trusting it (watch for
any hardcoded `.cpu()` calls or numpy round-trips that would silently
force a device transfer per-op and erase the benefit).

### 2.4 Check Open3D's CUDA build

`alignment_method: icp` uses Open3D's tensor geometry API
(`src/zreg/dataset.py:15-20`, `o3d.t.geometry`), which supports CUDA via
`o3d.core.Device("CUDA:0")` — but only if the installed Open3D wheel was
built with CUDA support (not guaranteed by a plain `pip install open3d`).
Verify with `python -c "import open3d as o3d; print(o3d.core.cuda.is_available())"`
on a HoreKa GPU node before relying on ICP-based configs there.

### 2.5 Revisit `max_points_per_frame` / `step`

The current `max_points_per_frame=1000, step=8` setup
(`baseline_experiments/README.md` "Spatial subsampling") was calibrated
specifically for this laptop's CPU and RAM ceiling. With real GPU
acceleration and HoreKa's much larger RAM per node, it may be worth
re-running the same calibration methodology (direct
`create_pairwise_distance_matrix` timing at a few candidate
`max_points_per_frame` values, this time on a HoreKa GPU node) to see
whether full point density (Kobitski ~16,572 / Shah ~4,113 pts/frame,
zero information loss from subsampling) becomes tractable within an
acceptable wall-clock budget. This would be a meaningful quality
improvement for the actual experiment results, not just a speed win —
worth doing once Phase 1 + 2.1-2.4 are confirmed working.

---

## Open questions before starting Phase 1

- Do you already have a HoreKa account + project allocation, or does that
  need requesting first?
- What's the actual partition name and account string for `#SBATCH`?
- Rough idea of how many GPU-hours/node-hours your allocation gives you,
  to size the `--nodes`/`--time` request sensibly for the first real run?
