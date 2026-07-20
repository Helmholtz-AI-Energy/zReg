#!/bin/bash
# Phase 51 smoke job — single-rank, proves env + data + GPU.
# Submit AFTER running scripts/setup_horeka.sh and transferring data.

#SBATCH --job-name=zreg-smoke
#SBATCH --partition=accelerated           # [VERIFY ON HOREKA] — check HoreKa GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hk-project-starter-p0024807/uitog/BA/zReg/baseline_experiments/logs/slurm-smoke-%j.out

module load devel/cuda/12.4
module load compiler/gnu
module load mpi/openmpi
# python3.12 is available by default — no module load needed

set -euo pipefail  # fail fast: broken module load, bad activate, or missing cd will abort job

VENV=~/regvenv_horeka  # explicit path used in srun — source activate alone does not carry across srun fork
source "$VENV/bin/activate"

cd /home/hk-project-starter-p0024807/uitog/BA/zReg
mkdir -p baseline_experiments/logs       # ensure --output parent dir exists before SLURM writes to it

export KMP_DUPLICATE_LIB_OK=TRUE         # harmless on Linux; matches HOREKA_PLAN.md template

# Run smoke evaluation via run_eval.py directly (not the orchestration script — that hardcodes
# the configs directory and would bypass the smoke config; see RESEARCH.md Pitfall 6).
# run_eval.py appends a datetime timestamp subdirectory to --output-dir; eval_report.json will be
# inside that timestamped subdir — check the SLURM log for the resolved path.
srun "$VENV/bin/python" run_eval.py \
  --config baseline_experiments/configs_horeka/smoke/selfcal/kobitski_ew06_alignment.yaml \
  --mode full \
  --output-dir baseline_experiments/experiments/smoke/kobitski_ew06_alignment

# After job completes, verify:
# 1. squeue shows job COMPLETED (not FAILED/CANCELLED)
# 2. This output file contains no Python tracebacks
# 3. eval_report.json exists under baseline_experiments/experiments/smoke/kobitski_ew06_alignment/<timestamp>/
# 4. python -c 'import torch; print(torch.cuda.is_available())' on compute node shows True
