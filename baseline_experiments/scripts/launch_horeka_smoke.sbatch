#!/bin/bash
# Phase 51 smoke job — single-rank, proves env + data + GPU.
# Submit AFTER running scripts/setup_horeka.sh and transferring data.
# Fill in all [VERIFY ON HOREKA] values before submitting.

#SBATCH --job-name=zreg-smoke
#SBATCH --partition=accelerated           # [VERIFY ON HOREKA] — check HoreKa GPU partition name
#SBATCH --account=<your-project>          # [VERIFY ON HOREKA] — fill in your project/account string
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/<name>/zReg/baseline_experiments/logs/slurm-smoke-%j.out  # [VERIFY ON HOREKA]

module load devel/cuda/12.x              # [VERIFY ON HOREKA]
module load compiler/gnu                  # [VERIFY ON HOREKA]
module load mpi/openmpi                   # [VERIFY ON HOREKA]
module load devel/python/3.12             # [VERIFY ON HOREKA]

set -euo pipefail  # fail fast: broken module load, bad activate, or missing cd will abort job

source ~/regvenv_horeka/bin/activate

cd /hkfs/work/workspace/<name>/zReg      # [VERIFY ON HOREKA] — absolute workspace repo checkout path
mkdir -p baseline_experiments/logs       # ensure --output parent dir exists before SLURM writes to it

export KMP_DUPLICATE_LIB_OK=TRUE         # harmless on Linux; matches HOREKA_PLAN.md template

# Run smoke evaluation via run_eval.py directly (not the orchestration script — that hardcodes
# the configs directory and would bypass the smoke config; see RESEARCH.md Pitfall 6).
# run_eval.py appends a datetime timestamp subdirectory to --output-dir; eval_report.json will be
# inside that timestamped subdir — check the SLURM log for the resolved path.
srun python run_eval.py \
  --config baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml \
  --mode full \
  --output-dir baseline_experiments/experiments/smoke/kobitski_ew06_alignment

# After job completes, verify:
# 1. squeue shows job COMPLETED (not FAILED/CANCELLED)
# 2. This output file contains no Python tracebacks
# 3. eval_report.json exists under baseline_experiments/experiments/smoke/kobitski_ew06_alignment/<timestamp>/
# 4. python -c 'import torch; print(torch.cuda.is_available())' on compute node shows True
