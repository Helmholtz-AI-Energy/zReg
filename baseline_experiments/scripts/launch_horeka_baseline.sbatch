#!/bin/bash
# Job 2 of 2: eval-only baseline phases.
# Runs three single-pass evaluations using params from the HPO phases.
# Submit AFTER launch_horeka.sbatch has completed successfully — this job reads
# best_params.json from selfcal/ and ground_truth/ output directories.
#
# No propulate/MPI needed: eval-only passes run on rank 0 only. Single node,
# single GPU keeps resource usage minimal.
#
# Runtime estimate (GPU-accelerated, step=8, max_points=1000):
#   baseline_no_hpo/ew06_vs_shah        :  10–25 min  (1 rank, eval-only, rigid CPD, ~756 pairs)
#   baseline_with_combined/ew06_vs_shah : 30–70 min  (16 ranks, 20 dev HPO trials, w=[5,10])
#   -----------------------------------------------------------------------
#   Total                               : 40–95 min  (well within 4 h wall clock)
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_baseline.sbatch
# Resume:  re-submit; _already_done() skips completed runs.

#SBATCH --job-name=zreg-baseline-eval
#SBATCH --partition=accelerated
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=120G
#SBATCH --output=/home/hk-project-starter-p0024807/uitog/BA/zReg/baseline_experiments/logs/slurm-%j.out

module load devel/cuda/12.4
module load compiler/gnu
module load mpi/openmpi

set -euo pipefail

VENV=~/regvenv_horeka
source "$VENV/bin/activate"

cd /home/hk-project-starter-p0024807/uitog/BA/zReg
mkdir -p baseline_experiments/logs

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# baseline_no_hpo: eval with config default_params (no HPO input needed).
python -u baseline_experiments/scripts/run_all.py \
  --phase baseline_no_hpo \
  --configs-dir baseline_experiments/configs_horeka \
  --verbose

# baseline_with_combined: HPO on the real cross-embryo task, warm-started from the
# merged selfcal+ground_truth params. Requires both selfcal and ground_truth phases
# to have produced best_params.json. Uses all 16 ranks via propulate.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase baseline_with_combined \
    --configs-dir baseline_experiments/configs_horeka \
    --clear-checkpoints \
    --verbose
