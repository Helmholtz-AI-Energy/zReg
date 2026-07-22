#!/bin/bash
# Job 2 of 2: eval-only baseline phases.
# Runs three single-pass evaluations using params from the HPO phases.
# Submit AFTER launch_horeka.sbatch has completed successfully — this job reads
# best_params.json from selfcal/ and ground_truth/ output directories.
#
# No propulate/MPI needed: eval-only passes run on rank 0 only. Single node,
# single GPU keeps resource usage minimal.
#
# Runtime estimate (1 rank, GPU-accelerated, step=8, max_points=1000, w=10):
#   baseline_no_hpo/ew06_vs_shah          :  5–15 min  (1 eval pass, ~756 pairs)
#   baseline_with_selfcal/ew06_vs_shah    :  5–15 min  (1 eval pass, ~756 pairs)
#   baseline_with_groundtruth/ew06_vs_shah:  5–15 min  (1 eval pass, ~756 pairs)
#   -----------------------------------------------------------------------
#   Total                                 : 15–45 min  (well within 1 h wall clock)
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_baseline.sbatch
# Resume:  re-submit; _already_done() skips completed runs.

#SBATCH --job-name=zreg-baseline-eval
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
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

# baseline_with_selfcal: eval with params merged from selfcal best_params.json.
# Requires selfcal phase to have produced best_params.json in all three selfcal dirs.
python -u baseline_experiments/scripts/run_all.py \
  --phase baseline_with_selfcal \
  --configs-dir baseline_experiments/configs_horeka \
  --verbose

# baseline_with_groundtruth: eval with params merged from ground_truth best_params.json.
# Requires ground_truth phase to have produced best_params.json in both ground_truth dirs.
python -u baseline_experiments/scripts/run_all.py \
  --phase baseline_with_groundtruth \
  --configs-dir baseline_experiments/configs_horeka \
  --verbose
