#!/bin/bash
# HPO on the cross-embryo paired task: Kobitski ew_06 → Shah sample-1.
# Runs alignment + kNN label transfer HPO directly on real paired data (no synthetic
# perturbation). Optimizer scores trials on proxy metrics + Shah label accuracy.
#
# Uses run_eval.py --mode full (not run_all.py) — a single srun call, no phase
# orchestration needed. Runs HPO then a final eval pass with best_params on rank 0.
# best_params.json and eval_report.json are written to a timestamped subdirectory of
# baseline_experiments/experiments/hpo_paired/ew06_vs_shah/ by rank 0.
#
# Runtime estimate (8 ranks / 2 nodes, GPU-accelerated, step=8, max_points=1000):
#   Paired Kobitski→Shah, 20 dev trials, rigid CPD, window=[5,10]: ~30–90 min
#   (paired runtime not benchmarked yet — wall clock is conservative)
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_hpo_paired.sbatch
# Resume:  re-submit; propulate resumes from checkpoint if output_dir/<timestamp>/
#          checkpoint files exist. Use a fresh output_dir or delete checkpoints to restart.

#SBATCH --job-name=zreg-hpo-paired
#SBATCH --partition=accelerated
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
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

srun --mpi=pmix --label \
  "$VENV/bin/python" -u run_eval.py \
    --config baseline_experiments/configs_horeka/hpo_paired/ew06_vs_shah.yaml \
    --mode full \
    --verbose
