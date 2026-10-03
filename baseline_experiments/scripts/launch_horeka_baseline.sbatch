#!/bin/bash
# Job 2 of 2: eval-only baseline phases.
# Runs three single-pass evaluations using params from the HPO phases.
# Submit AFTER launch_horeka.sbatch has completed successfully — this job reads
# best_params.json from selfcal/ and ground_truth/ output directories.
#
# Resources: 4 nodes x 4 GPUs (16 ranks). baseline_no_hpo runs eval-only as a single
# process on the batch host; baseline_with_combined runs Propulate HPO on all 16 ranks
# via srun.
#
# Runtime estimate (GPU-accelerated, step=8, max_points=1000):
#   baseline_no_hpo/ew06_vs_shah        :  10–25 min  (1 rank, eval-only, rigid CPD, ~756 pairs)
#   baseline_with_combined/ew06_vs_shah : 30–70 min  (16 ranks, 20 dev HPO trials, w=[5,10])
#   -----------------------------------------------------------------------
#   Total                               : 40–95 min  (well within 4 h wall clock)
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_baseline.sbatch
# Resume:  re-submit; _already_done() skips runs that have eval_report.json, unfinished
#          HPO resumes from Propulate checkpoints. Set ZREG_CLEAR_CHECKPOINTS=1 for one
#          submission to drop stale checkpoints (e.g. after a search-space change such as
#          Phase 63 D-08):  ZREG_CLEAR_CHECKPOINTS=1 sbatch <this script>
#          run_all.py records the search space next to the checkpoints and stops with an
#          error naming ZREG_CLEAR_CHECKPOINTS=1 if they were written under another one.
# Redo:    ZREG_FORCE=1 sbatch <this script> passes --force: every run of this job is
#          redone even if its eval_report.json exists. To redo one run only, delete that
#          run's eval_report.json instead (the error for a stale best_params.json names it).
# Env:     ZREG_VENV      venv to activate (default: $HOME/regvenv_horeka)
#          ZREG_REPO_DIR  repo checkout to cd into
#                         (default: /home/hk-project-starter-p0024807/uitog/BA/zReg)

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

VENV="${ZREG_VENV:-$HOME/regvenv_horeka}"
REPO_DIR="${ZREG_REPO_DIR:-/home/hk-project-starter-p0024807/uitog/BA/zReg}"

# Checkpoint clearing is opt-in: a plain re-submit resumes HPO from Propulate checkpoints.
CLEAR_FLAG=()
if [[ "${ZREG_CLEAR_CHECKPOINTS:-0}" == "1" ]]; then
  CLEAR_FLAG=(--clear-checkpoints)
fi
# Re-running completed runs is opt-in too (63-REVIEW WR-04).
FORCE_FLAG=()
if [[ "${ZREG_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

source "$VENV/bin/activate"

cd "${REPO_DIR}"
mkdir -p baseline_experiments/logs

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# baseline_no_hpo: eval with config default_params (no HPO input needed).
python -u baseline_experiments/scripts/run_all.py \
  --phase baseline_no_hpo \
  --configs-dir baseline_experiments/configs_horeka \
  ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
  --verbose

# baseline_with_combined: HPO on the real cross-embryo task, warm-started from the
# merged selfcal+ground_truth params. Requires both selfcal and ground_truth phases
# to have produced best_params.json. Uses all 16 ranks via propulate.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase baseline_with_combined \
    --configs-dir baseline_experiments/configs_horeka \
    ${CLEAR_FLAG[@]+"${CLEAR_FLAG[@]}"} \
    ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    --verbose
