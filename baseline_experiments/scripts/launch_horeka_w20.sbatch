#!/bin/bash
# window_size=20 split job — run AFTER launch_horeka.sbatch completes (or in parallel).
# Uses configs_horeka_w20/ which searches only window_size=[20].  Isolated from the
# main job because w=20+CPD trials are ~10x slower than w=5 and cause 4h timeout
# when mixed with w=5/10 trials in the same propulate population.
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_w20.sbatch
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

#SBATCH --job-name=zreg-baseline-w20
#SBATCH --partition=accelerated
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
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

# --clear-checkpoints is only passed when ZREG_CLEAR_CHECKPOINTS=1 (see header);
# otherwise unfinished HPO resumes from its Propulate checkpoints.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase selfcal \
    --configs-dir baseline_experiments/configs_horeka_w20 \
    ${CLEAR_FLAG[@]+"${CLEAR_FLAG[@]}"} \
    ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    --verbose

# ground_truth phase also needs w20 results; run it sequentially after selfcal.
# Both phases share the same MPI communicator so they must be separate srun calls.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase ground_truth \
    --configs-dir baseline_experiments/configs_horeka_w20 \
    ${CLEAR_FLAG[@]+"${CLEAR_FLAG[@]}"} \
    ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    --verbose
