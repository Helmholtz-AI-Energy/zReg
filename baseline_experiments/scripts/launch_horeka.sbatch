#!/bin/bash
# Job 1 of 2: HPO phases (selfcal + ground_truth).
# Runs the 5 optimize+eval experiments that require multi-rank propulate parallelism.
# Run BEFORE launch_horeka_baseline.sbatch — baseline phases need best_params.json
# from both selfcal and ground_truth to be present.
#
# Runtime estimate (16 ranks, GPU-accelerated, step=8, max_points=1000, w=[5,10]):
# cpd_penalty fixed to "rigid" — all trials run CPD; no cheap null trials remain,
# so the lower bound rises vs. the previous [null,rigid,affine] search space.
#   selfcal/kobitski_ew06_alignment : 30–50 min  (20 dev trials, all rigid CPD, w=[5,10])
#   selfcal/shah_alignment          :  5– 8 min  (20 dev trials, all rigid CPD, Shah cheaper)
#   selfcal/shah_label_transfer     :  7–12 min  (20 dev trials, same + KNN-LT overhead)
#   ground_truth/kobitski_ew06      : 30–50 min  (same search space as selfcal kobitski)
#   ground_truth/shah_sample1       :  7–12 min  (same search space as selfcal shah)
#   -----------------------------------------------------------------------
#   Total                           : 79–132 min  (well within 4 h wall clock)
#
# Pairs formula: n*k - k*(k-1)/2, k=1+2*W, n=x_samples (46 for Kobitski, 11 for Shah
# after step=8 subsampling). w=20 is isolated in launch_horeka_w20.sbatch.
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka.sbatch
# Resume:  re-submit; _already_done() skips runs that have eval_report.json, unfinished
#          HPO resumes from Propulate checkpoints. Set ZREG_CLEAR_CHECKPOINTS=1 for one
#          submission to drop stale checkpoints (e.g. after a search-space change such as
#          Phase 63 D-08):  ZREG_CLEAR_CHECKPOINTS=1 sbatch <this script>
# Env:     ZREG_VENV      venv to activate (default: $HOME/regvenv_horeka)
#          ZREG_REPO_DIR  repo checkout to cd into
#                         (default: /home/hk-project-starter-p0024807/uitog/BA/zReg)

#SBATCH --job-name=zreg-hpo
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

source "$VENV/bin/activate"

cd "${REPO_DIR}"
mkdir -p baseline_experiments/logs

export KMP_DUPLICATE_LIB_OK=TRUE
# Limit CUDA allocator fragmentation — caps individual split size without capping total VRAM.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# selfcal phase: 3 HPO runs (kobitski alignment, shah alignment, shah label-transfer).
# --clear-checkpoints is only passed when ZREG_CLEAR_CHECKPOINTS=1 (see header);
# otherwise unfinished HPO resumes from its Propulate checkpoints.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase selfcal \
    --configs-dir baseline_experiments/configs_horeka \
    ${CLEAR_FLAG[@]+"${CLEAR_FLAG[@]}"} \
    --verbose

# ground_truth phase: 2 HPO runs (kobitski alignment, shah alignment+LT).
# Must be a separate srun call — each srun re-initialises the MPI communicator.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase ground_truth \
    --configs-dir baseline_experiments/configs_horeka \
    ${CLEAR_FLAG[@]+"${CLEAR_FLAG[@]}"} \
    --verbose
