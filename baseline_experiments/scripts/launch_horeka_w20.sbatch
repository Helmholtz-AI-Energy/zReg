#!/bin/bash
# window_size=20 split job — run AFTER launch_horeka.sbatch completes (or in parallel).
# Uses configs_horeka_w20/ which searches only window_size=[20].  Isolated from the
# main job because w=20+CPD trials are ~10x slower than w=5 and cause 4h timeout
# when mixed with w=5/10 trials in the same propulate population.
#
# Submit:  sbatch baseline_experiments/scripts/launch_horeka_w20.sbatch
# Resume:  re-submit with same command; propulate will resume from its checkpoint.
#          Use --clear-checkpoints if the checkpoint is stale (pre-fix run).

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

VENV=~/regvenv_horeka
source "$VENV/bin/activate"

cd /home/hk-project-starter-p0024807/uitog/BA/zReg
mkdir -p baseline_experiments/logs

export KMP_DUPLICATE_LIB_OK=TRUE

# --clear-checkpoints removes stale *.pkl files before each HPO phase so
# propulate does not resume from a checkpoint with old float-index encoding.
# Remove the flag once the w20 job has run successfully at least once.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase selfcal \
    --configs-dir baseline_experiments/configs_horeka_w20 \
    --clear-checkpoints \
    --verbose

# ground_truth phase also needs w20 results; run it sequentially after selfcal.
# Both phases share the same MPI communicator so they must be separate srun calls.
srun --mpi=pmix --label \
  "$VENV/bin/python" -u baseline_experiments/scripts/run_all.py \
    --phase ground_truth \
    --configs-dir baseline_experiments/configs_horeka_w20 \
    --clear-checkpoints \
    --verbose
