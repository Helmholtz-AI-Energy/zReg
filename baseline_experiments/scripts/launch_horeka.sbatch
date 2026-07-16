#!/bin/bash
# Phase 52+ multi-rank target template.
# DO NOT SUBMIT until run_all.py has been made rank-aware in Phase 52 (PARA-01/PARA-02).
# Fill in all [VERIFY ON HOREKA] values before submitting.
# All SLURM resource values assume HoreKa accelerated partition — verify node/rank
# counts against your allocation before submitting.

#SBATCH --job-name=zreg-baseline
#SBATCH --partition=accelerated           # [VERIFY ON HOREKA] — check HoreKa GPU partition name
#SBATCH --nodes=4                         # [VERIFY ON HOREKA] — 4 nodes x 4 ranks = 16 total; adjust to allocation
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/home/hk-project-starter-p0024807/uitog/BA/zReg/baseline_experiments/logs/slurm-%j.out

module load devel/cuda/12.x              # [VERIFY ON HOREKA]
module load compiler/gnu                  # [VERIFY ON HOREKA]
module load mpi/openmpi                   # [VERIFY ON HOREKA]
module load devel/python/3.12             # [VERIFY ON HOREKA]

set -euo pipefail  # fail fast: broken module load, bad activate, or missing cd will abort job

source ~/regvenv_horeka/bin/activate

cd /home/hk-project-starter-p0024807/uitog/BA/zReg
mkdir -p baseline_experiments/logs       # ensure --output parent dir exists before SLURM writes to it

export KMP_DUPLICATE_LIB_OK=TRUE         # harmless on Linux; matches HOREKA_PLAN.md template

# --mpi=pspmix matches HoreKa's PMI implementation [VERIFY ON HOREKA]
# search_strategy in configs_horeka/ must be 'auto' or 'propulate' for multi-rank HPO
# — see Phase 52 PARA-03
srun --mpi=pspmix --label \
  python -u baseline_experiments/scripts/run_all.py \
    --phase all \
    --configs-dir baseline_experiments/configs_horeka \
    --verbose

# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors
#   grep "Loaded source dataset on cuda" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out
