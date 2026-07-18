#!/bin/bash
# Multi-rank validation test job for Phase 52 BUDG-04.
# Submit this BEFORE the full launch_horeka.sbatch to validate correct multi-rank
# behaviour on a short allocation (2 nodes x 2 ranks = 4 total, 30 min wall-clock).
#
# Passing criteria (D-13):
#   1. SLURM log shows rank IDs 0, 1, 2, and 3 (all 4 ranks participated).
#   2. best_params.json appears exactly once in the output directory (rank-0 wrote it, others did not).
#   3. No Python tracebacks in the log.
#
# Verification commands (run after job completes):
#   grep "rank" <slurm-output-file>   # should show rank 0, 1, 2, 3
#   find baseline_experiments/experiments/smoke/kobitski_ew06_alignment -name "best_params.json" | wc -l  # must be 1
#
# DO NOT SUBMIT until run_all.py has been made rank-aware (Phase 52 Plan 01, PARA-02).
# Fill in all [VERIFY ON HOREKA] values before submitting.

#SBATCH --job-name=zreg-multirank-test
#SBATCH --partition=accelerated           # [VERIFY ON HOREKA] — check HoreKa GPU partition name
#SBATCH --nodes=2                         # 2 nodes x 2 ranks = 4 total; small enough to schedule quickly
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=/home/hk-project-starter-p0024807/uitog/BA/zReg/baseline_experiments/logs/slurm-%j.out

module load devel/cuda/12.x              # [VERIFY ON HOREKA]
module load compiler/gnu                  # [VERIFY ON HOREKA]
module load mpi/openmpi                   # [VERIFY ON HOREKA]
# Python 3.9.21 is available by default — no module load needed

set -euo pipefail  # fail fast: broken module load, bad activate, or missing cd will abort job

source ~/regvenv_horeka/bin/activate

cd /home/hk-project-starter-p0024807/uitog/BA/zReg
mkdir -p baseline_experiments/logs       # ensure --output parent dir exists before SLURM writes to it

export KMP_DUPLICATE_LIB_OK=TRUE         # harmless on Linux; matches HOREKA_PLAN.md template

# --mpi=pspmix matches HoreKa's PMI implementation [VERIFY ON HOREKA]
# --phase selfcal runs only the selfcal HPO configs
# --configs-dir points at configs_horeka/smoke (single config: kobitski_ew06_alignment.yaml)
#
# NOTE on --configs-dir smoke vs full configs_horeka:
#   Option A (default, used here): --configs-dir baseline_experiments/configs_horeka/smoke
#     - Only kobitski_ew06_alignment.yaml is present in smoke/
#     - The other two selfcal configs (shah_alignment, shah_label_transfer) are absent
#     - run_all.py will raise FileNotFoundError for missing configs when --phase selfcal is used
#     - To avoid this, run_all.py must handle missing configs gracefully, OR use Option B
#   Option B (alternative): --configs-dir baseline_experiments/configs_horeka
#     - Uses the full configs_horeka tree (tier: dev, 20 trials each for all 3 selfcal configs)
#     - May exceed 30 min wall-clock; adjust --time accordingly
#
# tier: sanity in the smoke config limits to 5 fixed trials — job should finish in minutes
srun --mpi=pspmix --label \
  python -u baseline_experiments/scripts/run_all.py \
    --phase selfcal \
    --configs-dir baseline_experiments/configs_horeka/smoke \
    --verbose

# After job completes, verify passing criteria:
#   grep "rank" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out | head -20
#   find baseline_experiments/experiments/smoke/kobitski_ew06_alignment -name "best_params.json" | wc -l
#   grep -i "traceback\|error\|exception" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out | head -20

# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors
#   grep "Loaded source dataset on cuda" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out
