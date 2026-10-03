#!/bin/bash
# Generic HoreKa sbatch template for zReg experiments.
# Reads CONFIG and MODE from the environment (set via --export in orchestrator scripts).
# Default partition: accelerated (GPU). Override with: sbatch --partition=<name> ...
# Default resource profile is sized for propulate HPO (search_strategy: auto).
# Eval-only submissions get --ntasks-per-node=1 --gres=gpu:1 (plus smaller
# --nodes / --time) from the orchestrators; the log directory is created by the
# orchestrators (ZREG_SLURM_LOG_DIR) and passed as --output on the command line.
#
# Required env vars (set by orchestrator via --export=ALL,...):
#   CONFIG  — path to YAML config relative to repo root
#   MODE    — run_eval.py mode: full | optimize | eval
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --job-name=zreg-exp
#SBATCH --partition=accelerated
#SBATCH --output=/hkfs/work/workspace/scratch/%u-zreg/logs/slurm/%x-%j.out

srun --mpi=pmi2 --label bash /hkfs/work/workspace/scratch/${USER}-zreg/zReg/scripts/exp_horeka_srun.sh
