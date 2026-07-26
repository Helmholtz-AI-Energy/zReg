#!/bin/bash
# Generic JUWELS sbatch template for zReg experiments.
# Reads CONFIG and MODE from the environment (set via --export in orchestrator scripts).
# Default partition: booster (GPU). Override via sbatch command line if needed.
# Default resource profile is sized for propulate HPO (search_strategy: auto).
#
# Required env vars (set by orchestrator via --export=ALL,...):
#   CONFIG  — path to YAML config relative to repo root
#   MODE    — run_eval.py mode: full | optimize | eval
#
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=zreg-exp
#SBATCH --partition=booster
#SBATCH --account=tissuetwin
#SBATCH --output=/p/project1/tissuetwin/%u/logs/slurm/%x-%j.out

srun --mpi=pspmix --label bash /p/project1/tissuetwin/${USER}/zReg/scripts/exp_juwels_srun.sh
