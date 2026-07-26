#!/bin/bash
# Per-rank helper for JUWELS experiment runs. Sourced by each srun rank.
# Expects CONFIG and MODE to be set in the environment by the sbatch job.

ZREG_PROJECT="/p/project1/tissuetwin/${USER}"
ZREG_REPO="${ZREG_PROJECT}/zReg"
ZREG_VENV="${ZREG_PROJECT}/regvenv"

source "${ZREG_VENV}/bin/activate"
echo "[rank ${SLURM_PROCID}] activated venv, launching python..."

cd "${ZREG_REPO}"
python -u run_eval.py \
    --config "${CONFIG}" \
    --mode "${MODE:-full}"
