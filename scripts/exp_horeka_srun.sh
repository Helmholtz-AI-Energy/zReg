#!/bin/bash
# Per-rank helper for HoreKa experiment runs. Sourced by each srun rank.
# Expects CONFIG and MODE to be set in the environment by the sbatch job.
# Adjust module names and python version to match the HoreKa module tree.

ZREG_WORKSPACE="/hkfs/work/workspace/scratch/${USER}-zreg"
ZREG_REPO="${ZREG_WORKSPACE}/zReg"
ZREG_VENV="${ZREG_WORKSPACE}/regvenv"

module purge
module load compiler/gnu
module load mpi/openmpi
module load devel/python/3.12_gnu_12.3   # adjust to available Python 3.12 module

source "${ZREG_VENV}/bin/activate"
echo "[rank ${SLURM_PROCID}] activated venv, launching python..."

cd "${ZREG_REPO}"
python -u run_eval.py \
    --config "${CONFIG}" \
    --mode "${MODE:-full}" \
    ${WARM_START_FROM:+--warm-start-from "${WARM_START_FROM}"}
