#! /bin/bash

# One-time environment setup for HoreKa (KIT).
# Run this on the login node (not inside a job) before submitting launch_horeka.sbatch.
# Re-run whenever setup.cfg / pyproject.toml dependencies change.

set -e

WORKSPACE=/hkfs/work/workspace/scratch/<USER>-zreg
REPO="$WORKSPACE/zReg"
VENV="$WORKSPACE/regvenv"

module purge
module load compiler/gnu
module load mpi/openmpi
module load devel/python/3.10.5_gnu_12.1

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

# mpi4py must be built against the module-loaded OpenMPI (not a prebuilt wheel),
# otherwise multi-node srun communication over InfiniBand silently degrades or fails.
MPICC=$(which mpicc) pip install --no-binary mpi4py mpi4py

# propulate extra is what drives MPI-parallel HPO (auto-selected by
# optimizer.py's _detect_backend when SLURM_JOB_ID / MPI world_size > 1)
pip install -e "$REPO"[propulate]

echo "Setup complete. Venv: $VENV"
