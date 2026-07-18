#!/bin/bash
# setup_horeka.sh — one-time environment setup on HoreKa login node.
# Run ONCE from inside the workspace repo checkout (/hkfs/work/workspace/NAME/zReg).
# All lines marked [VERIFY ON HOREKA] must be checked against HoreKa's module tree
# before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Load modules
# ---------------------------------------------------------------------------
module load devel/cuda/12.4
module load compiler/gnu                  # [VERIFY ON HOREKA] — check gcc version required by mpi4py build
module load mpi/openmpi                   # [VERIFY ON HOREKA] — check exact OpenMPI module name
# python3.9 is the default; python3.12 is available directly without module load

# ---------------------------------------------------------------------------
# 2. Create virtual environment (matches JUWELS convention regvenv311)
#    All pip/python calls below use VENV explicitly — module loads on HPC
#    can corrupt PATH so "source activate" alone is not reliable.
# ---------------------------------------------------------------------------
VENV=~/regvenv_horeka
python3.12 -m venv "$VENV"
source "$VENV/bin/activate"
"$VENV/bin/pip" install --upgrade "pip==26.0.1"

# ---------------------------------------------------------------------------
# 3. Install zReg with propulate extra
#    propulate>=1.0,<2 + mpi4py>=3.1 per setup.cfg [options.extras_require]
# ---------------------------------------------------------------------------
"$VENV/bin/pip" install -e ".[propulate]"

# ---------------------------------------------------------------------------
# 4. Rebuild mpi4py against HoreKa's system MPI
#    The pip wheel bundles its own MPICH, which is incompatible with srun/pspmix.
#    Source build against the loaded mpi/openmpi module is mandatory for srun jobs.
#    --no-cache-dir forces a fresh build every run so a stale cached wheel from a
#    previous MPI module version is never reused.
# ---------------------------------------------------------------------------
export MPICC=$(which mpicc)  # ensure setuptools picks up the loaded MPI compiler, not system cc
"$VENV/bin/pip" install mpi4py --no-binary mpi4py --no-cache-dir --force-reinstall

# ---------------------------------------------------------------------------
# 5. Data transfer — run these commands BEFORE submitting any SLURM job.
#
#    On your LOCAL MACHINE (laptop), transfer real datasets to HoreKa workspace:
#
#      rsync -avz data/external/sample/ \
#        <user>@horeka.scc.kit.edu:/hkfs/work/workspace/<name>/zReg-data/sample/
#
#    [VERIFY ON HOREKA] — substitute <user> and <name> with your HoreKa username
#    and workspace name (created via: ws_allocate zreg-data <days>).
#
#    Then, ON HORKEA (this login node), create an absolute symlink inside the
#    repo checkout so all configs resolve data_path without any edits:
#
#      ln -s /hkfs/work/workspace/<name>/zReg-data/sample \
#             /hkfs/work/workspace/<name>/zReg/data/external/sample
#
#    [VERIFY ON HOREKA] — substitute <name> with your workspace name.
#    Use an ABSOLUTE symlink (not relative) — relative symlinks break when srun
#    changes working directory on the compute node.
# ---------------------------------------------------------------------------
echo "=== Data transfer instructions above — complete before submitting any job ==="

# ---------------------------------------------------------------------------
# 6. Verification (run on login node to confirm setup; GPU check is informational
#    only here — torch.cuda will return False on login node without a GPU;
#    the real CUDA check must be done on a compute node via srun)
# ---------------------------------------------------------------------------
echo "=== mpi4py sanity ==="
"$VENV/bin/python" -c "from mpi4py import MPI; print('mpi4py OK, rank', MPI.COMM_WORLD.Get_rank())"

echo "=== torch CUDA sanity ==="
"$VENV/bin/python" -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

mkdir -p baseline_experiments/logs  # required before SLURM can write --output files

echo "Setup complete."
