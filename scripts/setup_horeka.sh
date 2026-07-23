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
# 4. Install PyTorch for CUDA 12.4
#    torch is not a zReg extra — install explicitly against the loaded CUDA version.
# ---------------------------------------------------------------------------
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 5. Rebuild mpi4py against HoreKa's system MPI
#    The pip wheel bundles its own MPICH, which is incompatible with srun/pspmix.
#    Source build against the loaded mpi/openmpi module is mandatory for srun jobs.
#    --no-cache-dir forces a fresh build every run so a stale cached wheel from a
#    previous MPI module version is never reused.
# ---------------------------------------------------------------------------
export MPICC=$(which mpicc)  # ensure setuptools picks up the loaded MPI compiler, not system cc
"$VENV/bin/pip" install mpi4py --no-binary mpi4py --no-cache-dir --force-reinstall

# ---------------------------------------------------------------------------
# 6. Data transfer — run these commands BEFORE submitting any SLURM job.
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
# 7. Verification (run on login node; torch.cuda.is_available() will be False
#    here — no GPU on login node. The real CUDA check runs on a compute node.)
# ---------------------------------------------------------------------------
echo "=== mpi4py sanity ==="
# Verify installation location (avoids import, which requires libmpi.so in LD_LIBRARY_PATH).
"$VENV/bin/pip" show mpi4py
# Confirm the .so links against the system OpenMPI loaded above, not bundled MPICH.
MPI_SO=$(find "$VENV" -name "MPI*.so" -path "*/mpi4py/*" 2>/dev/null | head -1)
if [[ -z "$MPI_SO" ]]; then
    echo "ERROR: mpi4py .so not found under $VENV — install failed"
    exit 1
fi
echo "mpi4py .so: $MPI_SO"
ldd "$MPI_SO" | grep -i libmpi || { echo "WARNING: no libmpi found in ldd — may be linked against wrong MPI"; }

echo "=== torch sanity ==="
# Confirm torch is installed in the venv.
"$VENV/bin/pip" show torch
# Read version.py directly — avoids importing torch._C which requires CUDA libs at runtime.
TORCH_VERSION_PY=$(find "$VENV" -maxdepth 8 -name "version.py" -path "*/torch/version.py" 2>/dev/null | head -1)
if [[ -z "$TORCH_VERSION_PY" ]]; then
    echo "ERROR: torch/version.py not found under $VENV — install failed"
    exit 1
fi
echo "torch version.py: $TORCH_VERSION_PY"
grep -E "^__version__|^cuda" "$TORCH_VERSION_PY"

mkdir -p baseline_experiments/logs  # required before SLURM can write --output files

echo "Setup complete."
