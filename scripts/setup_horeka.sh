#!/bin/bash
# setup_horeka.sh — one-time environment setup on HoreKa login node.
# Run ONCE from inside the workspace repo checkout (/hkfs/work/workspace/NAME/zReg).
# All lines marked [VERIFY ON HOREKA] must be checked against HoreKa's module tree
# before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Load modules
# ---------------------------------------------------------------------------
module load devel/cuda/12.x              # [VERIFY ON HOREKA] — check exact CUDA version available
module load compiler/gnu                  # [VERIFY ON HOREKA] — check gcc version required by mpi4py build
module load mpi/openmpi                   # [VERIFY ON HOREKA] — check exact OpenMPI module name
module load devel/python/3.12             # [VERIFY ON HOREKA] — check exact Python 3.12 module name

# ---------------------------------------------------------------------------
# 2. Create virtual environment (matches JUWELS convention regvenv311)
# ---------------------------------------------------------------------------
python3 -m venv ~/regvenv_horeka
source ~/regvenv_horeka/bin/activate

# ---------------------------------------------------------------------------
# 3. Install zReg with propulate extra
#    propulate>=1.0,<2 + mpi4py>=3.1 per setup.cfg [options.extras_require]
# ---------------------------------------------------------------------------
pip install -e ".[propulate]"

# ---------------------------------------------------------------------------
# 4. Rebuild mpi4py against HoreKa's system MPI
#    The pip wheel bundles its own MPICH, which is incompatible with srun/pspmix.
#    Source build against the loaded mpi/openmpi module is mandatory for srun jobs.
# ---------------------------------------------------------------------------
pip install mpi4py --no-binary mpi4py --force-reinstall

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
python -c "from mpi4py import MPI; print('mpi4py OK, rank', MPI.COMM_WORLD.Get_rank())"

echo "=== torch CUDA sanity ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

echo "Setup complete."
