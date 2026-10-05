#!/bin/bash
# Run the zReg test suite on a JUPITER booster node (GH200), where the CUDA-gated tests that
# skip on CPU-only machines actually execute.
#
# Submitted through jsc-mpc from the synced checkout, e.g.
#     submit_job(cluster="jupiter", project="zreg", nodes=1, walltime="01:00:00",
#                script="scripts/jupiter/run_gpu_tests.sh")
#
# Requires a one-time scripts/jupiter/cluster_provision_jupiter.sh run on the login node.
# Extra pytest arguments can be passed through ZREG_PYTEST_ARGS.
#
# The real multi-rank MPI tests start their ranks with srun (tests/_mpi_launch.py), so the job
# needs at least two tasks; one task per GPU is requested here.
#SBATCH --ntasks-per-node=4
set -euo pipefail

ROOT="${ZREG_CLUSTER_ROOT:-/e/project1/tissuetwin/herold2/jsc-mpc-runs}"
ENV_SH="${ZREG_ENV_SH:-$ROOT/venvs/zreg-jupiter.env.sh}"
# shellcheck disable=SC1090
source "$ENV_SH"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible on this node"
print("CUDA ok:", torch.cuda.get_device_name(0), "| devices:", torch.cuda.device_count(),
      "| torch", torch.__version__)
PY

# Test runs write experiments/runs/<timestamp>/ dirs; keep them out of the synced work tree
# so the next jsc-mpc sync does not refuse a dirty checkout.
OUT="${SCRATCH:-/e/scratch/tissuetwin}/herold2/zreg-gpu-tests/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
# Compute nodes have no git, so copy the work tree minus .git. The jsc-mpc sync keeps that checkout
# clean, which makes this the exact synced commit.
SRC="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$SRC/tests" ] || { echo "ERROR: $SRC is not a zReg checkout (submit from its root)" >&2; exit 1; }
tar -C "$SRC" --exclude=.git -cf - . | tar -x -C "$OUT"
cd "$OUT"
export PYTHONPATH="$OUT/src:$OUT${PYTHONPATH:+:$PYTHONPATH}"   # APPEND: the module stack lives on PYTHONPATH

# -rs lists every skip reason, so the remaining CPU-only gaps are visible in the log.
# shellcheck disable=SC2086
python -m pytest tests baseline_experiments/tests -q -rs -o addopts="" -p no:cacheprovider \
    --junitxml="$OUT/junit.xml" ${ZREG_PYTEST_ARGS:-}
