#!/bin/bash
# Submit all Stage 1 alignment experiments on HoreKa.
# Each (method, tier) pair becomes a separate SLURM job.
# Tier 1 (synthetic)     → --mode full  (HPO + eval, 50 trials, 4 nodes, 4h)
# Tier 2 (semisynthetic) → --mode full  (HPO + eval, 20 trials, 2 nodes, 2h)
# Tier 3 (real)          → --mode eval  (eval only with default_params, 1 node, 1h)
#
# Usage: bash scripts/run_stage1_horeka.sh
# Run from repo root on HoreKa login node.
#
# NOTE: constrained_nonrigid_cpd requires VALID_CPD extension before submitting.
# See configs/experiments/stage1_alignment/constrained_nonrigid_cpd/synthetic.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH="${SCRIPT_DIR}/exp_horeka.sbatch"
CFG="${REPO_ROOT}/configs/experiments/stage1_alignment"
LOG="${REPO_ROOT}/experiments/runs/logs"
mkdir -p "${LOG}"

submit() {
    local cfg="$1" mode="$2" nodes="$3" time="$4"
    local name="s1-$(basename "$(dirname "$cfg")")-$(basename "$cfg" .yaml)"
    local jid
    jid=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes="${nodes}" \
        --time="${time}" \
        --export=ALL,CONFIG="${cfg}",MODE="${mode}" \
        "${SBATCH}")
    echo "$(date -Iseconds) submitted ${name} → job ${jid}" | tee -a "${LOG}/stage1_horeka.log"
}

echo "=== Stage 1 Alignment — HoreKa submissions ==="

# --- Tier 1: Fully synthetic (50 trials, full HPO) ---
submit "${CFG}/rigid_cpd/synthetic.yaml"                full 4 4:00:00
submit "${CFG}/affine_cpd/synthetic.yaml"               full 4 4:00:00
submit "${CFG}/nonrigid_cpd/synthetic.yaml"             full 4 4:00:00
# submit "${CFG}/constrained_nonrigid_cpd/synthetic.yaml" full 4 4:00:00  # needs VALID_CPD extension

# --- Tier 2: Semi-synthetic (20 trials, dev HPO) ---
submit "${CFG}/rigid_cpd/semisynthetic.yaml"                full 2 2:00:00
submit "${CFG}/affine_cpd/semisynthetic.yaml"               full 2 2:00:00
submit "${CFG}/nonrigid_cpd/semisynthetic.yaml"             full 2 2:00:00
# submit "${CFG}/constrained_nonrigid_cpd/semisynthetic.yaml" full 2 2:00:00

# --- Tier 3: Real data (eval only, use best params from synthetic HPO) ---
submit "${CFG}/rigid_cpd/real.yaml"                eval 1 1:00:00
submit "${CFG}/affine_cpd/real.yaml"               eval 1 1:00:00
submit "${CFG}/nonrigid_cpd/real.yaml"             eval 1 1:00:00
# submit "${CFG}/constrained_nonrigid_cpd/real.yaml" eval 1 1:00:00

echo "All Stage 1 jobs submitted. See ${LOG}/stage1_horeka.log for job IDs."
