#!/bin/bash
# Submit all Stage 2 label transfer experiments on HoreKa.
# Each (method, tier) pair becomes a separate SLURM job.
# Tier 1 (synthetic)     → --mode full  (HPO + eval, 50 trials, 4 nodes, 4h)
# Tier 2 (semisynthetic) → --mode full  (HPO + eval, 20 trials, 2 nodes, 2h)
# Tier 3 (real)          → --mode eval  (eval only with default_params, 1 node, 1h)
#
# Usage: bash scripts/run_stage2_horeka.sh
# Run from repo root on HoreKa login node.
# Run AFTER Stage 1 completes and update default_params in real tier configs
# with the best alignment params found in Stage 1.
#
# NOTE: Set egnn_checkpoint_path / pointnet2_checkpoint_path in the egnn/pointnet2
# configs before submitting those jobs.
# NOTE: hybrid_knn_cpd uses knn_voting as placeholder until hybrid is implemented.
#
# Eval-mode submissions request a single rank (--ntasks-per-node=1 --gres=gpu:1);
# HPO (full/optimize) keeps the sbatch template's multi-rank default.
# SLURM logs go to ZREG_SLURM_LOG_DIR
# (default: /hkfs/work/workspace/scratch/${USER}-zreg/logs/slurm),
# created before the first submission and passed as --output=<dir>/%x-%j.out.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH="${SCRIPT_DIR}/exp_horeka.sbatch"
CFG="${REPO_ROOT}/configs/experiments/stage2_label_transfer"
LOG="${REPO_ROOT}/experiments/runs/logs"
mkdir -p "${LOG}"
SLURM_LOG_DIR="${ZREG_SLURM_LOG_DIR:-/hkfs/work/workspace/scratch/${USER}-zreg/logs/slurm}"
mkdir -p "${SLURM_LOG_DIR}"

submit() {
    local cfg="$1" mode="$2" nodes="$3" time="$4"
    local name="s2-$(basename "$(dirname "$cfg")")-$(basename "$cfg" .yaml)"
    local res_flags=()
    if [[ "$mode" == "eval" ]]; then
        res_flags=(--ntasks-per-node=1 --gres=gpu:1)
    fi

    local jid
    jid=$(sbatch --parsable \
        ${res_flags[@]+"${res_flags[@]}"} \
        --output="${SLURM_LOG_DIR}/%x-%j.out" \
        --job-name="${name}" \
        --nodes="${nodes}" \
        --time="${time}" \
        --export=ALL,CONFIG="${cfg}",MODE="${mode}" \
        "${SBATCH}")
    echo "$(date -Iseconds) submitted ${name} → job ${jid}" | tee -a "${LOG}/stage2_horeka.log"
}

echo "=== Stage 2 Label Transfer — HoreKa submissions ==="

# --- Tier 1: Fully synthetic (50 trials, full HPO) ---
submit "${CFG}/knn/synthetic.yaml"           full 4 4:00:00
submit "${CFG}/cpd_weighted/synthetic.yaml"  full 4 4:00:00
submit "${CFG}/egnn/synthetic.yaml"          full 4 4:00:00
submit "${CFG}/pointnet2/synthetic.yaml"     full 4 4:00:00
submit "${CFG}/hybrid_knn_cpd/synthetic.yaml" full 4 4:00:00

# --- Tier 2: Semi-synthetic (20 trials, dev HPO) ---
submit "${CFG}/knn/semisynthetic.yaml"           full 2 2:00:00
submit "${CFG}/cpd_weighted/semisynthetic.yaml"  full 2 2:00:00
submit "${CFG}/egnn/semisynthetic.yaml"          full 2 2:00:00
submit "${CFG}/pointnet2/semisynthetic.yaml"     full 2 2:00:00
submit "${CFG}/hybrid_knn_cpd/semisynthetic.yaml" full 2 2:00:00

# --- Tier 3: Real data (eval only) ---
submit "${CFG}/knn/real.yaml"           eval 1 1:00:00
submit "${CFG}/cpd_weighted/real.yaml"  eval 1 1:00:00
submit "${CFG}/egnn/real.yaml"          eval 1 1:00:00
submit "${CFG}/pointnet2/real.yaml"     eval 1 1:00:00
submit "${CFG}/hybrid_knn_cpd/real.yaml" eval 1 1:00:00

echo "All Stage 2 jobs submitted. See ${LOG}/stage2_horeka.log for job IDs."
