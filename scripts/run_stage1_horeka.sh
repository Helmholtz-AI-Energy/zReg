#!/bin/bash
# Submit all Stage 1 alignment experiments on HoreKa.
# Each method runs as a three-job chain: synthetic → semisynthetic → real.
#
#   synthetic     --mode full  (HPO + eval, dev tier: 25 trials, 2 nodes, 2h)
#   semisynthetic --mode full  (HPO + eval, dev tier: 25 trials, 2 nodes, 2h)
#                 depends on synthetic; seeded with synthetic best_params
#   real          --mode eval  (eval only, 1 node, 1h)
#                 depends on semisynthetic; uses semisynthetic best_params
#
# Best params are forwarded via --warm-start-from (EXT-04): each downstream job
# receives the base output_dir of its predecessor; _load_warm_start() in
# run_eval.py finds the most-recent timestamped subdir automatically.
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
RUNS="${REPO_ROOT}/experiments/runs/stage1_alignment"
LOG="${REPO_ROOT}/experiments/runs/logs"
mkdir -p "${LOG}"

# submit cfg mode nodes time [dep_jid] [warm_start_from]
# Prints job ID to stdout; logs human-readable line to stderr + log file.
submit() {
    local cfg="$1" mode="$2" nodes="$3" time="$4"
    local dep="${5:-}" warm="${6:-}"
    local name="s1-$(basename "$(dirname "$cfg")")-$(basename "$cfg" .yaml)"

    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:${dep}"

    local export_str="ALL,CONFIG=${cfg},MODE=${mode}"
    [[ -n "$warm" ]] && export_str="${export_str},WARM_START_FROM=${warm}"

    local jid
    jid=$(sbatch --parsable \
        ${dep_flag:+"$dep_flag"} \
        --job-name="${name}" \
        --nodes="${nodes}" \
        --time="${time}" \
        --export="${export_str}" \
        "${SBATCH}")
    echo "$(date -Iseconds) submitted ${name} → job ${jid}${dep:+ (after ${dep})}" \
        | tee -a "${LOG}/stage1_horeka.log" >&2
    echo "${jid}"
}

echo "=== Stage 1 Alignment — HoreKa submissions ===" >&2

# ---------------------------------------------------------------------------
# rigid_cpd
# ---------------------------------------------------------------------------
jid=$(submit "${CFG}/rigid_cpd/synthetic.yaml"     full 2 2:00:00)
jid=$(submit "${CFG}/rigid_cpd/semisynthetic.yaml" full 2 2:00:00 \
          "$jid" "${RUNS}/rigid_cpd/synthetic")
submit      "${CFG}/rigid_cpd/real.yaml"           eval 1 1:00:00 \
          "$jid" "${RUNS}/rigid_cpd/semisynthetic"

# ---------------------------------------------------------------------------
# affine_cpd
# ---------------------------------------------------------------------------
jid=$(submit "${CFG}/affine_cpd/synthetic.yaml"     full 2 2:00:00)
jid=$(submit "${CFG}/affine_cpd/semisynthetic.yaml" full 2 2:00:00 \
          "$jid" "${RUNS}/affine_cpd/synthetic")
submit      "${CFG}/affine_cpd/real.yaml"           eval 1 1:00:00 \
          "$jid" "${RUNS}/affine_cpd/semisynthetic"

# ---------------------------------------------------------------------------
# nonrigid_cpd
# ---------------------------------------------------------------------------
jid=$(submit "${CFG}/nonrigid_cpd/synthetic.yaml"     full 2 2:00:00)
jid=$(submit "${CFG}/nonrigid_cpd/semisynthetic.yaml" full 2 2:00:00 \
          "$jid" "${RUNS}/nonrigid_cpd/synthetic")
submit      "${CFG}/nonrigid_cpd/real.yaml"           eval 1 1:00:00 \
          "$jid" "${RUNS}/nonrigid_cpd/semisynthetic"

# submit "${CFG}/constrained_nonrigid_cpd/synthetic.yaml" full 2 2:00:00  # needs VALID_CPD extension

echo "All Stage 1 chains submitted. See ${LOG}/stage1_horeka.log for job IDs." >&2
