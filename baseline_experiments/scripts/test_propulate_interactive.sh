#!/bin/bash
# Test propulate search in an interactive cluster session.
#
# Prerequisites:
#   - Active salloc with at least N_RANKS tasks allocated
#   - regvenv_horeka activated (or let the script use the full venv path)
#   - Run from repo root: bash baseline_experiments/scripts/test_propulate_interactive.sh
#
# Usage:
#   bash baseline_experiments/scripts/test_propulate_interactive.sh [N_RANKS]
#   N_RANKS defaults to 4.
#
# What it tests:
#   Test 1 — clean propulate run: verifies best_params.json written by rank 0,
#             checkpoint files present, no crash.
#   Test 2 — stale checkpoint resume: re-runs into the same checkpoint dir with
#             a narrowed search space (null removed from cpd_penalty). Verifies
#             stale individuals are skipped without crashing (fix in
#             eval/search_strategies.py PropulateSearch.search()).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=~/regvenv_horeka
N_RANKS="${1:-4}"
SMOKE_CONFIG="${REPO_ROOT}/baseline_experiments/configs_horeka/smoke/selfcal/kobitski_ew06_alignment.yaml"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
TEST_OUT="${REPO_ROOT}/baseline_experiments/experiments/propulate_test_${RUN_ID}"
NARROWED_CONFIG="${TEST_OUT}/narrowed.yaml"
HELPER_SCRIPT="${TEST_OUT}/run_direct.py"
PASS=0
FAIL=0

log()  { echo "[propulate-test] $*"; }
pass() { echo "  [PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

mkdir -p "${TEST_OUT}"
log "Repo root   : ${REPO_ROOT}"
log "Output root : ${TEST_OUT}"
log "MPI ranks   : ${N_RANKS}"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Narrowed config: remove null from cpd_penalty search space.
# Checkpoint individuals written with null will be undecodable against this
# narrowed space, triggering the stale-skip fix in PropulateSearch.search().
# Only the first cpd_penalty line (under search_space:) is edited;
# the default_params cpd_penalty: null line is left as-is.
# ---------------------------------------------------------------------------
sed 's/cpd_penalty: \[null, /cpd_penalty: [/' "${SMOKE_CONFIG}" > "${NARROWED_CONFIG}"
log "narrowed.yaml cpd_penalty: $(grep 'cpd_penalty:' "${NARROWED_CONFIG}" | head -1 | xargs)"

# ---------------------------------------------------------------------------
# Helper script: calls HyperparamOptimizer directly with an explicit output_dir.
# Bypasses run_eval.py's timestamp-subdir logic so propulate resumes from the
# exact checkpoint files written in Test 1.
# ---------------------------------------------------------------------------
cat > "${HELPER_SCRIPT}" << 'PYEOF'
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

repo_root = os.environ["REPO_ROOT"]
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))

from eval.config import EvalConfig
from eval.runners import HyperparamOptimizer

config_path, output_dir = sys.argv[1], sys.argv[2]
config = EvalConfig.from_yaml(config_path)
config = config.model_copy(update={"output_dir": output_dir})
HyperparamOptimizer(config).run()
PYEOF

export REPO_ROOT  # propagated to compute ranks via srun --export=ALL

# ---------------------------------------------------------------------------
# TEST 1: Clean propulate run via run_eval.py
# ---------------------------------------------------------------------------
log ""
log "=== TEST 1: clean propulate run ==="

srun --mpi=pmix -n "${N_RANKS}" --label \
  "${VENV}/bin/python" -u run_eval.py \
  --config "${SMOKE_CONFIG}" \
  --mode optimize \
  --output-dir "${TEST_OUT}/clean" \
  --verbose 2>&1 | tee "${TEST_OUT}/test1.log"

# run_eval.py appends a timestamp subdir — find it via best_params.json
TSTAMP_DIR="$(find "${TEST_OUT}/clean" -name "best_params.json" -printf '%h\n' 2>/dev/null | head -1 || true)"

if [[ -n "${TSTAMP_DIR}" ]]; then
    pass "best_params.json found: ${TSTAMP_DIR}"
    log "       contents: $(cat "${TSTAMP_DIR}/best_params.json")"
else
    fail "best_params.json not found — optimizer did not complete"
fi

N_CKPT="$(find "${TSTAMP_DIR:-${TEST_OUT}/clean}" -name "*.pickle" -o -name "*.pkl" 2>/dev/null | wc -l)"
if [[ "${N_CKPT}" -gt 0 ]]; then
    pass "checkpoint files present (${N_CKPT} file(s))"
    find "${TSTAMP_DIR}" -name "*.pickle" -o -name "*.pkl" | sed 's/^/         /'
else
    fail "no checkpoint files found (*.pickle / *.pkl)"
fi

if grep -q "Traceback" "${TEST_OUT}/test1.log"; then
    fail "Python traceback in Test 1 log"
else
    pass "no traceback in Test 1 log"
fi

# ---------------------------------------------------------------------------
# TEST 2: Stale checkpoint resume via direct HyperparamOptimizer call
# ---------------------------------------------------------------------------
log ""
log "=== TEST 2: stale checkpoint resume ==="

if [[ -z "${TSTAMP_DIR:-}" ]]; then
    log "SKIP: Test 1 produced no checkpoint dir — cannot run Test 2"
    FAIL=$((FAIL + 1))
else
    log "Checkpoint dir : ${TSTAMP_DIR}"
    log "Narrowed config: ${NARROWED_CONFIG}"

    srun --mpi=pmix -n "${N_RANKS}" --label \
      "${VENV}/bin/python" -u "${HELPER_SCRIPT}" \
      "${NARROWED_CONFIG}" "${TSTAMP_DIR}" 2>&1 | tee "${TEST_OUT}/test2.log"

    if grep -q "Skipping stale checkpoint" "${TEST_OUT}/test2.log"; then
        pass "stale checkpoint individuals skipped (fix is active)"
    else
        fail "no 'Skipping stale checkpoint' warning — stale path not triggered"
        log "  Hint: propulate may not have sampled cpd_penalty=null in Test 1."
        log "  Re-run to get a different random population, or the fix is correct"
        log "  but this population happened not to contain any null individuals."
    fi

    if grep -q "Traceback" "${TEST_OUT}/test2.log"; then
        fail "Python traceback in Test 2 — fix did not prevent crash"
    else
        pass "no traceback in Test 2 (stale individuals handled gracefully)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=== Summary: ${PASS} passed  ${FAIL} failed ==="
log "Logs: ${TEST_OUT}/test1.log  ${TEST_OUT}/test2.log"
[[ "${FAIL}" -eq 0 ]]
