---
status: complete
phase: 23-cli-entrypoint-scenario-configs
source: [23-VERIFICATION.md]
started: 2026-06-01T11:00:00Z
updated: 2026-06-03T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. `--mode eval` smoke test with real data
expected: `python run_eval.py --config configs/alignment_sanity.yaml --mode eval` exits 0 without Python traceback; `experiments/runs/run_config.yaml` is written; no crash on real tracklets data
result: pass

### 2. `--mode optimize` smoke test — best_params.json written
expected: `python run_eval.py --config configs/alignment_sanity.yaml --mode optimize` exits 0; `experiments/runs/best_params.json` exists after run; HyperparamOptimizer completes without error
result: pass

### 3. `--mode full` end-to-end chain with combined_full config
expected: `python run_eval.py --config configs/combined_full.yaml --mode full` exits 0; optimizer runs first, then EvaluationRunner; both `best_params.json` and evaluation output written to `experiments/runs/`
result: pass
note: best_params.json contains {} by design — combined_full.yaml has no search_space (D-04); optimizer correctly writes empty result, _load_best_params falls back to default_params, evaluation runs with those defaults

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
