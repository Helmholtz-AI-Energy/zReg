---
phase: 23-cli-entrypoint-scenario-configs
verified: 2026-06-01T00:00:00Z
status: human_needed
score: 9/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run `python run_eval.py --config configs/alignment_sanity.yaml --mode eval` from repo root with real tracklets data present"
    expected: "Command completes without error; EvalReport written to output_dir; no stack trace on stdout/stderr"
    why_human: "End-to-end test requires real tracklets file at data/external/sample/kobitski_data/... and real EvaluationRunner execution against real data — not safe to mock in automated checks without defeating the goal"
  - test: "Run `python run_eval.py --config configs/alignment_sanity.yaml --mode optimize` and verify best_params.json + search_history.json are written to output_dir"
    expected: "HyperparamOptimizer.run() completes; output_dir/best_params.json and output_dir/search_history.json exist after run"
    why_human: "Requires real data at tracklets path; optimizer produces real results only with live file access"
  - test: "Run `python run_eval.py --config configs/combined_full.yaml --mode full` and verify full chain (optimizer then runner) completes"
    expected: "Optimizer runs first, writes best_params.json, then EvaluationRunner runs with those params; no crash"
    why_human: "True end-to-end smoke test of the full-mode chain — phase goal states 'all 5 configs run end-to-end without errors'; requires real data; 23-VALIDATION.md explicitly defers this to post-merge"
---

# Phase 23: CLI Entrypoint & Scenario Configs Verification Report

**Phase Goal:** Deliver `run_eval.py` CLI (`--config`, `--mode optimize|eval|full`) and 5 scenario YAML configs in `configs/`; `full` mode chains Optimizer → EvaluationRunner; all 5 configs run end-to-end without errors.
**Verified:** 2026-06-01
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `run_eval.py` exists at repo root with argparse CLI exposing `--config`, `--mode`, `--output-dir`, `--verbose` | VERIFIED | File confirmed at repo root; `--help` exits 0 and prints all 4 flags |
| 2 | `--mode optimize` calls `HyperparamOptimizer(config).run()` exactly once and does not call `EvaluationRunner` | VERIFIED | `TestCLIModeDispatch::test_optimize_mode_calls_optimizer` PASSED; code at `run_eval.py:172` |
| 3 | `--mode eval` calls `EvaluationRunner(config, params).run()` with params loaded from `best_params.json` when it exists | VERIFIED | `TestCLIModeDispatch::test_eval_mode_calls_runner` PASSED; `MockRunner.call_args[0][1] == {"window_size": 7}` assertion passes |
| 4 | `--mode full` chains Optimizer first, then reads `best_params.json`, then calls EvaluationRunner (Pitfall 5 order) | VERIFIED | `test_full_mode_calls_optimizer_then_runner` PASSED; side_effect writes json then runner sees it; code lines 178-180 in correct order |
| 5 | Config errors produce one-line `Error: EvalConfig:` message to stderr with exit code 1, no Traceback | VERIFIED | `test_config_error_clean_message` PASSED; manual check `python run_eval.py --config /nonexistent.yaml --mode optimize` → `Error: EvalConfig: file not found: /nonexistent.yaml`, exit 1 |
| 6 | `run_config.yaml` is written to `output_dir` before pipeline executes (reproducibility) | VERIFIED | `test_run_config_yaml_written_before_pipeline` PASSED; code line 168 `_write_run_config` before lines 172-180 dispatch |
| 7 | All 5 scenario YAML configs exist in `configs/` with correct D-04 field values | VERIFIED | All 5 files present; `EvalConfig.from_yaml` on all 5 prints `OK`; `test_config_has_d04_values` for all 5 scenarios PASSED |
| 8 | `tests/test_cli.py` has 18 passing tests (8 FRAME-11 + 10 FRAME-12) with no subprocess, all via direct import + mock | VERIFIED | `pytest tests/test_cli.py` → `18 passed, 0 skipped, 0 failed` |
| 9 | Full test suite remains green (no regression) | VERIFIED | `pytest tests/ -x -q` → `741 passed, 17 skipped` |

**Score:** 9/9 truths verified

---

### Deferred Items

None — no items deferred to later phases.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `run_eval.py` | CLI entrypoint with argparse, EvalConfig loading, mode dispatch, run_config.yaml write, best_params.json load | VERIFIED | 187 lines; contains `def main`, `_build_parser`, `_write_run_config`, `_load_best_params`, `EvalConfigError`, `HyperparamOptimizer(config).run(`, `EvaluationRunner(config, params).run(` |
| `tests/test_cli.py` | FRAME-11 + FRAME-12 unit tests via direct import + unittest.mock.patch | VERIFIED | 291 lines; 6 test classes, 10 test methods + 10 parametrised variants = 18 tests total; all pass |
| `configs/alignment_sanity.yaml` | Sanity-tier alignment-only scenario (tier: sanity, n_trials: 3, n_synthetic: 20, run_alignment: true, run_label_transfer: false) | VERIFIED | File exists with exact D-04 values; loads via EvalConfig.from_yaml |
| `configs/alignment_dev.yaml` | Dev-tier alignment-only scenario (tier: dev, n_trials: 10, n_synthetic: 50, run_alignment: true, run_label_transfer: false) | VERIFIED | File exists with exact D-04 values; loads via EvalConfig.from_yaml |
| `configs/label_transfer_sanity.yaml` | Sanity-tier label-transfer-only scenario (tier: sanity, n_trials: 3, n_synthetic: 20, run_alignment: false, run_label_transfer: true) | VERIFIED | File exists with exact D-04 values; loads via EvalConfig.from_yaml |
| `configs/label_transfer_dev.yaml` | Dev-tier label-transfer-only scenario (tier: dev, n_trials: 10, n_synthetic: 50, run_alignment: false, run_label_transfer: true) | VERIFIED | File exists with exact D-04 values; loads via EvalConfig.from_yaml |
| `configs/combined_full.yaml` | Full-tier combined scenario (tier: full, n_trials: 20, n_synthetic: 100, run_alignment: true, run_label_transfer: true) | VERIFIED | File exists with exact D-04 values; loads via EvalConfig.from_yaml |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `run_eval.py` | `eval.config.EvalConfig.from_yaml` | `from eval.config import EvalConfig, EvalConfigError` + call inside `main()` | VERIFIED | Line 69 import; line 146 `EvalConfig.from_yaml(args.config)` |
| `run_eval.py` | `eval.runners.HyperparamOptimizer` | `from eval.runners import HyperparamOptimizer` + call in optimize/full branches | VERIFIED | Line 70 import; lines 172, 178 `HyperparamOptimizer(config).run()` |
| `run_eval.py` | `eval.runners.EvaluationRunner` | `from eval.runners import EvaluationRunner` + call in eval/full branches | VERIFIED | Line 70 import; lines 175, 180 `EvaluationRunner(config, params).run()` |
| `run_eval.py` | `{output_dir}/run_config.yaml` | `shutil.copy` in `_write_run_config` before mode dispatch | VERIFIED | Line 103 `shutil.copy(config_path, dst)`; called at line 168 before dispatch block |
| `run_eval.py` | `{output_dir}/best_params.json` | `json.load` in `_load_best_params` | VERIFIED | Line 116 `params = json.load(f)` |
| `tests/test_cli.py` | `run_eval.main` | `import run_eval` + `run_eval.main([...])` | VERIFIED | Line 26 `import run_eval`; used throughout all test classes |
| `configs/*.yaml` | `eval.config.EvalConfig` | `EvalConfig.from_yaml(path)` with `extra="forbid"` validation | VERIFIED | All 5 configs load cleanly; Python one-liner confirms `OK` |
| `tests/test_cli.py::TestScenarioConfigs` | `configs/*.yaml` | `pytest.mark.parametrize` over `_SCENARIO_TABLE` + `EvalConfig.from_yaml` | VERIFIED | `_SCENARIO_TABLE` at line 253-259; parametrize at lines 267, 281 |

---

### Data-Flow Trace (Level 4)

`run_eval.py` is a CLI dispatcher — it does not render dynamic data itself. Data flow is through mock in tests (no real I/O). The data-flow trace for the actual runtime path (real data through EvaluationRunner) is in the Human Verification section (requires real tracklets).

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `run_eval.py` → `_load_best_params` | `params` dict | `json.load` from `{output_dir}/best_params.json` | Yes (when file exists from prior optimizer run) | VERIFIED (code path correct; real data test is human-only) |
| `run_eval.py` → `_write_run_config` | YAML bytes | `shutil.copy` of `args.config` | Yes (exact byte copy) | VERIFIED |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `--help` exits 0 and shows all 4 flags | `python run_eval.py --help` | Exits 0; output contains `--config`, `--mode`, `--output-dir`, `--verbose` | PASS |
| Invalid config path → exit 1 with clean error | `python run_eval.py --config /nonexistent.yaml --mode optimize; echo $?` | stderr: `Error: EvalConfig: file not found: /nonexistent.yaml`; exit: 1; no Traceback | PASS |
| All 5 configs load via EvalConfig.from_yaml | Python one-liner loading all 5 configs | Prints `OK`; exit 0 | PASS |
| 18 test_cli.py tests pass | `pytest tests/test_cli.py -x -q` | `18 passed` | PASS |
| Full test suite not regressed | `pytest tests/ -x -q` | `741 passed, 17 skipped` | PASS |

---

### Probe Execution

No probe scripts declared in PLAN.md or discoverable in `scripts/` for this phase.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FRAME-11 | 23-01-PLAN.md | `run_eval.py` CLI entrypoint — `--config`, `--mode optimize|eval|full`, clean EvalConfigError messages, `run_config.yaml` copy | SATISFIED | `run_eval.py` implements all spec points; 8 FRAME-11 tests pass; `--help` exits 0 |
| FRAME-12 | 23-02-PLAN.md | 5 scenario YAML configs in `configs/`; all 5 run without errors end-to-end | SATISFIED (partial — automated load validated; end-to-end with real data is human verification) | All 5 configs exist; all load via `EvalConfig.from_yaml`; 10 FRAME-12 tests pass; end-to-end smoke deferred per 23-VALIDATION.md |

**Note on REQUIREMENTS.md:** FRAME-11 and FRAME-12 are still marked `[ ]` (pending) in `.planning/REQUIREMENTS.md` lines 57-58 and `Pending` in the traceability table at lines 123-124. The implementations are complete; the requirements file was not updated to reflect closure. This is a documentation-only gap — the phase goal is achieved in the codebase.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `run_eval.py` | 120 | `return {}` | Info | Legitimate empty-dict fallback in `_load_best_params` when `best_params.json` is absent — not a stub; this is the documented D-01 behavior |

No blockers. No debt markers (TBD/FIXME/XXX). No unreachable placeholders.

---

### Human Verification Required

**The phase goal states "all 5 configs run end-to-end without errors."** Automated checks confirm config loading and unit-test mock dispatch. The genuine end-to-end test (real data + real runner execution) was explicitly deferred in `23-VALIDATION.md` to post-merge and is not achievable programmatically.

#### 1. Eval-mode smoke — alignment_sanity

**Test:** From repo root, run `python run_eval.py --config configs/alignment_sanity.yaml --mode eval`
**Expected:** Command completes with exit code 0; `experiments/runs/` (or configured output_dir) contains `run_config.yaml` and `eval_report.json`; no stack trace on stdout/stderr
**Why human:** Requires real tracklets file at `data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets`; EvaluationRunner executes real alignment pipeline

#### 2. Optimize-mode smoke

**Test:** From repo root, run `python run_eval.py --config configs/alignment_sanity.yaml --mode optimize --output-dir /tmp/zreg_smoke`
**Expected:** Exit code 0; `/tmp/zreg_smoke/best_params.json` and `/tmp/zreg_smoke/search_history.json` exist; `/tmp/zreg_smoke/run_config.yaml` is a byte-for-byte copy of `configs/alignment_sanity.yaml`
**Why human:** Requires real tracklets data; HyperparamOptimizer runs real trial loop

#### 3. Full-mode chain smoke

**Test:** From repo root, run `python run_eval.py --config configs/combined_full.yaml --mode full --output-dir /tmp/zreg_full`
**Expected:** Optimizer runs first (writes `best_params.json`), then EvaluationRunner runs with those params; no crash; both `best_params.json` and `eval_report.json` present in output_dir
**Why human:** Validates the critical "Optimizer → EvaluationRunner" chain in `full` mode with real data; this is the central integration claim of the phase goal

---

### Gaps Summary

No gaps. All automated must-haves are VERIFIED. The only outstanding item is the end-to-end smoke test with real data, which was explicitly deferred by the plan (23-VALIDATION.md) and requires human execution.

**Documentation note:** `.planning/REQUIREMENTS.md` checkbox items for FRAME-11 (line 57) and FRAME-12 (line 58) plus the traceability table rows (lines 123-124) still show "Pending" status. These should be updated to Complete/checked after human smoke tests pass.

---

_Verified: 2026-06-01_
_Verifier: Claude (gsd-verifier)_
