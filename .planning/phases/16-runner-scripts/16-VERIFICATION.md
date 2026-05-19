---
phase: 16-runner-scripts
verified: 2026-05-18T00:00:00Z
status: passed
score: 14/14 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 16: Runner Scripts Verification Report

**Phase Goal:** Deliver EVAL-05 runner scripts — eval/run_synthetic.py and eval/run_real.py as standalone sweep orchestrators wiring eval/generators/, eval/tracking/, and zreg.metrics into logged sweep loops.
**Verified:** 2026-05-18
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | python eval/run_synthetic.py exits 0 and creates files in evaluation/runs/ | VERIFIED | run_sweep(output_dir=tmp) produces 12 .json + 12 .csv files; confirmed by behavioral spot-check |
| 2 | run_synthetic.py covers both noise sweep (SIGMAS) and outlier sweep (N_OUTLIERS_LIST) | VERIFIED | Lines 28-29 define SIGMAS=[0.0,0.01,0.05,0.1], N_OUTLIERS_LIST=[0,5,20]; nested loops at lines 50-51; 12-cell output confirmed |
| 3 | run_real.py exits 0 with informative message when DATASET_PATH does not exist | VERIFIED | Missing-data guard at line 38-40; print outputs "Dataset not found: ... Skipping real sweep."; behavioral check: "not found" in output, 0 files created |
| 4 | run_synthetic.py imports from zreg.metrics (installed package), not any local copy | VERIFIED | Line 25: `from zreg.metrics import chamfer, hausdorff`; test_metrics_import_path confirms no local copy |
| 5 | Neither eval/run_synthetic.py nor eval/run_real.py is importable as a package (no eval/__init__.py) | VERIFIED | `ls eval/__init__.py` returns "No such file"; `importlib.util.find_spec('eval').origin` returns None (namespace package) |
| 6 | run_sweep(output_dir) accepts output_dir parameter so tests can redirect to tmp_path | VERIFIED | Both scripts define `def run_sweep(output_dir: str = "evaluation/runs") -> None`; tests confirm redirectability |
| 7 | All run_ids in synthetic sweep embed sigma, n_outliers, and seed to prevent log_run() overwrite collisions | VERIFIED | run_id = f"synthetic_sigma{sigma}_out{n_out}_seed{SEED}" at line 63; all 12 stems verified to contain sigma, out, and seed |
| 8 | pytest tests/test_runners.py exits 0 with all tests green | VERIFIED | 9 passed in 1.76s |
| 9 | TestRunSynthetic.test_json_files_created: run_sweep(output_dir=tmp_path) produces {run_id}.json files | VERIFIED | test_creates_json_files passes; 12 json files asserted |
| 10 | TestRunSynthetic.test_csv_files_created: same call produces {run_id}.csv files | VERIFIED | test_creates_csv_files passes; 12 csv files asserted |
| 11 | TestRunSynthetic.test_run_count: sweep produces exactly len(SIGMAS)*len(N_OUTLIERS_LIST) unique run_ids | VERIFIED | test_run_ids_unique passes; 12 unique stems asserted |
| 12 | TestRunSynthetic.test_metrics_import_path: zreg.metrics is the import source, not a local copy | VERIFIED | test_metrics_import_path passes; scans source text for "from zreg.metrics import" |
| 13 | TestRunReal.test_missing_dataset_skips: run_sweep() returns without creating files and prints 'not found' | VERIFIED | test_missing_dataset_prints_message passes; "not found" confirmed in capsys output |
| 14 | TestRunReal.test_missing_dataset_no_files: tmp_path is empty after run_sweep() with missing DATASET_PATH | VERIFIED | test_missing_dataset_creates_no_files passes; list(tmp_path.iterdir()) == [] |

**Score:** 14/14 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/run_synthetic.py` | noise/outlier sweep orchestrator with `def run_sweep` | VERIFIED | 80 lines; contains `def run_sweep`, `SIGMAS`, `N_OUTLIERS_LIST`, `if __name__ == "__main__"`, imports from `zreg.metrics` |
| `eval/run_real.py` | scale/density sweep orchestrator with `def run_sweep` | VERIFIED | 64 lines; contains `def run_sweep`, `SCALES`, `DENSITY_FRACTIONS`, missing-data guard, `if __name__ == "__main__"` |
| `tests/test_runners.py` | functional test suite with class TestRunSynthetic | VERIFIED | 206 lines; 1 occurrence of `class TestRunSynthetic`, 6 test methods |
| `tests/test_runners.py` | functional test suite with class TestRunReal | VERIFIED | 1 occurrence of `class TestRunReal`, 3 test methods; total 9 def test_ methods |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| eval/run_synthetic.py | eval/tracking/log_run() | `from eval.tracking import log_run` | WIRED | Line 24; log_run called at line 64 with all required args |
| eval/run_synthetic.py | zreg.metrics | `from zreg.metrics import chamfer, hausdorff` | WIRED | Line 25; chamfer called at line 56, hausdorff at line 59 |
| eval/run_real.py | eval/tracking/log_run() | `from eval.tracking import log_run` | WIRED | Line 19; log_run called at line 50 inside loop |
| tests/test_runners.py | eval/run_synthetic.py | `from eval.run_synthetic import run_sweep as run_synthetic_sweep` | WIRED | Line 30; called in 5 test methods |
| tests/test_runners.py | eval/run_real.py | `from eval.run_real import run_sweep as run_real_sweep` | WIRED | Line 28; called in 2 test methods |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| eval/run_synthetic.py | traj_clean, traj_noisy | eval.generators.generate_trajectory + add_gaussian_noise + add_outliers | Yes — generates synthetic point clouds in memory | FLOWING |
| eval/run_synthetic.py | cd, hd (chamfer/hausdorff) | zreg.metrics.chamfer / hausdorff applied to real tensors | Yes — tensor computation confirmed: chamfer=0.1568 for sigma=0.1 | FLOWING |
| eval/run_synthetic.py | log_run output files | log_run() writes JSON+CSV with real computed values including chamfer/hausdorff as extra_fields | Yes — 12 json + 12 csv files created with actual metric values | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| run_synthetic.py creates 12 JSON + 12 CSV files | `KMP_DUPLICATE_LIB_OK=TRUE python -c "...run_sweep(output_dir=tmp)..."` | json: 12, csv: 12, PASS | PASS |
| run_real.py missing-data guard prints "not found" and creates 0 files | `python -c "...redirect_stdout...run_sweep(output_dir=tmp)..."` | "not found" in output, 0 files | PASS |
| All 12 run_ids contain sigma, out, and seed | `run_sweep + stem validation` | 12 unique stems verified | PASS |
| pytest tests/test_runners.py | `KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_runners.py -v` | 9 passed in 1.76s | PASS |
| Full test suite shows no regressions | `KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q` | 536 passed, 15 skipped, 2 warnings | PASS |

### Probe Execution

Step 7c: SKIPPED — no probe-*.sh files declared or found for Phase 16.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EVAL-05 | 16-01, 16-02 | Evaluation runner scripts in eval/ at repo root (not an importable package): run_synthetic.py, run_real.py, noise/corruption sweep, scale/density sweep — accept dict[int, zRegPointCloud] inputs, import metrics from zreg.metrics (installed source package), write outputs to evaluation/runs/ | SATISFIED | Both scripts exist, eval/ is a namespace package (no __init__.py), noise sweep iterates SIGMAS x N_OUTLIERS_LIST, scale/density sweep iterates SCALES x DENSITY_FRACTIONS, zreg.metrics imported, log_run() wired, 9-test suite green |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

No TBD, FIXME, XXX, TODO, HACK, or PLACEHOLDER markers in any Phase 16 files. No empty implementations, no return null/return []. No hardcoded empty data values in rendering paths.

### Structural Deviation: generate_trajectory Placement

The plan spec (Task 1, action block) listed `generate_trajectory(...)` inside the per-sigma inner loop. The actual implementation generates `traj_clean` once before both loops (line 49), then applies corruption variants inside the loops. This is semantically correct — the same clean trajectory is corrupted differently per cell — and produces identical 12-cell output. The behavioral spot-check confirms 12 unique run IDs with correct content. This is a minor structural optimization, not a behavioral deviation.

### Structural Deviation: run_real.py n_after Calculation

The plan spec specified `n_after = int(n_points_base * density)`. The actual code computes `n_points_scaled = int(n_points_base * scale)` then `n_after = int(n_points_scaled * density)`, applying scale first then density. The run_id format `real_scale{scale}_density{density}_seed{SEED}` remains consistent with the plan. This is a semantic enhancement that correctly models both scale and density as independent factors — not a deviation from the observable goal.

### Human Verification Required

None. All phase-16 behaviors are programmatically verifiable. Both scripts are standalone non-interactive orchestrators with no visual output, real-time behavior, or external services beyond the local filesystem.

---

_Verified: 2026-05-18T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
