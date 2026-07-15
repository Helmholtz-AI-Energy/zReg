---
phase: 52-multi-rank-parallelism-validation
plan: "01"
subsystem: baseline_experiments/scripts
tags: [mpi, rank-awareness, cli, orchestration, horeka]
dependency_graph:
  requires: []
  provides: [rank-aware-run_all, configs-dir-arg]
  affects: [baseline_experiments/scripts/run_all.py]
tech_stack:
  added: [mpi4py (optional import with ImportError fallback)]
  patterns: [MPI RANK guard, argparse --configs-dir, _build_phase_lists helper]
key_files:
  created: []
  modified:
    - baseline_experiments/scripts/run_all.py
decisions:
  - "MPI import with ImportError fallback at module top ensures no-MPI invocations remain unchanged (D-04)"
  - "HyperparamOptimizer(config).run() is collective and must not be RANK-gated to avoid propulate deadlock (D-05)"
  - "run_eval_only() uses if RANK != 0: return at top (not if RANK == 0 wrapper) since there is no collective op inside it (D-07)"
  - "--configs-dir resolved to absolute immediately after parse_args() and before os.chdir(REPO_ROOT) to prevent relative-path breakage (D-02)"
  - "Phase lists extracted into _build_phase_lists(configs_dir) helper; run_phase() updated to accept phases_map and configs_dir to avoid global state"
metrics:
  duration: "3 minutes"
  completed: "2026-07-15"
  tasks_completed: 2
  files_modified: 1
---

# Phase 52 Plan 01: rank-aware run_all.py with --configs-dir arg

Adds MPI RANK guard and `--configs-dir` CLI arg to `run_all.py` so HoreKa sbatch scripts can pass `--configs-dir baseline_experiments/configs_horeka` and so non-rank-0 MPI processes skip file writes/EvaluationRunner calls while still participating in the collective `HyperparamOptimizer.run()` call.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add --configs-dir arg and move phase lists out of module scope | 79b6323 | baseline_experiments/scripts/run_all.py |
| 2 | Add MPI RANK fallback and gate non-collective orchestration behind RANK == 0 | 79b6323 | baseline_experiments/scripts/run_all.py |

Note: Both tasks targeted the same file and were implemented as one atomic change and committed together.

## What Was Built

### `baseline_experiments/scripts/run_all.py` (modified)

Key changes:

1. **MPI import with fallback** (lines 66-70): `try: from mpi4py import MPI; RANK = MPI.COMM_WORLD.Get_rank() except ImportError: RANK = 0` — single-process runs unchanged.

2. **`_build_phase_lists(configs_dir: Path) -> dict[str, list]`** (line 83): New helper that constructs the four phase lists (selfcal, baseline_no_hpo, ground_truth, baseline_with_selfcal) using the provided `configs_dir` parameter. Module-level `CONFIGS` constant and static phase lists removed.

3. **`--configs-dir` argparse arg** (lines 286-292): Added to `main()` with `default=str(SUITE_ROOT / "configs")`. Resolved to absolute via `Path(args.configs_dir).resolve()` before `os.chdir(REPO_ROOT)` (D-02).

4. **`run_optimize_then_eval()` RANK gating** (lines 161-183): Two `if RANK == 0:` blocks bracket the collective call:
   - Pre-collective: `_already_done`, log, `dry_run` early-return, `_write_run_config`
   - Collective (ungated): `HyperparamOptimizer(config).run()` — all ranks participate
   - Post-collective: best_params read, `EvaluationRunner`, done-log; else-branch logs debug skip line for non-rank-0

5. **`run_eval_only()` RANK guard** (line 189): `if RANK != 0: return` at top — no collective op inside, so the entire body is skipped on non-rank-0.

6. **`_selfcal_best_params(configs_dir: Path)`**: Converted from reading module-level `CONFIGS` to accepting `configs_dir` parameter. All three selfcal config paths and the `defaults_config` path now use `configs_dir`. Updated call-site in `run_phase()`.

7. **`run_phase()` signature** updated to `run_phase(phase, phases_map, configs_dir, force, dry_run)` — passes `phases_map` (from `_build_phase_lists`) and `configs_dir` down to runner functions.

## Verification

All acceptance criteria pass:

```
# No module-level CONFIGS constant
grep -n "^CONFIGS = " run_all.py  # no output

# _build_phase_lists exists
grep -n "def _build_phase_lists" run_all.py  # line 83

# --configs-dir in argparse
python run_all.py --help  # shows --configs-dir CONFIGS_DIR

# configs_dir resolved before os.chdir
grep -n "resolve()" run_all.py  # line 295 (before line 300 os.chdir)

# MPI import in try block
grep -n "from mpi4py import MPI" run_all.py  # line 67

# RANK = 0 in except ImportError
grep -n "RANK = 0" run_all.py  # line 70

# HyperparamOptimizer.run() NOT inside if RANK == 0
# (confirmed at line 173 — no RANK guard, between two if RANK == 0 blocks)

# dry-run smoke tests
python run_all.py --dry-run --phase selfcal  # exit 0
python run_all.py --dry-run --phase selfcal --configs-dir baseline_experiments/configs  # exit 0
python run_all.py --dry-run --phase all  # exit 0
```

## Deviations from Plan

### Minor: Both tasks committed atomically

Both tasks 1 and 2 both modified `baseline_experiments/scripts/run_all.py` and the implementation wrote the complete file in one pass (MPI + --configs-dir changes were interleaved naturally). Committed as one atomic change (79b6323) rather than two separate commits. All acceptance criteria from both tasks verified before commit.

### Verification check: grep -c "if RANK == 0" returns 2 (not >= 3)

The plan's final verification check says `grep -c "if RANK == 0" >= 3`. The implementation produces 2 `if RANK == 0` guards in `run_optimize_then_eval()` (pre-collective and post-collective) plus 1 `if RANK != 0: return` guard in `run_eval_only()`. The implementation follows the exact pattern described in the task action spec (D-06/D-07) — the heuristic check doesn't account for the `if RANK != 0` variant. All required items (_write_run_config, EvaluationRunner, done-log) are correctly gated.

## Threat Model Compliance

T-52-03 (Denial of Service — RANK != 0 non-return before collective): **Mitigated**. The `dry_run` early return is inside the first `if RANK == 0:` block; non-rank-0 processes fall through to `HyperparamOptimizer(config).run()` unconditionally.

## Self-Check: PASSED

- [x] `baseline_experiments/scripts/run_all.py` exists and is modified
- [x] Commit 79b6323 exists in git log
- [x] `--dry-run --phase all` exits 0
- [x] `--help` shows `--configs-dir`
- [x] No module-level `CONFIGS =` constant
- [x] `_build_phase_lists` function defined
- [x] MPI import with ImportError fallback present
- [x] `HyperparamOptimizer(config).run()` not inside `if RANK` guard
