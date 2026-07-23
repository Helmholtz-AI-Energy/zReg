---
phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
plan: 03
subsystem: cli
tags: [argparse, cli, label-transfer, benchmarking, eval-framework, pointnet2, egnn]

# Dependency graph
requires:
  - phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
    plan: 02
    provides: LabelTransferBenchmark orchestration class (compare_methods/run_leakage_guard/save_report), exported from eval.runners
  - phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
    plan: 01
    provides: MethodBenchmarkResult/BenchmarkReport frozen pydantic models, save_checkpoint-based smoke-checkpoint pattern
provides:
  - benchmark_label_transfer.py — repo-root argparse CLI wrapping LabelTransferBenchmark, mirroring run_eval.py's sys.path/argparse convention
  - tests/test_benchmark_cli.py — CLI smoke test proving main(argv) runs end-to-end against inline smoke checkpoints
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI entry point wraps run_leakage_guard (always-run synthetic dimension) then optionally compare_methods (real-data dimension, guarded by a data_path != 'unused' sentinel check and try/except FileNotFoundError/ValueError)"
    - "--held-out-seeds 'start:stop' string parsed into range(int, int) at the CLI boundary — never hardcoded inside the wrapped class (Pattern 3 / A2)"

key-files:
  created:
    - benchmark_label_transfer.py
    - tests/test_benchmark_cli.py
  modified: []

key-decisions:
  - "Real-data dimension checked via `config.data_path != 'unused'` sentinel (matching train_label_transfer.py's own 'unused-training-entry-point-does-not-load-real-data'-style convention) rather than a None check, since EvalConfig.data_path is a required non-Optional str field with no None default"
  - "Footer uses `raise SystemExit(main())` per the plan's literal interface spec, rather than run_eval.py's `sys.exit(main())` — functionally equivalent, kept as specified"

patterns-established:
  - "Fourth repo-root CLI entry point in this codebase (after run_eval.py, train_label_transfer.py — this is the third counting only the eGNN/PointNet++ track), closing D-01's 'no code changes needed later, just point at the real checkpoint(s)' loop for the benchmarking machinery"

requirements-completed: [D-01]

# Metrics
duration: ~20min
completed: 2026-07-15
---

# Phase 49 Plan 3: benchmark_label_transfer.py CLI Entry Point Summary

**Repo-root `benchmark_label_transfer.py` CLI wraps `LabelTransferBenchmark`, mirroring `run_eval.py`'s argparse/sys.path convention — closes D-01's promise that the benchmarking machinery is user-runnable end-to-end with zero code changes once real checkpoints/data are available.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2 completed
- **Files modified:** 2 (both new)

## Accomplishments
- `benchmark_label_transfer.py` at repo root: `_build_parser()` (`--config`, `--output-dir`, `--held-out-seeds` default `"0:8"`, `--n-classes` default 4, `--methods`, `--verbose`) + `main(argv=None) -> int` mirroring `run_eval.py`'s `EvalConfigError` handling, timestamped `output_dir` resolution (T-49-06), and `if __name__ == "__main__"` footer exactly
- `--held-out-seeds` is parsed from caller input into `range(int(start), int(stop))` at the CLI boundary and forwarded to `LabelTransferBenchmark.run_leakage_guard` — never hardcoded internally (Pattern 3 / A2), satisfying the plan's must_have truth
- Always-run synthetic held-out dimension (`run_leakage_guard`) seeds `results`/`notes`; optional real-data dimension (`compare_methods` with `has_ground_truth=False`) only attempted when `config.data_path != "unused"`, wrapped in `try/except (FileNotFoundError, ValueError)` recording a note instead of crashing (T-49-07 / D-03)
- A single merged `BenchmarkReport` is written via `benchmark.save_report(report, config.output_dir)`
- `tests/test_benchmark_cli.py`: 3 tests — full end-to-end run against inline `PointNet2LabelTransfer`/`EGNNLabelTransfer` smoke checkpoints (writes `benchmark_report.json` with 4 results, each entry carrying `method`/`dataset_source`/`knn_consistency`/`latency_seconds`/`error`), invalid-config returns 1, `--held-out-seeds` parses as caller-supplied
- Full repo test suite (the whole eGNN/PointNet++ track, Phases 45-49) confirmed green: 1312 passed, 18 skipped, 1 xpassed (was 1309 at 49-02 close; +3 new tests), zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add benchmark_label_transfer.py repo-root CLI wrapping LabelTransferBenchmark** - `200312c` (feat)
2. **Task 2: CLI smoke test invoking main(argv) against smoke checkpoints** - `365ed65` (test)

**Plan metadata:** (this commit, below)

## Files Created/Modified
- `benchmark_label_transfer.py` - Repo-root CLI entry point wrapping `LabelTransferBenchmark`
- `tests/test_benchmark_cli.py` - CLI smoke test suite (3 tests): end-to-end report generation, invalid-config exit code, held-out-seeds parsing

## Decisions Made
- Used a `data_path != "unused"` sentinel check (string equality) rather than checking for `None`, since `EvalConfig.data_path: str` has no `None`/`Optional` variant — matches the existing `"unused"`/`"unused-training-entry-point-does-not-load-real-data"` sentinel convention already established by `tests/test_benchmark_runner.py` and `train_label_transfer.py` respectively.
- Followed the plan's literal `raise SystemExit(main())` footer spec rather than copying `run_eval.py`'s `sys.exit(main())` verbatim — both are functionally identical (`sys.exit` raises `SystemExit` internally); kept as the plan specified since it was an explicit interface directive, not an oversight to correct.

## Deviations from Plan

None — plan executed exactly as written. Both tasks' acceptance criteria and verify commands passed on the first attempt with no auto-fixes needed.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. The CLI is immediately runnable against smoke checkpoints; running it against real trained checkpoints (and, optionally, real mounted Kobitski/Shah data) requires only different `--config`/`--held-out-seeds` values, no code changes — fulfilling D-01's promise.

## Next Phase Readiness

This is the final plan in the final phase (49) of the entire eGNN/PointNet++ label-transfer track (Phases 45→46→47→48→49). All planned machinery is complete:
- Models (`PointNet2LabelTransfer`, `EGNNLabelTransfer`) — Phase 47
- Training pipeline (`train_label_transfer.py`) — Phase 47
- `LabelTransferStage` 4-way dispatch integration — Phase 48
- `LabelTransferBenchmark` orchestration + this plan's CLI wrapper — Phase 49

The only remaining step is external to this repo: the user trains real checkpoints on their cluster and points `benchmark_label_transfer.py --config <yaml> --held-out-seeds <disjoint-range>` at them — no code changes required.

No blockers. Full repo test suite green (1312 passed, 18 skipped, 1 xpassed).

---
*Phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods*
*Completed: 2026-07-15*

## Self-Check: PASSED

- FOUND: benchmark_label_transfer.py
- FOUND: tests/test_benchmark_cli.py
- FOUND commit: 200312c
- FOUND commit: 365ed65
