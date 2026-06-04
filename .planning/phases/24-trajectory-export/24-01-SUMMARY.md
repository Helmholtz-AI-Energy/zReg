---
phase: 24-trajectory-export
plan: "01"
subsystem: eval/tracking
tags: [trajectory-export, csv, json, stdlib, tdd]
dependency_graph:
  requires:
    - eval/types.py (AlignResult, LabelResult)
    - eval/config.py (EvalConfig)
    - eval/tracking/tracking.py (auto-capture block pattern)
  provides:
    - eval/tracking/trajectory.py (export_trajectory function)
    - eval/tracking/__init__.py (re-exports export_trajectory)
  affects:
    - eval/runners/eval_runner.py (Plan 24-02 will call export_trajectory)
tech_stack:
  added: []
  patterns:
    - TDD RED/GREEN cycle
    - stdlib-only file I/O (csv, json, uuid, subprocess, importlib.metadata)
    - auto-capture block (git_hash, zreg_version, timestamp) matching tracking.py
key_files:
  created:
    - eval/tracking/trajectory.py
    - tests/test_trajectory.py
  modified:
    - eval/tracking/__init__.py
decisions:
  - "export_trajectory uses zRegPointCloud instances (not plain dicts) in aligned_cloud — pydantic AlignResult validation requires zRegPointCloud; test fixtures updated accordingly"
  - "D-07/D-08 label pos source: use aligned_cloud['pos'] when align ran, else dataset['pos']"
  - "Single run_id UUID generated once per call; shared across both metadata files when both stages ran"
  - "Returned list order: align_csv, align_meta, label_csv, label_meta (absent stages omitted)"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-04"
  tasks_completed: 2
  files_created: 2
  files_modified: 1
  tests_added: 22
  tests_total: 763
---

# Phase 24 Plan 01: Trajectory Export — export_trajectory Summary

**One-liner:** stdlib-only export_trajectory writes per-point CSV trajectory files and 11-field metadata JSON per EXT-01, exposed from eval.tracking.

## What Was Built

`eval/tracking/trajectory.py` — new file implementing `export_trajectory(result, dataset, config, output_dir) -> list[str]`. The function writes up to four files depending on which stage results are present in `result["align"]` and `result["label"]`:

- `align_trajectory.csv` — 5-column (frame_idx, point_idx, x, y, z), one row per point per frame, positions from `AlignResult.aligned_cloud`
- `align_metadata.json` — 11-field dict (run_id, frame_count, frame_indices, data_path, params_used, tier, n_trials, n_synthetic, git_hash, zreg_version, timestamp)
- `label_trajectory.csv` — 6-column (adds label), positions from aligned_cloud (D-07) or dataset (D-08)
- `label_metadata.json` — same 11 fields with label stage params_used

`eval/tracking/__init__.py` — updated to re-export `export_trajectory` alongside `log_run`; both names in `__all__`.

## TDD Cycle

- **RED** commit `43a8eef`: 22 failing tests across 6 test classes
- **GREEN** commit `0a648cc`: implementation + fixture fix (zRegPointCloud instances required by pydantic)
- **Task 2** commit `0d9adb2`: `__init__.py` update

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test fixture used plain dict instead of zRegPointCloud**

- **Found during:** Task 1 GREEN phase (14 tests still failing after initial implementation)
- **Issue:** `AlignResult.aligned_cloud` has type `dict[int, zRegPointCloud]` — pydantic v2's `arbitrary_types_allowed=True` still enforces `isinstance` check. Test helper `_make_zreg_pc` returned `{"pos": tensor}` plain dict, causing `ValidationError: Input should be an instance of zRegPointCloud`.
- **Fix:** Changed `_make_zreg_pc` to construct `zRegPointCloud()` dict subclass and assign `pc["pos"] = pos`. Changed `_make_align_result` to pass the dataset dict directly (already contains `zRegPointCloud` instances) rather than re-wrapping with `dict(v)`.
- **Files modified:** `tests/test_trajectory.py`
- **Commit:** `0a648cc` (combined with GREEN implementation)

## Known Stubs

None — all exported paths are live file writes; no placeholder data.

## Threat Flags

No new threat surface beyond what was documented in the plan's threat model. `export_trajectory` writes only to `output_dir` (caller-controlled, under EvaluationRunner control per T-24-01), auto-captures git hash via subprocess (T-24-02, same pattern as tracking.py), and CSV size is bounded by dataset (T-24-03).

## Verification Results

1. `python -c "from eval.tracking import export_trajectory, log_run; print('ok')"` — exits 0
2. `eval/tracking/trajectory.py` imports: csv, datetime, importlib.metadata, json, subprocess, uuid, pathlib.Path, typing.Any, eval.config.EvalConfig — no third-party packages
3. `__all__` in `eval/tracking/__init__.py` contains both `"log_run"` and `"export_trajectory"`
4. 763 tests pass, 17 skipped (741 pre-existing + 22 new in tests/test_trajectory.py)

## Self-Check: PASSED
