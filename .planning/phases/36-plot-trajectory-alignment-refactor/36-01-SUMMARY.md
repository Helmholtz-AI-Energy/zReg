---
phase: 36
plan: "36-01"
subsystem: eval/viz
tags: [viz, matplotlib, refactor, viz-02]
dependency_graph:
  requires: [phase-25-visualisation-refactor, phase-33-cpd-aligned-trajectory]
  provides: [four-figure-alignment-output]
  affects: [eval/viz.py, tests/test_viz.py, tests/test_eval_runner.py]
tech_stack:
  added: []
  patterns:
    - module-level private helpers extracted for figure writing and frame selection
    - pre-computed per-cloud per-frame subsampled arrays to avoid redundant numpy ops
key_files:
  created: []
  modified:
    - eval/viz.py
    - tests/test_viz.py
    - tests/test_eval_runner.py
decisions:
  - _write_single_cloud_figure and _write_superposed_figure share _ax_style and _save_fig helpers to avoid subplot/style/save boilerplate duplication
  - _deduplicate_frames extracted as module-level function; used by both alignment and label branches
  - per_frame_target dict only populated when target is not None; _write_superposed_figure receives None to skip target scatter
  - warp_path first-occurrence-per-target-index semantics match Phase 33/35 CPD output keying
  - test_eval_runner.py tests updated as Rule 1 fix (old alignment_trajectory.pdf stem no longer produced)
metrics:
  duration_seconds: 344
  completed_date: "2026-06-22"
  tasks_completed: 6
  files_changed: 3
---

# Phase 36 Plan 01: plot_trajectory Alignment Figure Refactor Summary

**One-liner:** Replaced single superimposed alignment_trajectory figure with four independent 1x3 figure pairs (source/target/aligned/superposed) using module-level private helpers for deduplication, subsampling, and figure writing.

## What Was Built

Refactored the alignment branch of `plot_trajectory` in `eval/viz.py` to produce four independent 1x3 figure pairs (PDF + PNG each = 6-8 files) instead of the previous single 1x3 figure. The label branch was untouched.

### New Figure Stems

| Stem | Content | Colour | Written when |
|------|---------|--------|--------------|
| `alignment_source_trajectory` | Source cloud per frame | `#3a7abf` (blue) | always |
| `alignment_target_trajectory` | Target cloud per frame | `#38a058` (green) | only when `target` is not None |
| `alignment_aligned_trajectory` | Aligned source per frame | `#e07b39` (orange) | always |
| `alignment_superposed_trajectory` | All three superposed | blue + orange + green | always |

### Private Helpers Added

- `_deduplicate_frames(candidates)` — order-preserving deduplication for 1/2-frame edge cases; used by both alignment and label branches
- `_warp_source_map(warp_path)` — builds target_idx -> source_idx dict using first-occurrence-per-target semantics
- `_subsample(arr, max_pts=4000)` — fixed-seed RNG subsampling; called once per cloud per frame before figure loops
- `_ax_style(ax, title)` — applies tick/label/pane style to 3-D axes (fontsize=6 ticks, fontsize=7 labels, pane.fill=False)
- `_save_fig(fig, base, paths)` — saves PDF + PNG, closes fig, extends paths list
- `_write_single_cloud_figure(...)` — creates 1x3 figure for a single cloud type
- `_write_superposed_figure(...)` — creates 1x3 figure superposing source, aligned, and optional target

### Test Updates

- Updated 6 existing `TestPlotTrajectory` tests for new file counts and stems
- Added 3 new tests: `test_alignment_with_target_writes_8_files`, `test_alignment_without_target_skips_target_figure`, `test_no_figure_leak_4panel`
- Updated 1 `TestVizCoverageGaps` test for new stems
- Fixed 2 `TestEvaluationRunnerSavePlots` tests referencing old `alignment_trajectory.pdf` stem (Rule 1 bug fix — old stem no longer produced)

## Commits

| Task | Commit | Files |
|------|--------|-------|
| Tasks 1-6 (all) | 433dd16 | eval/viz.py, tests/test_viz.py, tests/test_eval_runner.py |

## Success Criteria Verification

1. `plot_trajectory(align_result, None, dataset, None, tmp_path, target=target_ds)` writes exactly 8 files — PASS (test_alignment_with_target_writes_8_files)
2. `plot_trajectory(align_result, None, dataset, None, tmp_path)` (no target) writes 6 files — PASS (test_alignment_without_target_skips_target_figure)
3. `plot_trajectory(None, None, dataset, None, tmp_path)` returns `[]` — PASS (test_neither_stage_returns_empty_list)
4. Old stem `alignment_trajectory` no longer produced — PASS (assertion in test_alignment_without_target_skips_target_figure)
5. Label branch unchanged — PASS (test_label_only_writes_label_files unchanged)
6. No matplotlib figure leaks — PASS (test_no_figure_leak + test_no_figure_leak_4panel)
7. All 969 existing tests pass; 3 new tests added — PASS (972 total, 18 skipped)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_eval_runner.py references to old alignment_trajectory.pdf stem**
- **Found during:** Full test suite run after implementation
- **Issue:** `TestEvaluationRunnerSavePlots.test_save_plots_true_creates_alignment_trajectory_pdf` asserted `alignment_trajectory.pdf` exists; `test_run_alignment_false_omits_alignment_trajectory_pdf` checked for `alignment_trajectory.pdf` in plot_paths — both broken by VIZ-02 refactor
- **Fix:** Updated assertions to use `alignment_source_trajectory.pdf` (the new first output stem); updated docstring
- **Files modified:** `tests/test_eval_runner.py`
- **Commit:** 433dd16

## Known Stubs

None.

## Threat Flags

None — pure visualization layer, no network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

- eval/viz.py exists and contains 4 private figure helpers plus updated plot_trajectory
- tests/test_viz.py contains 3 new test methods
- tests/test_eval_runner.py updated with new stem assertions
- Commit 433dd16 exists: confirmed via git log
- 972 tests pass, 18 skipped
