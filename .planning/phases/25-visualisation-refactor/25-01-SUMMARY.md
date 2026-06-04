---
phase: 25-visualisation-refactor
plan: 01
subsystem: eval
tags: [matplotlib, pydantic, plot_trajectory, eval_viz, eval_config, tdd]

# Dependency graph
requires:
  - phase: 24-trajectory-export
    provides: export_trajectory, EvalReport.trajectory_paths, EvaluationRunner wiring
  - phase: 21-evaluation-runner-visualisation
    provides: eval/viz.py with plot_point_cloud, plot_metrics_summary (FRAME-08 patterns)
  - phase: 20-labeltransferstage
    provides: LabelResult with transferred_labels dict[int, torch.Tensor]
  - phase: 19-alignmentstage
    provides: AlignResult with aligned_cloud dict[int, zRegPointCloud]
  - phase: 17-framework-config-datafactory
    provides: EvalConfig pydantic v2 BaseModel with extra="forbid"
provides:
  - plot_trajectory(align_result, label_result, dataset, label_names, output_dir) -> list[str]
  - plot_metrics(report, path) -> None (renamed from plot_metrics_summary)
  - EvalConfig.label_names field (dict[int, str] | None = None)
  - Two independent 1x3 figure pairs: alignment_trajectory.pdf/png and label_trajectory.pdf/png
  - EvaluationRunner.run() save_plots branch wired to plot_trajectory + plot_metrics
affects: [phase-25-plan-02, eval-runner-tests, visualisation-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "plot_trajectory returns list[str] of written paths (0/2/4 files, mirrors export_trajectory)"
    - "Two independent 1x3 figure pairs replace single 2x3 figure (D-03)"
    - "matplotlib.colormaps.get_cmap('tab10') — non-deprecated colormap API"
    - "label_names.get(lab, str(lab)) fallback pattern for optional legend labels"
    - "Frame selection: sorted_keys[0], sorted_keys[len//2], sorted_keys[-1] (mirrors trajectory.py)"

key-files:
  created: []
  modified:
    - eval/viz.py
    - eval/config.py
    - eval/runners/eval_runner.py
    - tests/test_viz.py
    - tests/test_eval_runner.py

key-decisions:
  - "plot_trajectory returns list[str] (empty when both inputs None) — mirrors export_trajectory pattern"
  - "Two independent file pairs replace single 2x3 figure — eliminates blank-panel complexity (D-03)"
  - "label_names: dict[int, str] | None = None added to EvalConfig as explicit pydantic field (D-08)"
  - "generate_labels produces 1-D int64 color tensors; fake_label_result fixture uses .long() not [:, 0].long()"
  - "EvaluationRunner wired to plot_trajectory per D-10; test_eval_runner updated to assert alignment_trajectory.pdf"

patterns-established:
  - "Pattern: plot_trajectory as two-stage conditional function (align_result guard + label_result guard)"
  - "Pattern: matplotlib.colormaps.get_cmap (not cm.get_cmap) for non-deprecated colormap access"

requirements-completed:
  - EXT-02

# Metrics
duration: 6min
completed: 2026-06-04
---

# Phase 25 Plan 01: Visualisation Refactor (plot_trajectory + EvalConfig.label_names) Summary

**Replaced plot_point_cloud with plot_trajectory (two independent 1x3 PDF+PNG pairs, conditional on stage results), renamed plot_metrics_summary to plot_metrics, and added label_names field to EvalConfig — satisfying EXT-02**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-04T13:11:00Z
- **Completed:** 2026-06-04T13:17:00Z
- **Tasks:** 3 (TDD: RED → GREEN → GREEN)
- **Files modified:** 5

## Accomplishments

- Implemented `plot_trajectory` in eval/viz.py: produces alignment_trajectory.pdf/png and/or label_trajectory.pdf/png depending on which stages ran; returns list[str] of written paths
- Renamed `plot_metrics_summary` to `plot_metrics`; removed `plot_point_cloud` entirely; `__all__` updated
- Added `label_names: dict[int, str] | None = None` field to EvalConfig; passes pydantic v2 validation
- Updated EvaluationRunner.run() save_plots branch to use plot_trajectory + plot_metrics (D-10)
- All 9 test_viz.py tests pass (TestPlotMetrics 3 + TestPlotTrajectory 6); full suite at 789 passing, 17 skipped

## Task Commits

Each task was committed atomically:

1. **Task 1: Wave 0 — Write failing stubs in tests/test_viz.py (RED)** - `7116aef` (test)
2. **Task 2: Implement eval/viz.py — GREEN** - `0ae200b` (feat)
3. **Task 3: Add EvalConfig.label_names + populate TestPlotTrajectory stubs — GREEN** - `cfcf73c` (feat)

_TDD plan: test(RED) → feat(GREEN) → feat(GREEN+full-stubs)_

## Files Created/Modified

- `eval/viz.py` — Removed plot_point_cloud; added plot_trajectory (D-01..D-07); renamed plot_metrics_summary to plot_metrics; __all__ updated
- `eval/config.py` — Added label_names: dict[int, str] | None = None field after metric_weights; updated class docstring
- `eval/runners/eval_runner.py` — Updated import and save_plots branch to use plot_trajectory + plot_metrics per D-10
- `tests/test_viz.py` — Rewrote: removed TestPlotPointCloud, renamed TestPlotMetricsSummary to TestPlotMetrics, added TestPlotTrajectory (6 tests), added fake_label_result fixture
- `tests/test_eval_runner.py` — Updated test_save_plots_true_creates_point_cloud_pdf → test_save_plots_true_creates_alignment_trajectory_pdf

## Decisions Made

- `plot_trajectory` returns `list[str]` of paths written (empty when both inputs are None) — mirrors the `export_trajectory` Phase 24 pattern for consistency
- Two independent file pairs (alignment_trajectory + label_trajectory) replace the originally-planned single 2×3 figure — eliminates blank-panel complexity when only one stage runs (D-03)
- `label_names: dict[int, str] | None = None` uses no `Field()` wrapper — simple default suffices with pydantic v2
- `matplotlib.colormaps.get_cmap("tab10")` over deprecated `cm.get_cmap` per plan RESEARCH constraints

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed fake_label_result fixture: color tensor already 1-D**
- **Found during:** Task 2 (running pytest after GREEN implementation)
- **Issue:** Plan's Pitfall 2 note said "generate_labels produces color tensor shape (N,3); use color[:, 0].long()" — but the actual generate_labels output produces 1-D int64 tensors (shape (N,)), causing IndexError: too many indices for tensor of dimension 1
- **Fix:** Changed `synthetic_dataset_3[k]['color'][:, 0].long()` to `synthetic_dataset_3[k]['color'].long()` in the fake_label_result fixture
- **Files modified:** tests/test_viz.py
- **Verification:** 4 TestPlotTrajectory errors resolved; all 9 test_viz.py tests pass
- **Committed in:** `0ae200b` (Task 2 commit, included fixture fix)

**2. [Rule 3 - Blocking] Updated EvaluationRunner imports and save_plots branch**
- **Found during:** Task 3 (running full pytest suite after implementing Task 3)
- **Issue:** eval/runners/eval_runner.py still imported `plot_metrics_summary, plot_point_cloud` from eval.viz — causing ImportError that made test_eval_runner.py and test_cli.py uncollectable
- **Fix:** Updated import to `plot_metrics, plot_trajectory`; rewired save_plots branch per D-10; updated test_eval_runner.py to assert `alignment_trajectory.pdf` instead of `point_cloud.pdf`
- **Files modified:** eval/runners/eval_runner.py, tests/test_eval_runner.py
- **Verification:** 789 tests pass, 17 skipped; no regressions
- **Committed in:** `cfcf73c` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 3 blocking)
**Impact on plan:** Both fixes necessary for correctness. The fixture fix resolves an inaccurate plan pitfall note; the runner fix was required to keep the full suite green after the viz API rename. No scope creep.

## Issues Encountered

- The plan's RESEARCH Pitfall 2 about `generate_labels` producing `(N,3)` color tensors was inaccurate for this environment — the actual tensors are 1-D int64. Auto-fixed via Rule 1.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- EXT-02 satisfied: plot_trajectory + plot_metrics exported from eval.viz; EvalConfig.label_names available
- Phase 25 Plan 02 can proceed: EvaluationRunner save_plots branch is wired to the new API
- 789 tests pass (17 skipped) — baseline established for Phase 25 Plan 02

## Self-Check: PASSED

- FOUND: eval/viz.py
- FOUND: eval/config.py
- FOUND: eval/runners/eval_runner.py
- FOUND: tests/test_viz.py
- FOUND: .planning/phases/25-visualisation-refactor/25-01-SUMMARY.md
- FOUND commit: 7116aef (test(25-01): RED stubs)
- FOUND commit: 0ae200b (feat(25-01): GREEN implementation)
- FOUND commit: cfcf73c (feat(25-01): label_names + stubs GREEN)
