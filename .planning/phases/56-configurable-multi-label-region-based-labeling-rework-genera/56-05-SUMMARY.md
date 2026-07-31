---
phase: 56-configurable-multi-label-region-based-labeling-rework-genera
plan: 05
subsystem: testing
tags: [pytest, n_classes-rename, generate_labels, DataFactory, synthetic-data]

# Dependency graph
requires:
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 02)
    provides: "generate_labels() renamed n_classes -> n_labels"
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 03)
    provides: "DataFactory.generate_training_triple/generate_training_set renamed n_classes -> n_labels (sentinel precedence)"
provides:
  - "Full test suite green again after the n_classes -> n_labels rename cascade from Plans 56-02/56-03"
  - "No remaining generate_labels/generate_training_triple/generate_training_set call anywhere in tests/ using n_classes="
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: [tests/test_benchmark_runner.py, tests/test_data_factory_training_triples.py, tests/test_zreg_models_pointnet2.py, tests/test_train_label_transfer.py]

key-decisions:
  - "Task 1 (5 files: test_optimizer.py, test_viz.py, test_trajectory_export.py, test_label_transfer_stage.py, test_eval_runner.py) required no edits — Plan 56-02's executor already fixed every generate_labels(..., n_classes=...) call site in these files as a Rule-1 deviation. Verified via re-grep and full acceptance-criteria test run (208 passed) rather than re-editing already-correct code."
  - "The plan's own Task 1 verify grep (`generate_labels(` -A2 | grep n_classes=) produces one textual false positive in test_label_transfer_stage.py:675 — a docstring sentence legitimately mentions both the unrelated model-hyperparameter n_classes=4 and the already-renamed generate_labels(..., n_labels=4) in the same line. No further edit needed; this is expected per the plan's own interfaces section."

requirements-completed: [D-04]

# Metrics
duration: ~15min
completed: 2026-07-31
---

# Phase 56 Plan 05: Remaining Test-Suite n_classes -> n_labels Rename Cascade Summary

**Closed the last 4 unfixed test-file call sites (test_benchmark_runner.py's generate_training_set, plus test_data_factory_training_triples.py/test_zreg_models_pointnet2.py/test_train_label_transfer.py's generate_training_triple/generate_training_set calls) for the n_classes -> n_labels rename, bringing the full 1398-test suite back to green except the 2 already-logged pre-existing test_icp_registration.py full-suite-order flakes.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-31 (immediately after Plan 56-04 completed)
- **Completed:** 2026-07-31T15:03:09Z
- **Tasks:** 2
- **Files modified:** 4 (Task 2 only; Task 1 required no edits)

## Accomplishments
- Verified (via re-grep and a full test run of the 5 affected files, 208 passed) that Task 1's target call sites in `tests/test_optimizer.py`, `tests/test_viz.py`, `tests/test_trajectory_export.py`, `tests/test_label_transfer_stage.py`, and `tests/test_eval_runner.py` are already fully renamed to `n_labels=` by Plan 56-02's documented Rule-1 deviation — no changes needed.
- Renamed the remaining 5 real call sites across 4 files to close out D-04's rename cascade:
  - `tests/test_benchmark_runner.py:307` — `generate_training_set(seeds, n_classes=N_CLASSES)` -> `n_labels=N_CLASSES`
  - `tests/test_data_factory_training_triples.py:213, 225` — two `generate_training_triple(seed=0, n_classes=6)` -> `n_labels=6`
  - `tests/test_zreg_models_pointnet2.py:104` — `generate_training_triple(seed=1, n_classes=n_classes)` -> `n_labels=n_classes` (local variable name `n_classes` on the right-hand side deliberately untouched)
  - `tests/test_train_label_transfer.py:123, 146, 205` — two `generate_training_triple(..., n_classes=N_CLASSES)` and one `generate_training_set(range(16), n_classes=N_CLASSES)` -> `n_labels=N_CLASSES`
- Confirmed unrelated `n_classes` usages in `test_train_label_transfer.py` (model-hyperparameter dicts at lines 53-54, `args.n_classes` at line 226, `test_main_validates_n_classes_one` test name at line 261) remain completely untouched.
- Full suite (`python -m pytest tests/ -q`): **1374 passed, 22 skipped, 1 xpassed, 2 failed** — the 2 failures are exactly the already-logged `test_icp_registration.py::test_icp_translation_recovery`/`test_icp_rotation_recovery` full-suite-order flakes (see `deferred-items.md`), confirmed pre-existing and out of scope. All 14 `n_classes=` `TypeError`s from Plan 56-03's baseline are resolved.

## Task Commits

1. **Task 1: Rename n_classes to n_labels at generate_labels call sites (5 files)** - no commit (verified already satisfied by Plan 56-02's deviation fix `e8bdc24`; no changes made)
2. **Task 2: Rename n_classes to n_labels at generate_training_triple/generate_training_set call sites (4 files)** - `1327986` (test)

## Files Created/Modified
- `tests/test_benchmark_runner.py` - `_build_holdout_pair`'s `generate_training_set` call renamed `n_classes` -> `n_labels`
- `tests/test_data_factory_training_triples.py` - `TestLabelVsIdDiscipline`'s two `generate_training_triple` calls renamed `n_classes` -> `n_labels`
- `tests/test_zreg_models_pointnet2.py` - `TestBowlSparseRobustness`'s `generate_training_triple` call renamed keyword only (local var `n_classes` untouched)
- `tests/test_train_label_transfer.py` - three call sites (`test_train_step_forward_backward`, `test_loss_decreases_over_a_handful_of_steps`, `test_mps_train_step_runs_on_mps`) renamed `n_classes` -> `n_labels`; model-hyperparameter dicts and CLI-arg usages left unchanged

## Decisions Made
- Task 1 was verified rather than re-edited: re-grepping each of the 5 target files confirmed every `generate_labels(` call site already used `n_labels=`, a direct consequence of Plan 56-02's documented cross-file Rule-1 deviation. Re-running the 5 files' test suite (208 passed) served as positive confirmation instead of touching already-correct code.
- The plan's own Task 1 automated verify command (`grep "generate_labels(" -A2 | grep "n_classes="`) surfaces one textual match in `test_label_transfer_stage.py:675`, a docstring sentence describing an unrelated model hyperparameter (`n_classes=4`) that happens to share a line with the already-renamed `generate_labels(..., n_labels=4)` reference. This is not an unfixed call site and needed no edit — consistent with the plan's own interfaces section, which explicitly calls out this exact docstring line as already-updated.

## Deviations from Plan

None - plan executed exactly as written. Task 1 was a no-op as anticipated by the plan's own "Post-56-02 note"; Task 2's 5 real call sites were fixed with pure mechanical kwarg renames, no other logic touched.

## Issues Encountered

None. Full suite verification matched expectations exactly: 14 pre-existing `n_classes=` `TypeError`s resolved, 2 pre-existing `test_icp_registration.py` full-suite-order flakes remain (already logged in `deferred-items.md` from Plan 56-02, confirmed out of scope, not attempted).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 56 (Configurable multi-label region-based labeling rework) is now fully complete: all 5 plans (56-01 through 56-05) executed, `generate_labels()`/`DataFactory.generate_training_triple`/`generate_training_set`'s `n_classes` -> `n_labels` rename cascade (D-04) is closed across production code, `eval/`, and the entire `tests/` suite.
- No blockers. The only remaining suite-level issue is the pre-existing, unrelated `test_icp_registration.py` full-suite-order flake, logged for future investigation but explicitly out of scope for this phase.
- Next milestone action (per STATE.md): `/gsd:plan-phase 54` — Budget Calibration & Full-Suite Gate (BUDG-02, BUDG-03), which requires real HoreKa timing data from Phases 52/53.

---
*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Completed: 2026-07-31*

## Self-Check: PASSED

- FOUND: tests/test_benchmark_runner.py
- FOUND: tests/test_data_factory_training_triples.py
- FOUND: tests/test_zreg_models_pointnet2.py
- FOUND: tests/test_train_label_transfer.py
- FOUND: 1327986 (Task 2 commit)
