---
phase: 56-ground-truth-field-consistency
plan: 01
subsystem: evaluation-framework
tags: [pytorch, pydantic, ground-truth, label-transfer, f1, data-factory]

# Dependency graph
requires:
  - phase: 38-color-to-label-rename
    provides: pc["label"] as the canonical class-label field (CLN-01/02)
  - phase: 31-synthetic-pipeline
    provides: EvalConfig.transform_spec, generate_target/get_synthetic_ground_truth, pipeline_mode="synthetic"
provides:
  - EvalConfig.ground_truth_field (Literal["id", "label"] = "label", D-01/D-02)
  - DataFactory._correspondence_idx per-frame original-index map, threaded through
    drop_points()/sample_new_points(), reset at the start of every generate_target() call
  - get_ground_truth()/get_synthetic_ground_truth() reading pc[config.ground_truth_field]
    instead of hardcoded pc["id"]
  - get_synthetic_ground_truth() gathering y_true by tracked correspondence (not truncating
    positionally) when dropout_fraction/n_new_points change point counts, with -1 sentinel
    for uncorrelated new points
  - eval_runner._run_single WR-01 truncation re-scoped (comment only) as a paired-mode-only
    defensive fallback
affects: [57-synthetic-labeled-subsample-pair-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reusable per-frame correspondence-index map (dict[int, torch.Tensor]) threaded
      through augmentation methods that change point counts, following the
      AlignResult.estep_results per-frame diagnostic-dict precedent"

key-files:
  created: []
  modified:
    - eval/config.py
    - eval/data_factory.py
    - eval/runners/eval_runner.py
    - tests/test_data_factory.py
    - tests/test_eval_runner.py

key-decisions:
  - "ground_truth_field is a closed Literal['id','label'] field (D-02) — no free-form
    override, following the pipeline_mode Literal pattern rather than the
    Field(description=...) + validator pattern used for plain str fields"
  - "Correspondence map stored as a DataFactory instance attribute (self._correspondence_idx),
    matching the existing _synthetic_target/_source_dataset/_transform_spec convention,
    rather than returned alongside the augmented dataset"
  - "eval_runner's WR-01 positional-truncation block is left functionally unchanged —
    only its comment is re-scoped — since it remains the correct fallback for paired-mode
    heterogeneous real-data cases where no correspondence can be computed"

requirements-completed: [GT-01, GT-02]

# Metrics
duration: ~40min (active; spans a session gap between Task 1 and Task 2 commits)
completed: 2026-08-03
---

# Phase 56 Plan 01: Ground-Truth Field Consistency Summary

**F1 ground truth now reads `pc["label"]` by default (with a validated `"id"` escape hatch) and `get_synthetic_ground_truth()` gathers y_true by a tracked correspondence index instead of silently truncating positionally when dropout/new-points change point counts.**

## Performance

- **Duration:** ~40 min active work (git commits span a longer wall-clock gap due to a session interruption between Task 1 and Task 2)
- **Started:** 2026-08-03T14:32:04+02:00 (Phase 56 execution session start)
- **Completed:** 2026-08-03T23:10:27Z
- **Tasks:** 2 completed
- **Files modified:** 5 (eval/config.py, eval/data_factory.py, eval/runners/eval_runner.py, tests/test_data_factory.py, tests/test_eval_runner.py)

## Accomplishments

- `EvalConfig.ground_truth_field: Literal["id", "label"] = "label"` — F1 ground truth defaults to the field `LabelTransferStage.run()` actually transfers, fixing GT-01's field mismatch (previously hardcoded to `pc["id"]`, a per-point tracking identity)
- `DataFactory._correspondence_idx` — a reusable per-frame original-index correspondence map threaded through `drop_points()`/`sample_new_points()`, composing correctly when both are applied in sequence and resetting cleanly at the start of every `generate_target()` call
- `get_synthetic_ground_truth()` now gathers `y_true` by that tracked correspondence instead of relying on identity/positional alignment, so its length matches the target's actual post-dropout/new-points point count (GT-02); newly-added points get a `-1` sentinel that `compute_f1` already excludes
- `eval_runner._run_single`'s WR-01 truncation comment re-scoped to document it as a paired-mode-only defensive fallback, no longer the correctness mechanism for synthetic mode

## Task Commits

Each task was committed atomically:

1. **Task 1: EvalConfig.ground_truth_field contract + correspondence-tracking producers** - `1e24d0a` (feat)
2. **Task 2: GT-extraction consumers + eval_runner WR-01 re-scoping** - `6d57066` (fix)

**Plan metadata:** (this commit)

## Files Created/Modified

- `eval/config.py` - Added `ground_truth_field: Literal["id", "label"] = "label"` field + Attributes docstring entry
- `eval/data_factory.py` - Added `self._correspondence_idx`; reset in `generate_target()`; correspondence tracking in `drop_points()`/`sample_new_points()`; `get_ground_truth()`/`get_synthetic_ground_truth()` rewritten to read `config.ground_truth_field` and gather by correspondence
- `eval/runners/eval_runner.py` - WR-01 comment block re-scoped to document paired-mode-only fallback status (no code change)
- `tests/test_data_factory.py` - New tests for `ground_truth_field` validation, correspondence tracking (fresh + chained), GT-02 gather behavior (dropout-only, new-points-only, combined, `compute_f1` compatibility); updated `id`-path tests to explicitly set `ground_truth_field="id"`; added default-`"label"`-path test; clarifying docstring notes on coincidentally-passing tests
- `tests/test_eval_runner.py` - New synthetic-mode integration test proving GT-02's non-degenerate F1 under a post-dropout/new-points ground truth with a sentinel entry

## Decisions Made

- `ground_truth_field` uses the `pipeline_mode`-style bare `Literal` field pattern (no custom `@field_validator`) per the plan's explicit instruction, since pydantic validates `Literal` membership automatically at construction.
- The correspondence map is a `DataFactory` instance attribute (`self._correspondence_idx`), matching the established `_synthetic_target`/`_source_dataset`/`_transform_spec` producer/consumer convention already used for `generate_target()` state, rather than a value threaded through return values.
- One test literal in the plan's Task 2 acceptance criteria (`dropout_fraction: 0.3` on a 20-point source producing exactly 10 post-dropout points) doesn't match the `round(n * (1 - fraction))` formula actually used by `drop_points` (`round(20 * 0.7) = 14`, not 10). Rather than hard-code the plan's arithmetic, the corresponding test (`test_gather_matches_target_length_after_dropout`) asserts `gt[k].shape[0] == factory._synthetic_target[k]["pos"].shape[0]` dynamically — this proves the same GT-02 correctness property (length matches the actual target, not the original source count) without asserting an incorrect literal. The combined-transform test (`dropout_fraction=0.5, n_new_points=3` → 13) does match the plan's math exactly and is asserted literally.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's dropout-only test literal was arithmetically incorrect**
- **Found during:** Task 2 (writing the GT-02 length-matching test)
- **Issue:** The plan's acceptance criteria asserted that `dropout_fraction: 0.3` on a 20-point source produces `gt[k].shape[0] == 10`, but `drop_points`'s actual formula (`round(n * (1.0 - fraction))`) produces `round(20 * 0.7) = 14`, not 10.
- **Fix:** Wrote the test to assert dynamic equality against `factory._synthetic_target[k]["pos"].shape[0]` (the authoritative post-dropout count) plus a `target_n < 20` sanity check and a value-correctness check via the correspondence index, instead of hard-coding the plan's incorrect literal `10`. The combined-transform acceptance criterion (`dropout_fraction=0.5, n_new_points=3` → 13), whose arithmetic does check out (`round(20*0.5)=10`, `+3=13`), is asserted literally as specified.
- **Files modified:** tests/test_data_factory.py
- **Verification:** `pytest tests/test_data_factory.py -k TestGetSyntheticGroundTruth` passes; test explicitly checks `target_n < 20` and gathered-value correctness, which is a strictly stronger proof of GT-02 than the plan's literal would have been.
- **Committed in:** 6d57066 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — corrected an arithmetic error in the plan's own acceptance-criteria text; did not affect production code)
**Impact on plan:** No scope creep — the underlying GT-02 fix (`get_synthetic_ground_truth`'s correspondence gather) matches the plan's `<action>` block exactly; only a test assertion's literal value was corrected.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- GT-01 and GT-02 are fixed and covered by tests; `EvalConfig.ground_truth_field` and `DataFactory._correspondence_idx` are the reusable building blocks Phase 57 (Synthetic Labeled Subsample-Pair Generation) needs for its own "index subset with tracked correspondence" requirement (D-04).
- This plan intentionally left GT-03 (Shah/Kobitski config audit and stale-comment fixes, and the fully-synthetic `stage2_label_transfer/*/synthetic.yaml` degenerate-label audit) to the sibling plan 56-02, which touches `baseline_experiments/configs/ground_truth/*.yaml` and `configs/experiments/stage2_label_transfer/*/synthetic.yaml` — disjoint files from this plan's scope.
- Full regression suite (`pytest -q`) passes except one pre-existing, unrelated failure (`tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery`), confirmed via `git stash` to fail identically on the pre-Task-2 tree and to pass in isolation (test-order-dependent RNG-state flakiness in an unrelated ICP test, out of this plan's scope per the Scope Boundary rule).

---
*Phase: 56-ground-truth-field-consistency*
*Completed: 2026-08-03*

## Self-Check: PASSED

All modified files verified present on disk; both task commits (`1e24d0a`, `6d57066`) verified present in `git log --oneline --all`.
