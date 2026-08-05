---
phase: 58-synthetic-labeled-subsample-pair-generation
plan: 01
subsystem: evaluation-framework
tags: [pytorch, pydantic, ground-truth, label-transfer, data-factory, hpo]

# Dependency graph
requires:
  - phase: 57-ground-truth-field-consistency
    provides: EvalConfig.ground_truth_field, DataFactory._correspondence_idx threaded through
      drop_points()/sample_new_points(), get_synthetic_ground_truth() correspondence-gather
  - phase: 31-synthetic-pipeline
    provides: EvalConfig.transform_spec, generate_target()/get_synthetic_ground_truth(),
      pipeline_mode="synthetic"
  - phase: 46-training-data-pipeline
    provides: generate_training_triple()'s sample_ball/sample_bowl + generate_labels
      synthesize-from-scratch geometry pattern (mirrored, not called, by this plan's method)
provides:
  - DataFactory.generate_subsample_pair(dataset, transform_spec) -> (source_view, target_view)
  - DataFactory._subsample_source_view instance attribute (D-01/D-02/D-03/D-05)
  - eval/config.py transform_spec docstring documenting "subsample_pair" as a third
    recognised "type" value (D-04, docs only, no schema change)
affects: [58-02, 58-03, 58-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sibling-method composition: generate_subsample_pair() calls drop_points() twice
      (source/target views) and optionally generate_target() (D-03 perturbation) rather
      than reimplementing correspondence tracking or transform application"
    - "Explicit self._correspondence_idx = None reset before each of two sequential
      drop_points() calls on the same base dataset, with immediate local-variable capture
      after each call, to avoid drop_points()'s prior-state composition behavior"

key-files:
  created: []
  modified:
    - eval/data_factory.py
    - eval/config.py
    - tests/test_data_factory.py

key-decisions:
  - "Perturbation keys explicitly whitelisted to rotation_deg/rotation_axis/scale_factor/
    sigma when layering the D-03 conditional transform onto the target view — dropout_fraction/
    n_outliers/n_new_points/augment_seed are deliberately excluded because a second dropout
    on top of the view subsampling would require additional correspondence composition this
    method does not perform"
  - "When run_alignment=True and no explicit perturbation key is present, a default
    rotation+scale transform is derived from transform_spec['seed'] via random.Random(seed),
    mirroring generate_training_triple()'s own default transform construction rather than
    inventing a new default-generation convention"
  - "generate_target()'s internal overwrite of self._source_dataset/self._correspondence_idx
    is explicitly restored immediately after the call (to base_dataset/target_corr) so the
    producer contract get_synthetic_ground_truth() expects is preserved without modifying
    that consumer method"

requirements-completed: [GT-04, GT-05]

# Metrics
duration: ~25min
completed: 2026-08-04
---

# Phase 58 Plan 01: Synthetic Labeled Subsample-Pair Generation Summary

**`DataFactory.generate_subsample_pair()` subsamples a labeled point cloud into two independently-dropped (source, target) views with tracked per-point correspondence, reusing Phase 57's `drop_points()`/`_correspondence_idx` machinery and remaining drop-in compatible with `get_synthetic_ground_truth()`.**

## Performance

- **Duration:** ~25 min active work
- **Started:** 2026-08-04T13:44:12+02:00
- **Completed:** 2026-08-04T13:51:00+02:00
- **Tasks:** 2 completed
- **Files modified:** 3 (eval/data_factory.py, eval/config.py, tests/test_data_factory.py)

## Accomplishments

- `DataFactory.generate_subsample_pair(dataset, transform_spec)` — new sibling method to the
  known-transform (`generate_target`) path: subsamples a base labeled point cloud TWICE
  (different seeds) into a `(source_view, target_view)` pair, reusing `drop_points()` and
  `self._correspondence_idx` rather than reimplementing index tracking
- Two input modes (D-02): an already-loaded dataset (real data via `load_real()`/`load_target()`,
  or any `dict[int, zRegPointCloud]`), or a synthesize-from-scratch mode (`transform_spec["synthesize"]
  = True`) mirroring `generate_training_triple()`'s `sample_ball`/`sample_bowl` + `generate_labels`
  geometry pipeline without calling or modifying that method
- D-03 conditional transform: when `config.run_alignment` is `True`, a rigid/noise-style
  perturbation (explicit or seed-derived default) is layered onto the target view via
  `generate_target()`; when `False`, source and target stay in the same coordinate frame
- Producer-contract compatibility: `self._source_dataset`/`self._correspondence_idx` are set to
  match `generate_target()`'s own contract after the call, so `get_synthetic_ground_truth()`
  works completely unmodified against `generate_subsample_pair()`'s output
- List-valued `transform_spec["seed"]` is rejected with a `ValueError` directing multi-seed
  callers to `HyperparamOptimizer` (D-06/D-07 — that resolution is an `optimizer.py`-level
  concern, out of this method's scope)
- `eval/config.py`'s `transform_spec` docstring documents `"subsample_pair"` as a third
  recognised `"type"` value (D-04, docs only, no schema change)

## Task Commits

Each task was committed atomically:

1. **Task 1: DataFactory.generate_subsample_pair() + _subsample_source_view attribute + config docstring** - `025cad0` (feat)
2. **Task 2: Tests for generate_subsample_pair()** - `c8b2e8b` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `eval/data_factory.py` - Added `self._subsample_source_view` instance attribute (`__init__`);
  new `generate_subsample_pair()` method placed between `generate_target()` and
  `generate_training_triple()`, composing `drop_points()`/`generate_target()` per the plan's
  7-step implementation (seed validation, fraction validation, base-dataset construction,
  twice-subsample with reset-before-each-call, optional D-03 perturbation, final state
  reconciliation, return)
- `eval/config.py` - `transform_spec` Attributes docstring entry extended with one sentence
  documenting `"subsample_pair"` as a third `"type"` value
- `tests/test_data_factory.py` - New `TestGenerateSubsamplePair` class (13 tests) placed after
  `TestGenerateTarget`, before `TestGenerateSynthetic`: validation-error coverage (list-seed,
  missing-dataset, both fraction boundaries on both `source_fraction`/`target_fraction`),
  `run_alignment=True` real-geometric-transform proof, `run_alignment=False` identity-subsample
  proof, source/target-are-different-selections proof, instance-state assertions, GT-extraction
  length/value-correctness, synthesize-mode labeled-pair proof, cross-instance reproducibility

## Decisions Made

- Followed the plan's corrected Task 1 step 4 exactly: `self._correspondence_idx = None` reset
  immediately before EACH of the two `drop_points()` calls (not just once before both), with the
  resulting correspondence tensor captured into a local variable (`source_corr`/`target_corr`)
  immediately after each call, before the next reset clears it. This is the bug the plan-checker
  caught in the original (uncorrected) plan draft — verified the implementation matches the fixed
  version, not the buggy one (see Success Criteria verification below).
- Excluded `dropout_fraction`/`n_outliers`/`n_new_points`/`augment_seed` from the whitelisted
  perturbation keys forwarded to `generate_target()` in the `run_alignment=True` branch, per the
  plan's explicit scope boundary (a second dropout on top of view subsampling would require
  correspondence composition this method does not implement).
- Test fixture (`_make_ds_100pts`) uses a single-frame, 100-point dataset with both `label`
  (`torch.arange(100, dtype=torch.long)`) and `id` populated, mirroring `TestDropPoints`'s
  `_make_ds_100pts` convention (distinct per-point values enable independent correspondence
  verification) rather than reusing that exact private helper across test classes.

## Deviations from Plan

None - plan executed exactly as written (including the corrected Task 1 step 4 the plan-checker
flagged prior to this execution).

## Issues Encountered

None. One pre-existing, unrelated test failure was observed during the full-suite verification
run (`tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery`) — this
test file was not touched by either commit in this plan (confirmed via `git log` showing its last
modification predates this plan), fails identically in isolation, and was already documented as a
pre-existing flaky/unrelated failure in Phase 57's summary. Out of this plan's scope per the Scope
Boundary rule; no action taken.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `DataFactory.generate_subsample_pair()` and `self._subsample_source_view` are the reusable
  building blocks Plans 58-02/58-03 (Wave 2) need to wire `transform_spec["type"] ==
  "subsample_pair"` dispatch into `eval/runners/eval_runner.py` and `eval/runners/optimizer.py`'s
  three call sites, per 58-PATTERNS.md's integration-point mapping.
- `get_synthetic_ground_truth()` requires no changes for the `subsample_pair` case — confirmed via
  the new `test_get_synthetic_ground_truth_matches_target_length_and_values` test — so downstream
  wiring in 58-02/58-03 only needs the `type == "subsample_pair"` dispatch branch itself, not any
  GT-extraction changes.
- Full regression suite (`pytest -q`, 1390 passed / 22 skipped / 1 xpassed) shows zero new
  failures beyond the single pre-existing, unrelated ICP flaky test noted above.

---
*Phase: 58-synthetic-labeled-subsample-pair-generation*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: eval/data_factory.py (DataFactory.generate_subsample_pair exists, verified via
  `python -c "from eval.data_factory import DataFactory; assert hasattr(DataFactory, 'generate_subsample_pair')"`)
- FOUND: eval/config.py (contains "subsample_pair" substring, verified via grep)
- FOUND: tests/test_data_factory.py (TestGenerateSubsamplePair class, 13 tests, all passing)
- FOUND commit 025cad0 in `git log --oneline --all`
- FOUND commit c8b2e8b in `git log --oneline --all`
