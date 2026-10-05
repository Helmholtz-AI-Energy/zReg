---
phase: 56-configurable-multi-label-region-based-labeling-rework-genera
plan: 04
subsystem: testing
tags: [pytest, pydantic, torch, labels, synthetic-data]

# Dependency graph
requires:
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 01)
    provides: "LabelComponentSpec/LabelSpec pydantic models and _component_score/_label_scores scoring primitives"
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 02)
    provides: "generate_labels() as the config-driven orchestrator (n_labels/label_specs paths, deterministic/probabilistic modes, D-07 fix)"
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 03)
    provides: "EvalConfig.label_generation field and DataFactory.generate_training_triple's three-way n_labels precedence"
provides:
  - "TestLabelRegionShapes/TestLabelAssignmentModes/TestLabelGenerationD07Regression test classes in tests/test_generators.py"
  - "tests/test_label_generation_config.py — LabelGenerationConfig/EvalConfig.label_generation pytest coverage, including an end-to-end config-vs-explicit-argument precedence test"
affects: [56-05-remaining-test-suite-fixes]

# Tech tracking
tech-stack:
  added: []
  patterns: ["_single_point_traj/_empty_traj per-class helper methods mirroring the deleted TestSphericalLabelGenerators style"]

key-files:
  created: [tests/test_label_generation_config.py]
  modified: [tests/test_generators.py]

key-decisions:
  - "End-to-end precedence test uses seed=2 (not seed=0) for DataFactory.generate_training_triple — seed=0 leaves one Voronoi label with zero assigned points for both n_labels=3 and n_labels=5, which would make the 'exactly N unique values' assertion fail on a correct implementation; seed=2 was verified to populate every label cell for both call variants"
  - "Probabilistic-majority test uses a >0.5 statistical threshold (not an exact count) per the plan's own framing, since torch.multinomial sampling is inherently stochastic even with a fixed seed's softmax weighting"

patterns-established:
  - "D-07 regression tests build minimal 2/3-frame trajectories with an identical single point position per frame and assert torch.equal across frames — the structural template for any future centers-must-be-frame-invariant regression test"

requirements-completed: [D-07, D-08, D-09, D-10, D-11, D-12, D-13]

# Metrics
duration: ~25min
completed: 2026-07-31
---

# Phase 56 Plan 04: Comprehensive Test Coverage for Configurable Multi-Label Region-Based Labeling Summary

**Added 24 new tests proving per-shape (voronoi/blob/cone) correctness, deterministic-vs-probabilistic assignment behavior, the D-11 mixture-within-a-label case, the D-07 center-stability regression fix, and full `LabelGenerationConfig`/`EvalConfig.label_generation` pydantic coverage including an end-to-end precedence test through `DataFactory.generate_training_triple`.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-31 (immediately after Plan 56-03 completed)
- **Completed:** 2026-07-31
- **Tasks:** 2
- **Files modified:** 2 (1 modified + 1 new), exactly matching the plan's declared `files_modified`

## Accomplishments
- `TestLabelRegionShapes` (7 tests): voronoi/blob/cone "point at own center wins" correctness, the D-11 mixture-within-a-label case (a two-component label beats a competing single-component label whose only component is far), and empty-frame (`N == 0`) handling for both the voronoi and blob/cone code paths.
- `TestLabelAssignmentModes` (6 tests): deterministic mode's bitwise-reproducible argmax behavior, probabilistic mode's valid-sample-from-distribution and statistical-majority behavior, both of D-10's documented probabilistic-mode error conditions (`n_labels` + probabilistic; missing `temperature` on a probabilistic voronoi component), and arbitrary/non-contiguous `label_id` round-tripping.
- `TestLabelGenerationD07Regression` (2 tests): the phase's headline regression test — two/three frames holding an identical single point position produce `torch.equal` labels, for both the `n_labels` simple path and a `label_specs` multi-component mixture path, proving region/component centers are resolved once per trajectory rather than redrawn per frame.
- `tests/test_label_generation_config.py` (10 tests, new file): mirrors `tests/test_alignment_preprocessing_config.py`'s structure exactly — defaults, explicit `label_specs`, both mutual-exclusivity `ValidationError` directions, extra-key-forbidden, `__all__` membership, `EvalConfig` backward-compat (`None` default) and nested-dict coercion, and an end-to-end precedence test proving `EvalConfig.label_generation` applies when the caller omits `n_labels` and that an explicit caller `n_labels` argument overrides it.
- Full suite: 1359 passed / 22 skipped / 1 xpassed / 17 failed — the 17 failures are unchanged from Plan 56-03's baseline (14 pre-existing `n_classes=` kwarg `TypeError`s deferred to Plan 56-05, 2 pre-existing `test_icp_registration.py` full-suite-order flakes already logged in `deferred-items.md`); this plan added 24 new passing tests (1335 -> 1359) with zero new failures.

## Task Commits

Each task was committed atomically:

1. **Task 1: Per-shape, mixture, and assignment-mode tests** - `7aadce8` (test)
2. **Task 2: LabelGenerationConfig and EvalConfig.label_generation tests** - `e6438ed` (test)

## Files Created/Modified
- `tests/test_generators.py` - added `LabelComponentSpec`/`LabelSpec` to the import block; added `TestLabelRegionShapes`, `TestLabelAssignmentModes`, `TestLabelGenerationD07Regression` classes (15 new tests total)
- `tests/test_label_generation_config.py` (new) - 10 tests for `LabelGenerationConfig`/`EvalConfig.label_generation`, mirroring `tests/test_alignment_preprocessing_config.py`'s established style

## Decisions Made
- The end-to-end precedence test's two `DataFactory.generate_training_triple(...)` calls use `seed=2` instead of the plan action block's illustrative `seed=0` — verified by brute-force search that `seed=0` produces a Voronoi partition where one of the 3 (or 5) labels gets zero points for this ball-shape/point-count combination (a legitimate, expected outcome of random nearest-center partitioning with few centers over a bounded point cloud, not a bug), which would make the "exactly N unique values" assertion fail regardless of whether the precedence logic is correct. `seed=2` was confirmed (along with several other seeds) to populate every label cell for both the omitted-`n_labels` and explicit-`n_labels=5` calls, so the assertion actually verifies the precedence invariant the test is designed to prove.
- The probabilistic-mode majority test asserts `> 0.5` (a statistical majority, not an exact count) per the plan's own "statistical, not exact-count" framing — appropriate given `torch.multinomial`'s inherent stochasticity even under a fixed seed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan action block's illustrative `seed=0` for the end-to-end precedence test does not produce the exact unique-label-count invariant it asserts**
- **Found during:** Task 2 verification
- **Issue:** Running `DataFactory(cfg).generate_training_triple(seed=0)` with `label_generation={"n_labels": 3}` produces only 2 unique label values (one Voronoi cell has zero nearest points among the ~198 sampled points), and the `seed=0, n_labels=5` override call produces only 4 unique values — both fail the plan's literal "assert exactly N unique values" wording even though the underlying precedence logic (config applies when omitted; explicit arg overrides) is correct. This is the same category of issue documented in Plan 56-01's summary (a plan-authored verify/test numeric expectation that doesn't hold for the specific example chosen, not an implementation bug).
- **Fix:** Brute-force-searched seeds 0-59 for one where both the `n_labels=3` (config) and `n_labels=5` (explicit override) calls populate every label cell; `seed=2` was the smallest such seed (114 points, all cells populated for both variants). Used `seed=2` in both test functions with an inline docstring/comment explaining why, so a future reader doesn't "fix" it back to `seed=0`.
- **Files modified:** `tests/test_label_generation_config.py` (test-file-only; no production code changed)
- **Verification:** Both precedence tests pass deterministically with `seed=2`; full `tests/test_label_generation_config.py` suite (10 tests) and `tests/test_generators.py` suite (76 tests including the 15 new ones) pass with no regressions.
- **Committed in:** `e6438ed` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 test-data bug in the plan's own illustrative seed choice, not in implementation)
**Impact on plan:** No scope creep — every test listed in the plan's action block was implemented with the exact assertions described; only the specific numeric seed argument in the end-to-end precedence test was substituted for one that actually exercises the invariant the test claims to prove, mirroring Plan 56-01's precedent for this exact class of issue.

## Issues Encountered
- Initial verification runs used a `cd /Users/valeriekieslinger/Documents/Hiwi/BA/zReg && ...` Bash invocation that silently resolved to the main repository checkout rather than this plan's worktree (`.../.claude/worktrees/generate-labels-vision-969cb0`) — caught immediately when the new test classes appeared to be "not collected" (pytest was running against the main repo's stale `tests/test_generators.py`, which still had the pre-Plan-56-02 `TestSphericalLabelGenerators` class). No commits were made from the wrong location; all subsequent commands were re-run with the correct worktree path and verified via `git rev-parse --show-toplevel`/`--git-dir` before every commit, consistent with the #3097 cwd-drift hazard documented in Plan 56-02's summary.
- Local environment requires `KMP_DUPLICATE_LIB_OK=TRUE` for ad-hoc `python -c` verification snippets (same pre-existing environment quirk noted in Plans 56-01/56-02/56-03's summaries); `pytest`'s own `conftest.py` import ordering was unaffected.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All of Phase 56's core correctness guarantees (per-shape scoring, assignment modes, D-11 mixture, D-07 center-stability) and its config-wiring surface (`LabelGenerationConfig`/`EvalConfig.label_generation`, including the `DataFactory` precedence chain) are now verified by dedicated, passing pytest coverage.
- Plan 56-05 (fixing the remaining 14 pre-existing `n_classes=` kwarg `TypeError`s across `tests/test_data_factory_training_triples.py`, `tests/test_benchmark_runner.py`, `tests/test_train_label_transfer.py`, `tests/test_zreg_models_pointnet2.py`) is unaffected by and independent of this plan's changes — those failures were present before this plan ran and remain identical in count and location afterward.
- No blockers for Plan 56-05.

---
*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Completed: 2026-07-31*

## Self-Check: PASSED

- FOUND: tests/test_generators.py
- FOUND: tests/test_label_generation_config.py
- FOUND: 7aadce8 (Task 1 commit)
- FOUND: e6438ed (Task 2 commit)
