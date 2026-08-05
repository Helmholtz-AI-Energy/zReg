---
phase: 58-synthetic-labeled-subsample-pair-generation
plan: 03
subsystem: evaluation-framework
tags: [pytorch, pydantic, ground-truth, label-transfer, data-factory, hpo, optuna]

# Dependency graph
requires:
  - phase: 58-01
    provides: "DataFactory.generate_subsample_pair(dataset, transform_spec) -> (source_view,
      target_view), DataFactory._subsample_source_view, producer-contract compatible with
      get_synthetic_ground_truth()"
provides:
  - "HyperparamOptimizer._is_subsample_pair_mode() predicate helper"
  - "run()'s pre-populate step subsample_pair branch (generate_subsample_pair() instead of
    generate_target(), single fixed seed for every trial per D-06)"
  - "_tier_dataset('dev'/'full') subsample_pair branch returning
    self._factory._subsample_source_view"
  - "_objective()'s tier-target and GT-selection branches for subsample_pair mode: isolated
    scratch DataFactory for sanity tier, pre-populated self._factory state for dev/full"
  - "TestOptimizerSubsamplePair test class (11 tests)"
affects: [58-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Three call-site dispatch keyed on a single _is_subsample_pair_mode() predicate method
      (rather than re-deriving the pipeline_mode/transform_spec check inline at each site),
      nested inside the existing pipeline_mode == 'synthetic' conditionals at all three
      optimizer.py call sites"
    - "scratch_factory = None declared before site 2, reassigned inside the sanity-tier
      subsample_pair branch, and read back (same instance) at site 3's GT-selection branch
      within the same _objective() call — carries isolated per-trial state across the two
      call sites without a class-level attribute"

key-files:
  created: []
  modified:
    - eval/runners/optimizer.py
    - tests/test_optimizer.py

key-decisions:
  - "Sanity-tier subsample_pair path always sets sanity_spec['synthesize'] = True regardless
    of the original transform_spec['synthesize'] value, matching the existing sanity-tier
    convention (Pitfall 5) that the sanity tier never touches real data"
  - "GT-selection for subsample_pair mode always reads via get_synthetic_ground_truth()
    (correspondence-based extraction) for both sanity and dev/full tiers, rather than the
    rigid/noise sanity branch's positional tier_dataset[...]['label'] read — subsample_pair's
    source/target views do not share point-for-point positional correspondence"
  - "Test mocking uses a DataFactory(...) side_effect that returns the shared main-factory mock
    on the first call and a fresh isolated MagicMock on every subsequent call, so sanity-tier
    trials' DataFactory(self.config) instantiation inside _objective is provably distinct from
    self._factory — mirrors the real isolation contract precisely enough to assert
    main_factory.generate_subsample_pair.assert_not_called() during sanity-only runs"

requirements-completed: [GT-06]

# Metrics
duration: ~20min
completed: 2026-08-04
---

# Phase 58 Plan 03: HyperparamOptimizer Subsample-Pair Wiring Summary

**All three `pipeline_mode == "synthetic"` call sites in `eval/runners/optimizer.py` (`run()`'s pre-populate step, `_tier_dataset()`, `_objective()`'s tier-target and GT-selection branches) now dispatch to `DataFactory.generate_subsample_pair()` for the default single-fixed-seed `subsample_pair` path, with zero changes to the existing rigid/noise or paired-mode branches.**

## Performance

- **Duration:** ~20 min active work
- **Started:** 2026-08-04
- **Completed:** 2026-08-04
- **Tasks:** 3 completed
- **Files modified:** 2 (eval/runners/optimizer.py, tests/test_optimizer.py)

## Accomplishments

- `HyperparamOptimizer._is_subsample_pair_mode()` — new predicate method:
  `pipeline_mode == "synthetic" AND transform_spec is not None AND
  transform_spec.get("type") == "subsample_pair"`, used at all three call sites instead of
  re-deriving the check inline
- `run()`'s pre-populate step: nested `if self._is_subsample_pair_mode():` branch calls
  `self._factory.generate_subsample_pair(base, transform_spec)` where `base` is `None` when
  `transform_spec["synthesize"]` is `True`, else `self._factory.load_real()`; the existing
  `generate_target()` call for rigid/noise mode is unchanged in the `else:` branch
- `_tier_dataset("dev")` / `_tier_dataset("full")`: return
  `self._factory._subsample_source_view` in subsample_pair mode, as the first statement in each
  branch, before the existing rigid/noise/paired logic; `_tier_dataset("sanity")` is textually
  unchanged
- `_objective()`'s tier-target branch (site 2): sanity tier builds an isolated
  `scratch_factory = DataFactory(self.config)` and calls
  `scratch_factory.generate_subsample_pair(None, sanity_spec)` with
  `sanity_spec["synthesize"] = True` forced regardless of the original spec; dev/full tier
  reuses the existing `tier_target` dict-comprehension verbatim
- `_objective()`'s GT-selection branch (site 3): sanity tier reads
  `scratch_factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]` (same scratch instance
  from site 2, via a `scratch_factory = None` local declared before site 2); dev/full tier reads
  `self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]` — identical to the
  existing rigid/noise dev/full line
- All three sites preserve every existing `type: rigid`/`noise` and paired-mode branch
  byte-for-byte unchanged (verified via `git diff` — only new nested branches added)
- `TestOptimizerSubsamplePair` (11 tests): `_is_subsample_pair_mode()` direct unit coverage
  (3 configs), `_tier_dataset("dev"/"full")` direct unit coverage, `run()` pre-populate dispatch
  proof (`generate_subsample_pair` called once, `generate_target` never called), sanity-tier
  isolation proof (main factory's `generate_subsample_pair` never called during sanity trials),
  and one real (non-mocked) end-to-end full-tier sweep asserting a non-empty
  `SearchResult.history` with finite scores

## Task Commits

Each task was committed atomically:

1. **Task 1: `_is_subsample_pair_mode()` helper + `run()` pre-populate + `_tier_dataset()`
   dev/full branches** - `59feeb5` (feat)
2. **Task 2: `_objective()` tier-target and GT-selection branches for subsample_pair** -
   `9105551` (feat)
3. **Task 3: Tests for optimizer.py subsample_pair single-seed wiring** - `fdb4a47` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `eval/runners/optimizer.py` - Added `_is_subsample_pair_mode()` method; nested subsample_pair
  branches inside all three existing `pipeline_mode == "synthetic"` conditionals (`run()`'s
  pre-populate block, `_tier_dataset()`'s `"dev"`/`"full"` branches, `_objective()`'s
  tier-target and GT-selection branches); `scratch_factory = None` local added before site 2 to
  thread the sanity-tier isolated `DataFactory` instance to site 3
- `tests/test_optimizer.py` - New `TestOptimizerSubsamplePair` class (11 tests) placed after
  `TestOptimizerSyntheticMode`, mirroring its `@patch("eval.runners.optimizer.MetricsEngine")`/
  `@patch("eval.runners.optimizer.DataFactory")` mocking convention, plus a `DataFactory(...)`
  `side_effect` helper distinguishing the main factory instance from per-trial scratch instances

## Decisions Made

- Followed the plan's exact nested-branch shape at all three call sites: `if
  self._is_subsample_pair_mode(): ... elif tier_name == "sanity": ... else: ...` (site 2/3) and
  `if self._is_subsample_pair_mode(): return ...` as the first statement in `_tier_dataset`'s
  `"dev"`/`"full"` branches — no restructuring of the existing branches beyond nesting the new
  check
- Test doubles for the isolated scratch `DataFactory` use a `side_effect` returning the shared
  main-factory mock on the first `DataFactory(...)` call and a fresh `MagicMock()` on every
  subsequent call, since `@patch("eval.runners.optimizer.DataFactory")`'s default
  `mock_factory_cls.return_value` would otherwise return the SAME object for both `self._factory`
  and every scratch instantiation, making it impossible to distinguish "main factory touched" from
  "isolated scratch factory touched" in assertions
- The `run()` pre-populate dispatch test and the sanity-isolation test both rely on `_objective`'s
  existing `try/except` (D-09: failed trials return `0.0`) to tolerate under-mocked stage
  execution — only the call-count assertions on `generate_subsample_pair`/`generate_target` are
  load-bearing, not full end-to-end correctness (that is covered separately by the real,
  non-mocked smoke test)

## Deviations from Plan

None - plan executed exactly as written, including the exact task order and interface contracts
documented in 58-03-PLAN.md's `<interfaces>` section.

## Issues Encountered

None. `pytest tests/test_optimizer.py -q` (full 40-test file) passes with zero regressions;
`pytest tests/test_optimizer.py -k "SubsamplePair or SyntheticMode" -q` (14 tests: 11 new + 3
existing) passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three `pipeline_mode == "synthetic"` call sites in `eval/runners/optimizer.py` now
  consistently dispatch to `generate_subsample_pair()`-derived state for the default
  single-fixed-seed path — the highest-risk item flagged in this plan's prompt (missing one of
  the three sites) was avoided; each site's dispatch was independently verified via `grep` for
  the exact expected call/attribute strings before committing
- Plan 58-04 (D-07's opt-in multi-seed averaging, gated to `tier == "full"`) can build directly
  on `_is_subsample_pair_mode()` and the `scratch_factory`-isolation pattern established here
  without needing further changes to the default single-seed path documented in this plan
- Full regression suite for the touched file (`tests/test_optimizer.py`, 40 tests) shows zero
  new failures

---
*Phase: 58-synthetic-labeled-subsample-pair-generation*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: eval/runners/optimizer.py (contains `_is_subsample_pair_mode` and
  `_subsample_source_view`, verified via `grep -n "_is_subsample_pair_mode\|_subsample_source_view"
  eval/runners/optimizer.py`)
- FOUND: eval/runners/optimizer.py (contains `scratch_factory.generate_subsample_pair` and
  `scratch_factory.get_synthetic_ground_truth`, verified via grep)
- FOUND: tests/test_optimizer.py (TestOptimizerSubsamplePair class, 11 tests, all passing,
  verified via `pytest tests/test_optimizer.py -k "SubsamplePair" -q`)
- FOUND commit 59feeb5 in `git log --oneline --all`
- FOUND commit 9105551 in `git log --oneline --all`
- FOUND commit fdb4a47 in `git log --oneline --all`
