---
phase: 58-synthetic-labeled-subsample-pair-generation
plan: 04
subsystem: evaluation-framework
tags: [pytorch, pydantic, ground-truth, label-transfer, data-factory, hpo, optuna]

# Dependency graph
requires:
  - phase: 58-03
    provides: "HyperparamOptimizer._is_subsample_pair_mode(), the three
      pipeline_mode=='synthetic' call-site dispatch shape in run()/_tier_dataset/
      _objective(), scratch_factory isolation pattern for the sanity tier"
provides:
  - "SUBSAMPLE_PAIR_MULTISEED_TIER module constant ('full')"
  - "_resolve_subsample_pair_seeds(transform_spec, tier_name) -> list[int] module-level helper"
  - "HyperparamOptimizer._score_subsample_pair_multiseed() method — averages a trial's
    score across N isolated per-seed subsample pairs"
  - "_objective()'s single new early-exit branch dispatching to the multiseed scorer
    when len(seeds) > 1"
  - "run()'s pre-populate list-seed skip guard"
  - "TestSubsamplePairMultiseed test class (12 tests)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Graceful-fallback-not-raise tier gate: _resolve_subsample_pair_seeds logs a
      warning and returns [seed[0]] for sanity/dev tiers with a list seed, rather
      than raising, because _objective's broad except Exception: return 0.0 would
      otherwise silently swallow a raised validation error and mask it as a
      generic failed-trial score"
    - "Fresh-per-seed DataFactory isolation inside a single trial: unlike the
      sanity-tier single scratch_factory pattern from 58-03 (one isolated instance
      per _objective call), _score_subsample_pair_multiseed constructs a NEW
      DataFactory(self.config) for EACH seed in the loop, since
      _correspondence_idx/_synthetic_target/_source_dataset are single-slot
      instance attributes that would otherwise be clobbered between seeds within
      the same trial (T-57-11)"
    - "Single early-exit branch inserted immediately after _objective's existing
      merged = {...} line — the only edit to _objective's body; when
      len(seeds) == 1 execution falls through completely unchanged into Plan
      58-03's existing logic (D-06 compliance, verified by a dedicated
      assert_not_called() test)"

key-files:
  created: []
  modified:
    - eval/runners/optimizer.py
    - tests/test_optimizer.py

key-decisions:
  - "Multi-seed averaging on tiers before 'full' (sanity/dev) in a tier=='full'
    run gracefully degrades rather than fully participating: run()'s pre-populate
    step skips generation entirely for a list seed (deferring ALL generation to
    per-trial scoring), so sanity/dev tiers' single-seed fallback path
    (_synthetic_target-based) has nothing pre-populated to read and every sanity/
    dev trial in that scenario fails via _objective's existing except Exception:
    return 0.0 handler — this does NOT crash the run (D-09 tolerates it) and the
    full tier still produces valid averaged trials, matching the plan's literal
    requirement ('does not crash') and the end-to-end smoke test's actual
    assertions (which only check the full SearchResult.history is non-empty,
    not that every tier produced trials)"
  - "last_metrics (the final seed's StageMetrics) is stored on the single averaged
    Trial as a representative sample for search_history.json logging only — the
    Trial's score field is the true source of truth (the average), not
    last_metrics"

requirements-completed: [GT-04, GT-06]

# Metrics
duration: ~25min
completed: 2026-08-04
---

# Phase 58 Plan 04: D-07 Multi-Seed Averaging for HyperparamOptimizer Summary

**`transform_spec["seed"]` may now optionally be a list of ints for `subsample_pair` HPO configs; on the `full` tier, a trial's score is averaged across the N resulting subsample pairs via `_score_subsample_pair_multiseed()`, isolated per seed through a fresh `DataFactory` instance — while the default single-int-seed path (Plan 58-03) receives zero further edits inside `_objective`.**

## Performance

- **Duration:** ~25 min active work
- **Started:** 2026-08-04
- **Completed:** 2026-08-04
- **Tasks:** 3 completed
- **Files modified:** 2 (eval/runners/optimizer.py, tests/test_optimizer.py)

## Accomplishments

- `SUBSAMPLE_PAIR_MULTISEED_TIER: str = "full"` — new module-level constant, placed
  after `DEV_N_TRIALS`, gating D-07's multi-seed averaging to the `full` tier only
- `_resolve_subsample_pair_seeds(transform_spec, tier_name) -> list[int]` — new
  module-level helper, placed after `_apply_transform_to_dataset`. Int seed
  (the default) always returns `[seed]`; a list seed on the `full` tier returns
  `list(seed)` unchanged; a list seed on any other tier logs a warning and
  gracefully falls back to `[seed[0]]` (never raises, since `_objective`'s broad
  `except Exception: return 0.0` would silently swallow a raised exception);
  an empty list or a non-int/non-list seed raises `ValueError`
- `run()`'s pre-populate block: a small guard now reads
  `transform_spec.get("seed", 42)` before calling `generate_subsample_pair()`
  and skips that call entirely (`pass`) when `seed` is a list — deferring all
  generation to per-trial multiseed scoring inside `_objective()`. The existing
  single-int-seed body is otherwise byte-for-byte unchanged
- `HyperparamOptimizer._score_subsample_pair_multiseed(self, params, merged,
  tier_name, seeds, history_out) -> float` — new method placed immediately
  after `_objective`. Resolves `base` once (`None` if synthesizing, else
  `self._factory.load_real()`), then for each seed: builds
  `per_seed_spec = {**transform_spec, "seed": s}`, constructs a FRESH
  `DataFactory(self.config)`, calls `generate_subsample_pair(base,
  per_seed_spec)`, runs the identical stage-execution/metric-assembly sequence
  `_objective` uses for a single pair (`AlignmentStage`/`LabelTransferStage`/
  `MetricsEngine.compute_stage_metrics`/`compute_score`), and appends each
  seed's score. After the loop, appends exactly ONE `Trial` with
  `score=avg_score` and returns `avg_score`. Does not catch its own exceptions —
  they propagate to `_objective`'s existing `except Exception: return 0.0`
  handler (D-09 consistency)
- `_objective()`'s only change: a new early-exit branch inserted immediately
  after `merged = {**self._default_params, **params}` and before the existing
  site-2 tier-target dispatch — `if self._is_subsample_pair_mode(): seeds =
  _resolve_subsample_pair_seeds(...); if len(seeds) > 1: return
  self._score_subsample_pair_multiseed(...)`. When `len(seeds) == 1` (every
  default int-seed call, plus the sanity/dev graceful fallback), execution
  falls through completely unchanged into Plan 58-03's existing logic
- `TestSubsamplePairMultiseed` (12 tests): `_resolve_subsample_pair_seeds`'s
  full decision table (6 tests), `_objective`'s early-exit dispatch proof
  (single int seed never calls the multiseed scorer; list seed on full tier
  does — 2 tests), averaging-correctness proof using distinct
  `compute_score` `side_effect` values per seed (1 test), and one real
  (non-mocked) end-to-end 3-seed full-tier sweep (1 test), plus 2 additional
  assertions folded into the averaging-correctness test (Trial count, tier
  field)

## Task Commits

Each task was committed atomically:

1. **Task 1: `SUBSAMPLE_PAIR_MULTISEED_TIER` + `_resolve_subsample_pair_seeds()` +
   `run()` list-seed guard** - `13559f1` (feat)
2. **Task 2: `_score_subsample_pair_multiseed()` + `_objective()` early-exit
   branch** - `d8522ef` (feat)
3. **Task 3: Tests for D-07 multi-seed averaging** - `2a7d48b` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `eval/runners/optimizer.py` - Added `SUBSAMPLE_PAIR_MULTISEED_TIER` constant,
  `_resolve_subsample_pair_seeds()` module function, `run()`'s list-seed
  pre-populate skip guard, `HyperparamOptimizer._score_subsample_pair_multiseed()`
  method, and `_objective()`'s single new early-exit branch
- `tests/test_optimizer.py` - New `TestSubsamplePairMultiseed` class (12 tests)
  placed after `TestOptimizerSubsamplePair`, plus an updated import block
  pulling in `_resolve_subsample_pair_seeds` and `SUBSAMPLE_PAIR_MULTISEED_TIER`

## Decisions Made

- Followed the plan's exact method placement (`_score_subsample_pair_multiseed`
  immediately after `_objective`; `_resolve_subsample_pair_seeds` immediately
  after `_apply_transform_to_dataset`) and the exact early-exit insertion point
  (`_objective`, immediately after `merged = {...}`)
- Verified via direct execution (not just reading the plan) that a `tier="full"`
  run with a list seed causes sanity/dev tiers' trials to fail gracefully
  (via `_objective`'s existing D-09 `except Exception: return 0.0` handler)
  because `run()`'s pre-populate step defers all generation for list seeds —
  this does not crash the run, and the `full` tier still produces valid
  averaged trials, matching both the plan's literal "does not crash"
  requirement and the end-to-end smoke test's actual assertions
- Test 5 (averaging correctness) constructs `HyperparamOptimizer` with a
  `DataFactory` `side_effect` list of exactly 3 mocks: the 1st satisfies
  `self._factory`'s construction inside `__init__`; the 2nd/3rd are the two
  per-seed FRESH scratch factories `_score_subsample_pair_multiseed` builds
  inside its loop — proven distinct from `self._factory` via
  `main_factory.generate_subsample_pair.assert_not_called()`

## Deviations from Plan

None - plan executed exactly as written, including the exact task order,
method placement, and interface contracts documented in 58-04-PLAN.md's
`<interfaces>` section.

## Issues Encountered

None specific to this plan. `pytest tests/test_optimizer.py -k "Multiseed or
SubsamplePair or SyntheticMode" -q` (21 tests) passes; `pytest
tests/test_optimizer.py -q` (full 50-test file) passes with zero regressions.

**Full project test suite** (`KMP_DUPLICATE_LIB_OK=TRUE python3 -m pytest -q`,
run as the final plan in this phase per the orchestrator's instructions):
1410 passed, 22 skipped, 1 xpassed, 1 failed
(`tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery`).
This failure is unrelated to this plan's scope — ICP registration code was not
touched by any commit in Phase 58 — and was confirmed to be a pre-existing,
test-order-dependent flake: running it in isolation
(`pytest tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery -q`)
passes cleanly. No action taken (out of scope per the deviation-rules scope
boundary — pre-existing failures in unrelated files are not auto-fixed).

## User Setup Required

None - no external service configuration required.

## D-08 Confirmation

D-08 (train/val split for `subsample_pair`-generated ground truth) was
explicitly deferred per 58-CONTEXT.md and is NOT implemented anywhere in this
plan. `EvalConfig.val_split`, `DataFactory.prepare_split()`, and
`split_seeds()` remain untouched and unconsumed by `HyperparamOptimizer`.

## Next Phase Readiness

- Phase 58 (Synthetic Labeled Subsample-Pair Generation) is now complete:
  `DataFactory.generate_subsample_pair()` (58-01), `EvaluationRunner` dispatch
  (58-02), `HyperparamOptimizer` single-fixed-seed wiring (58-03), and D-07's
  opt-in multi-seed averaging (58-04) are all committed on this branch
- No further plans are queued in this phase; the full project test suite was
  run as the final integration check (see Issues Encountered above)

---
*Phase: 58-synthetic-labeled-subsample-pair-generation*
*Completed: 2026-08-04*

## Self-Check: PASSED

- FOUND: eval/runners/optimizer.py (contains `SUBSAMPLE_PAIR_MULTISEED_TIER`,
  `_resolve_subsample_pair_seeds`, `_score_subsample_pair_multiseed`, verified
  via `grep -n` on eval/runners/optimizer.py)
- FOUND: tests/test_optimizer.py (`TestSubsamplePairMultiseed` class, 12 tests,
  all passing, verified via `pytest tests/test_optimizer.py -k "Multiseed" -q`)
- FOUND commit 13559f1 in `git log --oneline --all`
- FOUND commit d8522ef in `git log --oneline --all`
- FOUND commit 2a7d48b in `git log --oneline --all`
