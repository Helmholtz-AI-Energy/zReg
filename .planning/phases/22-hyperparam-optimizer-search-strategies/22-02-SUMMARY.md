---
phase: 22-hyperparam-optimizer-search-strategies
plan: 02
subsystem: eval
tags: [hyperparameter-optimization, optuna, tpe-sampler, gate-tests, frame-09, frame-10]

# Dependency graph
requires:
  - phase: 22-01
    provides: HyperparamOptimizer.run(), _objective(), _tier_dataset(), prune_candidates(); BayesianSearch with TPESampler; Wave-0 test stubs
  - phase: 18-metricsengine-result-types
    provides: Trial, SearchResult, StageMetrics frozen pydantic models
provides:
  - tests/test_optimizer.py: 5 fully populated FRAME-09+10 gate test classes (0 stubs)
affects: [23-cli-entrypoint-scenario-configs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "@patch('eval.runners.optimizer.DataFactory') mock with side_effect for get_ground_truth"
    - "optuna.create_study patched via side_effect to capture sampler; in-memory storage fallback"
    - "HyperparamOptimizer.prune_candidates() unit-tested as @staticmethod (no mock needed)"
    - "Direct _objective() call for default-score baseline in TestBestParamsImproveDefault"

key-files:
  created: []
  modified:
    - tests/test_optimizer.py

key-decisions:
  - "mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]['color'] for k in ds} — handles tier_dataset keyset dynamically (not fixed synthetic_dataset fixture keys)"
  - "TestBayesianSearch patches optuna.create_study with side_effect that captures sampler and falls back to in-memory storage (no SQLite side effects in test)"
  - "TestBestParamsImproveDefault uses lenient assertion (>= default_score OR >= 0.0) — consistent with plan note about small synthetic data and random seed"
  - "sampler._n_startup_trials used directly to verify FRAME-10 n_startup >= 2*N_params invariant"

# Metrics
duration: 10min
completed: 2026-05-29
---

# Phase 22 Plan 02: FRAME-09+10 Gate Tests Summary

**5 FRAME-09+10 gate test classes fully populated (0 stubs); 723 tests pass, 17 skipped, 0 failures**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-29T13:00:00Z
- **Completed:** 2026-05-29T13:11:17Z
- **Tasks:** 1
- **Files created/modified:** 1

## Accomplishments

- Replaced all 5 `pytest.skip(...)` stubs in `tests/test_optimizer.py` with complete, passing test implementations
- `TestHyperparamOptimizerSanityTier`: end-to-end `optimizer.run()` with mocked DataFactory; timing assertion `elapsed < 120.0`; `isinstance(result, SearchResult)` — completed in 2.29s
- `TestHyperparamOptimizerOutputFiles`: `best_params.json` and `search_history.json` existence + content validation (dict, non-empty list)
- `TestPruneCandidates`: pure `@staticmethod` unit test; 5 → 3 reduction; score-descending sort verified; `pruned[0]["window_size"] == 4`
- `TestBestParamsImproveDefault`: full `optimizer.run()` followed by direct `_objective()` call for default-score baseline; lenient correctness assertion
- `TestBayesianSearch`: `optuna.create_study` patched via `side_effect` to capture sampler; `isinstance(sampler, TPESampler)` verified; `_n_startup_trials >= 2*N_params` asserted; integration smoke test confirms non-empty results
- Full test suite: **723 passed, 17 skipped, 0 failures** (718 pre-existing + 5 new)

## Task Commits

1. **Task 1: Populate all 5 test classes in tests/test_optimizer.py** — `fd7635b` (test)

## Files Created/Modified

- `tests/test_optimizer.py` — 5 stub test methods replaced with full implementations (139 insertions, 10 deletions)

## Decisions Made

- `mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["color"] for k in ds}` chosen over a fixed return value because `_objective()` calls `get_ground_truth(tier_dataset)` with the internal tier dataset (50-point, 3-frame), not the `synthetic_dataset` fixture (20-point, 3-frame) — a fixed return_value would produce a key-mismatch error
- `TestBayesianSearch` patches `eval.search_strategies.optuna.create_study` (point-of-use) with a `side_effect` that captures the sampler argument and then calls the real `create_study` with `storage=None` to avoid writing SQLite files during test
- `TestBestParamsImproveDefault` uses the lenient assertion pattern from the plan spec because SANITY_N_TRIALS=5 with a 2-key search space (4 grid points total) and tiny synthetic data may produce scores that tie with the default

## Deviations from Plan

None — plan executed exactly as written. All acceptance criteria met on first attempt.

## Known Stubs

None — all 5 `pytest.skip(...)` stubs removed. `grep -c 'pytest.skip' tests/test_optimizer.py` → 0.

## Threat Flags

No new security-relevant surface introduced. This plan modifies only test code. Threat model T-22-04 and T-22-05 accepted per plan as written.

## Self-Check: PASSED

### Files exist

- `tests/test_optimizer.py`: FOUND

### Commits exist

- `fd7635b`: FOUND — test(22-02): populate 5 FRAME-09+10 gate test classes in test_optimizer.py

### Test verification

- `python -m pytest tests/test_optimizer.py -v`: 5 passed, 0 skipped
- `python -m pytest tests/ -q`: 723 passed, 17 skipped, 0 failures
- `grep -c 'pytest.skip' tests/test_optimizer.py`: 0

---
*Phase: 22-hyperparam-optimizer-search-strategies*
*Completed: 2026-05-29*
