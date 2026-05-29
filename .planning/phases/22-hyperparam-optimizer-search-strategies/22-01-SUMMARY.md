---
phase: 22-hyperparam-optimizer-search-strategies
plan: 01
subsystem: eval
tags: [optuna, hyperparameter-optimization, grid-search, random-search, bayesian-search, tpe, sqlite, pydantic-v2]

# Dependency graph
requires:
  - phase: 21-evaluationrunner-visualisation
    provides: EvaluationRunner stage-invocation pattern; AlignmentStage + LabelTransferStage + MetricsEngine interfaces; Trial/SearchResult frozen pydantic models
  - phase: 18-metricsengine-result-types
    provides: Trial, SearchResult, StageMetrics frozen pydantic models in eval/types.py
  - phase: 19-alignmentstage
    provides: AlignmentStage.REQUIRED_PARAMS, AlignmentStage.run()
  - phase: 20-labeltransferstage
    provides: LabelTransferStage.REQUIRED_PARAMS, LabelTransferStage.run()
provides:
  - optuna>=4.0,<5 installed and declared in setup.cfg install_requires
  - eval/search_strategies.py: GridSearch, RandomSearch, BayesianSearch classes
  - eval/runners/optimizer.py: HyperparamOptimizer with run(), _objective(), _tier_dataset(), prune_candidates(), save_best_params()
  - tests/test_optimizer.py: 5 Wave-0 stub test classes (FRAME-09+10 gate)
  - eval/runners/__init__.py: exports HyperparamOptimizer alongside EvaluationRunner
affects: [22-02-plan, 23-cli-entrypoint-scenario-configs]

# Tech tracking
tech-stack:
  added: [optuna>=4.0,<5 (TPE Bayesian HPO, SQLite study persistence)]
  patterns:
    - macOS-ARM import order enforced: zreg.* -> torch -> optuna -> eval.*
    - Strategy pattern: GridSearch/RandomSearch/BayesianSearch share (search_space, objective_fn, n_trials, warm_start) interface
    - Tiered execution with warm-start seeding: prune_candidates() top-k -> warm_start for next tier
    - _objective calls stages directly (not EvaluationRunner) for lightweight trial execution
    - model_dump() + json.dump() for all pydantic model serialisation (never model_dump_json)

key-files:
  created:
    - eval/search_strategies.py
    - eval/runners/optimizer.py
    - tests/test_optimizer.py
  modified:
    - setup.cfg (added optuna>=4.0,<5 to install_requires)
    - eval/runners/__init__.py (added HyperparamOptimizer export)

key-decisions:
  - "SANITY_N_TRIALS=5 and DEV_N_TRIALS=20 as module-level constants in optimizer.py"
  - "BayesianSearch n_startup_trials clamped via min(max(10, 2*n_params), n_trials-1) to prevent startup-phase exhaustion on sanity tier (Pitfall 3)"
  - "_tier_dataset('sanity') calls generate_labels(generate_trajectory(...)) so pc['id'] is populated for get_ground_truth() (Pitfall 5)"
  - "_default_params dict in HyperparamOptimizer.__init__ covers all 9 required stage keys; merged with trial params before stage calls (Pitfall 4)"
  - "T-22-01: search_space validation at __init__ — each value must be non-empty list"
  - "T-22-02: output_dir resolved via Path.resolve() as first line of run() to guard against path traversal"

patterns-established:
  - "Strategy interface: search(search_space, objective_fn, [n_trials], [output_dir], [warm_start]) -> list[tuple[dict, float]]"
  - "Warm-start via warm_start list prepend (Grid/Random) or study.enqueue_trial(skip_if_exists=True) (Bayesian)"
  - "Tiered HPO: TIER_SEQUENCE[:index+1] ceiling, prune_candidates() between tiers"
  - "Test scaffold Wave 0: all 5 FRAME gate test classes stubbed with pytest.skip before any implementation"

requirements-completed: [FRAME-09, FRAME-10]

# Metrics
duration: 30min
completed: 2026-05-29
---

# Phase 22 Plan 01: HyperparamOptimizer & Search Strategies Summary

**Optuna 4.x HPO machinery with GridSearch/RandomSearch/BayesianSearch strategies, tiered HyperparamOptimizer with warm-start pruning, and Wave-0 test scaffold for FRAME-09+10**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-05-29T11:30:00Z
- **Completed:** 2026-05-29T12:00:00Z
- **Tasks:** 3
- **Files created/modified:** 5

## Accomplishments

- Installed optuna 4.8.0 and declared `optuna>=4.0,<5` in setup.cfg
- Created `eval/search_strategies.py` with GridSearch (itertools.product), RandomSearch (random.choice), and BayesianSearch (Optuna TPE, SQLite, enqueue_trial warm-start)
- Created `eval/runners/optimizer.py` with HyperparamOptimizer: tiered run(), lightweight _objective() calling stages directly (D-07), _tier_dataset() with labelled sanity dataset, prune_candidates(), save_best_params()
- Updated `eval/runners/__init__.py` to export HyperparamOptimizer alongside EvaluationRunner
- Created `tests/test_optimizer.py` Wave-0 scaffold: 5 stub test classes (pytest.skip), importable with try/except guards
- Full test suite: 718 passed, 22 skipped (0 new failures)

## Task Commits

1. **Task 1: Wave 0 — optuna dependency + test scaffold** - `fc6ed73` (chore)
2. **Task 2: eval/search_strategies.py — GridSearch, RandomSearch, BayesianSearch** - `ba65526` (feat)
3. **Task 3: eval/runners/optimizer.py + update eval/runners/__init__.py** - `240ae6b` (feat)

## Files Created/Modified

- `setup.cfg` — added `optuna>=4.0,<5` to install_requires
- `tests/test_optimizer.py` — 5 Wave-0 stub test classes (FRAME-09-SC1, SC2, SC4; FRAME-10-SC3, TPE gate)
- `eval/search_strategies.py` — GridSearch (Cartesian product + warm-start dedup), RandomSearch (random.choice), BayesianSearch (Optuna TPE, SQLite, enqueue_trial)
- `eval/runners/optimizer.py` — HyperparamOptimizer orchestrator with all 5 public methods
- `eval/runners/__init__.py` — added HyperparamOptimizer import and __all__ entry

## Decisions Made

- `SANITY_N_TRIALS = 5`, `DEV_N_TRIALS = 20` as module-level constants (RESEARCH Q3 resolved)
- BayesianSearch.n_startup clamped with `min(n_startup, n_trials - 1)` for small trial budgets (Pitfall 3)
- `_tier_dataset("sanity")` calls `generate_labels(generate_trajectory(...))` — not `DataFactory.generate_synthetic()` — to ensure `pc["id"]` is populated (Pitfall 5)
- `_default_params` dict merges with trial params before stage calls to prevent `validate_params` failures when search_space is a subset of required keys (Pitfall 4)
- Wave-0 test scaffold uses `try/except ImportError` guards for `HyperparamOptimizer` and strategy imports so the file is importable before implementation

## Deviations from Plan

None — plan executed exactly as written. All acceptance criteria met on first attempt.

## Issues Encountered

None.

## Next Phase Readiness

- Plan 22-02 can populate the 5 stub test classes in `tests/test_optimizer.py`
- `HyperparamOptimizer.run()` is fully functional — Plan 22-02 tests can call it directly
- `eval/runners/__init__.py` exports both runners; Phase 23 CLI can import `HyperparamOptimizer` from `eval.runners`
- `best_params.json` and `search_history.json` output format is stable for Phase 23 CLI consumption

## Self-Check: PASSED

### Files exist

- `eval/search_strategies.py`: FOUND
- `eval/runners/optimizer.py`: FOUND
- `tests/test_optimizer.py`: FOUND
- `.planning/phases/22-hyperparam-optimizer-search-strategies/22-01-SUMMARY.md`: FOUND

### Commits exist

- `fc6ed73`: FOUND — chore(22-01): add optuna>=4.0,<5 dep + Wave 0 test scaffold
- `ba65526`: FOUND — feat(22-01): add eval/search_strategies.py
- `240ae6b`: FOUND — feat(22-01): add HyperparamOptimizer + update eval/runners/__init__.py

---
*Phase: 22-hyperparam-optimizer-search-strategies*
*Completed: 2026-05-29*
