---
phase: 42-sobol-quasi-random-search
plan: "01"
subsystem: eval
tags: [search-strategy, sobol, quasi-random, hpo, config]
requires: []
provides: [SobolSearch, SOBOL_MIN_TRIALS, sobol_seed, sobol_randomize, sobol-dispatcher-branch]
affects: [eval/config.py, eval/search_strategies.py, eval/runners/optimizer.py]
tech-stack:
  added: [scipy.stats.qmc.Sobol]
  patterns: [stateless-search, qmc-quantization, sobol-min-trials-fallback]
key-files:
  created: []
  modified:
    - eval/config.py
    - eval/search_strategies.py
    - eval/runners/optimizer.py
decisions:
  - sobol_seed=42 and sobol_randomize=True as flat EvalConfig fields (D-01/D-02/D-03)
  - SOBOL_MIN_TRIALS=8 at module level; below threshold falls back to RandomSearch (D-10)
  - seed/randomize as search() call-args, not constructor args (D-05, consistent with BayesianSearch.output_dir)
  - sobol_randomize=False silently ignores seed - Van der Corput mode (D-04)
metrics:
  duration: 8 minutes
  completed: 2026-06-30T08:26:33Z
  tasks-completed: 3
  tasks-total: 3
  files-changed: 3
requirements-completed: [OPT-04-01, OPT-04-02, OPT-04-03, OPT-04-04, OPT-04-06]
---

# Phase 42 Plan 01: Sobol Quasi-Random Search - Production Code Summary

**One-liner:** Sobol quasi-random search added as default HPO strategy via scipy.stats.qmc.Sobol with SOBOL_MIN_TRIALS=8 fallback to RandomSearch.

## What Was Built

Three targeted changes to wire Sobol sequence search as the new default hyperparameter search strategy:

1. **`eval/config.py`** — Extended `EvalConfig` with `"sobol"` in the `search_strategy` Literal, changed default from `"grid"` to `"sobol"`, added `sobol_seed: int = 42` and `sobol_randomize: bool = True` as flat primitive fields after `n_trials`.

2. **`eval/search_strategies.py`** — Added `SobolSearch` class between `RandomSearch` and `BayesianSearch`, `SOBOL_MIN_TRIALS: int = 8` at module level, `from scipy.stats import qmc` after `import optuna`, and `"SobolSearch"` to `__all__`. The class implements empty-space guard, small-budget fallback to `RandomSearch`, warm-start prepend (deduplication), Sobol sampling with index clamping at `len-1`, and results accumulation.

3. **`eval/runners/optimizer.py`** — Added `SobolSearch` to the search_strategies import and inserted `elif strategy_name == "sobol":` branch (after `random`, before `bayesian`) passing `self.config.sobol_seed` and `self.config.sobol_randomize`. Return value not captured (side effects via `_objective` append to `all_history`).

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 — EvalConfig sobol fields | d4482fe | feat(42-01): extend EvalConfig with sobol_seed, sobol_randomize, sobol default |
| Task 2 — SobolSearch class | e2d654e | feat(42-01): add SobolSearch class with SOBOL_MIN_TRIALS=8 fallback |
| Task 3 — optimizer dispatcher | 53320a1 | feat(42-01): wire SobolSearch dispatcher in HyperparamOptimizer |

## Verification Results

### Smoke Test (all assertions passed)

- `EvalConfig(data_path='/tmp/x').search_strategy == 'sobol'` — OPT-04-02
- `EvalConfig(data_path='/tmp/x', search_strategy='grid').search_strategy == 'grid'` — OPT-04-03
- `SobolSearch().search({'a': [1,2,3], 'b': ['x','y']}, ..., n_trials=8, seed=42)` returns 8 results with valid values — OPT-04-01
- `EvalConfig(data_path='/tmp/x', sobol_seed=7, sobol_randomize=False)` sets correct field values — OPT-04-04
- `'sobol' in inspect.getsource(HyperparamOptimizer)` — OPT-04-06

### Regression Tests

Full suite: **976 passed, 18 skipped** (no regression from v1.2 baseline).

Note: When running `pytest tests/ --ignore=tests/test_search_strategies.py --ignore=tests/test_optimizer.py`, a pre-existing flaky test `test_cpd_type_affine` fails due to test-ordering state dependency. This failure exists in the baseline and is completely unrelated to Sobol search changes — the test passes in isolation and in the full suite.

## Self-Check

### Files Exist
- `eval/config.py` — modified (contains `sobol_seed: int = 42`, `sobol_randomize: bool = True`, `"sobol"` in Literal, default `"sobol"`)
- `eval/search_strategies.py` — modified (contains `SobolSearch`, `SOBOL_MIN_TRIALS: int = 8`, `from scipy.stats import qmc`)
- `eval/runners/optimizer.py` — modified (contains `elif strategy_name == "sobol":`, `self.config.sobol_seed`, `self.config.sobol_randomize`)

### Commits Exist
- d4482fe: feat(42-01): extend EvalConfig with sobol_seed, sobol_randomize, sobol default
- e2d654e: feat(42-01): add SobolSearch class with SOBOL_MIN_TRIALS=8 fallback
- 53320a1: feat(42-01): wire SobolSearch dispatcher in HyperparamOptimizer

## Self-Check: PASSED

All 5 criteria verified:
- [x] EvalConfig.search_strategy defaults to "sobol"
- [x] EvalConfig has sobol_seed: int = 42 and sobol_randomize: bool = True as flat fields
- [x] SobolSearch class present in eval/search_strategies.py with SOBOL_MIN_TRIALS=8
- [x] eval/runners/optimizer.py dispatches to SobolSearch on "sobol"
- [x] No regression in existing tests (976 passed full suite)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries beyond those documented in the plan's threat model (T-42-01 through T-42-SC).
