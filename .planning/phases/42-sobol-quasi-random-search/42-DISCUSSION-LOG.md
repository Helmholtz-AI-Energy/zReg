# Phase 42: Sobol Quasi-Random Search as Default - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-29
**Phase:** 42-sobol-quasi-random-search
**Areas discussed:** Config field shape, Scramble default, Small n_trials behavior

---

## Config field shape

| Option | Description | Selected |
|--------|-------------|----------|
| Flat fields | `sobol_seed` / `sobol_randomize` directly on `EvalConfig`, consistent with `n_trials` / `search_strategy` | ✓ |
| Nested SobolConfig | `sobol_config: SobolConfig \| None = None` sub-model, consistent with Phase 41's `AlignmentPreprocessingConfig` | |

**User's choice:** Flat fields

| Option | Description | Selected |
|--------|-------------|----------|
| sobol_seed = 0 | Conventional quasi-random default | |
| sobol_seed = 42 | Consistent with `BayesianSearch(seed=42)` and `random.Random(42)` in `prune_candidates` | ✓ |

**User's choice:** 42

| Option | Description | Selected |
|--------|-------------|----------|
| Seed ignored when randomize=False (no warning) | Classical Van der Corput: seed has no effect | ✓ |
| Warn when seed != 42 and randomize=False | Alert user that seed is unused | |

**User's choice:** Seed silently ignored when `randomize=False`

---

## Scramble default

| Option | Description | Selected |
|--------|-------------|----------|
| sobol_randomize = True | Owen scrambling: better uniformity, reproducible via seed. scipy recommended. | ✓ |
| sobol_randomize = False | Classical deterministic Van der Corput: always identical, seed unused | |

**User's choice:** True (scrambled by default)

| Option | Description | Selected |
|--------|-------------|----------|
| Raise ValueError on empty search space | scipy.stats.qmc.Sobol requires d >= 1 | |
| Return [] silently | Consistent with GridSearch empty Cartesian product behavior | ✓ |

**User's choice:** Return `[]` silently on empty search space

---

## Small n_trials behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Use Sobol regardless | scipy handles any n_trials >= 1; no fallback needed | |
| Warn when n_trials < 8 | Log debug warning; user sees it with -v | |
| Fall back to RandomSearch below n_trials < 8 | Random equivalent in practice for small budgets | ✓ |

**User's choice:** Fall back to `RandomSearch` when `n_trials < 8`

| Option | Description | Selected |
|--------|-------------|----------|
| Log at debug level | `logging.debug(...)` — visible in test output with -v | ✓ |
| Silent fallback | No logging; behavior implicit | |

**User's choice:** Log at `logging.debug`

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded SOBOL_MIN_TRIALS = 8 | Implementation detail, not exposed in config | ✓ |
| Configurable via EvalConfig | `sobol_min_trials: int = 8` in YAML | |

**User's choice:** Hardcoded constant

---

## Claude's Discretion

- Whether `SobolSearch` receives `seed` / `randomize` as constructor args or `search()` call args
- Whether to log at INFO vs DEBUG when Sobol strategy is selected in the dispatcher
- Whether `SobolSearch` validates non-empty lists in `search_space` before sampling

## Deferred Ideas

None — discussion stayed within phase scope.
