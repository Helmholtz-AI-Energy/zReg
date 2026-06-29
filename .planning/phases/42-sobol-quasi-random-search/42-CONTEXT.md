# Phase 42: Sobol Quasi-Random Search as Default - Context

**Gathered:** 2026-06-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `SobolSearch` to `eval/search_strategies.py` and change the `HyperparamOptimizer` default from `"grid"` to `"sobol"`. Specifically:
- New `SobolSearch` class alongside `GridSearch`, `RandomSearch`, `BayesianSearch`, `PropulateSearch`
- Two new flat fields on `EvalConfig`: `sobol_seed: int = 42` and `sobol_randomize: bool = True`
- `search_strategy` Literal extended to include `"sobol"`; default changed from `"grid"` to `"sobol"`
- `eval/runners/optimizer.py` dispatcher wired with `"sobol"` branch
- Existing configs without `search_strategy` now inherit `sobol` (intentional per OPT-04-06)
- Configs with explicit `search_strategy: grid` or `search_strategy: bayesian` are unchanged

`AlignmentStage`, `LabelTransferStage`, `EvaluationRunner`, all other stages — untouched.

</domain>

<decisions>
## Implementation Decisions

### Config Field Placement
- **D-01:** `sobol_seed` and `sobol_randomize` are **flat fields** directly on `EvalConfig` — not wrapped in a sub-model. Consistent with `n_trials`, `tier`, and `search_strategy` as top-level primitives.
- **D-02:** `sobol_seed: int = 42` — default 42, consistent with `BayesianSearch` (`TPESampler(seed=42)`) and `prune_candidates` (`random.Random(42)`).
- **D-03:** `sobol_randomize: bool = True` — scrambled Owen sequence by default (scipy's recommended mode; better uniformity, reproducible via `sobol_seed`).
- **D-04:** `sobol_seed` is silently ignored when `sobol_randomize=False` (classical Van der Corput; no warning needed).

### SobolSearch Class Behavior
- **D-05:** `SobolSearch.search()` mirrors `RandomSearch.search()` signature: `search(search_space, objective_fn, n_trials, seed, randomize, warm_start=None) -> list[tuple[dict, float]]`.
- **D-06:** Mapping categorical search space → Sobol: generate `n_trials` points in `[0, 1]^d`, then quantize each dimension `i` to `list_choices[i][int(floor(val * len(choices)))]` (clamp index to `len - 1` to avoid off-by-one at val=1.0).
- **D-07:** Warm-start: prepend warm-start params (evaluated first) before Sobol-sampled candidates, mirroring `RandomSearch` behavior.
- **D-08:** Empty search space (`d=0`): return `[]` silently — consistent with `GridSearch` (empty Cartesian product = no trials).

### Small n_trials Fallback
- **D-09:** `SOBOL_MIN_TRIALS = 8` — hardcoded constant inside `SobolSearch` (not configurable via `EvalConfig`).
- **D-10:** When `n_trials < SOBOL_MIN_TRIALS`: fall back to `RandomSearch` (using `sobol_seed` as the random seed for reproducibility). Log at `logging.debug`: `"SobolSearch: n_trials=%d < 8, using RandomSearch fallback"`.
- **D-11:** Fallback applies to Sobol-generated trials only. Warm-start params are always evaluated first regardless of n_trials count.

### Claude's Discretion
- Whether `SobolSearch` accepts `seed` / `randomize` as constructor args or as `search()` call args — planner picks (search-call args are consistent with how `output_dir` is passed in `BayesianSearch`).
- Whether to log at `INFO` vs `DEBUG` for the new `"sobol"` strategy being selected in the optimizer dispatcher.
- Whether `SobolSearch` validates that all `search_space` values are non-empty lists before sampling.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS-v1.4.md` §OPT-04 — Full OPT-04 requirements (OPT-04-01 through OPT-04-06); includes YAML config examples, backward compat spec, and `sobol_seed` / `sobol_randomize` field names

### Existing Search Strategy Infrastructure
- `eval/search_strategies.py` — `GridSearch`, `RandomSearch`, `BayesianSearch`, `PropulateSearch`; add `SobolSearch` here; follow the existing class structure and `search()` signature conventions
- `eval/runners/optimizer.py` — `HyperparamOptimizer`; `search_strategy` dispatcher (lines ~249–315); `SANITY_N_TRIALS = 5`, `DEV_N_TRIALS = 20`; wire `"sobol"` branch here
- `eval/config.py` — `EvalConfig` Pydantic v2 model; `search_strategy: Literal[...]` field (line 187); add `"sobol"` to Literal, change default from `"grid"` to `"sobol"`, add `sobol_seed` and `sobol_randomize` flat fields

### Test Patterns
- `tests/test_search_strategies.py` — unit tests for all existing search strategies; add `TestSobolSearch` class following the same structure
- `tests/test_optimizer.py` — integration tests for `HyperparamOptimizer`; add test for Sobol being used when `search_strategy` is absent

### Backward Compatibility
- `configs/*.yaml` — existing scenario configs; do NOT add `search_strategy` to them — they should naturally inherit Sobol as the new default per OPT-04-06

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/search_strategies.py::RandomSearch.search()` — `SobolSearch` falls back to this for `n_trials < 8`; import and delegate directly rather than re-implementing
- `eval/search_strategies.py::GridSearch.search()` — warm-start deduplication pattern (seen set + candidates list) can be adapted for `SobolSearch` warm-start prepend
- `tests/test_search_strategies.py::TestBayesianSearch._obj` — minimal objective function pattern for unit tests

### Established Patterns
- **Flat primitive fields on EvalConfig**: `search_strategy`, `n_trials`, `tier` set the pattern; `sobol_seed` and `sobol_randomize` should follow (not a nested sub-model)
- **`zreg.*` before `torch` import order**: enforced in `eval/search_strategies.py:42-44` and `tests/conftest.py:20-24`; `SobolSearch` only uses `scipy.stats.qmc` (no torch), so no import-order risk — but document this in module-level note
- **Literal Pydantic v2 field validators**: `search_strategy` uses `Literal["grid", "random", "bayesian", "propulate", "auto"]` — extend to add `"sobol"` and change default
- **`__all__` exports**: `eval/search_strategies.py` exports `["GridSearch", "RandomSearch", "BayesianSearch", "PropulateSearch"]` — add `"SobolSearch"` here
- **Seed=42 convention**: `BayesianSearch` uses `seed=42` for `TPESampler`; `prune_candidates` uses `random.Random(42)`; `sobol_seed=42` is consistent

### Integration Points
- `eval/config.py` line 187: `search_strategy: Literal[...] = "grid"` → change default to `"sobol"` and add `"sobol"` to Literal
- `eval/runners/optimizer.py` lines ~249–315: add `elif strategy_name == "sobol":` branch calling `SobolSearch().search(...)`; pass `self.config.sobol_seed`, `self.config.sobol_randomize`
- `eval/search_strategies.py`: new `SobolSearch` class + `SOBOL_MIN_TRIALS = 8` constant

</code_context>

<specifics>
## Specific Ideas

- The `SOBOL_MIN_TRIALS = 8` threshold is a power-of-2 boundary (2^3). Below this, Sobol's uniformity advantage over random sampling is negligible — the fallback to `RandomSearch` with `sobol_seed` gives reproducibility without Sobol overhead.
- `sobol_seed = 42` default deliberately mirrors `BayesianSearch`'s `TPESampler(seed=42)` and `random.Random(42)` in `prune_candidates` — makes all stochastic strategies use the same seed by default.
- `sobol_randomize = True` default follows scipy's recommendation: scrambled sequences have better uniformity properties at all sample sizes, and with a fixed `sobol_seed` they are fully reproducible.
- The sanity tier runs with `SANITY_N_TRIALS = 5` — this is below `SOBOL_MIN_TRIALS = 8`, so sanity-tier runs will always use `RandomSearch` fallback. This is expected and acceptable behavior.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 42-sobol-quasi-random-search*
*Context gathered: 2026-06-29*
