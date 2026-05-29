# Phase 22: HyperparamOptimizer & Search Strategies - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `eval/search_strategies.py` (GridSearch, RandomSearch, BayesianSearch via Optuna 4.x with SQLite storage) and `eval/runners/optimizer.py` (`HyperparamOptimizer`) with sanity/dev/full tier logic, candidate pruning between tiers, and JSON output (`best_params.json` + `search_history.json`). This phase produces the search machinery; it is not called directly by users — Phase 23 CLI orchestrates it.

</domain>

<decisions>
## Implementation Decisions

### Tier Execution Flow

- **D-01:** `run()` stops at `config.tier` — if `tier="sanity"`, only the sanity tier runs; if `tier="dev"`, runs sanity then dev; if `tier="full"`, runs all three in sequence. The config ceiling is respected, not overridden.
- **D-02:** Each tier differs in **both** n_trials and dataset size. Sanity uses a small synthetic subset + few trials (fast). Dev uses a larger real subset. Full uses the complete dataset + all `n_trials` from config. `_tier_dataset(tier)` handles the dataset slicing per tier.
- **D-03:** `prune_candidates(history, keep_top_k)` prunes the **candidate list** (top-k `Trial` param dicts by score), not the search space itself. The search space is unchanged between tiers.
- **D-04:** Pruned top-k params from tier N are used as **warm-start seeds** for tier N+1. They are seeded into the Optuna study as fixed trials (or prepended to the candidate list for Grid/Random). The next tier still explores; it just starts from known-good regions.

### Search Space Format

- **D-05:** Unified format across all three strategies — **list of values per param**:
  ```yaml
  search_space:
    window_size: [3, 5, 7]
    k_neighbours: [3, 5, 10]
    cpd_penalty: [null, "l2"]
    step: [1, 2]
  ```
  GridSearch exhausts all combinations. RandomSearch samples from the lists. Optuna treats each list as a set of categorical choices (TPE over categorical space).
- **D-06:** Search space dict keys are **exact param names** passed to stages — no mapping layer. `window_size` in `search_space` becomes `window_size` in `AlignmentStage.run(dataset, params)` and `LabelTransferStage.run(dataset, params)` directly.

### Objective Function

- **D-07:** `_objective(params) → float` calls stages + `MetricsEngine` **directly** — does NOT call `EvaluationRunner.run()`. No JSON writing, no plots, no `log_run()`. Pipeline: instantiate stages → run → `MetricsEngine.compute_score()` → return float. This keeps each trial lightweight enough for sanity tier to complete in under 2 minutes on a laptop.
- **D-08:** Dataset passed to `_objective()` is the **tier-specific subset** provided by `_tier_dataset(tier)`, not the full DataFactory output.
- **D-09:** If a trial raises an exception (bad param combo, stage failure), `_objective()` returns `0.0` and logs a warning. The search continues — one failed trial does not abort the run.

### Optuna Study Lifecycle

- **D-10:** Study name is fixed as `"zreg-hpo"` per `output_dir`. Re-running `HyperparamOptimizer` with the same `output_dir` resumes the existing study (`load_if_exists=True`).
- **D-11:** SQLite storage file lives at `{output_dir}/optuna.db` — co-located with `best_params.json` and `search_history.json`. Easy to delete for a clean restart.
- **D-12:** GridSearch and RandomSearch are **stateless** — all trials computed in memory. `save_best_params(result, path)` writes `best_params.json` + `search_history.json` at the end. No SQLite, no incremental writes for these strategies.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FRAME-09 — `HyperparamOptimizer` full spec (run, _objective, _tier_dataset, prune_candidates, save_best_params)
- `.planning/REQUIREMENTS.md` §FRAME-10 — `SearchStrategies` full spec (GridSearch, RandomSearch, Optuna 4.x, TPE sampler, n_startup_trials >= 2×N_params, SQLite)

### Existing Result Types (Phase 22 consumes and produces these)
- `eval/types.py` — `Trial`, `SearchResult` (frozen pydantic v2 models that optimizer populates); also `StageMetrics` (returned by MetricsEngine, stored in each Trial)
- `eval/config.py` — `EvalConfig` (search_space, search_strategy, tier, n_trials, output_dir, metric_weights fields all consumed by optimizer)

### Upstream Stages & MetricsEngine (called inside _objective)
- `eval/stages/alignment.py` — `AlignmentStage` (called in _objective with params from search_space)
- `eval/stages/label_transfer.py` — `LabelTransferStage` (called in _objective)
- `eval/metrics.py` — `MetricsEngine` (compute_score returns the float _objective returns)
- `eval/data_factory.py` — `DataFactory` (used by _tier_dataset to slice tier-appropriate dataset)

### Phase 23 Downstream Consumer
- `.planning/ROADMAP.md` §Phase 23 — CLI reads `best_params.json` produced by this phase; optimizer output format must be stable

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/runners/eval_runner.py` (`EvaluationRunner`) — **do not call inside _objective** (too heavy); but study its stage-invocation pattern as the reference for how to call AlignmentStage + LabelTransferStage + MetricsEngine directly
- `eval/types.py` `Trial` and `SearchResult` — already defined and frozen; optimizer populates these, does not redefine them
- `eval/config.py` `EvalConfig` — `search_space`, `search_strategy`, `tier`, `n_trials`, `output_dir` fields are all directly consumed; no new config fields needed for Phase 22
- `eval/runners/__init__.py` — regular package init exists; `optimizer.py` slots in here

### Established Patterns
- Frozen pydantic v2 models with `model_copy(update={...})` for mutations (from Phase 18/21)
- `logging.getLogger(__name__)` for all logging; `config.verbose` gates verbosity
- `eval/` is a namespace directory (no `__init__.py`); `eval/runners/` is a regular package
- macOS ARM import order: `zreg.dataset` → `zreg.*` → `torch` — enforce in `optimizer.py` and `search_strategies.py`

### Integration Points
- `eval/runners/optimizer.py` is the new file; `eval/search_strategies.py` is the new file
- `tests/test_optimizer.py` is the new test file (success criterion 5 from ROADMAP)
- Phase 23 CLI will import `HyperparamOptimizer` from `eval.runners.optimizer` and read `best_params.json` from `output_dir`

</code_context>

<specifics>
## Specific Details

- Optuna version constraint: `optuna >=4.0,<5` with TPE sampler; `n_startup_trials >= 2×N_params` per FRAME-10
- `best_params.json` and `search_history.json` written to `output_dir` (not a subdirectory)
- `search_history.json` serializes the full `SearchResult.history` as a list of Trial dicts
- Success criterion 1: sanity tier completes on laptop in under 2 minutes — _objective must stay lightweight (D-07)
- Success criterion 3: best params from optimizer improve score vs default params — test via a separate EvaluationRunner call in `tests/test_optimizer.py`, not inside the optimizer itself
- Success criterion 4: `prune_candidates()` demonstrably reduces candidate count — test should assert `len(pruned) < len(history)`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 22-HyperparamOptimizer & Search Strategies*
*Context gathered: 2026-05-29*
