# Phase 22: HyperparamOptimizer & Search Strategies - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-29
**Phase:** 22-HyperparamOptimizer & Search Strategies
**Areas discussed:** Tier execution flow, Search space format, Objective function weight, Optuna study lifecycle

---

## Tier Execution Flow

### Q1: Does run() run all tiers or stop at config.tier?

| Option | Description | Selected |
|--------|-------------|----------|
| Stop at config.tier | If "sanity", only run sanity. Config controls the ceiling. | ✓ |
| Always run all three | Every run() executes sanity→dev→full regardless of config.tier. | |

**User's choice:** Stop at config.tier

---

### Q2: What differs between tiers?

| Option | Description | Selected |
|--------|-------------|----------|
| Both n_trials and dataset size | Sanity: small synthetic subset + few trials. Dev: larger real subset. Full: whole dataset + all n_trials. | ✓ |
| Only n_trials differ | All tiers evaluate on the same dataset. | |
| Only dataset size differs | Same number of trials, smaller data. | |

**User's choice:** Both n_trials and dataset size

---

### Q3: What does prune_candidates() prune?

| Option | Description | Selected |
|--------|-------------|----------|
| Prune the candidate list | Keeps top-k Trial param dicts by score as warm-start seeds. Search space unchanged. | ✓ |
| Prune the search space | Removes poor-performing param values, narrowing the space. | |

**User's choice:** Prune the candidate list

---

### Q4: How are pruned params used in the next tier?

| Option | Description | Selected |
|--------|-------------|----------|
| Warm-start seeds | Top-k added as fixed trials to Optuna study / seeded into Random/Grid. Next tier still explores. | ✓ |
| Filter: only evaluate pruned params | Next tier evaluates only exact top-k param sets — no new exploration. | |
| Passed as initial_params to EvaluationRunner only | Pruned params inform final EvaluationRunner call but don't affect strategy. | |

**User's choice:** Warm-start seeds

---

## Search Space Format

### Q1: Unified format or strategy-specific?

| Option | Description | Selected |
|--------|-------------|----------|
| Unified format for all three | One dict structure all strategies interpret. | ✓ |
| Strategy-specific formats | Grid, Random, Optuna each get their own format. | |

**User's choice:** Unified format

---

### Q2: What unified format?

| Option | Description | Selected |
|--------|-------------|----------|
| List of values per param | `window_size: [3, 5, 7]` — Grid exhausts, Random samples, Optuna treats as categorical. | ✓ |
| Min/max/type per param | More expressive for Optuna continuous ranges, heavier YAML, awkward for categoricals. | |

**User's choice:** List of values per param
**Notes:** User asked for pros/cons before deciding. Key factor: all current hyperparams are small categorical sets, not continuous ranges. Optuna over categorical space is still useful for learning promising combinations.

---

### Q3: Are search_space keys exact param names?

| Option | Description | Selected |
|--------|-------------|----------|
| Exact param names, no mapping | Keys passed directly to stages as params dict. Zero indirection. | ✓ |
| Mapped through a schema | Mapping layer translates keys to stage-specific names. | |

**User's choice:** Exact param names, no mapping

---

## Objective Function Weight

### Q1: Does _objective call EvaluationRunner or stages directly?

| Option | Description | Selected |
|--------|-------------|----------|
| Stages + MetricsEngine directly | No EvaluationRunner, no file I/O. Fast per trial. | ✓ |
| EvaluationRunner with save_plots=False | Reuses orchestration but carries EvalReport overhead. | |

**User's choice:** Stages + MetricsEngine directly

---

### Q2: What dataset does _objective evaluate on?

| Option | Description | Selected |
|--------|-------------|----------|
| Tier-specific subset via _tier_dataset() | Sanity: small synthetic. Dev: larger real subset. Full: complete dataset. | ✓ |
| Always the full dataset | Tier only controls n_trials, not data size. | |

**User's choice:** Tier-specific subset

---

### Q3: What happens on trial failure?

| Option | Description | Selected |
|--------|-------------|----------|
| Return 0.0 and log a warning | Failed trial scores worst-possible. Run continues. | ✓ |
| Raise and abort the search | Any exception propagates up and stops run(). | |

**User's choice:** Return 0.0 and log a warning

---

## Optuna Study Lifecycle

### Q1: Study name?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed "zreg-hpo" per output_dir | Simple, deterministic, tied to output_dir. Re-run resumes the study. | ✓ |
| Derived from config hash | Unique study per config. No collisions but SQLite accumulates. | |
| User-specified via config field | Full control but adds config surface area. | |

**User's choice:** Fixed name per output_dir ("zreg-hpo")

---

### Q2: Where does the SQLite file live?

| Option | Description | Selected |
|--------|-------------|----------|
| Inside output_dir | `{output_dir}/optuna.db` — co-located with best_params.json. | ✓ |
| Repo root | `optuna.db` at repo root, shared across all runs. | |

**User's choice:** Inside output_dir

---

### Q3: Do GridSearch/RandomSearch write persistent state?

| Option | Description | Selected |
|--------|-------------|----------|
| Stateless — results in memory only | save_best_params() writes at the end. No SQLite, no incremental writes. | ✓ |
| Write search_history.json incrementally | Allows partial recovery if process killed mid-search. | |

**User's choice:** Stateless

---

## Claude's Discretion

- Strategy interface design (Protocol/ABC for GridSearch/RandomSearch/BayesianSearch) — implementation detail not discussed
- SQLite study naming mechanism (hash derivation vs string literal) — decided as "zreg-hpo" by user
- Exact tier trial count multipliers (e.g., sanity=5% of n_trials, dev=30%, full=100%) — planner to decide based on 2-minute constraint

## Deferred Ideas

None — discussion stayed within phase scope.
