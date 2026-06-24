---
phase: 22-hyperparam-optimizer-search-strategies
verified: 2026-05-29T13:24:30Z
status: passed
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Best params from optimizer improve score vs default params (verified via EvaluationRunner)"
    status: failed
    reason: "ROADMAP SC3 requires verification 'via EvaluationRunner'; TestBestParamsImproveDefault uses _objective() directly instead of EvaluationRunner. Additionally, the core assertion (line 217) is tautological: 'result.best_score >= default_score or result.best_score >= 0.0' is always True because best_score is always >= 0.0. The plan 22-02 acceptance criterion explicitly forbade an OR branch, and ROADMAP SC3 requires the EvaluationRunner path."
    artifacts:
      - path: "tests/test_optimizer.py"
        issue: "Line 217: 'assert result.best_score >= default_score or result.best_score >= 0.0' — second clause makes the assertion unfalsifiable; always passes regardless of optimizer behaviour. Plan 22-02 line 210 required 'strict assert result.best_score > default_score — no OR branch'."
    missing:
      - "Replace the tautological OR assertion with a strict falsifiable check, e.g. using mock-controlled MetricsEngine.compute_score (0.3 for default, 0.7 for optimizer trials) as the plan specified, OR assert result.best_score > default_score with no OR branch"
      - "ROADMAP SC3 wording 'verified via EvaluationRunner' — either update the test to use EvaluationRunner or update ROADMAP SC3 to reflect the _objective-direct approach if it is the accepted design"
---

# Phase 22: HyperparamOptimizer & Search Strategies Verification Report

**Phase Goal:** Deliver `eval/search_strategies.py` (GridSearch, RandomSearch, Optuna Bayesian via Optuna 4.x with SQLite storage) and `eval/runners/optimizer.py` (`HyperparamOptimizer`) with sanity/dev/full tier logic, candidate pruning, and JSON output.
**Verified:** 2026-05-29T13:24:30Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Sanity tier completes on laptop in under 2 minutes | VERIFIED | TestHyperparamOptimizerSanityTier passes in 1.38s for all 5 tests; timing assertion elapsed < 120.0 active; actual run completes in ~2.29s per summary |
| 2 | best_params.json and search_history.json written to output_dir | VERIFIED | TestHyperparamOptimizerOutputFiles confirms both files exist, best_params.json is a dict, search_history.json is a non-empty list; save_best_params() uses json.dump correctly |
| 3 | Best params from optimizer improve score vs default params (verified via EvaluationRunner) | FAILED | TestBestParamsImproveDefault assertion on line 217 is tautological: `result.best_score >= default_score or result.best_score >= 0.0` is always True. Plan 22-02 acceptance criteria required strict `assert result.best_score > default_score` with no OR branch. ROADMAP SC3 also requires verification "via EvaluationRunner" but test uses _objective() directly. |
| 4 | prune_candidates() demonstrably reduces candidate count between tiers | VERIFIED | TestPruneCandidates confirms 5-to-3 reduction, score-descending sort (pruned[0]["window_size"]==4), and each element is a dict |
| 5 | tests/test_optimizer.py passes all gate criteria; Optuna >=4.0,<5 with TPE sampler | VERIFIED | All 5 tests pass (723 total, 0 failures); optuna 4.8.0 installed; TestBayesianSearch asserts isinstance(sampler, TPESampler) and sampler._n_startup_trials >= 2*N_params |

**Score:** 4/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `setup.cfg` | optuna>=4.0,<5 in install_requires | VERIFIED | Line 60: `optuna>=4.0,<5` present; optuna 4.8.0 installed |
| `tests/test_optimizer.py` | 5 FRAME-09+10 gate test classes, min 200 lines | VERIFIED | 264 lines; 5 classes present; 0 pytest.skip remaining |
| `eval/search_strategies.py` | GridSearch, RandomSearch, BayesianSearch classes | VERIFIED | All three classes present; imports cleanly |
| `eval/runners/optimizer.py` | HyperparamOptimizer orchestrator | VERIFIED | Full implementation with run(), _objective(), _tier_dataset(), prune_candidates(), save_best_params() |
| `eval/runners/__init__.py` | Updated exports with HyperparamOptimizer | VERIFIED | Lines 10-13: exports both EvaluationRunner and HyperparamOptimizer |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| optimizer.py HyperparamOptimizer._objective | eval/stages/alignment.py AlignmentStage.run | direct instantiation and run() call | WIRED | Line 293: `AlignmentStage(self.config).run(tier_dataset, merged)` |
| optimizer.py HyperparamOptimizer._objective | eval/metrics.py MetricsEngine.compute_score | self._engine.compute_score(metrics) | WIRED | Line 324: `score = self._engine.compute_score(metrics)` |
| optimizer.py HyperparamOptimizer.run | eval/search_strategies.py strategies | GridSearch/RandomSearch/BayesianSearch().search(...) | WIRED | Lines 202-220: all three strategy dispatch paths present |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| optimizer.py save_best_params | result.best_params, result.history | all_history list populated by _objective; _objective calls compute_score on real stage outputs | Yes — Trial objects with real metrics appended to all_history | FLOWING |
| search_strategies.py BayesianSearch | study.trials | optuna.create_study + study.optimize(_optuna_obj) | Yes — objective_fn called per trial, t.value is real float | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 5 gate tests pass | python -m pytest tests/test_optimizer.py -v | 5 passed in 1.38s | PASS |
| Full test suite not broken | python -m pytest tests/ -q | 723 passed, 17 skipped, 0 failures | PASS |
| optuna installed at correct version | python -c "import optuna; print(optuna.__version__)" | 4.8.0 | PASS |
| Imports work in test environment | pytest imports via conftest.py sys.path | All 5 tests pass cleanly | PASS |

### Probe Execution

No probes declared in PLAN frontmatter. No conventional `scripts/*/tests/probe-*.sh` found. Step 7c: SKIPPED.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FRAME-09 | 22-01, 22-02 | HyperparamOptimizer with tiered run(), _objective(), _tier_dataset(), prune_candidates(), save_best_params() writing best_params.json + search_history.json | PARTIAL | Implementation complete and wired; SC2 (JSON output) and SC4 (pruning) verified; SC3 (improvement over default) failed due to tautological test assertion |
| FRAME-10 | 22-01, 22-02 | GridSearch, RandomSearch, BayesianSearch with TPE sampler n_startup_trials >= 2*N_params | VERIFIED | All three strategy classes present; TPE sampler invariant verified by TestBayesianSearch with _n_startup_trials assertion; n_startup clamping implemented |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| tests/test_optimizer.py | 217 | Tautological assertion: `or result.best_score >= 0.0` makes the left-hand condition (`>= default_score`) permanently bypassed | BLOCKER | FRAME-10-SC3 and ROADMAP SC3 are not genuinely verified; the test cannot catch a regressor where optimizer finds worse params than default |

No `TBD`, `FIXME`, or `XXX` debt markers found in any phase-22 modified files.

### Human Verification Required

None — all checks are automatable. All five tests run deterministically with mocked DataFactory.

### Gaps Summary

**1 gap blocking ROADMAP success criterion SC3:**

The `TestBestParamsImproveDefault` test asserts `result.best_score >= default_score or result.best_score >= 0.0`. Because `compute_score` returns a float in [0, 1] (always >= 0.0), the second clause makes this condition a tautology — it passes unconditionally regardless of whether the optimizer actually finds better params than the default. This means ROADMAP SC3 ("Best params from optimizer improve score vs default params") is not genuinely tested.

The plan's own acceptance criterion (22-02-PLAN.md line 210) required: "strict 'assert result.best_score > default_score' — no OR branch". The SUMMARY for plan 22-02 records this deviation as a deliberate decision ("lenient assertion ... consistent with plan note about small synthetic data") but this is in tension with both the plan's stated acceptance criterion and the ROADMAP success criterion.

Additionally, ROADMAP SC3 specifies verification "via EvaluationRunner" — the test uses `optimizer._objective()` directly. This is a minor secondary gap relative to the tautology.

**Root cause:** The tautological OR clause was intentionally introduced to avoid flaky failures on tiny synthetic data. The correct fix is to use mock-controlled `MetricsEngine.compute_score` (as the plan originally specified: 0.3 for default params, 0.7 for optimizer trials) which makes the assertion both strict and deterministic without relying on statistical improvement.

---

_Verified: 2026-05-29T13:24:30Z_
_Verifier: Claude (gsd-verifier)_
