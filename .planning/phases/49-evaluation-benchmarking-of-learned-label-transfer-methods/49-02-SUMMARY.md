---
phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
plan: 02
subsystem: testing
tags: [pydantic, pytest, label-transfer, benchmarking, eval-framework, cpd, pointnet2, egnn]

# Dependency graph
requires:
  - phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
    plan: 01
    provides: MethodBenchmarkResult/BenchmarkReport frozen pydantic models, benchmark_smoke_checkpoint fixture, write_shah_fixture_csv helper, empirical proof cpd_weighted accepts raw source/target + align_result
  - phase: 48-labeltransferstage-integration-for-learned-methods
    provides: LabelTransferStage 4-way method dispatch (knn_voting/cpd_weighted/pointnet2/egnn)
  - phase: 44-cpd-weighted-label-transfer-method
    provides: AlignmentStage(alignment_method="cpd") + AlignResult.estep_results
provides:
  - eval.runners.LabelTransferBenchmark — compare_methods/run_leakage_guard/save_report orchestration class
  - Exported from eval.runners (eval/runners/__init__.py)
  - Full D-02.1-4 test coverage (F1/knn comparison, real-data f1=None, latency>0, held-out-seed leakage guard) plus graceful degradation + save_report round-trip
affects: [49-03-cli-or-report-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single guarded AlignmentStage call (config.model_copy(update={'alignment_method': 'cpd'})) shared across a per-method loop, feeding every method the SAME raw source/target pair"
    - "Per-method try/except (ValueError, RuntimeError) graceful degradation recorded as MethodBenchmarkResult.error, never crashing the comparison run"

key-files:
  created: []
  modified:
    - eval/runners/benchmark_runner.py
    - eval/runners/__init__.py
    - tests/test_benchmark_runner.py

key-decisions:
  - "cpd_penalty must default to a valid CPD-type string ('rigid'), never a numeric magnitude — AlignmentStage.VALID_CPD = (None, 'rigid', 'affine', 'nonrigid'); a float default made validate_params raise unconditionally whenever cpd_weighted was requested without an explicit override"
  - "F1 is computed via torch.cat across all paired frames' transferred_labels vs. target[k]['label'], never per-frame-only, aggregating the comparison signal across the full held-out set"
  - "knn_consistency is always computed on the last paired frame only (tk = keys[-1]), mirroring EvaluationRunner's own knn_consistency frame-selection convention"

patterns-established:
  - "LabelTransferBenchmark is additive to EvaluationRunner — the multi-method 'compare N methods on the same data' orchestration tier, distinct from EvaluationRunner's one-runner-one-method-one-report design"

requirements-completed: [D-01, D-02.1, D-02.2, D-02.3, D-02.4, D-03]

# Metrics
duration: ~15min (continuation session; Task 1 orchestration class was already committed from a prior session)
completed: 2026-07-15
---

# Phase 49 Plan 2: LabelTransferBenchmark Orchestration Class + Dimension Tests Summary

**`LabelTransferBenchmark.compare_methods` runs all four `LabelTransferStage` methods against an identical raw source/target pair with one guarded `AlignmentStage` call, computing F1/knn_consistency/latency per method and recording failures gracefully — covering all four of Phase 49's D-02 evaluation dimensions.**

## Performance

- **Duration:** ~15 min (this session picked up mid-flight: Task 1's `LabelTransferBenchmark` class was already implemented and committed from a prior session; this session found and fixed a latent bug in it, then committed Task 2's test suite)
- **Tasks:** 2 completed (Task 1 pre-existing commit `5e2f175` + this session's `4c2a475` fix; Task 2 `0806f56`)
- **Files modified:** 3 (`eval/runners/benchmark_runner.py`, `eval/runners/__init__.py`, `tests/test_benchmark_runner.py`)

## Accomplishments
- `LabelTransferBenchmark` class in `eval/runners/benchmark_runner.py`: `compare_methods()` (single guarded `AlignmentStage` call only when `cpd_weighted` is requested, per-method `try/except` graceful degradation, F1 only when `has_ground_truth=True`, latency/knn always computed for successful methods), `run_leakage_guard()` (explicit required `held_out_seeds`, never hardcoded), `save_report()` (mirrors `EvaluationRunner.save_report` exactly — `model_dump()` + `json.dump(indent=2)`, never `model_dump_json()`)
- Exported from `eval.runners` (`eval/runners/__init__.py`)
- Fixed a latent bug (found this session, not caught by Task 1's own inline verify command since it never exercised `cpd_weighted` end-to-end): the internal `AlignmentStage` call's `cpd_penalty` default was `1.0` (a float), but `AlignmentStage.VALID_CPD = (None, "rigid", "affine", "nonrigid")` — this made `validate_params` raise unconditionally whenever `cpd_weighted` was requested without an explicit `params["cpd_penalty"]` override. Fixed the default to `"rigid"`, mirroring 49-01's own Wave-0 precedent.
- `tests/test_benchmark_runner.py` extended with 6 new test classes covering all D-02 dimensions: `TestCompareMethods` (D-01/D-02.1), `TestLatency` (D-02.3), `TestRealDataDimension` (D-02.2/Pitfall 1 — F1 always `None` against Shah-CSV-format real data), `TestLeakageGuard` (D-02.4 — `split_seeds()`-derived `held_out_seeds`), `TestErrorHandling` (D-03/Pitfall 1/3 — a missing-checkpoint method records `error` while a sibling method succeeds), `TestSaveReportRoundTrip`
- Zero regressions: full suite 1309 passed, 18 skipped, 1 xpassed (was 1303 at 49-01 close; +6 new tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement LabelTransferBenchmark orchestration class + save_report + export** - `5e2f175` (feat) — pre-existing commit from a prior session, verified present and correct in structure
2. **Task 1 bugfix (Rule 1 — auto-fixed, found during Task 2 verification):** `4c2a475` (fix) — corrected `cpd_penalty` default from `1.0` to `"rigid"`
3. **Task 2: Benchmark dimension tests (D-02.1-4 + graceful degradation)** - `0806f56` (test)

**Plan metadata:** (this commit, below)

## Files Created/Modified
- `eval/runners/benchmark_runner.py` - `LabelTransferBenchmark` class (`compare_methods`, `run_leakage_guard`, `save_report`); `cpd_penalty` default fixed from `1.0` to `"rigid"`
- `eval/runners/__init__.py` - `LabelTransferBenchmark` import + `__all__` export (already present from prior session)
- `tests/test_benchmark_runner.py` - Added `TestCompareMethods`, `TestLatency`, `TestRealDataDimension`, `TestLeakageGuard`, `TestErrorHandling`, `TestSaveReportRoundTrip`, plus shared `BENCHMARK_PARAMS`/`_build_benchmark_config`/`_build_holdout_pair` helpers reusing Plan 01's `benchmark_smoke_checkpoint` fixture and `write_shah_fixture_csv` helper

## Decisions Made
- Fixed `cpd_penalty`'s internal default to `"rigid"` (a valid `AlignmentStage.VALID_CPD` string) rather than the plan's literal `1.0` — see Deviations below.
- Test params (`BENCHMARK_PARAMS`) supply `cpd_penalty="rigid"` explicitly for clarity even though the class now defaults it correctly, documenting the constraint at the call site.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid `cpd_penalty` default type in `compare_methods`**
- **Found during:** Task 2 (writing `TestCompareMethods`, which exercises `cpd_weighted` end-to-end for the first time since Task 1 was written)
- **Issue:** 49-02-PLAN.md's Task 1 action text and 49-RESEARCH.md's Pattern 1 skeleton both specify `params.get("cpd_penalty", 1.0)` as the internal `AlignmentStage` call's default. `AlignmentStage.VALID_CPD = (None, "rigid", "affine", "nonrigid")` — a numeric `1.0` is not a valid CPD-type value, so `AlignmentStage.validate_params` raised unconditionally whenever `"cpd_weighted"` was requested without an explicit `params["cpd_penalty"]` override, breaking the plan's own must_have that every successful method (including `cpd_weighted`) produces populated `f1_score`/`knn_consistency`.
- **Fix:** Changed the default in `eval/runners/benchmark_runner.py` to `params.get("cpd_penalty", "rigid")`, mirroring 49-01-SUMMARY.md's own precedent ("only 'rigid' dispatches correctly upstream"). Updated the docstring accordingly.
- **Files modified:** `eval/runners/benchmark_runner.py`
- **Verification:** `pytest tests/test_benchmark_runner.py -q` — all 9 tests pass including `TestCompareMethods::test_all_four_methods_produce_results`, which exercises `cpd_weighted` end-to-end.
- **Committed in:** `4c2a475` (fix)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary for correctness — without this fix, `cpd_weighted` would fail unconditionally in every `compare_methods()` call, violating the plan's own must_have truth claims. No scope creep.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `LabelTransferBenchmark` is complete, exported, and fully tested across all four D-02 dimensions plus graceful degradation and report round-trip.
- Plan 03 (CLI or report integration, if planned) can build on `LabelTransferBenchmark.compare_methods`/`run_leakage_guard`/`save_report` directly — no further changes needed to this class.
- No blockers.

---
*Phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods*
*Completed: 2026-07-15*

## Self-Check: PASSED

- FOUND: eval/runners/benchmark_runner.py
- FOUND: .planning/phases/49-evaluation-benchmarking-of-learned-label-transfer-methods/49-02-SUMMARY.md
- FOUND commit: 5e2f175
- FOUND commit: 4c2a475
- FOUND commit: 0806f56
