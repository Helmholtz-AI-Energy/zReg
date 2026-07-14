---
phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
plan: 01
subsystem: testing
tags: [pydantic, pytest, cpd, label-transfer, benchmarking, eval-framework]

# Dependency graph
requires:
  - phase: 48-labeltransferstage-integration-for-learned-methods
    provides: LabelTransferStage with pointnet2/egnn/cpd_weighted/knn_voting methods, MODEL_REGISTRY, _load_learned_model
  - phase: 44-cpd-weighted-label-transfer-method
    provides: AlignmentStage(alignment_method="cpd") + AlignResult.estep_results, LabelTransferStage cpd_weighted branch
provides:
  - eval.types.MethodBenchmarkResult and eval.types.BenchmarkReport frozen pydantic models
  - Empirical proof (RESEARCH Assumption A1 / Pattern 2) that cpd_weighted accepts RAW source/target alongside a real align_result — no aligned_cloud substitution needed
  - benchmark_smoke_checkpoint fixture (pointnet2/egnn smoke-test checkpoint factory) for Plan 02/03 reuse
  - write_shah_fixture_csv helper (synthetic Shah-format CSV writer) for Plan 02/03 reuse
affects: [49-02-comparison-runner, 49-03-cli-or-report-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Frozen pydantic v2 result models (ConfigDict(frozen=True, arbitrary_types_allowed=True)) — MethodBenchmarkResult/BenchmarkReport mirror AlignResult/LabelResult/TrainingTriple exactly"
    - "Wave-0 empirical de-risking test before orchestration code is written"

key-files:
  created: [tests/test_benchmark_runner.py]
  modified: [eval/types.py]

key-decisions:
  - "cpd_weighted's align_result.estep_results is index-space (pmat), independent of whether source/target positions are raw or CPD-aligned — confirmed empirically, not just by code reading"
  - "Reused generate_trajectory + generate_labels (not DataFactory.generate_training_triple) for the Wave-0 test's source/target, matching the existing AlignmentStage/LabelTransferStage test conventions"

patterns-established:
  - "Pattern 6 benchmark result types now available for Plan 02's comparison runner"

requirements-completed: [D-01, D-02.2]

# Metrics
duration: 20min
completed: 2026-07-14
---

# Phase 49 Plan 1: Benchmark Types + Wave-0 Assumption Verification Summary

**Added `MethodBenchmarkResult`/`BenchmarkReport` frozen pydantic models to `eval/types.py` and empirically proved that `cpd_weighted` succeeds on raw (non-CPD-aligned) source/target when paired with a real `AlignmentStage(alignment_method="cpd")` result — de-risking Plan 02's comparison-runner design before it's built.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-14T15:55:00Z
- **Completed:** 2026-07-14T16:18:45Z
- **Tasks:** 2 completed
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- `MethodBenchmarkResult` and `BenchmarkReport` added to `eval/types.py`, frozen, in `__all__`, JSON-serializable via `model_dump()` + `json.dump()`
- `tests/test_benchmark_runner.py::TestCpdWeightedRawInput::test_cpd_weighted_raw_input` empirically proves RESEARCH Assumption A1 / Pattern 2: `cpd_weighted` does NOT require `align_result.aligned_cloud` — raw source/target plus the `align_result` object from a single `AlignmentStage(alignment_method="cpd")` run is sufficient
- `benchmark_smoke_checkpoint` fixture and `write_shah_fixture_csv` helper shipped for Plan 02/03 reuse
- Zero regressions: full suite 1303 passed, 18 skipped, 1 xpassed (was 1300 passed at Phase 48 close; +3 new tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add MethodBenchmarkResult + BenchmarkReport frozen models to eval/types.py** - `72628ad` (feat)
2. **Task 2: Wave-0 Pattern-2 verification test + shared fixtures + type round-trip test** - `d50e74a` (test)

**Plan metadata:** (pending — final commit below)

## Files Created/Modified
- `eval/types.py` - Added `MethodBenchmarkResult` (method, dataset_source, f1_score, knn_consistency, latency_seconds, n_pairs, error) and `BenchmarkReport` (params, results, notes) frozen pydantic models; both appended to `__all__`
- `tests/test_benchmark_runner.py` - New file: `TestCpdWeightedRawInput` (Wave-0 Assumption A1 proof), `TestBenchmarkReportRoundTrip` (Task 1 type coverage), `TestShahFixtureCsvLoadable` (fixture-writer proof), `benchmark_smoke_checkpoint` fixture, `write_shah_fixture_csv` helper

## Decisions Made
- Built the Wave-0 test's labeled source/target via `generate_trajectory` + `generate_labels(n_classes=4)` (mirroring `test_alignment_stage.py`/`test_label_transfer_stage.py` conventions) rather than `DataFactory.generate_training_triple` — kept the test self-contained and consistent with existing `AlignmentStage`/`LabelTransferStage` test fixtures, since neither the plan nor the interfaces required a specific choice.
- Used `cpd_penalty="rigid"` in the Wave-0 test's `align_params` per the documented Pitfall 3 precedent (`test_alignment_stage.py`: "only 'rigid' dispatches correctly upstream") to avoid an unrelated non-rigid/affine CPD flake masking the Assumption A1 result.
- Added a third test class (`TestShahFixtureCsvLoadable`) beyond the plan's two named test classes to directly exercise the `must_haves.truths` claim that `write_shah_fixture_csv`'s output is "loadable via `DataFactory.load_real(data_format=csv)`" — the plan's acceptance criteria only grepped for the header string, but the truth claim is a runtime behavior, so it is now asserted, not just structurally implied.

## Deviations from Plan

None - plan executed exactly as written. The one addition (`TestShahFixtureCsvLoadable`) is additive test coverage for an already-specified `must_haves.truths` claim, not a change in scope, approach, or files (still `tests/test_benchmark_runner.py`).

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `MethodBenchmarkResult`/`BenchmarkReport` are ready for Plan 02's comparison runner to populate.
- The `cpd_weighted` raw-input contract is now empirically verified: Plan 02 can safely design its comparison runner to run `AlignmentStage` once per dataset source and feed the same raw source/target into every `LabelTransferStage.run()` call (including `cpd_weighted`), rather than needing per-method aligned-cloud bookkeeping.
- `benchmark_smoke_checkpoint` and `write_shah_fixture_csv` are available in `tests/test_benchmark_runner.py` for Plan 02/03 to import or otherwise reuse.
- No blockers for Plan 02.

---
*Phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods*
*Completed: 2026-07-14*

## Self-Check: PASSED

- FOUND: eval/types.py
- FOUND: tests/test_benchmark_runner.py
- FOUND: .planning/phases/49-evaluation-benchmarking-of-learned-label-transfer-methods/49-01-SUMMARY.md
- FOUND commit: 72628ad
- FOUND commit: d50e74a
