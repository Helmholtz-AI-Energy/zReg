---
phase: 49-evaluation-benchmarking-of-learned-label-transfer-methods
verified: 2026-07-15T07:02:44Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
---

# Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods Verification Report

**Phase Goal:** Apply the existing F1/knn_consistency metrics plus additional evaluation dimensions
from Phase 45's strategy to benchmark eGNN/PointNet++ against knn_voting/cpd_weighted baselines
across this project's dataset sources. Per D-01, verified this session via a freshly-initialized
smoke-test checkpoint — the real meaningful comparison run happens once the user has trained real
weights externally.

**Verified:** 2026-07-15T07:02:44Z
**Status:** passed
**Re-verification:** No — initial verification

This is also the final phase of the eGNN/PointNet++ track (Phases 45→46→47→48→49); the track-level
coherence checks (item 10 below, plus a full repo-wide diff-stat) are included alongside the
phase-goal checks.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `MethodBenchmarkResult`/`BenchmarkReport` frozen pydantic models exist, JSON-round-trip | ✓ VERIFIED | `eval/types.py:415-508`; `TestBenchmarkReportRoundTrip::test_json_round_trip` passes |
| 2 | cpd_weighted succeeds (Assumption A1/Pattern 2) fed RAW source/target + a real `align_result` | ✓ VERIFIED | `tests/test_benchmark_runner.py::TestCpdWeightedRawInput::test_cpd_weighted_raw_input` — independently re-run this session, 1 passed |
| 3 | `LabelTransferBenchmark.compare_methods` runs `AlignmentStage` AT MOST ONCE, only when `cpd_weighted` requested, via `config.model_copy(update={"alignment_method": "cpd"})` | ✓ VERIFIED | Source read `eval/runners/benchmark_runner.py:171-181` — single `if "cpd_weighted" in self.methods:` guard before the method loop, `model_copy` used, never the caller's raw `config.alignment_method` |
| 4 | F1 is NEVER computed when `has_ground_truth=False` (real-data dimension) | ✓ VERIFIED | Source read `benchmark_runner.py:197-201` — `f1 = None` unconditionally, only overwritten inside `if has_ground_truth:` block; test `TestRealDataDimension::test_real_data_f1_none_no_crash` asserts `f1_score is None` for all 4 results |
| 5 | Each method's `ValueError`/`RuntimeError` is caught and recorded as `error`, other methods still succeed | ✓ VERIFIED | Source read `benchmark_runner.py:184-227` — per-method `try/except (ValueError, RuntimeError)`, loop continues; `TestErrorHandling` confirms pointnet2 fails gracefully while knn_voting still succeeds in the same run |
| 6 | `run_leakage_guard`'s `held_out_seeds` has no default (required argument) | ✓ VERIFIED | `inspect.signature` this session: `(self, held_out_seeds, params, n_classes, dataset_source='synthetic_holdout')` — `held_out_seeds` has no default |
| 7 | `benchmark_label_transfer.py` CLI parses `--held-out-seeds` from argv into a `range` and forwards it (not hardcoded) | ✓ VERIFIED | Source read `benchmark_label_transfer.py:155-169,231` — `_parse_held_out_seeds` splits `"start:stop"` and builds `range(int, int)`; `TestCliHeldOutSeedsParsed` confirms caller value flows through unmodified |
| 8 | Pattern 2/A1 empirical test genuinely passes (not just claimed) | ✓ VERIFIED | Independently re-ran `pytest tests/test_benchmark_runner.py -k cpd_weighted_raw_input -q` this session — 1 passed |
| 9 | `cpd_penalty` bug fix (`1.0` → `"rigid"`) is actually present; `AlignmentStage.VALID_CPD` requires a string/None type | ✓ VERIFIED | `benchmark_runner.py:178`: `params.get("cpd_penalty", "rigid")`; `eval/stages/alignment.py:92`: `VALID_CPD: tuple = (None, "rigid", "affine", "nonrigid")` |
| 10 | No point-count-regime guard, timeout, or automatic subsampling introduced (D-03) | ✓ VERIFIED | `grep -iE "timeout|subsampl|max_points|signal.alarm"` across `benchmark_runner.py`/`benchmark_label_transfer.py` returns only docstring mentions explaining the *absence* of such a guard — no code implements one |
| 11 | Track-level: all four `LabelTransferStage` methods still present, stage unmodified by Phase 49 | ✓ VERIFIED | `git diff --stat 39c0bf6..HEAD -- eval/stages/label_transfer.py` is empty (file untouched); `VALID_METHODS = ("knn_voting", "cpd_weighted", "pointnet2", "egnn")` confirmed at `label_transfer.py:126` |
| 12 | Full test suite passes with the count SUMMARY.md claims (~1312) | ✓ VERIFIED | Independently re-ran `pytest -q` this session — `1312 passed, 18 skipped, 1 xpassed` — exact match to 49-03-SUMMARY.md's claim |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/types.py` | `MethodBenchmarkResult`, `BenchmarkReport` frozen pydantic models | ✓ VERIFIED | Both classes present (lines 415, 475), `ConfigDict(frozen=True, arbitrary_types_allowed=True)`, both in `__all__` |
| `eval/runners/benchmark_runner.py` | `LabelTransferBenchmark` orchestration class + `save_report` | ✓ VERIFIED | 300 lines; `compare_methods`, `run_leakage_guard`, `save_report` all present and substantive (not stubs) |
| `eval/runners/__init__.py` | `LabelTransferBenchmark` export | ✓ VERIFIED | Imported and in `__all__` |
| `benchmark_label_transfer.py` | Repo-root CLI wrapping `LabelTransferBenchmark` | ✓ VERIFIED | 262 lines; `main`, `_build_parser`, `_parse_held_out_seeds`, `_parse_methods` all present, argparse + sys.path convention mirrored from `run_eval.py` |
| `tests/test_benchmark_runner.py` | Wave-0 + dimension tests | ✓ VERIFIED | 443 lines, 9 test functions across 7 test classes, all passing |
| `tests/test_benchmark_cli.py` | CLI smoke test | ✓ VERIFIED | 149 lines, 3 test functions, all passing |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `benchmark_runner.py` | `eval.stages.LabelTransferStage.run` | per-method loop, `align_result` only for `cpd_weighted` | ✓ WIRED | Line 191: `align_result=align_result if method == "cpd_weighted" else None` |
| `benchmark_runner.py` | `eval.stages.AlignmentStage.run` | single guarded call under `"cpd_weighted" in methods` | ✓ WIRED | Lines 172-181 |
| `benchmark_runner.py` | `zreg.metrics.compute_f1`/`knn_consistency` | metric computation per method | ✓ WIRED | Lines 47, 201, 204 |
| `benchmark_runner.py` | `eval.types.BenchmarkReport`/`MethodBenchmarkResult` | construction + `save_report` | ✓ WIRED | Lines 206-229, 295-298 |
| `benchmark_label_transfer.py` | `eval.runners.LabelTransferBenchmark` | `run_leakage_guard` + `compare_methods` + `save_report` | ✓ WIRED | Lines 235, 238, 247, 255 |
| `benchmark_label_transfer.py` | `eval.config.EvalConfig.from_yaml` | config load + `EvalConfigError` handling | ✓ WIRED | Lines 214-218 |

### Data-Flow Trace (Level 4)

Not applicable in the traditional UI-data sense (this is a research/CLI library phase, not a
rendering component) — data-flow was instead traced through the try/except and F1-gating logic
above (truths 3-5), which is the equivalent "does data really flow correctly" check for this
domain. Confirmed: `has_ground_truth` genuinely gates F1 computation at the source-code level, not
merely in a test's assumption.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cpd_weighted raw-input assumption (Pattern 2/A1) | `pytest tests/test_benchmark_runner.py -k cpd_weighted_raw_input -q` | 1 passed | ✓ PASS |
| Full CLI + runner test suite | `pytest tests/test_benchmark_cli.py tests/test_benchmark_runner.py -q` | 12 passed | ✓ PASS |
| Full repo suite | `pytest -q` | 1312 passed, 18 skipped, 1 xpassed | ✓ PASS |
| `run_leakage_guard` signature has required `held_out_seeds` | `inspect.signature(...)` | no default on `held_out_seeds` | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared or discovered for this phase. SKIPPED (no
probe-based verification convention used in this phase — pytest is the verification mechanism, run
independently above).

### Requirements Coverage

No formal `.planning/REQUIREMENTS.md` IDs exist for this phase (ad hoc, per 49-CONTEXT.md/49-RESEARCH.md). Verified against D-01 through D-03 / D-02.1-4 instead:

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| D-01 | 49-01, 49-02, 49-03 | Build + smoke-test machinery via freshly-initialized checkpoint | ✓ SATISFIED | `benchmark_smoke_checkpoint` fixture, `TestCompareMethods`, CLI end-to-end smoke test all pass against untrained checkpoints |
| D-02.1 | 49-02 | F1/knn_consistency comparison across 4 methods | ✓ SATISFIED | `TestCompareMethods` — all 4 methods present, f1/knn populated for successes |
| D-02.2 | 49-01, 49-02 | Generalization across dataset sources (synthetic + real-format) | ✓ SATISFIED | `write_shah_fixture_csv` + `TestRealDataDimension` exercises the real Shah CSV code path end-to-end |
| D-02.3 | 49-02 | CPU inference-latency benchmarking | ✓ SATISFIED | `TestLatency` — `latency_seconds > 0` for every successful method |
| D-02.4 | 49-02 | Train/eval leakage guard via held-out synthetic variants | ✓ SATISFIED | `run_leakage_guard` + `TestLeakageGuard`, explicit `held_out_seeds` via `split_seeds` |
| D-03 | 49-02, 49-03 | Document scale mismatch, don't block; no new guard/timeout/subsampling | ✓ SATISFIED | Graceful `try/except` degradation confirmed; no guard code found anywhere in phase 49's diff |

No orphaned requirements — D-01 through D-03/D-02.1-4 all mapped to at least one plan's `requirements` frontmatter field and independently confirmed above.

### Anti-Patterns Found

None. Scanned `eval/types.py`, `eval/runners/benchmark_runner.py`, `eval/runners/__init__.py`,
`benchmark_label_transfer.py`, `tests/test_benchmark_runner.py`, `tests/test_benchmark_cli.py` for
`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/"not yet implemented"/"coming soon" — zero matches.
No empty-return stubs, no hardcoded-empty data flowing to output.

### Human Verification Required

None. This is a backend/library/CLI phase with no UI surface; every claimed behavior (F1 gating,
AlignmentStage call-count, graceful degradation, CLI argument forwarding, cpd_penalty type fix) was
independently verified via direct source read plus live pytest re-execution in this session, not
merely inferred from SUMMARY.md prose. The deferred "real, meaningful benchmark run against a
trained checkpoint" is explicitly out of scope for this phase by design (D-01) — not a gap requiring
human sign-off now.

### Gaps Summary

No gaps. All 12 observable truths, all 6 required artifacts, and all 6 key links verified directly
against the current source. The one prior-session claim flagged for extra scrutiny (49-02-SUMMARY.md's
Task 1 commit `5e2f175` reportedly from "a prior session") checks out: the commit exists in git log
with an authentic earlier timestamp (2026-07-14 18:24:30, same day as 49-01's commits, before
49-02's other commits on 2026-07-15), and its content matches the plan's specification. The
`cpd_penalty` bug-fix claim (`1.0` → `"rigid"`) is confirmed present in the current source and is a
real, necessary fix given `AlignmentStage.VALID_CPD`'s string-only contract. The full test suite was
independently re-run end-to-end (not trusted from SUMMARY.md) and reproduces the exact claimed count
(1312 passed / 18 skipped / 1 xpassed). Track-level check confirms `eval/stages/label_transfer.py`
was not touched by Phase 49 and all four label-transfer methods remain intact — the eGNN/PointNet++
track (Phases 45→49) is left in a coherent, working state.

---

*Verified: 2026-07-15T07:02:44Z*
*Verifier: Claude (gsd-verifier)*
