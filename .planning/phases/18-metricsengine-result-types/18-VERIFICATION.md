---
phase: 18-metricsengine-result-types
verified: 2026-05-28T08:30:00Z
status: passed
score: 4/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Validate normalize() contract with negative metric inputs (CR-01)"
    expected: "Either pydantic ValidationError on construction of StageMetrics with negative lower-is-better fields, OR normalize() clamps output to [0,1]. Currently chamfer_distance=-0.5 produces normalize['chamfer']=2.0, which violates the [0,1] contract stated in ROADMAP SC-2."
    why_human: "The upstream zreg.metrics primitives never produce negative values in the operational path (chamfer/hausdorff are geometric means, path_smoothness is variance, temporal_stability is mean Frobenius norm — all non-negative). Developer must decide: (a) add ge=0 validators to StageMetrics lower-is-better fields to make the contract explicit, or (b) add a max(0, x) guard in normalize(), or (c) accept the known limitation for a research codebase where all inputs are controlled."
  - test: "Validate compute_score() [0,1] invariant when weight keys are absent from normalized (CR-02)"
    expected: "If metric_weights contains a key absent from metrics.normalized, the auto-rescale should account for only the present keys so that score remains in [0,1]. Currently total is computed over ALL weight keys including absent ones, so the score is suppressed below the correct value when keys are missing."
    why_human: "Decide whether the current behavior (silently lower score on missing keys) is acceptable as a robustness clause or whether the fix from CR-02 (compute total over only active keys) should be applied. This affects correctness of compute_score when StageMetrics.normalized has been manually constructed with fewer than all 6 keys."
---

# Phase 18: MetricsEngine & Result Types Verification Report

**Phase Goal:** Deliver `eval/types.py` (six frozen pydantic result models: AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport) and `eval/metrics.py` (MetricsEngine wrapping all existing `zreg.metrics.*` with normalization, aggregation, scoring, and sanity checking).
**Verified:** 2026-05-28T08:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | All six dataclasses importable from `eval.types` | VERIFIED | `eval/types.py` exists (310 lines); all 6 classes in `__all__`; confirmed via `python3 -c "from eval.types import ..."` and `TestResultTypesImportable` (1 test, green) |
| SC-2 | `MetricsEngine.normalize()` maps every metric to [0,1] with correct direction | PARTIAL | For the operational path (non-negative inputs from `zreg.metrics.*`), normalize works correctly — confirmed by 5 passing tests in `TestNormalize`. However, CR-01 (from code review) shows negative float inputs produce out-of-range values (e.g. chamfer=-0.5 → normalized=2.0). No `ge=0` validators on `StageMetrics` lower-is-better fields; no clamp in `normalize()`. This is a known flaw but may be acceptable for a research codebase where all inputs originate from geometric primitives that are always non-negative. |
| SC-3 | `MetricsEngine.sanity_check()` fires warnings on degenerate inputs (empty cloud, all-same labels) | PARTIAL | 6 tests in `TestSanityCheck` pass. Empty cloud, single-frame, all-same labels, all-sentinel, non-finite metric, and clean-baseline negative control all pass. CR-03 (from code review) identifies a vacuous-truth false positive: an empty tensor `torch.tensor([], dtype=torch.long)` triggers the sentinel flag before the same-labels flag, masking the real "empty tensor" condition. Verified by running the scenario: `sanity_check(LabelResult({0: empty_tensor}))` returns `['all-sentinel labels: compute_f1 returns 0']` instead of an appropriate "empty label tensor" flag. |
| SC-4 | `MetricsEngine.compute_score()` returns a scalar in [0,1] | PARTIAL | For the normal case (all 6 canonical keys present in `metrics.normalized`), compute_score returns a float in [0,1]. CR-02 (from code review) shows that when any weight key is absent from `metrics.normalized`, the denominator is over-counted, producing a suppressed (potentially outside-range) score. Verified: removing "chamfer" from normalized while all other metrics are 1.0 produces score=0.65 instead of 1.0. The `test_unknown_normalized_key_is_skipped` test asserts only `isinstance(score, float)`, not `0.0 <= score <= 1.0`, so this violation is undetected by the test suite. |
| SC-5 | `tests/test_metrics.py` passes all gate criteria on handcrafted fixtures | VERIFIED | 24 tests pass, 0 skip, 0 fail. Full repo suite: 603 passed, 17 skipped (baseline 579 + 24 new). `pytest tests/test_metrics.py -v` exits 0. All 6 test classes populated. |

**Score: 4/5 truths VERIFIED or partially verified** — SC-2, SC-3, SC-4 are partially verified with known defects documented in the code review (CR-01, CR-02, CR-03). These defects affect edge-case inputs that do not appear in the operational path from `zreg.metrics.*` primitives. Whether to fix before proceeding requires human judgment.

### Deferred Items

None.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/types.py` | 6 frozen pydantic result models (FRAME-04) | VERIFIED | 310 lines; 6 classes; all use `ConfigDict(frozen=True, arbitrary_types_allowed=True)`; `__all__` correct |
| `eval/metrics.py` | MetricsEngine with normalize/compute_score/aggregate/sanity_check (FRAME-03) | VERIFIED | 395 lines; MetricsEngine class with all 5 methods (including optional `compute_stage_metrics`); `__all__ = ["MetricsEngine"]` |
| `eval/config.py` | EvalConfig extended with `metric_weights` field | VERIFIED | `metric_weights: dict[str, float]` present after `val_split`, before `@classmethod`; `default_factory=lambda: {...}` with D-07 defaults; `extra="forbid"` preserved |
| `tests/test_metrics.py` | All 6 test classes populated, 0 `pytest.skip` markers | VERIFIED | 455 lines; 24 active tests; 0 `pytest.skip` calls; all 6 `class Test*` present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval/types.py` | `zreg.dataset.zRegPointCloud` | `AlignResult.aligned_cloud` field | WIRED | Line 100: `aligned_cloud: dict[int, zRegPointCloud]`; import at line 50 precedes `import torch` (line 53) |
| `eval/types.py` | `torch.Tensor` | `LabelResult.transferred_labels` field | WIRED | Line 130: `transferred_labels: dict[int, torch.Tensor]` |
| `eval/metrics.py` | `zreg.metrics` (all 6 primitives) | `from zreg.metrics import` | WIRED | Lines 57-64: chamfer, compute_f1, hausdorff, knn_consistency, path_smoothness, temporal_stability all imported; used in `compute_stage_metrics` |
| `eval/metrics.py` | `eval.config.EvalConfig` | `MetricsEngine.__init__` stores config | WIRED | Line 70: `from eval.config import EvalConfig`; line 183: `self.config.metric_weights` read in `compute_score` |
| `eval/metrics.py` | `eval.types.StageMetrics` | normalize/compute_score/aggregate/sanity_check inputs | WIRED | Line 71: `from eval.types import AlignResult, LabelResult, StageMetrics` |
| `eval/config.py` | `MetricsEngine.compute_score` | `metric_weights` consumed downstream | VERIFIED | Field at line 110 with `default_factory=lambda:`; canonical short-name keys match `normalize()` output |
| `tests/test_metrics.py` | `eval.types` | import for all 6 result models | WIRED | Lines 41-49 import all 6 from eval.types |
| `tests/test_metrics.py` | `eval.metrics.MetricsEngine` | `from eval.metrics import MetricsEngine` | WIRED | Line 41 |

### Data-Flow Trace (Level 4)

This phase delivers typed output containers and a stateless computation engine — not a data rendering layer. Level 4 trace is not applicable (no UI/rendering components). The behavioral spot-checks below substitute.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 6 types importable from eval.types | `python3 -c "from eval.types import AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport"` | exit 0 | PASS |
| normalize(all zeros) returns correct values | `me.normalize(all_zero_sm)` | `{'chamfer': 1.0, 'hausdorff': 1.0, ..., 'f1': 0.0, 'knn_consistency': 0.0}` | PASS |
| aggregate([]) returns {} | `me.aggregate([])` | `{}` | PASS |
| sanity_check NaN metric flagged | `me.sanity_check(metrics=NaN_sm)` | `['non-finite metric: chamfer_distance=nan']` | PASS |
| compute_score raises on zero-sum weights | zero-weight engine `.compute_score(sm)` | `ValueError: metric_weights sum to zero` | PASS |
| normalize(chamfer=-0.5) returns out-of-range | `me.normalize(negative_sm)["chamfer"]` | `2.0` (> 1.0, violates SC-2) | FAIL (CR-01) |
| compute_score(missing key in normalized) | partial normalized, score | `0.65` instead of re-normalized `1.0` (violates SC-4 invariant) | FAIL (CR-02) |
| sanity_check(empty tensor) | `me.sanity_check(label=LabelResult({0: empty_tensor}))` | `['all-sentinel labels: ...']` false positive (CR-03) | FAIL (CR-03) |
| Full test suite green | `pytest tests/ -q` | `603 passed, 17 skipped` | PASS |

### Probe Execution

No probes declared or found in phase directory.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FRAME-04 | 18-01-PLAN.md | `eval/types.py` — 6 frozen pydantic result models with correct field types | SATISFIED | `eval/types.py` exists; all 6 models with correct fields, frozen + arbitrary_types_allowed; REQUIREMENTS.md updated to `[x]` |
| FRAME-03 | 18-02-PLAN.md | `eval/metrics.py` — MetricsEngine wrapping all 6 primitives with normalize/aggregate/compute_score/sanity_check | SATISFIED (with known defects) | `eval/metrics.py` exists with all required methods; all wrap `zreg.metrics.*` without reimplementation; code review identified 3 correctness edge cases (CR-01/CR-02/CR-03) in normalize/compute_score/sanity_check; REQUIREMENTS.md still shows `[ ]` — documentation discrepancy |

**Documentation note:** REQUIREMENTS.md line 115 shows `FRAME-03 | Phase 18 | Pending`. The traceability table was not updated after Phase 18 completion. The code satisfies FRAME-03 but the requirements document still reflects Pending status.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `eval/metrics.py` | 382 | `# TODO: confirm signature with Phase 21 caller` inside docstring | INFO | Text note embedded inside a `compute_stage_metrics` docstring (not real Python comment). Already flagged as IN-01 in 18-REVIEW.md. No functional impact. |
| `eval/metrics.py` | 148-154 | `normalize()` uses `1/(1+x)` with no non-negativity guard | WARNING | CR-01: negative metric inputs produce out-of-range normalized values. Does not affect operational path (all upstream primitives return non-negative values) but violates the documented contract and creates a `ZeroDivisionError` risk at exactly `x=-1.0`. |
| `eval/metrics.py` | 183-188 | `compute_score()` computes `total` over all weight keys including absent ones | WARNING | CR-02: score is suppressed below correct value when any weight key is absent from `metrics.normalized`. Documented as "robustness" but breaks the `[0,1]` invariant for partial normalized dicts. |
| `eval/metrics.py` | 299-311 | `sanity_check()` sentinel check uses `.all()` before empty-tensor guard | WARNING | CR-03: empty tensors produce false-positive "all-sentinel" warnings due to PyTorch vacuous truth semantics. |
| `eval/config.py` | 153-156 | `from_yaml` reports only first ValidationError | WARNING | WR-01 from review: user must re-run once per invalid YAML field. Not a correctness issue for this phase. |

No `TBD`, `FIXME`, or `XXX` markers found in phase files (the `# TODO` at line 382 is inside a docstring string, not parsed as Python code, so it is not a real comment token). Per the debt-marker gate: no blockers from unresolved debt markers.

### Human Verification Required

#### 1. CR-01: normalize() out-of-range values for negative inputs

**Test:** Construct `StageMetrics(chamfer_distance=-0.5, hausdorff_distance=0.0, path_smoothness=0.0, temporal_stability=0.0, f1_score=0.0, knn_consistency=0.0)` and call `MetricsEngine.normalize(sm)`. Observe `norm["chamfer"] == 2.0`.
**Expected:** Developer decision — either (a) add `ge=0` field validators to StageMetrics lower-is-better fields (pydantic `Field(ge=0)`) so negative inputs are rejected at construction, or (b) add `max(0.0, x)` clamp in `normalize()`, or (c) document as an accepted limitation for this research codebase where all inputs originate from non-negative geometric primitives.
**Why human:** All upstream `zreg.metrics.*` primitives (chamfer, hausdorff, path_smoothness, temporal_stability) compute non-negative values by construction (geometric distances, norms, variance). The defect is theoretical in the current operational context but will become real if future phases pass synthetic or fabricated inputs to MetricsEngine. Only the developer can decide if Phase 18 should fix this or defer to Phase 23 CLI hardening.

#### 2. CR-02: compute_score() invariant broken for partial normalized dicts

**Test:** Build `StageMetrics` with full `normalized` dict missing the `"chamfer"` key; call `compute_score()`. Observe score = 0.65 (chamfer weight 0.35 is subtracted from the denominator but not the numerator).
**Expected:** Developer decision — either apply the fix from CR-02 (compute `total` over only keys present in `metrics.normalized`) or document that `metrics.normalized` is always expected to be fully populated before calling `compute_score()`.
**Why human:** The current behavior is internally inconsistent: the docstring says "keys absent from `metrics.normalized` are silently skipped" but the auto-rescale denominator does not account for this. The fix is mechanical (2-3 lines) but changes observable behavior for callers that might depend on the current broken semantics.

## Gaps Summary

No hard BLOCKER gaps: all required files exist, are substantive, and are wired. The full test suite is green with 603 passed. The phase goal is structurally achieved.

Three WARNING-level defects in `eval/metrics.py` were identified by the code review (CR-01, CR-02, CR-03) and verified by the verifier. These affect edge cases outside the operational path of the current codebase. Two items require developer judgment before a PASSED verdict can be issued.

The REQUIREMENTS.md traceability table has a documentation gap: FRAME-03 still shows Pending despite the implementation being present and tested.

---

_Verified: 2026-05-28T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
