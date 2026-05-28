---
phase: 18-metricsengine-result-types
reviewed: 2026-05-28T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - eval/config.py
  - eval/metrics.py
  - eval/types.py
  - tests/test_metrics.py
findings:
  critical: 3
  warning: 4
  info: 1
  total: 8
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-05-28
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Four source files reviewed: `eval/config.py` (EvalConfig YAML loader), `eval/types.py` (six pydantic result models), `eval/metrics.py` (MetricsEngine), and `tests/test_metrics.py` (test suite). The result-model layer in `eval/types.py` is structurally sound. The critical issues cluster in `eval/metrics.py`: the `normalize()` contract of returning values in `[0,1]` is broken for negative inputs, `compute_score()` has a logic error when weight keys are absent from `normalized`, and `sanity_check()` misfires on empty tensors due to Python/PyTorch vacuous-truth semantics. Three findings are blockers because they produce incorrect outputs silently.

---

## Critical Issues

### CR-01: `normalize()` crashes or returns out-of-range values for negative metric inputs

**File:** `eval/metrics.py:148-154`
**Issue:** The formula `1.0 / (1.0 + x)` is only in `(0, 1]` when `x >= 0`. `StageMetrics` fields are typed `float` with no non-negativity constraint, so negative values are accepted by pydantic without error. A negative chamfer/hausdorff/path_smoothness/temporal_stability value (possible if upstream metric code returns signed residuals or if a future caller passes synthetic data) produces values outside `[0,1]`, breaks `compute_score`'s claimed `[0,1]` scalar range, and causes a `ZeroDivisionError` at exactly `x = -1.0`.

Verified outputs:
- `x = -0.5` → `normalize = 2.0` (> 1.0)
- `x = -1.0` → `ZeroDivisionError`
- `x = -2.0` → `normalize = -1.0` (< 0.0)

**Fix:** Either add `ge=0` field validators to the four lower-is-better fields in `StageMetrics`, or guard in `normalize()`:
```python
# Option A — guard in StageMetrics (preferred: catches bad data at source)
# In eval/types.py, add to the four lower-is-better fields:
from pydantic import field_validator
@field_validator("chamfer_distance", "hausdorff_distance",
                 "path_smoothness", "temporal_stability")
@classmethod
def must_be_non_negative(cls, v: float) -> float:
    if v < 0:
        raise ValueError(f"metric must be >= 0, got {v!r}")
    return v

# Option B — defensive guard in normalize() only
"chamfer": 1.0 / (1.0 + max(0.0, metrics.chamfer_distance)),
```

---

### CR-02: `compute_score()` produces a score outside `[0,1]` when weight keys are absent from `normalized`

**File:** `eval/metrics.py:183-188`
**Issue:** `total = sum(w.values())` always sums ALL weights in `config.metric_weights`, including weights for keys that are NOT present in `metrics.normalized`. The subsequent sum only accumulates contributions from keys that ARE present. This means when any key is skipped, the denominator is larger than the sum of the numerator weights, and the result is silently lower than it would be if the weights were properly re-normalised over only the present keys. In the worst case (all keys absent except one), the score can collapse to near zero even when the one present metric is perfect. This violates the documented `[0,1]` invariant.

Example with default weights: if `"chamfer"` (weight 0.35) is absent from `normalized` but all five remaining metrics are at their maximum (1.0), the score is `0.65` not `1.0`. The docstring calls this "silently skipped" robustness, but the contract of auto-rescaling is broken.

**Fix:** Compute `total` over only the keys that are actually present in `norm`:
```python
def compute_score(self, metrics: StageMetrics) -> float:
    w = self.config.metric_weights
    norm = metrics.normalized
    # Only include keys present in both dicts
    active = {k: w[k] for k in w if k in norm}
    total = sum(active.values())
    if total == 0:
        raise ValueError("metric_weights sum to zero")
    return sum(norm[k] * active[k] / total for k in active)
```

---

### CR-03: `sanity_check()` emits a false "all-sentinel" warning for empty label tensors

**File:** `eval/metrics.py:300-304`
**Issue:** In Python/PyTorch, `.all()` on an empty tensor returns `True` (vacuous truth). If `transferred_labels` contains a frame mapped to an empty tensor (`torch.tensor([], dtype=torch.long)`), the check `(tensor == -1).all().item()` returns `True`, triggering the "all-sentinel labels" warning even though the frame has zero labels. This is a false positive that masks the real problem (empty label tensor) and simultaneously suppresses the "all-same labels" flag for the same frame by setting `same_flagged = True`.

Verified: `torch.tensor([], dtype=torch.long).unique().numel()` returns `0` (which is `<= 1`), so an empty tensor would also independently trigger the all-same path.

**Fix:** Add an explicit empty-tensor guard before the sentinel check:
```python
for tensor in label.transferred_labels.values():
    if tensor.numel() == 0:
        flags.append("empty label tensor: frame has 0 labels")
        continue
    if not sentinel_flagged and (tensor == -1).all().item():
        ...
```

---

## Warnings

### WR-01: `EvalConfig.from_yaml` silently discards all but the first validation error

**File:** `eval/config.py:154-156`
**Issue:** When a YAML file has multiple invalid fields, `e.errors()[0]` extracts only the first `pydantic.ValidationError` entry. Users fixing a config file must re-run once per invalid field, seeing only one error per invocation. The second and subsequent errors are silently discarded.

**Fix:** Report all errors, not just the first:
```python
except ValidationError as e:
    msgs = []
    for err in e.errors():
        field = ".".join(str(x) for x in err["loc"])
        msgs.append(f"field '{field}': {err['msg']}")
    raise EvalConfigError(
        "EvalConfig: " + "; ".join(msgs)
    ) from e
```

---

### WR-02: `aggregate()` uses population std (`pstdev`) without documenting the choice

**File:** `eval/metrics.py:235`
**Issue:** `statistics.pstdev` computes the **population** standard deviation (divides by N). When `results` represents a sample of trials drawn from a broader distribution (the normal use-case for a hyper-parameter search), the statistically appropriate measure is the **sample** standard deviation (divides by N-1, `statistics.stdev`). Population std is systematically biased low for small N. Callers using `agg["chamfer_distance"]["std"]` as a confidence interval or for model-selection comparisons will get a biased result. The docstring acknowledges the single-element guard but does not explain why `pstdev` was chosen over `stdev`.

**Fix:** Either switch to `statistics.stdev` (sample std) and return `0.0` for N=1 as the guard already does, or prominently document in the docstring that `pstdev` is intentional and explain its implications:
```python
# If sample std is intended:
"std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
```

---

### WR-03: `test_unknown_normalized_key_is_skipped` does not assert score is in `[0,1]`

**File:** `tests/test_metrics.py:269-286`
**Issue:** The test removes `"chamfer"` from `normalized` and calls `compute_score`. It only asserts `isinstance(score, float)` — it does NOT assert the score is in `[0,1]`. With the current (buggy) implementation (CR-02), the score is `0.65` when all remaining normalized values are `1.0` and chamfer weight is `0.35`. The test passes without detecting the contract violation. This test was designed to catch a `KeyError` (robustness to missing keys) but fails to guard the more important invariant (score remains in `[0,1]`).

**Fix:**
```python
score = eng.compute_score(sm_partial)
assert isinstance(score, float)
assert 0.0 <= score <= 1.0  # add this assertion
```

---

### WR-04: `normalize()` is not tested for negative inputs, leaving CR-01 undetected

**File:** `tests/test_metrics.py:109-187`
**Issue:** The entire `TestNormalize` class tests only non-negative inputs. There is no test for negative `chamfer_distance`, `hausdorff_distance`, `path_smoothness`, or `temporal_stability`. CR-01 (ZeroDivisionError at x=-1.0, values > 1.0 for -1 < x < 0) is therefore undetected by the test suite. A regression test for negative inputs should either assert a `ValueError` (if `StageMetrics` adds the validator from the CR-01 fix) or document the expected out-of-range behavior.

**Fix:** Add a negative-input test to `TestNormalize`:
```python
def test_negative_lower_is_better_raises(self):
    """StageMetrics must reject negative lower-is-better values (CR-01 guard)."""
    with pytest.raises(pydantic.ValidationError):
        StageMetrics(
            chamfer_distance=-0.5,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
```

---

## Info

### IN-01: `compute_stage_metrics` has an inline TODO comment in production code

**File:** `eval/metrics.py:382`
**Issue:** `# TODO: confirm signature with Phase 21 caller` is a leftover planning note embedded in a public method's docstring Notes section. TODO markers in shipped source code indicate incomplete design decisions and create noise in `grep`-based searches.

**Fix:** Remove the comment once Phase 21 wires in the caller, or track it as a ticket/planning item rather than an inline comment. If the signature is genuinely preliminary, document it as a version note rather than a TODO:
```python
# Notes: signature is preliminary pending Phase 21 EvaluationRunner integration.
```

---

_Reviewed: 2026-05-28_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
