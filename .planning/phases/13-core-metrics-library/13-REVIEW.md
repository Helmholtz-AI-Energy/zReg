---
phase: 13-core-metrics-library
reviewed: 2026-05-15T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/zreg/metrics/__init__.py
  - src/zreg/metrics/alignment.py
  - src/zreg/metrics/label_transfer.py
  - tests/test_alignment_metrics.py
  - tests/test_eval_metrics.py
  - tests/test_label_transfer_metrics.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-05-15
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

The metrics library is well-structured and covers the stated contract: bidirectional Chamfer distance, percentile Hausdorff, DTW path smoothness, k-NN label consistency, temporal stability, and weighted-F1 with sentinel masking. Validation, GPU safety, and error messaging are consistently handled across the two production modules. The test suites are thorough for happy paths and documented error cases.

Three groups of defects survive closer inspection:

1. **`path_smoothness` is broken for realistic DTW paths** — any path containing a "vertical" or "horizontal" step (where `Δi = 0` or `Δj = 0`) causes the slope to spike to ~10^6, inflating the smoothness variance by 10^12 or more. This makes the metric unreliable for real DTW output.
2. **Type and error-type contract violations** — `hausdorff` and `chamfer` raise `IndexError`/`RuntimeError` in cases the docstring promises raise `ValueError`; `temporal_stability` silently accepts an invalid single-element list and returns `0.0` without type-checking.
3. **Test comment inaccuracies** — two test comments contain wrong intermediate values that could mislead future maintainers.

---

## Critical Issues

### CR-01: `path_smoothness` produces absurd values for valid DTW paths containing axis-aligned steps

**File:** `src/zreg/metrics/alignment.py:161-163`

**Issue:** DTW paths legitimately contain "vertical" steps (`Δi = 0, Δj > 0`) and "horizontal" steps (`Δi > 0, Δj = 0`). The slope formula `dt2 / (dt1 + 1e-6)` uses a 1e-6 epsilon to avoid division by zero, but this maps `Δi = 0` to a slope of `Δj * 1_000_000` rather than signalling a discontinuity or treating the move differently. Feeding a 4-point path with one vertical step produces a smoothness value on the order of 10^12 instead of something near 0:

```python
path = [(0, 0), (1, 1), (1, 2), (2, 3)]   # vertical step at index 2
path_smoothness(path)   # => 999_998_029_824.0  (not ≈ 0)
```

A constant-speed diagonal path broken by a single valid DTW axis-aligned step goes from a smoothness near 0 to a smoothness near 10^12 — a discontinuous, order-of-magnitude-wrong result. No test exercises this case.

**Fix:** Replace the floating-point slope with a vector-based curvature that is well-defined for axis-aligned steps, for example by computing the cross-product magnitude of consecutive step vectors:

```python
def path_smoothness(path: list[tuple[int, int]]) -> float:
    if len(path) < 3:
        return 0.0

    # Step vectors
    steps = [
        (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        for k in range(1, len(path))
    ]
    # Signed curvature: cross-product of consecutive 2-D step vectors
    # (di1, dj1) x (di2, dj2) = di1*dj2 - di2*dj1
    cross = [
        steps[i][0] * steps[i + 1][1] - steps[i + 1][0] * steps[i][1]
        for i in range(len(steps) - 1)
    ]
    t = torch.tensor(cross, dtype=torch.float64)
    return float(t.var(unbiased=False).item())
```

Alternatively, if slope-based smoothness is intentional, add an explicit guard:

```python
if dt1 == 0:
    slopes.append(float("inf"))
```

and return `float("inf")` or raise `ValueError` for paths that contain vertical steps, so callers know the metric is undefined for such paths.

---

## Warnings

### WR-01: `chamfer` and `hausdorff` raise `IndexError` (not `ValueError`) for empty point clouds

**File:** `src/zreg/metrics/alignment.py:68-70` and `src/zreg/metrics/alignment.py:126-128`

**Issue:** Both functions accept `shape (N, 3)` with `N = 0` — the shape guard checks `ndim == 2` and `shape[1] == 3`, which passes for a `(0, 3)` tensor. The subsequent `dist.min(dim=1)` call then raises an `IndexError: min(): Expected reduction dim 0 to have non-zero size`. The docstring promises `ValueError` for invalid input.

```python
chamfer(torch.zeros(0, 3), torch.ones(5, 3))   # => IndexError
hausdorff(torch.zeros(0, 3), torch.ones(5, 3)) # => IndexError
```

**Fix:** Add an explicit empty-cloud guard after the shape checks:

```python
if source.shape[0] == 0:
    raise ValueError("source must be non-empty (N > 0)")
if target.shape[0] == 0:
    raise ValueError("target must be non-empty (M > 0)")
```

---

### WR-02: `hausdorff` raises `RuntimeError` (not `ValueError`) for out-of-range `percentile`

**File:** `src/zreg/metrics/alignment.py:129-130`

**Issue:** The docstring states that the `percentile` parameter is in `[0, 100]`, but there is no explicit guard. Passing `percentile=110.0` reaches `torch.quantile` which raises a `RuntimeError: quantile() q values must be in the range [0, 1]`, not the `ValueError` callers would expect based on the function's documented exception contract.

**Fix:** Add a guard before computation:

```python
if not (0.0 <= percentile <= 100.0):
    raise ValueError(
        f"percentile must be in [0, 100], got {percentile}"
    )
```

---

### WR-03: `temporal_stability` silently returns `tensor(0.0)` for a single-element list containing an invalid type

**File:** `src/zreg/metrics/alignment.py:340-341`

**Issue:** The early-return guard fires for `len(transforms) < 2` before any type checking occurs. A list with one invalid element (e.g., `[42]` or `["bad"]`) returns `tensor(0.0)` silently instead of raising `TypeError`, violating the documented exception contract. The test `test_unsupported_type_raises_type_error` only exercises `[1, 2]` (length 2) and therefore misses this case.

```python
temporal_stability([42])        # => tensor(0.0) — no error raised
temporal_stability([42, 43])    # => TypeError: Unsupported...
```

**Fix:** Validate all elements regardless of list length, either before the early return or as a dedicated validation pass:

```python
for tf in transforms:
    if not isinstance(tf, (RigidTransformation, AffineTransformation)):
        raise TypeError(
            f"Unsupported transformation type: {type(tf).__name__}"
        )
if len(transforms) < 2:
    return torch.tensor(0.0)
```

---

### WR-04: `temporal_stability` returns a CPU `tensor(0.0)` regardless of input device for lists of length 0 or 1

**File:** `src/zreg/metrics/alignment.py:340-341`

**Issue:** `torch.tensor(0.0)` always creates a CPU tensor. When a caller passes a list containing a single CUDA transform, or an empty list that is meant to be used in a CUDA pipeline, the returned `tensor(0.0)` is on CPU while the transforms are on CUDA. This is inconsistent with the 2+ element path which returns a tensor on whichever device the matrices are on.

**Fix:** Infer the device from the first element when available:

```python
if len(transforms) < 2:
    if transforms:
        ref = _to_matrix(transforms[0])
        return torch.tensor(0.0, dtype=ref.dtype, device=ref.device)
    return torch.tensor(0.0)
```

---

## Info

### IN-01: Test comment in `test_alignment_metrics.py` contains wrong slope and delta values

**File:** `tests/test_alignment_metrics.py:162-163`

**Issue:** The comment above `test_varying_slopes_nonzero` reads:

```
# 4 points → slopes: 1.0, 2.0, 0.5 → deltas: [1.0, -1.5] → variance > 0
path = [(0, 0), (1, 1), (2, 3), (3, 3)]
```

The third slope is `(3-3) / (3-2 + 1e-6) ≈ 0.0`, not `0.5`. The deltas are therefore `[1.0, -2.0]`, not `[1.0, -1.5]`. The assertion `result > 0.0` still passes, but the comment is incorrect and could mislead future maintainers auditing the test's intent.

**Fix:** Update the comment to match the actual computation:

```python
# 4 points → slopes: 1.0, 2.0, 0.0 → deltas: [1.0, -2.0] → variance > 0
```

---

### IN-02: `path_smoothness` accumulates slopes with `float32` precision via `torch.tensor(deltas)`

**File:** `src/zreg/metrics/alignment.py:166`

**Issue:** `deltas` is a Python list of `float` values (64-bit), but `torch.tensor(deltas)` defaults to `torch.float32`. This silently downcasts the inputs before computing variance. For typical DTW paths the error is negligible, but the behaviour is surprising because pure Python `float` arithmetic was used to build the list.

**Fix:** Pass an explicit dtype:

```python
return float(torch.tensor(deltas, dtype=torch.float64).var(unbiased=False).item())
```

---

### IN-03: `path_smoothness` docstring claims path needs "at least 2 slopes for a slope-change to exist" but the threshold should be stated as "at least 4 points for non-trivially-zero variance"

**File:** `src/zreg/metrics/alignment.py:148-149`

**Issue:** The docstring states: _"Returns 0.0 if fewer than 3 points are provided (need at least 2 slopes for a slope-change to exist)."_ However, with exactly 3 points there are 2 slopes and 1 delta, and `torch.Tensor.var(unbiased=False)` of a single-element tensor is always `0.0` — so the function also returns 0.0 for 3-point paths. The effective threshold for a potentially non-zero result is 4 points (2 deltas). The comment in `test_alignment_metrics.py:156-158` for `test_two_points_returns_zero` also states the wrong reason ("need >=2 slopes for variance" when the guard fires at `len < 3`, not at the delta stage).

**Fix:** Update the docstring:

```
Returns 0.0 if fewer than 4 points are provided; 3-point paths produce exactly
one slope-change delta and therefore always yield zero variance.
```

---

_Reviewed: 2026-05-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
