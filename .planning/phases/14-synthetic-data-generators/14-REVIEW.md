---
phase: 14-synthetic-data-generators
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - eval/generators/__init__.py
  - eval/generators/corruption.py
  - eval/generators/generators.py
  - eval/generators/labels.py
  - eval/generators/transforms.py
  - tests/conftest.py
  - tests/test_generators.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-05-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

This review covers the synthetic data generator package (`eval/generators/`) and its companion test suite. The implementation is structurally sound: immutability contracts are upheld (deep-copy before modify), seed reproducibility is consistent, `ValueError` guards are present for clearly invalid inputs, and dtype/shape contracts for `zreg.metrics.compute_f1` compatibility are met.

Three warnings were found. The most substantive is a latent sign-flip defect in the w-coordinate clamping logic in `transforms.py` (inherited from `homogeneous.py`). The second is an undocumented and unvalidated `scale` parameter in `add_outliers` that accepts negative values with no error. The third is that `deepcopy` silently consumes torch RNG state between `manual_seed` and the actual `randn` calls, which is not documented and makes the effective random stream non-obvious to callers.

Two informational findings cover an untested code path (`fps-idx` sentinel extension) and a probabilistically-passing nondeterminism test.

---

## Warnings

### WR-01: w-coordinate clamp uses `min=eps` only, causing sign flip on negative w

**File:** `eval/generators/transforms.py:106`
**Issue:** The defensive w-divide clamps `w` with `torch.clamp(w, min=eps)`. This prevents division by zero for small positive `w` but silently flips the sign of any negative `w` value (clamped from, say, `-1.0` up to `+eps`), which then divides the xyz coordinates by a near-zero positive number instead of a negative number. The result is catastrophically wrong coordinates rather than a clean error.

For `apply_rigid` and `apply_affine` as currently implemented, `M[3,3] = 1.0` always, so `w` is always exactly `1.0` and the clamp never fires. The defect is therefore latent and unreachable through the public API today. However, `_apply_matrix` is a non-private helper (single underscore) that any future caller could pass an arbitrary matrix to, and the inherited bug from `src/zreg/transforms/homogeneous.py` (which has the same flaw in production code) represents an uncontained design error.

The correct guard should preserve the sign of `w`:

**Fix:**
```python
# Current (wrong for negative w):
w = torch.clamp(w, min=torch.finfo(pos.dtype).eps)

# Correct: clamp magnitude, preserve sign
eps = torch.finfo(pos.dtype).eps
w = torch.where(w.abs() < eps, torch.full_like(w, eps) * w.sign().clamp(min=1), w)
# Or more simply for the affine/rigid use-case where w is always 1:
# assert (w > 0).all(), "Unexpected negative homogeneous coordinate"
```

---

### WR-02: `add_outliers` accepts negative `scale` without raising `ValueError`

**File:** `eval/generators/corruption.py:60-103`
**Issue:** The docstring describes `scale` as the "standard deviation of the outlier point distribution" and implicitly documents it as a non-negative value. However, unlike `sigma` in `add_gaussian_noise` (which raises `ValueError` for negative values at line 49-50), `add_outliers` performs no validation on `scale`. Passing `scale=-5.0` silently succeeds and produces `torch.randn(...) * -5.0`, which is mathematically equivalent to `scale=5.0` but semantically contradicts the documented API. This API asymmetry is a latent correctness hazard when `scale` is derived from a user-controlled parameter.

**Fix:**
```python
# Add after line 103 (the n_outliers check):
if scale < 0:
    raise ValueError(f"scale must be >= 0, got {scale}")
```

---

### WR-03: `deepcopy` silently consumes torch RNG between `manual_seed` and `randn` calls

**File:** `eval/generators/corruption.py:53-56`, `eval/generators/labels.py:71-75`
**Issue:** In `add_gaussian_noise`, `generate_labels`, and `add_outliers`, `torch.manual_seed(seed)` is called, and then `copy.deepcopy(trajectory)` is immediately called before any `torch.randn*` call. Copying `torch.Tensor` objects via `deepcopy` is implemented in PyTorch in a way that advances the global RNG state. The practical consequence is that the noise/label values produced are NOT the values one would obtain from a fresh `torch.manual_seed(seed)` followed directly by `torch.randn(...)`. Reproducibility between two calls with the same trajectory and seed is preserved (same deepcopy path = same RNG offset), but:

1. The effective noise values differ from the naive expectation documented implicitly by the seed contract.
2. If the trajectory's tensor shapes change between two "same-seed" calls (e.g., different `n_points`), the deepcopy RNG offset changes and the per-frame noise values will differ for frames after frame 0.

The fix is to perform the deepcopy before calling `manual_seed`, or to capture the RNG state after deepcopy and restore it:

**Fix:**
```python
# Pattern: deepcopy first, seed second
def add_gaussian_noise(trajectory, sigma=0.01, seed=42):
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    result = copy.deepcopy(trajectory)      # deepcopy before seeding
    if seed is not None:
        torch.manual_seed(seed)             # seed after deepcopy
    for pc in result.values():
        noise = torch.randn_like(pc["pos"]) * sigma
        pc["pos"] = pc["pos"] + noise
    return result
```

Apply the same reordering in `add_outliers` (line 104-106) and `generate_labels` (line 71-72).

---

## Info

### IN-01: `fps-idx` sentinel extension path in `add_outliers` has no test coverage

**File:** `eval/generators/corruption.py:140-148` / `tests/test_generators.py`
**Issue:** The `add_outliers` function extends `pc["fps-idx"]` with sentinel `-1` values when that field is not `None`. The `trajectory_data` fixture in `conftest.py` does not set `fps-idx` (it defaults to `None`), so this branch is never exercised by the test suite. The analogous `id` and `color` branches are tested. A missing test for `fps-idx` leaves the only non-trivial branch in `add_outliers` completely uncovered.

**Fix:** Add a test case that creates a trajectory with a non-`None` `fps-idx` tensor and asserts that `add_outliers` extends it with the correct number of `-1` sentinels:
```python
def test_add_outliers_extends_fps_idx_with_sentinel(self):
    traj = {0: zRegPointCloud(pos=torch.randn(10, 3))}
    traj[0]["fps-idx"] = torch.arange(10, dtype=torch.long)
    result = add_outliers(traj, n_outliers=4, seed=42)
    assert result[0]["fps-idx"].shape[0] == 14
    assert torch.all(result[0]["fps-idx"][-4:] == -1)
```

---

### IN-02: `test_seed_none_is_nondeterministic` is a probabilistic (non-deterministic) test

**File:** `tests/test_generators.py:74-79`
**Issue:** The test asserts that two consecutive `seed=None` calls produce different position tensors. This is correct in practice but is technically probabilistic — if the global RNG is in a state where two draws happen to produce identical 50×2×3 float32 values, the test fails. The comment acknowledges this ("It would be astronomically unlikely") but the test itself has no seed fixture, making the probability nonzero in theory. In a CI environment where prior test order may set RNG state deterministically, this is a long-term fragility concern.

**Fix:** Replace the probabilistic assertion with a deterministic one: verify that calling with the same `seed=None` trajectory (after a fixed manual seed) does not produce the same output as calling with a different fixed seed:
```python
def test_seed_none_does_not_fix_output(self):
    """seed=None continues from current RNG; different initial RNG state -> different output."""
    torch.manual_seed(0)
    traj_a = generate_trajectory(50, 2, seed=None)
    torch.manual_seed(1)
    traj_b = generate_trajectory(50, 2, seed=None)
    assert not all(torch.equal(traj_a[i]["pos"], traj_b[i]["pos"]) for i in traj_a)
```

---

_Reviewed: 2026-05-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
