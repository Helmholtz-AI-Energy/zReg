---
phase: 09-distance-transform-restructure
reviewed: 2026-04-28T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/zreg/distances/_protocol.py
  - src/zreg/distances/__init__.py
  - src/zreg/distances/sw_varients.py
  - src/zreg/pairwise_distance_matrix.py
  - src/zreg/transforms.py
findings:
  critical: 1
  warning: 7
  info: 5
  total: 13
status: fixes_applied
fix_commit: a95354f
---

# Phase 09: Code Review Report

**Reviewed:** 2026-04-28T00:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed 5 Python source files in the distance and transform modules. Found 1 critical issue (hardcoded `p=2` ignoring parameter), 7 warnings (logic errors, missing error handling, undefined behavior), and 5 info items (code quality improvements).

The critical issue in `minkowski_distance()` causes incorrect results when `p != 2`. Several warnings involve unchecked tensor dimensions, missing validation for edge cases, and potential division by zero. Code quality issues include unused imports, commented-out code, and debug print statements.

## Critical Issues

### CR-01: Hardcoded p=2 in minkowski_distance ignores parameter

**File:** `src/zreg/distances/general.py:81`
**Issue:** The `minkowski_distance()` function accepts a `p` parameter but always calls `torch.cdist(x, y, p=2)`, ignoring the caller's `p` value. This causes incorrect distance calculations when `p != 2` (e.g., Manhattan distance with `p=1` will compute Euclidean instead).

**Fix:**
```python
# Line 81: Replace
return torch.cdist(x, y, p=2)
# with
return torch.cdist(x, y, p=p)
```

This ensures the function respects the `p` parameter as documented.

## Warnings

### WR-01: Unchecked division by zero in proj_onto_unit_sphere

**File:** `src/zreg/distances/sw_varients.py:33`
**Issue:** `proj_onto_unit_sphere()` divides by the L2 norm without checking for zero-length vectors. If all input vectors are zero, this produces NaN results. While unlikely in normal usage, this could cause silent failures in edge cases.

**Fix:**
```python
def proj_onto_unit_sphere(vectors):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    norm = torch.sqrt(torch.sum(vectors**2, dim=2, keepdim=True))
    # Clamp to avoid division by zero
    norm = torch.clamp(norm, min=torch.finfo(vectors.dtype).eps)
    return vectors / norm
```

### WR-02: Missing validation for dimension constraints in BaseWD

**File:** `src/zreg/distances/sw_varients.py:167-169`
**Issue:** `BaseWD.forward()` checks `x.ndim < 3` to add batch dimension, but does not validate that `x.ndim` is at least 2. If a 1D tensor is passed, `unsqueeze(0)` creates a 2D tensor, which still violates the expected shape `(batch, n_points, dim)`. This leads to incorrect bmm operations downstream.

**Fix:**
```python
def forward(self, x, y, *args, **kwargs):
    _validate_tensors(x, y, names=["x", "y"])

    # Validate minimum dimensions
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(
            f"Expected x and y to have at least 2 dimensions (n_points, dim), "
            f"but got x.ndim={x.ndim}, y.ndim={y.ndim}"
        )

    xsqueeze = False
    if x.ndim < 3 or self.nobatchdim:
        x = x.unsqueeze(0)
        xsqueeze = True
    # ... rest of method
```

### WR-03: Potential race condition in AdaptiveSlicedWassersteinDistance file cleanup

**File:** `src/zreg/distances/sw_varients.py:280-290`
**Issue:** The `remove_history()` method uses `try/except FileNotFoundError` to handle race conditions, but the file is written without any locking mechanism. In parallel MPI execution (as used in `pairwise_distance_matrix.py`), multiple processes could attempt to write/delete the same file simultaneously, leading to corrupted data or incomplete writes.

**Fix:**
Use process-specific temporary files instead of a shared file:
```python
# In __init__ (line 228):
if self.projs_history is not None:
    # Append rank ID to filename for MPI safety
    import os
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    base, ext = os.path.splitext(self.projs_history)
    self.projs_history = f"{base}_rank{rank}{ext}"
```

### WR-04: Off-by-one error in window boundary calculation

**File:** `src/zreg/pairwise_distance_matrix.py:139-140`
**Issue:** The window maximum calculation uses `y_samples + 1`, but the loop iterates `for j in range(window_min, window_max)`. Since `max(y)` returns the highest key (likely equal to `len(y) - 1` for 0-indexed dicts), adding 1 assumes all indices exist. If `y` has gaps (e.g., keys [0, 1, 3, 5]), this causes KeyError when accessing `y[j]`.

**Fix:**
```python
# Lines 138-140: Replace
window_max = i + window
if window_max > y_samples + 1:
    window_max = y_samples + 1
# with
window_max = min(i + window + 1, max(y.keys()) + 1)
```

Same issue exists in `create_pairwise_distance_matrix_given_rigid_rot` at lines 338-340.

### WR-05: Inconsistent condition number validation in RigidTransformation.__mul__

**File:** `src/zreg/transforms.py:283-288`
**Issue:** The validation checks `cond.item() > 1e6` but only raises an error inside that branch. The error message duplicates the `det` check and references both `det` and `cond`, but the error is only raised for high condition numbers, not for invalid determinants. This makes the determinant check ineffective.

**Fix:**
```python
# Lines 277-288: Replace
det = torch.det(rot_composed)
if abs(det.item() - 1.0) > 1e-6:
    raise ValueError(
        f"RigidTransformation composition produced invalid rotation: "
        f"det={det.item():.6f}, expected 1.0"
    )
cond = torch.linalg.cond(rot_composed)
if cond.item() > 1e6:
    raise ValueError(
        f"RigidTransformation composition produced invalid rotation: "
        f"det={det.item():.6f}, cond={cond.item():.2e}"
    )
# with
det = torch.det(rot_composed)
cond = torch.linalg.cond(rot_composed)
if abs(det.item() - 1.0) > 1e-6 or cond.item() > 1e6:
    raise ValueError(
        f"RigidTransformation composition produced invalid rotation: "
        f"det={det.item():.6f}, cond={cond.item():.2e} (expected det=1.0, cond<1e6)"
    )
```

### WR-06: Missing error handling for bmm shape mismatch

**File:** `src/zreg/distances/sw_varients.py:58`
**Issue:** `compute_practical_moments_sw()` performs `x.bmm(projections.transpose(1, 2))` without checking that `x.size(2) == projections.size(2)` (i.e., that the number of dimensions matches). If `x` has a different dimension than expected, this raises a cryptic runtime error instead of a clear validation error.

**Fix:**
```python
# After line 56, add validation:
if x.size(2) != dim:
    raise ValueError(
        f"Expected x to have dimension {dim} to match projections, "
        f"but got x.size(2)={x.size(2)}"
    )
if y.size(2) != dim:
    raise ValueError(
        f"Expected y to have dimension {dim} to match projections, "
        f"but got y.size(2)={y.size(2)}"
    )
```

### WR-07: Undefined behavior when num_projs > dim in OrthogonalSlicedWassersteinDistance

**File:** `src/zreg/distances/sw_varients.py:356-366`
**Issue:** `torch.nn.init.orthogonal_()` requires the matrix to have `rows >= cols`, but the code creates a matrix with shape `(num_projs, dim)`. If `num_projs > dim`, this raises an error. The function does not validate this constraint upfront.

**Fix:**
```python
def _forward(self, x, y, *args, **kwargs):
    """
    x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
    """
    dim = x.shape[2]
    if self.num_projs > dim:
        raise ValueError(
            f"OrthogonalSlicedWassersteinDistance requires num_projs <= dim, "
            f"but got num_projs={self.num_projs}, dim={dim}"
        )

    projections = torch.zeros(
        (x.shape[0], self.num_projs, x.shape[2]),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
    )
    # ... rest of method
```

## Info

### IN-01: Unused import Variable in sw_varients.py

**File:** `src/zreg/distances/sw_varients.py:6`
**Issue:** `from torch.autograd import Variable` is imported but only used in `MaxSlicedWassersteinDistance._forward()` at line 307. In modern PyTorch (1.0+), `Variable` is deprecated and tensors have `requires_grad` by default. The usage at line 307-310 can be simplified.

**Fix:**
```python
# Line 307-310: Replace
projections = Variable(
    minibatch_rand_projections(batchsize=x.size(0), dim=dim, num_projections=1),
    requires_grad=True,
)
# with
projections = minibatch_rand_projections(batchsize=x.size(0), dim=dim, num_projections=1)
projections.requires_grad = True
```

Then remove the `Variable` import from line 6.

### IN-02: Unused imports in sw_varients.py

**File:** `src/zreg/distances/sw_varients.py:7-9`
**Issue:** `pathlib.Path`, `os`, and `tempfile` are imported but:
- `os` is only used in `remove_history()` for a safe file deletion (line 287), which could use `Path.unlink()` instead
- `tempfile` is used in lines 275-277 to create a throwaway file (never read back)
- `Path` is used in line 283 and 550 in pairwise_distance_matrix.py

The tempfile usage (lines 275-277) creates a file that is immediately deleted when the context exits, making the write operation pointless.

**Fix:**
Remove the tempfile block (lines 275-277) entirely since it has no observable effect. Update `remove_history()` to use `Path.unlink()`:
```python
def remove_history(self):
    if self.projs_history is None:
        return
    file = Path(self.projs_history)
    if file.exists():
        try:
            file.unlink()
        except FileNotFoundError:
            pass
```

Then remove `import os` if no longer needed.

### IN-03: Commented-out code in general.py

**File:** `src/zreg/distances/general.py:83-89`
**Issue:** Lines 83-89 contain commented-out alternative implementations of the Minkowski distance. This suggests incomplete refactoring or uncertainty about the implementation.

**Fix:**
Remove the commented code entirely if `torch.cdist()` is the canonical implementation. If these alternatives serve as documentation, move them to a docstring example or comment explaining why `torch.cdist()` is preferred.

### IN-04: Debug print statement in pairwise_distance_matrix.py

**File:** `src/zreg/pairwise_distance_matrix.py:228, 413`
**Issue:** `print("end of first iteration")` is left in production code. This bypasses the logging system and produces unstructured output.

**Fix:**
```python
# Lines 228, 413: Replace
print("end of first iteration")
# with
log.debug("Completed first iteration")
```

### IN-05: Typo in filename sw_varients.py

**File:** `src/zreg/distances/sw_varients.py`
**Issue:** The filename is misspelled as "sw_varients.py" (should be "sw_variants.py"). While this doesn't affect functionality, it violates naming conventions and could confuse contributors.

**Fix:**
Rename the file to `sw_variants.py` and update all imports:
```bash
git mv src/zreg/distances/sw_varients.py src/zreg/distances/sw_variants.py
```

Then update `__init__.py` line 19:
```python
from .sw_variants import (
```

---

_Reviewed: 2026-04-28T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
