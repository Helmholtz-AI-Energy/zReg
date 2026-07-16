---
phase: 07-cpd-deep-restructure
reviewed: 2026-04-20T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/zreg/cpd/__init__.py
  - src/zreg/cpd/_registration.py
  - src/zreg/cpd/_types.py
  - src/zreg/cpd/affine.py
  - src/zreg/cpd/base.py
  - src/zreg/cpd/kernels.py
  - src/zreg/cpd/nonrigid.py
  - src/zreg/cpd/rigid.py
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: fixes_applied
fix_commit: 1f7e74d
---

# Phase 07: Code Review Report

**Reviewed:** 2026-04-20T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

This review covers the CPD (Coherent Point Drift) package restructuring, including the abstract base class, rigid/affine/nonrigid implementations, type definitions, kernel utilities, and convenience functions. The review identified 2 critical issues (type mismatch and potential null pointer access), 5 warnings (missing validation, unused parameters, inconsistent API), and 3 info-level items (unused parameters, documentation improvements).

The codebase demonstrates good separation of concerns with the abstract base class pattern, proper use of type hints, and comprehensive docstrings. However, there are several bugs related to None-checking and type inconsistencies that could cause runtime failures.

## Critical Issues

### CR-01: Type Mismatch in log_freq Default Value

**File:** `src/zreg/cpd/base.py:58`
**Issue:** The `log_freq` parameter is declared as `int` but has a default value of `True` (boolean). This causes a type error and will fail type checking.

```python
# Current (line 58)
log_freq: int = True,
```

**Fix:**
```python
log_freq: int = 100,
```

The default should be an integer. Based on usage patterns in subclasses (rigid.py, affine.py, nonrigid.py), 100 is the standard default. If -1 is meant to disable logging, use that instead.

### CR-02: Potential Null Pointer Access Before None Check

**File:** `src/zreg/cpd/rigid.py:64`
**Issue:** Lines 64-69 access `source.dtype` and `source.device` when `source` could be `None` (as declared in the type hint on line 47). This will raise an `AttributeError` if a RigidCPD is instantiated without a source.

```python
# Current (lines 64-69)
fact = {"dtype": source.dtype, "device": source.device}
self._tf_type = tf.RigidTransformation
self._update_scale = update_scale
self.transform = None
self._tf_init_params = tf_init_params
self._tf_init_params.update(fact)
```

**Fix:**
```python
fact = {}
if source is not None:
    fact = {"dtype": source.dtype, "device": source.device}
self._tf_type = tf.RigidTransformation
self._update_scale = update_scale
self.transform = None
self._tf_init_params = tf_init_params
self._tf_init_params.update(fact)
```

Alternatively, if `source` is required, remove `| None` from the type hint and validate early:
```python
if source is None:
    raise ValueError("source is required for RigidCPD initialization")
fact = {"dtype": source.dtype, "device": source.device}
```

## Warnings

### WR-01: Missing None Check Before Normalization

**File:** `src/zreg/cpd/nonrigid.py:49-51`
**Issue:** If `source` is `None`, lines 50-51 will raise an `AttributeError` when trying to normalize. The type hint allows `None` but there's no guard.

```python
# Current (lines 49-51)
if self._source is not None:
    self._normalized_source, _ = normalize_point_cloud(self._source)
    self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)
```

**Fix:** The code is actually correct - it has a None check. However, `_normalized_source` and `_tf_obj` are not initialized when `source` is None, which could cause issues later. Add initialization:

```python
self._normalized_source = None
self._tf_obj = None
if self._source is not None:
    self._normalized_source, _ = normalize_point_cloud(self._source)
    self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)
```

Same issue in `ConstrainedNonRigidCPD.__init__` at lines 225-227.

### WR-02: Unused Parameter use_cuda

**File:** Multiple files (base.py:58, affine.py:38, nonrigid.py:41, rigid.py:51)
**Issue:** The `use_cuda` parameter is accepted in all CPD class constructors but is never used or stored. This suggests incomplete implementation or dead parameter.

**Fix:** Either implement CUDA support by using this parameter to move tensors to GPU:
```python
self._use_cuda = use_cuda
if use_cuda and torch.cuda.is_available():
    self._source = self._source.cuda()
```

Or remove the parameter from all signatures if CUDA device selection is handled externally (more likely given the codebase pattern).

### WR-03: Inconsistent Parameter Order in maximization_step

**File:** `src/zreg/cpd/base.py:223-224` vs `src/zreg/cpd/rigid.py:120-122`
**Issue:** In `base.py` the signature is `target_colors, source_colors` but in `rigid.py` the order is reversed to `source_colors, target_colors`. While both use keyword arguments, this inconsistency is error-prone.

**Fix:** Standardize to alphabetical order (`source_colors, target_colors`) throughout:
```python
# base.py line 223-224
target_colors: torch.Tensor | None = None,
source_colors: torch.Tensor | None = None,

# Should be:
source_colors: torch.Tensor | None = None,
target_colors: torch.Tensor | None = None,
```

Update the method call on line 254 accordingly.

### WR-04: Missing Validation in set_source Methods

**File:** `src/zreg/cpd/nonrigid.py:63-65`, `src/zreg/cpd/nonrigid.py:239-241`
**Issue:** The `set_source` methods in NonRigidCPD and ConstrainedNonRigidCPD override the base class but don't call `_validate_tensors` like the base class does (base.py:83-85).

**Fix:**
```python
def set_source(self, source: torch.Tensor, source_colors: torch.Tensor | None = None) -> None:
    """Set source and initialize the non-rigid transformation object."""
    from ..validation import _validate_tensors
    _validate_tensors(source, names=["source"])
    if source_colors is not None:
        _validate_tensors(source, source_colors, names=["source", "source_colors"])

    self._source = source
    self._normalized_source, _ = normalize_point_cloud(self._source)
    self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)
```

### WR-05: Empty transformation Initialization

**File:** `src/zreg/cpd/base.py:68`
**Issue:** `self.transformation` is set to `None` in the base class but there's no type hint for this attribute. This makes it unclear what type is expected and could cause type checker warnings.

**Fix:** Add type annotation in the class:
```python
class CoherentPointDrift(ABC):
    """Abstract base class for Coherent Point Drift algorithm."""

    _N_DIM = 3
    _N_COLOR = 3

    transformation: tf.Transformation | None  # Add this line

    def __init__(self, ...):
```

Or at least document in the docstring under "Attributes" section.

## Info

### IN-01: Unused Parameter sigma2_p in Affine M-step

**File:** `src/zreg/cpd/affine.py:84`
**Issue:** The `sigma2_p` parameter (previous variance) is declared but never used in the affine maximization step, unlike rigid and nonrigid variants which do use it.

**Fix:** If the previous variance is not needed for affine transformation, remove it from the signature or add a comment explaining why:
```python
sigma2_p: float | None = None,  # Unused: affine doesn't require prior sigma2
```

Or if it should be used for numerical stability, investigate if the algorithm needs it.

### IN-02: Hardcoded Rotation Matrix in Rigid Initialization

**File:** `src/zreg/cpd/rigid.py:88-94`
**Issue:** The initial rotation matrix is hardcoded with a comment "found empirically for Shah->Kobiski data". This is dataset-specific and will not generalize to other data.

```python
# Lines 88-94
rot = torch.tensor(
    [[-0.0, -1.0, 0.0], [1.0, -0.0, 0.5], [0.0, 0.5, 1.0]],
    dtype=self._source.dtype,
    device=self._source.device,
)
self.transformation.rot = rot
```

**Fix:** Use identity matrix as default or make this configurable:
```python
# Default to identity rotation
rot = torch.eye(3, dtype=self._source.dtype, device=self._source.device)
# Allow override via tf_init_params if provided
if 'initial_rotation' in self._tf_init_params:
    rot = self._tf_init_params['initial_rotation']
self.transformation.rot = rot
```

### IN-03: Convergence Check Could Fail on First Iterations

**File:** `src/zreg/cpd/base.py:387-389`
**Issue:** The convergence check uses `.diff()` which requires at least 2 elements. While `running_avg` is initialized with 4 elements, the check on iteration 0 will compare 4 initialized values that may not represent actual convergence state.

```python
# Line 387-389
running_avg[i % running_avg.shape[0]] = res.q

if running_avg.abs().diff().mean().abs() < tol:
```

**Fix:** Only check convergence after filling the running average:
```python
running_avg[i % running_avg.shape[0]] = res.q

# Only check after we have at least one full cycle
if i >= running_avg.shape[0] and running_avg.abs().diff().mean().abs() < tol:
    if self.log_freq > 0:
        log.info(
            f"Hit tolerance in iteration {i} (criteria: {res.q:.4f}), exiting."
        )
    break
```

---

_Reviewed: 2026-04-20T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
