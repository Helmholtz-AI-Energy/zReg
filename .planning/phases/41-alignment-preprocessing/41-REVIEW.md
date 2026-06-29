---
phase: 41-alignment-preprocessing
reviewed: 2026-06-29T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - eval/config.py
  - eval/stages/alignment.py
  - eval/types.py
  - src/zreg/__init__.py
  - src/zreg/preprocessing.py
  - tests/test_alignment_preprocessing_config.py
  - tests/test_alignment_stage.py
  - tests/test_eval_config.py
  - tests/test_preprocessing.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 41: Code Review Report

**Reviewed:** 2026-06-29T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed all nine source files introduced or modified in Phase 41 (alignment
preprocessing: PCA principal-axes rotation and velocity-landmark detection).
The core logic in `src/zreg/preprocessing.py` and the pydantic models in
`eval/config.py` are largely sound. Two blockers were found:

1. Three tests in `test_eval_config.py` use a stale match string that no
   longer matches the actual validator error message (the message was extended
   to include `'swd'` when Phase 39/40 added that method, but the tests were
   not updated). All three will fail at runtime.

2. The `swd_variant` parameter from `params` is never forwarded from `run()`
   to `_build_aligned_cloud`. When `alignment_method='swd'`, the user-
   configured variant is silently discarded and the hardcoded default `"aswd"`
   is used instead.

---

## Critical Issues

### CR-01: Stale test match strings — three tests will fail

**File:** `tests/test_eval_config.py:55`, `:63`, `:70`

**Issue:** `TestEvalConfigAlignmentMethodValidation` has three tests that call
`pytest.raises(ValueError, match="alignment_method must be 'cpd' or 'icp'")`.
The actual validator message (in `eval/config.py:275`) is:

```
alignment_method must be 'cpd', 'icp', or 'swd'; got 'invalid_method'
```

`pytest.raises(match=...)` uses `re.search(pattern, str(exception))`. The
pattern `"alignment_method must be 'cpd' or 'icp'"` does **not** appear as a
substring in the actual message — confirmed with `re.search` returning
`False`. All three tests (`test_alignment_method_invalid_value_raises`,
`test_alignment_method_case_sensitive`, `test_alignment_method_empty_string_raises`)
will raise `Failed: DID NOT RAISE <class 'ValueError'>` or a match-failure
depending on pytest version.

The message was updated when `'swd'` was added to `alignment_method` but the
corresponding test expectations were not.

**Fix:**
```python
# Replace in all three failing tests:
with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
```

---

### CR-02: `swd_variant` from params is never forwarded to `_build_aligned_cloud`

**File:** `eval/stages/alignment.py:269-278`, `479`

**Issue:** `run()` calls `_build_aligned_cloud` with only these keyword
arguments:

```python
aligned_cloud = self._build_aligned_cloud(
    source=working_source,
    target=target,
    source_sub=source_sub,
    target_sub=target_sub,
    warp_path=result.warping_path,
    cpd_penalty=params["cpd_penalty"],
    alignment_method=params["alignment_method"],
    stored_transforms=result.stored_transforms,
    # swd_variant is NOT passed
)
```

Inside `_build_aligned_cloud`, the SWD branch reads the variant from `**kwargs`:

```python
elif alignment_method == "swd":
    swd_variant = kwargs.get("swd_variant", "aswd")  # Always "aswd" — kwargs is empty
    swd_num_iterations = kwargs.get("swd_num_iterations", 50)
```

Because `run()` never passes `swd_variant` (or `swd_num_iterations`) as a
keyword argument, `kwargs` is always `{}` in this branch. When
`alignment_method='swd'`, any value the user puts in `params["swd_variant"]`
(validated by `validate_params` at line 179-184) is silently discarded and
`"aswd"` is used unconditionally. The test suite masks this because the test
happens to configure `swd_variant="aswd"` (the same as the hidden default).

**Fix:**
```python
# In run(), add the swd kwargs when alignment_method is 'swd':
aligned_cloud = self._build_aligned_cloud(
    source=working_source,
    target=target,
    source_sub=source_sub,
    target_sub=target_sub,
    warp_path=result.warping_path,
    cpd_penalty=params["cpd_penalty"],
    alignment_method=params["alignment_method"],
    stored_transforms=result.stored_transforms,
    swd_variant=params.get("swd_variant", self.config.swd_variant),
    swd_num_iterations=params.get("swd_num_iterations", 50),
)
```

---

## Warnings

### WR-01: `validate_params` silently mutates the caller's `params` dict

**File:** `eval/stages/alignment.py:147-148`

**Issue:** When `alignment_method` is absent from the caller-supplied `params`,
`validate_params` writes directly into it:

```python
if "alignment_method" not in params:
    params["alignment_method"] = self.config.alignment_method
```

This is a side effect on an object owned by the caller. Callers that hold a
reference to `params` after calling `run()` will observe the dict unexpectedly
mutated. Neither the `validate_params` docstring nor the `run()` docstring
documents this mutation. The docstring notes that `params_used` is a shallow
copy to prevent Pitfall 7, but the mutation happens before the copy is taken,
meaning the caller's dict is altered regardless.

**Fix:** Operate on a local copy inside `validate_params` (and adjust `run()`
to pass the copy):

```python
def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)  # never mutate the caller's dict
    if "alignment_method" not in params:
        params["alignment_method"] = self.config.alignment_method
    # ... rest of validation ...
    return params  # return the (possibly augmented) copy
```

Then in `run()`:
```python
params = self.validate_params(params)  # validated copy, caller's dict untouched
```

---

### WR-02: `detect_velocity_landmarks` crashes when consecutive frames have different point counts

**File:** `src/zreg/preprocessing.py:101-103`

**Issue:**

```python
diffs = trajectory[k]["pos"] - trajectory[prev_k]["pos"]
```

This subtraction assumes `trajectory[k]["pos"]` and `trajectory[prev_k]["pos"]`
have identical shapes (`[N, 3]`). Real datasets (tracklets, CSVs) can produce
trajectories where different frames have different numbers of tracked points.
When that happens, PyTorch will raise a `RuntimeError: The size of tensor a
(M) must match the size of tensor b (N) at non-singleton dimension 0` with no
indication of which frame pair caused the failure.

The docstring says shape is `[N, 3]` but does not state that N must be
constant across frames, leaving callers unaware of the constraint.

**Fix:** Add a guard before the subtraction:

```python
prev_pos = trajectory[prev_k]["pos"]
curr_pos = trajectory[k]["pos"]
if curr_pos.shape[0] != prev_pos.shape[0]:
    raise ValueError(
        f"detect_velocity_landmarks: frame {k} has {curr_pos.shape[0]} points "
        f"but frame {prev_k} has {prev_pos.shape[0]}; point counts must be equal"
    )
diffs = curr_pos - prev_pos
```

---

### WR-03: ICP (and SWD) always apply spatial registration, ignoring `cpd_penalty=None`

**File:** `eval/stages/alignment.py:407-497`

**Issue:** The dispatch logic in `_build_aligned_cloud` treats `cpd_penalty`
as relevant only for the CPD branch:

```python
if cpd_penalty is None and alignment_method == "cpd":
    aligned[tk] = matched_source_frame       # temporal-only
elif cpd_penalty is not None and alignment_method == "cpd":
    ...                                       # CPD spatial registration
elif alignment_method == "icp":
    icp.register(...)                         # ALWAYS registers
elif alignment_method == "swd":
    aligner.register(...)                     # ALWAYS registers
```

When `alignment_method="icp"` or `"swd"`, spatial registration is applied
unconditionally even if `cpd_penalty=None`. The `_build_aligned_cloud` docstring
describes `cpd_penalty` as controlling whether spatial registration is applied,
creating a false expectation for callers of ICP/SWD methods. The test
`test_alignment_stage_icp_specific_behavior` asserts this is intentional, but
it is asymmetric and undocumented at the method signature level.

**Fix:** Either document this asymmetry explicitly in `_build_aligned_cloud`'s
docstring, or introduce a separate `apply_spatial_registration: bool` parameter
that gives callers a uniform way to opt out of spatial registration for all
three methods.

---

### WR-04: `velocity_threshold` accepts negative values without error

**File:** `eval/config.py:54`

**Issue:**

```python
velocity_threshold: float = 0.5
```

No lower-bound constraint is declared. A negative threshold causes
`detect_velocity_landmarks` to flag every frame except frame 0 (since
`velocity >= 0 > threshold` is always true). This is almost certainly not the
user's intent, and the silent acceptance of negative thresholds makes
misconfigured YAML files very hard to debug.

**Fix:**

```python
velocity_threshold: float = Field(default=0.5, ge=0.0)
```

---

## Info

### IN-01: Dead runtime import in `_apply_preprocessing` suppressed by `noqa`

**File:** `eval/stages/alignment.py:575`

**Issue:**

```python
from eval.config import AlignmentPreprocessingConfig  # noqa: F401
```

This import is inside the function body and is marked with `# noqa: F401`
(imported but unused). The type annotation for the `config` parameter is a
string literal (`"AlignmentPreprocessingConfig | None"`), so Python never
evaluates it at runtime and the import is genuinely dead code at the function
level. `AlignmentPreprocessingConfig` is already available via the module-level
`from eval.config import EvalConfig` if it were added there. The `noqa`
suppresses a correct lint warning.

**Fix:** Remove the in-function import. If IDE/type-checker support is needed,
add `AlignmentPreprocessingConfig` to the module-level import:

```python
from eval.config import AlignmentPreprocessingConfig, EvalConfig
```

And use it directly in the type annotation (no string literal needed).

---

### IN-02: `alignment_method` uses `str` + `field_validator` instead of `Literal`

**File:** `eval/config.py:210-213`

**Issue:** Every other constrained string field in `EvalConfig` uses `Literal`
for validation (`search_strategy`, `tier`, `pipeline_mode`). `alignment_method`
departs from this pattern by using a plain `str` field with a `field_validator`:

```python
alignment_method: str = Field(default="cpd", ...)

@field_validator("alignment_method")
@classmethod
def validate_alignment_method(cls, v: str) -> str:
    if v not in ("cpd", "icp", "swd"):
        raise ValueError(...)
    return v
```

`Literal["cpd", "icp", "swd"]` would achieve the same validation with better
IDE completion, static analysis, and pydantic's native error messages (which
include the allowed values automatically).

**Fix:**

```python
alignment_method: Literal["cpd", "icp", "swd"] = "cpd"
# Remove validate_alignment_method entirely
```

---

### IN-03: `compute_pca_rotation` has no input shape validation

**File:** `src/zreg/preprocessing.py:26-65`

**Issue:** The function expects `[N, 3]` tensors and silently produces
incorrect results for other shapes. A `[N, 2]` input (2-D cloud) produces a
`[2, 2]` rotation matrix that cannot be applied to 3-D point clouds without
a crash, but the error message at the crash site will reference matrix
dimensions rather than pointing to this function. An `[N, 1]` input returns
a `[1, 1]` scalar. No guard exists.

**Fix:**

```python
if source_cloud.ndim != 2 or source_cloud.shape[1] != 3:
    raise ValueError(
        f"compute_pca_rotation: source_cloud must have shape [N, 3]; "
        f"got {tuple(source_cloud.shape)}"
    )
if target_cloud.ndim != 2 or target_cloud.shape[1] != 3:
    raise ValueError(
        f"compute_pca_rotation: target_cloud must have shape [M, 3]; "
        f"got {tuple(target_cloud.shape)}"
    )
```

---

_Reviewed: 2026-06-29T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
