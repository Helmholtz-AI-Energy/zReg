---
phase: 20-labeltransferstage
reviewed: 2026-05-29T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - eval/stages/__init__.py
  - eval/stages/label_transfer.py
  - tests/test_label_transfer_stage.py
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: fixes_applied
---

# Phase 20: Code Review Report

**Reviewed:** 2026-05-29
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed `LabelTransferStage` and its test suite. The implementation is generally
clean and correctly wires through to `transfer_colors` via `KNN_VOTING`. The
bool-exclusion guards are applied correctly (WR-01 pattern from Phase 19), and the
shallow-copy of `params_used` is handled properly.

Two structural bugs were found: one in the production code (`run()` crashes with
`IndexError` on empty datasets because `validate_params` does not guard against
missing `dataset` keys) and one `k_neighbours` range validity gap (`k > n_source`
is not caught). There are also two test-quality issues worth fixing before the
suite is relied upon for regression coverage.

---

## Critical Issues

### CR-01: `run()` crashes with `IndexError` on empty dataset

**File:** `eval/stages/label_transfer.py:196`

**Issue:** When `dataset={}` is passed with otherwise valid `params`, `validate_params`
returns normally (it inspects only `params`, not `dataset`). Execution then reaches
`transferred[sorted_keys[0]]` where `sorted_keys = []`, producing:

```
IndexError: list index out of range
```

This means the D-08 guarantee ("validate_params is called first so callers get
`ValueError`, not internal errors") is violated for the dataset-shape dimension.
The existing tests `test_run_empty_params_raises_value_error` and
`test_run_empty_dataset_empty_params_raises_value_error` only exercise the case
where `params={}` (which raises before the dataset is touched), leaving the
`dataset={}` + valid-params path uncovered and broken.

**Fix:** Add a dataset length guard at the top of `run()`, immediately after
`validate_params`:

```python
def run(self, dataset, params):
    self.validate_params(params)

    if not dataset:
        raise ValueError("dataset must be non-empty; got 0 frames")

    sorted_keys = sorted(dataset.keys())
    ...
```

---

## Warnings

### WR-01: `k_neighbours > n_source_points` raises uncaught `RuntimeError` from torch

**File:** `eval/stages/label_transfer.py:201-207`

**Issue:** `validate_params` correctly enforces `k_neighbours >= 1`, but there is
no guard ensuring `k_neighbours <= len(source_frame["pos"])`. When the caller
passes a dataset where any source frame has fewer points than `k_neighbours`,
`torch.topk` inside `_transfer_colors_knn_voting` raises:

```
RuntimeError: k (N) is too big for dimension size (M)
```

This is an opaque runtime error that bypasses the documented `ValueError` contract
and gives the caller no hint about what went wrong or how to fix it.

The stage cannot validate this at `validate_params` time (dataset is not available
then), so the guard must be placed in `run()` just before the loop:

```python
for k in range(1, len(sorted_keys)):
    src_frame = dataset[sorted_keys[k - 1]]
    n_src = src_frame["pos"].shape[0]
    if params["k_neighbours"] > n_src:
        raise ValueError(
            f"k_neighbours={params['k_neighbours']} exceeds source frame "
            f"{sorted_keys[k - 1]} point count ({n_src})"
        )
    tgt_frame = dataset[sorted_keys[k]]
    ...
```

### WR-02: Duplicate test body — `test_run_empty_dataset_empty_params_raises_value_error` is identical to `test_run_empty_params_raises_value_error`

**File:** `tests/test_label_transfer_stage.py:291-299`

**Issue:** Both methods in `TestRunValidatesFirst` have exactly the same body:
they both call `stage.run({}, {})` and assert `ValueError("Missing required param")`.
The second test adds no new coverage and its docstring claim ("before touching the
dataset") is false — the dataset is not touched in _either_ case because `params={}`
causes the missing-key check to fire first.

```python
# line 291-299 — identical bodies
def test_run_empty_params_raises_value_error(self, stage):
    with pytest.raises(ValueError, match="Missing required param"):
        stage.run({}, {})

def test_run_empty_dataset_empty_params_raises_value_error(self, stage):
    with pytest.raises(ValueError, match="Missing required param"):
        stage.run({}, {})
```

The second test should instead cover `dataset={}` with _valid_ params to catch the
`IndexError` identified in CR-01. Rename and rewrite it:

```python
def test_run_valid_params_empty_dataset_raises_value_error(self, stage, good_params):
    """run({}, valid_params) raises ValueError before indexing empty sorted_keys."""
    with pytest.raises(ValueError):
        stage.run({}, good_params)
```

### WR-03: Mid-file module-level imports in test file break import ordering convention

**File:** `tests/test_label_transfer_stage.py:364-366`

**Issue:** Three module-level imports appear after the first set of test classes,
annotated with `# noqa: E402`:

```python
from zreg.generators import add_gaussian_noise  # noqa: E402
from zreg.metrics.label_transfer import compute_f1  # noqa: E402
from eval.types import AlignResult  # noqa: E402
```

These are not inside a function or fixture — they execute at import time in
documentation order. The `noqa: E402` suppression masks the linter warning that
would otherwise flag this structural issue. More importantly, the project enforces
a strict `zreg.*` before `torch` import order for macOS-ARM libomp reasons
(documented in `conftest.py:20-24`). Placing imports mid-file makes it harder to
audit compliance with that constraint.

Move all three imports to the top of the file, grouped after the existing
`from zreg.dataset import zRegPointCloud` and `from zreg.generators import ...`
lines, and before the `import torch` line:

```python
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_trajectory, generate_labels
from zreg.generators import add_gaussian_noise
from zreg.metrics.label_transfer import compute_f1

import torch

from eval.config import EvalConfig
from eval.stages import LabelTransferStage, PipelineStage
from eval.stages.label_transfer import LabelTransferStage as LabelTransferStageDirect
from eval.types import LabelResult, AlignResult
```

---

## Info

### IN-01: Frame-0 reference pass-through is not documented at the `run()` call site via a type hint or runtime check

**File:** `eval/stages/label_transfer.py:196`

**Issue:** The docstring for `run()` and the module-level docstring both document
the frame-0 pass-through hazard ("callers must treat the source dataset as
read-only after calling `run()`"), but nothing at the `run()` call site enforces
or signals this. A caller who mutates `dataset[0]["color"]` after `run()` returns
will silently corrupt `result.transferred_labels[0]` because both point to the
same tensor object.

The `LabelResult` model uses `ConfigDict(frozen=True)` which blocks attribute
reassignment but does NOT prevent in-place tensor mutation (documented in
`eval/types.py` Pitfall 1). The issue is that `transferred_labels[keys[0]]` is
not a copy.

This is a low-severity concern because it is documented, but a `# NOTE:` comment
directly on line 196 referencing the pitfall number would help future readers:

```python
transferred[sorted_keys[0]] = dataset[sorted_keys[0]]["color"]
# NOTE: reference, not copy — D-02/Pitfall 1. Callers must not mutate
# dataset[sorted_keys[0]]["color"] after run() returns.
```

---

_Reviewed: 2026-05-29_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
