---
phase: 24-trajectory-export
reviewed: 2026-06-04T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - eval/tracking/trajectory.py
  - eval/tracking/__init__.py
  - eval/types.py
  - eval/runners/eval_runner.py
  - tests/test_trajectory.py
  - tests/test_trajectory_export.py
  - tests/test_metrics.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 24: Code Review Report

**Reviewed:** 2026-06-04T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 24 delivers `export_trajectory` (EXT-01), wires it into `EvaluationRunner.run()`, adds the `trajectory_paths` field to `EvalReport`, and provides two test modules covering the new functionality. All 43 tests pass. The implementation is structurally sound and correct for the happy path. The issues found are: one hanging risk from an unguarded `subprocess.run` call, one silent data-integrity failure when `labels` and `pos` are different lengths, one `KeyError` risk from direct dict access on an unvalidated `result` argument, and one test class that makes live git calls without mocking. No security vulnerabilities or data-loss bugs were found; the findings are robustness and reliability issues.

## Warnings

### WR-01: `subprocess.run` has no `timeout` — can hang indefinitely

**File:** `eval/tracking/trajectory.py:93-97`
**Issue:** `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)` is called without a `timeout` argument. In CI environments, slow mounts, or NFS-backed repos, this call can block the entire Python process indefinitely. The `except Exception` handler on line 102 only fires if the process raises an exception (e.g. `FileNotFoundError` when git is absent); it does not protect against a subprocess that hangs. `tracking.py` has the same omission (as acknowledged in the docstring: "verbatim copy"), but that does not reduce the risk here.

**Fix:**
```python
_git_result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    timeout=5,  # seconds — fall through to "unknown" on TimeoutExpired
)
```
Catch `subprocess.TimeoutExpired` inside the existing `except Exception` block, or add it explicitly before the broad catch.

---

### WR-02: Silent `IndexError` / wrong-label write when `labels` length < `pos` length

**File:** `eval/tracking/trajectory.py:170-173`
**Issue:** In the label-trajectory write loop, `labels` is looked up by `frame_idx` and then indexed by `i` up to `len(pos)`:

```python
labels = label_result.transferred_labels[frame_idx]   # line 170
for i in range(len(pos)):                              # line 171
    ...
    label = int(labels[i].item())                      # line 173
```

If `transferred_labels[frame_idx]` has fewer elements than `pos`, line 173 raises `IndexError` and the CSV is written partially (truncated at the point of failure). If it has *more* elements, the extra labels are silently dropped. Neither condition is checked. The docstring (line 57) states `pos` is shape `(N, 3)` and `transferred_labels` are "1-D label tensor per frame", implying they should match, but no assertion enforces this invariant. A mismatch propagates silently until the `IndexError` crash, leaving a corrupt partial CSV on disk.

**Fix:** Add a length check before the inner loop:
```python
labels = label_result.transferred_labels[frame_idx]
if len(labels) != len(pos):
    raise ValueError(
        f"frame {frame_idx}: transferred_labels length {len(labels)} "
        f"!= pos length {len(pos)}"
    )
for i in range(len(pos)):
    ...
```

---

### WR-03: `result["align"]` and `result["label"]` accessed with bare dict syntax — `KeyError` on malformed input

**File:** `eval/tracking/trajectory.py:118,156`
**Issue:** The function contract says `result` must have keys `"align"` and `"label"`, but both are accessed with `result["align"]` and `result["label"]` (no `.get()`). If a caller passes a dict with only one key (e.g. `{"align": ...}` missing `"label"`), line 156 raises an unguarded `KeyError`. `EvaluationRunner._run_single` always populates both keys, so the happy path is fine, but `export_trajectory` is a public API in `eval.tracking.__all__` and callers outside `EvaluationRunner` may omit a key.

**Fix:** Use `.get()` with a `None` default:
```python
align_result_val = result.get("align")
label_result_val = result.get("label")
if align_result_val is not None:
    ...
if label_result_val is not None:
    ...
```
Alternatively, add a guard at the top of the function:
```python
if "align" not in result or "label" not in result:
    raise KeyError("result dict must contain both 'align' and 'label' keys")
```

---

### WR-04: Three test classes in `test_trajectory_export.py` make live `subprocess.run(git …)` calls

**File:** `tests/test_trajectory_export.py:109,154,203`
**Issue:** `TestExportTrajectoryAlignOnly._run`, `TestExportTrajectoryLabelOnly._run`, and `TestExportTrajectoryCombined._run` all call `export_trajectory(...)` without patching `subprocess.run` or `importlib.metadata.version`. This means every test method in these three classes makes a real `git rev-parse HEAD` call. In environments without git on `PATH`, the call falls back to `"unknown"` (safe but noisy). In environments with a slow or unresponsive git (network drives, some CI setups), it adds latency and can trigger the WR-01 hang. The `TestExportTrajectoryMetadata` class (line 259) correctly patches both — the other three classes should follow the same pattern.

**Fix:** Extract the mock context into a module-level fixture and apply it to the `_run` helpers in all three classes:
```python
@pytest.fixture(autouse=True)
def patch_auto_capture():
    with (
        patch("eval.tracking.trajectory.subprocess.run", return_value=_MOCK_GIT_RESULT),
        patch("eval.tracking.trajectory.importlib.metadata.version", return_value="0.0.1"),
    ):
        yield
```
Or inline the context manager in each `_run` helper, matching what `TestExportTrajectoryMetadata._run` already does.

---

## Info

### IN-01: `test_align_csv_xyz_are_floats` assertion is fragile — fails for NaN positions

**File:** `tests/test_trajectory.py:242`
**Issue:** The float-format check is:
```python
assert "." in first_row["x"] or "e" in first_row["x"].lower()
```
This passes for normal floats (`"1.234"`, `"-2.3e-5"`) but **fails** for `"nan"` or `"inf"` (neither contains `"."` nor `"e"`). A better assertion for the actual intent ("the column contains a floating-point value") is simply `float(first_row["x"])` raises no exception. The test one line later (`assert isinstance(float(first_row["x"]), float)`) is the correct check; the first assertion is redundant and fragile.

**Fix:** Remove lines 242-243 and rely solely on:
```python
assert isinstance(float(first_row["x"]), float)
```

---

### IN-02: `export_trajectory` silently produces a header-only CSV for an empty `dataset`

**File:** `eval/tracking/trajectory.py:126-130`
**Issue:** When `dataset = {}`, `sorted(dataset.keys())` is empty, so the write loop never executes, producing a CSV with only the header row. The function still returns the two file paths and reports `frame_count=0`. This is not an error per the spec, but it is not documented and no test covers it. A caller inspecting `frame_count=0` may be unaware that the CSV body is empty.

**Fix:** Add documentation to the docstring noting that an empty `dataset` produces header-only CSV files, or add a guard:
```python
if not dataset:
    raise ValueError("dataset must be non-empty")
```

---

### IN-03: `EvalReport.trajectory_paths` field position is tested by index arithmetic, which is brittle

**File:** `tests/test_metrics.py:499-511`
**Issue:** `test_trajectory_paths_field_order_after_plot_paths` asserts:
```python
fields.index("trajectory_paths") == fields.index("plot_paths") + 1
```
This couples the test to the exact field ordering inside `EvalReport`. If any field is inserted between `plot_paths` and `trajectory_paths` (or reordered for any reason), this test fails despite no functional regression. The intent — that `trajectory_paths` exists and is logically grouped near `plot_paths` — is better expressed as a membership test plus a relative-order test:
```python
assert "trajectory_paths" in fields
assert fields.index("trajectory_paths") < fields.index("sanity_flags")
```
The second test (`test_trajectory_paths_field_order_before_sanity_flags`, line 506) already expresses this weaker contract correctly.

**Fix:** Replace the `== fields.index("plot_paths") + 1` assertion with an ordering check that is less sensitive to intermediate insertions, or remove it and rely on `test_trajectory_paths_field_order_before_sanity_flags`.

---

_Reviewed: 2026-06-04T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
