---
phase: 15-experiment-tracking-run-management
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - eval/tracking/__init__.py
  - eval/tracking/tracking.py
  - tests/test_tracking.py
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: fixes_applied
fix_commit: ea835bc
---

# Phase 15: Code Review Report

**Reviewed:** 2026-05-18
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the `eval/tracking` package (`__init__.py`, `tracking.py`) and its test suite (`tests/test_tracking.py`). The implementation is structurally sound and follows the established project conventions. One security vulnerability was found: `run_id` is interpolated directly into file paths without sanitization, enabling path traversal. Three warnings cover a subprocess hang risk, a missing test for the non-zero returncode fallback path, and silent overwrite behaviour on duplicate `run_id`. Two informational items note an unused import in the test file and missing type annotations on two parameters.

---

## Critical Issues

### CR-01: Path Traversal via Unsanitized `run_id` in Filename Construction

**File:** `eval/tracking/tracking.py:120-126`
**Issue:** `run_id` is interpolated directly into both output file paths without any sanitization or containment check. A caller-controlled value such as `"../evil"` resolves outside `output_dir`:

```
Path("evaluation/runs") / "../evil.json"  =>  "evaluation/evil.json"
```

An absolute component such as `"/etc/cron.d/x"` causes `Path` to discard `output_dir` entirely and resolve to `/etc/cron.d/x.json`. While the primary callers (Phase 16 runner scripts and Phase 17 Optuna trials) are trusted, the public API contract advertises `run_id` as a caller-provided `str` with no stated restrictions. An untrusted or programmatically generated `run_id` (e.g., derived from a dataset filename or trial parameter) can silently write files to arbitrary locations.

**Fix:** Reject or sanitize `run_id` before path construction. The minimal safe fix rejects any value containing a path separator or leading dot:

```python
import os

def _validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    if os.sep in run_id or (os.altsep and os.altsep in run_id):
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    if run_id.startswith("..") or run_id.startswith("/"):
        raise ValueError(f"run_id must not begin with '..' or '/': {run_id!r}")
```

Call `_validate_run_id(run_id)` at the top of `log_run()`, before any I/O. Alternatively, use `Path(run_id).name` to extract only the final path component and assert it equals `run_id`.

---

## Warnings

### WR-01: `subprocess.run` Called Without `timeout` — Can Hang Indefinitely

**File:** `eval/tracking/tracking.py:82-86`
**Issue:** The `git rev-parse HEAD` subprocess is invoked without a `timeout` parameter. In environments where `git` blocks (e.g., a network-mounted filesystem, a locked index, or a broken `GIT_SSH_COMMAND`), `log_run()` will stall indefinitely. The docstring promises that `git_hash` failures fall back silently to `"unknown"`, but this promise is only honoured for non-zero return codes and exceptions — a hung subprocess is not an exception and will never reach the fallback.

**Fix:**
```python
result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    timeout=5,          # seconds; tune as needed
)
```

Wrap the call in the existing `except Exception` block — `subprocess.TimeoutExpired` is a subclass of `SubprocessError` which is a subclass of `Exception`, so it will be caught and `git_hash` will correctly fall back to `"unknown"`.

---

### WR-02: Non-Zero `returncode` Fallback Path Not Tested

**File:** `tests/test_tracking.py:231-246`
**Issue:** The test suite exercises the git fallback via `side_effect=Exception(...)` (line 233) but never via `returncode != 0`. The implementation has two distinct code paths that produce `"unknown"`:

1. `result.returncode != 0` → `else: git_hash = "unknown"` (line 90)
2. `subprocess.run` raises → `except Exception: git_hash = "unknown"` (line 92)

Only path 2 is covered. A future refactor that accidentally removes the `else` branch (e.g., replacing `if result.returncode == 0` with `result.check_returncode()`) would not be caught by the current test suite.

**Fix:** Add a test for the non-zero returncode path:

```python
def test_git_hash_fallback_on_nonzero_returncode(self, tmp_path):
    """When subprocess.run returns returncode != 0, git_hash equals 'unknown'."""
    mock_result = MagicMock(stdout="", returncode=128)
    with patch("eval.tracking.tracking.subprocess.run", return_value=mock_result), \
         patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
        log_run(
            run_id=RUN_ID,
            dataset_path=DATASET_PATH,
            frame_indices=FRAME_INDICES,
            seed=SEED,
            n_points_before=N_POINTS_BEFORE,
            n_points_after=N_POINTS_AFTER,
            output_dir=str(tmp_path),
        )
    with open(tmp_path / f"{RUN_ID}.json") as f:
        data = json.load(f)
    assert data["git_hash"] == "unknown"
```

---

### WR-03: Silent Overwrite When `run_id` Is Reused

**File:** `eval/tracking/tracking.py:121, 127`
**Issue:** Both output files are opened with mode `"w"` (truncate-and-overwrite). If `log_run()` is called twice with the same `run_id` and `output_dir`, the second call silently destroys the first run's records without any warning. This is especially risky in Optuna sweep loops (Phase 17), where trial identifiers could collide if two trials share an `int`-to-`str` conversion that isn't globally unique. The docstring does not mention this behaviour.

**Fix (option A — document the behaviour):** Add to the docstring `Raises` section or a `Notes` section:

```
Notes
-----
If a file named ``{run_id}.json`` or ``{run_id}.csv`` already exists in
``output_dir``, it is silently overwritten.  Callers are responsible for
ensuring ``run_id`` uniqueness within a given ``output_dir``.
```

**Fix (option B — fail-fast on collision):** Open with exclusive-creation mode and raise a descriptive error:

```python
json_path = Path(output_dir) / f"{run_id}.json"
try:
    f = open(json_path, "x")   # exclusive create; raises FileExistsError if present
except FileExistsError:
    raise FileExistsError(
        f"Run ID {run_id!r} already exists at {json_path}. "
        "Use a unique run_id or remove the existing file."
    ) from None
```

Option A is the minimal change; option B is safer for production use.

---

## Info

### IN-01: Unused Import `io` in Test File

**File:** `tests/test_tracking.py:22`
**Issue:** `import io` is present at the top of the test module but `io` is never referenced anywhere in the file. This is dead code that may confuse future maintainers (suggesting `io.StringIO` or similar was originally planned).

**Fix:** Remove line 22:
```python
import io   # delete this line
```

---

### IN-02: Missing Type Annotations on `dataset_path`, `frame_indices`, and `seed` Parameters

**File:** `eval/tracking/tracking.py:23-26`
**Issue:** Three parameters lack PEP 484 type annotations in the function signature, while `n_points_before: int` and `n_points_after: int` are annotated. The docstring describes the expected types (`str or Path`, `list of int`, `int`) but they are not reflected in the signature, reducing static-analysis coverage.

```python
# Current
def log_run(
    run_id: str,
    dataset_path,        # unannotated
    frame_indices,       # unannotated
    seed,                # unannotated
    ...
```

**Fix:**
```python
import os
from typing import List, Union

def log_run(
    run_id: str,
    dataset_path: Union[str, Path],
    frame_indices: List[int],
    seed: int,
    n_points_before: int,
    n_points_after: int,
    output_dir: str = "evaluation/runs",
) -> str:
```

---

_Reviewed: 2026-05-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
