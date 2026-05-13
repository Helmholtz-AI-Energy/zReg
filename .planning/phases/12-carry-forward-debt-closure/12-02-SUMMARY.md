---
phase: 12-carry-forward-debt-closure
plan: "02"
subsystem: packaging
tags: [python, packaging, public-api, config, protocol]

requires:
  - phase: 10-code-quality-verification
    provides: "src/zreg/config.py with configure_pytorch()"
  - phase: 11.1-close-dtw-02
    provides: "DistanceMetric Protocol annotations in pairwise_distance_matrix.py and dtw/core.py"

provides:
  - "zreg.config accessible as top-level attribute (CARRY-04 closed)"
  - "CARRY-02 formally closed with evidence cited (no code change)"

affects:
  - "Any caller using `import zreg; zreg.config.configure_pytorch()`"

tech-stack:
  added: []
  patterns:
    - "Submodule re-export: `from . import config as config` following existing `from . import {name} as {name}` idiom"

key-files:
  created: []
  modified:
    - "src/zreg/__init__.py"
    - "tests/conftest.py"

key-decisions:
  - "config placed after utils (line 38) following immediate-neighbor style — no `# noqa: E402` comment added to match surrounding lines 37-42"
  - "CARRY-02 closed without code change: Phase 11.1 already wired DistanceMetric annotation in pairwise_distance_matrix.py:40 and dtw/core.py:83"
  - "conftest.py open3d import order fix applied as Rule 3 deviation (same fix as Phase 11.1) to unblock pytest on macOS ARM"

requirements-completed:
  - CARRY-02
  - CARRY-04

duration: 6min
completed: "2026-05-13"
---

# Phase 12 Plan 02: config Submodule Export and CARRY-02 Closure Summary

**One-line add of `from . import config as config` to src/zreg/__init__.py exposes the config submodule at the top level (CARRY-04), and CARRY-02 is formally closed with citation to Phase 11.1 evidence — zero additional code change required**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-13T12:42:00Z
- **Completed:** 2026-05-13T12:48:58Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- `import zreg; zreg.config` now resolves to `<module 'zreg.config' from '...'>` without AttributeError
- `zreg.config.configure_pytorch()` is callable via both `zreg.config.configure_pytorch` and `from zreg.config import configure_pytorch`
- All 9 existing submodule exports unchanged and still importable
- CARRY-02 formally closed: `DistanceMetric` used as type annotation in both consumer modules (Phase 11.1 evidence)
- 381 tests pass, 8 skipped

## Task Commits

Each task was committed atomically:

1. **Task 1: Add config submodule re-export and CARRY-02 closure** - `ed6ded2` (feat)

## Exact Diff Applied

`src/zreg/__init__.py` — one line added after `from . import utils as utils`:

```diff
 from . import utils as utils
+from . import config as config
 from . import cpd as cpd
```

## CARRY-02 Closure Note

**Status:** Closed without code change (per D-02 in CONTEXT.md)

**Evidence:** Phase 11.1 wired `DistanceMetric` as a type annotation in two consumer modules:

1. `src/zreg/pairwise_distance_matrix.py:40`
   ```python
   distance_metric: list[str | DistanceMetric] | str | DistanceMetric = "swd",
   ```

2. `src/zreg/dtw/core.py:83`
   ```python
   distance_metric: list[str | DistanceMetric] | str | DistanceMetric = "swd",
   ```

**Source:** `.planning/phases/11.1-close-dtw-02/11.1-01-SUMMARY.md` — confirms both files were modified in tasks 3–4 of Phase 11.1 (commits `3a6e7ff` and `47639c1`).

The CARRY-02 requirement states: "DistanceMetric Protocol used as annotation in at least one consumer." Both `pairwise_distance_matrix.py` and `dtw/core.py` now satisfy this. No further annotation needed.

## Files Created/Modified

- `src/zreg/__init__.py` — Added `from . import config as config` on line 38 (between utils and cpd)
- `tests/conftest.py` — Added open3d import guard before torch (Rule 3 deviation, see below)

## Decisions Made

- `config` positioned after `utils` (line 38) to group small utility modules together; no `# noqa: E402` comment added to match the immediate neighbors (lines 37–42 are without it)
- No `__all__` added to `__init__.py` per CONTEXT.md Claude's Discretion: existing pattern has no `__all__`
- CARRY-02: zero code change per D-02; Phase 11.1 work is sufficient evidence

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed macOS ARM libomp crash in tests/conftest.py**
- **Found during:** Task 1 test verification
- **Issue:** `tests/conftest.py` imported `torch` before `zreg.dataset` (which imports open3d), triggering macOS ARM libomp duplicate-initialization crash (SIGABRT, exit code 134). This is the same issue documented in Phase 11.1 SUMMARY — the fix existed in main but not in this worktree branch.
- **Fix:** Added `import open3d.t.geometry; import open3d.core` block at the top of conftest.py before the `import torch` line, guarded by `try/except` for non-macOS environments
- **Files modified:** `tests/conftest.py`
- **Verification:** 381 tests passed with 0 failures after fix
- **Committed in:** `ed6ded2` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking)
**Impact on plan:** Essential for test execution on macOS ARM. No scope creep.

## Test Counts

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 381 (via PYTHONPATH workaround) | 381 |
| Tests skipped | 8 | 8 |
| Tests failing | 0 | 0 |
| Coverage | 99% | 99% |

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The change adds a submodule re-export only.

## Self-Check: PASSED

- `src/zreg/__init__.py` — FOUND and contains `from . import config as config` on line 38
- `tests/conftest.py` — FOUND and contains open3d import guard before torch
- Commit `ed6ded2` — FOUND in git log
- `python -c "import zreg; zreg.config.configure_pytorch"` — exits 0
- 381 tests pass, 8 skipped, 0 failures

---
*Phase: 12-carry-forward-debt-closure*
*Completed: 2026-05-13*
