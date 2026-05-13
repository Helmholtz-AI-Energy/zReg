---
phase: 12-carry-forward-debt-closure
plan: "01"
subsystem: cpd, color_transfer
tags: [python, imports, cpd, color-transfer, cleanup, carry-forward]
requirements-completed: [CARRY-01, CARRY-03]
dependency-graph:
  requires: []
  provides: [collections-abc-callable-cpd, relative-import-color-transfer]
  affects: [src/zreg/cpd/base.py, src/zreg/cpd/_registration.py, src/zreg/color_transfer.py]
tech-stack:
  added: []
  patterns: [collections.abc.Callable for type annotations, relative intra-package imports]
key-files:
  modified:
    - src/zreg/cpd/base.py
    - src/zreg/cpd/_registration.py
    - src/zreg/color_transfer.py
decisions:
  - "Used two-line split (from typing import Any + from collections.abc import Callable) in _registration.py to keep Any sourced from typing while moving Callable to collections.abc — matches stdlib-block import style already in the file"
  - "No changes to downstream list[Callable] annotations or EstepResult use sites — symbol names are identical across the import sources"
metrics:
  duration: "507s (~8 min)"
  completed: "2026-05-13T12:53:28Z"
  tasks-completed: 2
  files-modified: 3
---

# Phase 12 Plan 01: Import Style Debt Closure (CARRY-01, CARRY-03) Summary

**One-liner:** Replace deprecated `typing.Callable` with `collections.abc.Callable` in two CPD files and convert the sole remaining absolute intra-package import in `color_transfer.py` to a relative import — zero behavior change, three files, three lines.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Convert typing.Callable to collections.abc.Callable in cpd/ | 71e11c9 | src/zreg/cpd/base.py, src/zreg/cpd/_registration.py |
| 2 | Convert absolute zreg.cpd import to relative .cpd import in color_transfer.py | 8442e1d | src/zreg/color_transfer.py |

## Exact Lines Changed

### src/zreg/cpd/base.py (line 4)
```
- from typing import Callable
+ from collections.abc import Callable
```

### src/zreg/cpd/_registration.py (line 7)
```
- from typing import Any, Callable
+ from typing import Any
+ from collections.abc import Callable
```

### src/zreg/color_transfer.py (line 11)
```
- from zreg.cpd import EstepResult
+ from .cpd import EstepResult
```

## Verification Results

- No `from typing import .*Callable` remains in `src/zreg/cpd/`: PASS
- No `from zreg.` absolute import remains in `color_transfer.py`: PASS
- `from collections.abc import Callable` present in both CPD files: PASS (2 matches)
- `from typing import Any` still in `_registration.py`: PASS
- `list[Callable]` downstream usage preserved: base.py (3 matches), _registration.py (4 matches)
- `EstepResult` usage: 5 occurrences in color_transfer.py (import + 4 use sites on lines 34, 55, 164, 179)
- Package imports cleanly: `import zreg`, `import zreg.color_transfer`, `import zreg.cpd`: PASS
- `from zreg.cpd.base import CoherentPointDrift; from zreg.cpd._registration import cpd_registration`: PASS
- `from zreg.color_transfer import transfer_colors, ColorTransferMethod`: PASS

## Test Suite Note

The test suite runs correctly (exit 0) when executed without output redirection in the foreground. A pre-existing macOS ARM OpenMP/libiomp conflict (`OMP Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`) causes SIGSEGV (exit 139) when test subprocesses are spawned with forked output capture. This is unrelated to the import changes made in this plan — both tests/test_cpd.py and tests/test_color_transfer.py exit 0 when run in the foreground without subshell capture. The workaround (`KMP_DUPLICATE_LIB_OK=TRUE`) eliminates the crash at the process level. The issue predates this plan and is documented in `dataset.py` line 9: "open3d must be imported before torch to avoid libomp conflict on macOS ARM."

## Requirements Closed

- **CARRY-01**: Both `cpd/base.py` and `cpd/_registration.py` now source `Callable` from `collections.abc`. `typing.Callable` is no longer used in the CPD package.
- **CARRY-03**: `color_transfer.py` uses a relative import (`from .cpd import EstepResult`). No `from zreg.cpd` line remains in the file.

## Deviations from Plan

None — plan executed exactly as written. All three one-line changes applied as specified, no additional fixes required.

## Known Stubs

None.

## Threat Flags

None — these are import statement changes only. No new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- [x] src/zreg/cpd/base.py exists with `from collections.abc import Callable` on line 4
- [x] src/zreg/cpd/_registration.py exists with `from collections.abc import Callable` on line 8
- [x] src/zreg/color_transfer.py exists with `from .cpd import EstepResult` on line 11
- [x] Commit 71e11c9 exists (Task 1)
- [x] Commit 8442e1d exists (Task 2)
