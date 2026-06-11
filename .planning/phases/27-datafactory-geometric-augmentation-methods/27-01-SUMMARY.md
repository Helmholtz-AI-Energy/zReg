---
phase: 27-datafactory-geometric-augmentation-methods
plan: "01"
subsystem: eval
tags:
  - data-factory
  - augmentation
  - geometric-transforms
  - tdd
dependency_graph:
  requires:
    - eval/data_factory.py (DataFactory base class from Phase 17)
    - src/zreg/generators/transforms.py (apply_rigid)
    - src/zreg/transforms/__init__.py (RigidTransformation)
  provides:
    - DataFactory.scale(dataset, factor)
    - DataFactory.rotate(dataset, rotation_matrix)
    - DataFactory.drop_points(dataset, fraction, seed)
    - DataFactory.sample_new_points(dataset, n_extra, seed)
  affects:
    - eval/data_factory.py
    - tests/test_data_factory.py
tech_stack:
  added:
    - import math (stdlib, for augment() Rodrigues extension in Plan 27-02)
    - from zreg.transforms import RigidTransformation (wraps rotation matrix for apply_rigid)
  patterns:
    - Immutable per-frame transform (new zRegPointCloud object per frame, no deepcopy needed)
    - fps-idx explicit pass-through pattern (mirrors add_outliers from corruption.py)
    - torch.manual_seed seeded once before loop (not inside loop)
    - Inner _extend helper for 1-D/-2D sentinel fill
key_files:
  created: []
  modified:
    - eval/data_factory.py
    - tests/test_data_factory.py
decisions:
  - "scale/drop_points/sample_new_points use new-zRegPointCloud pattern (not deepcopy) — sufficient for immutability since pos*factor creates a new tensor"
  - "rotate delegates to apply_rigid which internally calls copy.deepcopy via _apply_matrix — no additional copying needed"
  - "torch.manual_seed called once before loop in drop_points and sample_new_points — seeding inside loop would make all frames identical"
  - "sample_new_points uses per-frame bbox (not dataset-global) — consistent with each frame's own point cloud extent"
metrics:
  duration: "~4 min"
  completed: "2026-06-11"
  tasks_completed: 2
  tests_added: 18
  files_modified: 2
---

# Phase 27 Plan 01: DataFactory Geometric Augmentation Methods Summary

Four standalone geometric augmentation methods added to DataFactory in eval/data_factory.py using TDD (RED → GREEN cycle). All four methods follow the immutable contract — input dataset is never mutated.

## What Was Built

Four new DataFactory instance methods with full test coverage:

1. **`scale(dataset, factor)`** — multiplies every frame's `pos` by `factor`, preserves `fps-idx`, returns new dict. `pos * factor` creates a new tensor so no deepcopy needed.

2. **`rotate(dataset, rotation_matrix)`** — thin wrapper around `apply_rigid(dataset, RigidTransformation(rot=R, ...))`. Deep copy guaranteed by `_apply_matrix` in `src/zreg/generators/transforms.py`.

3. **`drop_points(dataset, fraction, seed=42)`** — uses `torch.randperm` seeded once before the loop, device-safe indexing. All fields (`pos`, `color`, `id`, `fps-idx`) indexed with same `idx` permutation.

4. **`sample_new_points(dataset, n_extra, seed=42)`** — per-frame bbox sampling, `_extend` inner helper for 1-D (sentinel -1) and 2-D (zeros) fields. `fps-idx` extended with -1 sentinels.

Import additions: `import math` (stdlib), `from zreg.transforms import RigidTransformation`. Removed `# noqa: F401` from `apply_rigid` import (now actively used).

## Test Classes

- **TestScale** (4 tests): pos scaling, immutability, factor zero, fps-idx preservation
- **TestRotate** (4 tests): identity noop, rotation changes pos, deep copy contract, correctness (90° Z: (1,0,0)→(0,1,0))
- **TestDropPoints** (5 tests): removes fraction, zero noop, id/fps-idx shape consistency, immutability
- **TestSampleNewPoints** (5 tests): adds n_extra, bbox containment, id sentinel -1, immutability, fps-idx extended

## Commits

| Task | Type | Commit | Description |
|------|------|--------|-------------|
| 2 (RED) | test | c858304 | Add failing tests for 4 methods (18 tests, 4 classes) |
| 1 (GREEN) | feat | f8fe8b4 | Implement scale, rotate, drop_points, sample_new_points |

## Verification Results

```
pytest tests/test_data_factory.py::TestScale tests/test_data_factory.py::TestRotate \
       tests/test_data_factory.py::TestDropPoints tests/test_data_factory.py::TestSampleNewPoints -x -q
18 passed
```

```
pytest tests/ -q
818 passed, 18 skipped, 0 errors
```

## Deviations from Plan

None — plan executed exactly as written.

The task ordering was:
- RED commit (test classes) before GREEN commit (implementation) — strict TDD compliance

Note: `grep -c "torch.manual_seed" eval/data_factory.py` returns 3 (not 2 as stated in acceptance criteria) because the docstring of `sample_new_points` contains the text `torch.manual_seed`. The actual code has exactly 2 calls (lines 373 and 422). Criterion satisfied in substance.

## Known Stubs

None — all four methods are fully implemented and wired. No placeholder values or TODO stubs.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. All operations are pure in-memory tensor transforms. No threat flags.

## Self-Check: PASSED

- eval/data_factory.py modified with 4 new methods: FOUND
- tests/test_data_factory.py with 4 new test classes: FOUND
- Commit c858304 (RED): FOUND
- Commit f8fe8b4 (GREEN): FOUND
- 818 passed, 18 skipped full suite: VERIFIED
