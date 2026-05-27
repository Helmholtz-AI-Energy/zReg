---
phase: 17-framework-config-datafactory
plan: "02"
subsystem: eval-framework
tags: [datafactory, frame-02, data-loading, synthetic-generation, augmentation, train-val-split]
dependency_graph:
  requires:
    - 17-01  # EvalConfig + EvalConfigError from eval/config.py
    - zreg.dataset  # load_data_from_tracklets, load_shah_from_csv, zRegPointCloud
    - zreg.generators  # generate_trajectory, add_gaussian_noise, add_outliers
  provides:
    - eval.data_factory.DataFactory  # lazy+cached data orchestrator for FRAME-02
    - tests/test_data_factory.py  # 21 tests (6 from 17-01 + 15 new)
  affects:
    - Phase 18 (MetricsEngine) — DataFactory is the data input source
    - Phase 19–23 — all downstream phases depend on DataFactory API
tech_stack:
  added: []
  patterns:
    - lazy-cache class with _field: T | None = None instance attributes (D-08, D-09)
    - dict-key dispatch in augment() reading config.augmentation_params
    - sorted random.sample split preserving temporal ordering (D-05)
key_files:
  created:
    - eval/data_factory.py
  modified:
    - tests/test_data_factory.py
decisions:
  - augment() dispatch: sigma→add_gaussian_noise first, n_outliers→add_outliers second (locks RESEARCH Open Question 1)
  - generate_synthetic() uses n_points=100 hardcoded (no per-config field in locked FRAME-01 spec)
  - prepare_split() uses int() floor truncation per D-07; 4-frame dataset with val_split=0.2 returns empty val silently
  - get_ground_truth() returns tensor references from pc["id"] (no deepcopy per D-11)
metrics:
  duration: "7 minutes"
  completed: "2026-05-27"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 1
  tests_added: 15
  tests_total: 21
---

# Phase 17 Plan 02: DataFactory — Lazy+Cached FRAME-02 Orchestrator Summary

DataFactory class in `eval/data_factory.py` wrapping `load_data_from_tracklets`, `load_shah_from_csv`, `generate_trajectory`, `add_gaussian_noise`, and `add_outliers` behind a single EvalConfig-driven lazy+cached API with 21 unit tests.

## What Was Built

### Task 1: eval/data_factory.py (commit 9797b45)

Created `eval/data_factory.py` as a new standalone module in the `eval/` namespace directory (no `eval/__init__.py`).

**DataFactory class with six FRAME-02 methods:**

- `__init__(config: EvalConfig)`: stores config, initializes `_real_dataset: dict | None = None` and `_synthetic_dataset: dict | None = None`. No I/O (D-08).
- `load_real()`: dispatches to `load_data_from_tracklets(path, device="cpu")` (destructuring the tuple — Pitfall 4) or `load_shah_from_csv(path, device="cpu")` (explicit device — Pitfall 5) per `config.data_format`. Raises `ValueError` for unknown format. Caches by reference (D-09).
- `generate_synthetic()`: calls `generate_trajectory(n_points=100, n_frames=config.n_synthetic, seed=42)`. Caches by reference (D-09).
- `augment(dataset)`: reads `config.augmentation_params` dict; applies `add_gaussian_noise` if key `"sigma"` present, then `add_outliers` if key `"n_outliers"` present. Empty dict is no-op (returns input unchanged).
- `prepare_split(dataset)`: `sorted(keys)` → `random.sample` → two sorted-key dicts (D-05). Single-frame or zero-val-count returns `(dataset, {})` silently (D-07, Pitfall 8).
- `get_ground_truth(dataset)`: extracts `{i: pc["id"] for i, pc in dataset.items()}` by default; loads external file via same loader when `config.ground_truth_path` is set (D-10, D-11).

**Import order** honors macOS ARM libomp safety: `zreg.dataset` → `zreg.generators` → `import torch` → `from eval.config import EvalConfig` (lines 20, 25, 35, 38).

### Task 2: tests/test_data_factory.py — 15 new tests (commit 0c7a487)

Replaced the commented `# Plan 17-02: from eval.data_factory import DataFactory` placeholder with a real import. Added `from zreg.dataset import zRegPointCloud` (before `import torch` for macOS ARM). Populated six stub classes with 15 test methods:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestDataFactoryConstruction | 1 | D-08 lazy init |
| TestLoadReal | 3 | tracklets dispatch, csv dispatch, D-09 cache |
| TestGenerateSynthetic | 2 | seed reproducibility, D-09 cache |
| TestAugment | 4 | no-op, sigma, n_outliers, both keys |
| TestPrepareSplit | 3 | 8+2 shape gate, sorted keys, 1-frame guard |
| TestGetGroundTruth | 2 | id extraction, external GT path |

All mocks patch at `eval.data_factory.load_*` (not `zreg.dataset.load_*`) per the test_tracking.py precedent.

## Verification Results

- `pytest tests/test_data_factory.py -x --tb=short` → **21 passed**
- `pytest tests/ -x --tb=short` → **579 passed, 17 skipped** (full repo suite green)
- `eval/__init__.py` does NOT exist (namespace directory preserved)
- `import torch` on line 35, after `from zreg.dataset` (line 20) and `from zreg.generators` (line 25)
- 3 patches at `eval.data_factory.load_*`; 0 patches at wrong namespace

## Deviations from Plan

None — plan executed exactly as written. All locked interface specifications in `<interfaces>` block were followed verbatim, including the `augment()` dispatch order (noise then outliers), the `prepare_split()` floor-truncation behavior, and the `get_ground_truth()` tensor-reference return contract.

## Known Stubs

None. All six DataFactory methods are fully implemented.

## Threat Flags

The threat surface introduced by `load_real()` and `get_ground_truth()` (file access from user-provided `data_path` / `ground_truth_path`) is already catalogued in the plan's threat model as T-17-05 (DoS via large .mat file — accepted) and T-17-06 (ValueError message leaks offending value — accepted, researcher-only inputs). No new surface was introduced beyond what the plan documented.

## Self-Check: PASSED

- `eval/data_factory.py` exists at `/Users/valeriekieslinger/Documents/Hiwi/BA/zReg/.claude/worktrees/agent-ab9dd81b15a5f7c72/eval/data_factory.py`
- Commit `9797b45` exists (feat(17-02): DataFactory)
- Commit `0c7a487` exists (test(17-02): 15 DataFactory tests)
- 21 tests passing; 579 repo tests green
