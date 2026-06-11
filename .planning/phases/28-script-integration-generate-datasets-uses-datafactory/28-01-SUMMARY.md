---
phase: 28-script-integration-generate-datasets-uses-datafactory
plan: "01"
subsystem: scripts
tags: [refactor, data-factory, augmentation, script-integration, df-02]
dependency_graph:
  requires: [eval/data_factory.py, eval/config.py]
  provides: [scripts/generate_datasets.py]
  affects: [data/synthetic/semi_synthetic/]
tech_stack:
  added: []
  patterns: [DataFactory(EvalConfig(...)).augment(dataset) — per-augmentation factory pattern]
key_files:
  created: []
  modified: [scripts/generate_datasets.py]
decisions:
  - "D-01: DataFactory(EvalConfig(data_path='', augmentation_params={KEY: value})).augment(dataset) per call"
  - "D-02: All three augmentation types route through DataFactory.augment via canonical dispatch keys"
  - "D-03: _augment_noise, _augment_scaling, _augment_dropout, apply_augmentation all removed"
  - "D-04/D-05: Dropout RNG divergence accepted and documented in code comment"
  - "D-06: eval.data_factory/eval.config imports placed after zreg.* and before import torch"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-11"
  tasks_completed: 2
  tasks_total: 3
  files_changed: 1
---

# Phase 28 Plan 01: Script Integration — generate_datasets uses DataFactory — Summary

**One-liner:** Refactored `scripts/generate_datasets.py` to route all three semi-synthetic augmentations (noise, scaling, dropout) through `DataFactory(EvalConfig(...)).augment(dataset)`, eliminating four local duplicate implementations and closing DF-02.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Capture pre-refactor parity baseline | (no commit — transient artefacts only) | /tmp/ scratch only |
| 2 | Refactor scripts/generate_datasets.py to call DataFactory.augment | 97c6d0b | scripts/generate_datasets.py |

## Lines Changed in scripts/generate_datasets.py

**Imports block:**
- Removed: `from zreg.generators import add_gaussian_noise` (line 33)
- Added: `sys.path.insert(0, str(ROOT))` (repo root on path for eval.* imports)
- Added: `from eval.data_factory import DataFactory` (after zreg.* imports, before torch)
- Added: `from eval.config import EvalConfig` (after zreg.* imports, before torch)

**Functions removed (39 lines deleted):**
- `def _augment_noise(dataset, sigma)` — lines 308-309
- `def _augment_scaling(dataset, factor)` — lines 312-316
- `def _augment_dropout(dataset, fraction)` — lines 319-331
- `def apply_augmentation(dataset, aug_type, param_name, value)` — lines 334-346
- `# Semi-synthetic augmentations` section header — lines 304-306

**Call site replacement (1 line → 4 lines):**
- Removed: `augmented = apply_augmentation(dataset, aug_type, param_name, value)`
- Added:
  ```python
  # Note: dropout RNG changed from np.random (pre-Phase 28) to torch.randperm via DataFactory.drop_points; dropout point selection differs but fraction and reproducibility are preserved (D-04/D-05).
  _aug_key = {"noise": "sigma", "scaling": "scale_factor", "dropout": "dropout_fraction"}[aug_type]
  _cfg = EvalConfig(data_path="", augmentation_params={_aug_key: value})
  augmented = DataFactory(_cfg).augment(dataset)
  ```

## Parity Results

| Augmentation Type | Parity Method | Result |
|-------------------|---------------|--------|
| Noise (sigma 1.0, 5.0, 10.0) | MD5 bit-identity | PASS — all 3 hashes match baseline |
| Scaling (factor 0.8, 1.2, 1.5) | MD5 bit-identity | PASS — all 3 hashes match baseline |
| Dropout (fraction 0.1, 0.2, 0.3) | Row count parity (±1) | PASS — exact match (diff=0 for all 3) |
| Dropout reproducibility | diff -r across two consecutive runs | PASS — byte-identical across 2 runs |

**Note on dropout RNG change (D-04/D-05):** The dropout implementation switched from NumPy `np.random.default_rng(42).choice()` (old) to PyTorch `torch.manual_seed(42)` + `torch.randperm()` via `DataFactory.drop_points`. The specific points selected differ between old and new, but the fraction kept is identical (`round(n*(1-fraction))` calculation is the same). The new implementation is fully reproducible (run-to-run byte-identity confirmed).

## Regression Test Results

```
826 passed, 18 skipped
```

Full regression suite (826 tests) passes. `test_data_factory.py` (47 tests) passes. The script change does not touch `eval/` — no `eval.*` tests are affected.

## RNG Divergence Comment

The following comment was added immediately above the `_aug_key` line inside `generate_semi_synthetic()`:
```
# Note: dropout RNG changed from np.random (pre-Phase 28) to torch.randperm via DataFactory.drop_points; dropout point selection differs but fraction and reproducibility are preserved (D-04/D-05).
```

## Checkpoint Status

Task 3 (`checkpoint:human-verify`) reached — awaiting human verification of the refactored script. The refactor is complete and all automated checks pass; the checkpoint asks a human reviewer to visually inspect the import block, removed functions, and call site.

## Deviations from Plan

**1. [Rule 3 - Blocking] generate_datasets.py was untracked — not committed to git**
- **Found during:** Task 1
- **Issue:** `scripts/generate_datasets.py` was listed in `files_modified` in the plan frontmatter, but the file existed only as an untracked file in the main repo (per git status `?? scripts/generate_datasets.py`). It was not present in the worktree.
- **Fix:** Copied the untracked file from the main repo into the worktree before starting Task 1. The commit (97c6d0b) adds it as a new tracked file (`create mode 100644 scripts/generate_datasets.py`).
- **Files modified:** scripts/generate_datasets.py

**2. [Rule 3 - Blocking] sys.path.insert needed for eval.* imports**
- **Found during:** Task 2
- **Issue:** The script uses a relative `ROOT = Path(__file__).resolve().parent.parent` path insertion for `src/`, but `eval.*` lives at the repo root (not under `src/`). Without `sys.path.insert(0, str(ROOT))`, `from eval.data_factory import DataFactory` would fail with ModuleNotFoundError.
- **Fix:** Added `sys.path.insert(0, str(ROOT))` after the existing `sys.path.insert(0, str(ROOT / "src"))` line.
- **Files modified:** scripts/generate_datasets.py

**3. Minor: plan's parity check step (step 5) compares paths differently**
- The plan's step 5 used `sort /tmp/...MD5SUMS.txt` for the "old" file. In practice, the hash extraction approach (`awk '{print $1}'` + sort) was used to compare only the hash portion, avoiding path prefix differences. Result: all 6 noise/scaling hashes match exactly.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. The script writes to the existing `data/synthetic/semi_synthetic/` directory via the pre-existing `save_as_csv` function.

## Self-Check

- [x] `scripts/generate_datasets.py` exists in worktree: `ls -la scripts/generate_datasets.py`
- [x] Commit 97c6d0b exists: confirmed via `git log --oneline`
- [x] No scratch directories remain: `ls /tmp/zreg_phase28*` returns no matches
- [x] 826 tests pass: confirmed via `pytest` full suite run
