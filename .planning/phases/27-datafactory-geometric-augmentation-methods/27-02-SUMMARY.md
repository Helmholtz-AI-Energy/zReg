---
phase: 27-datafactory-geometric-augmentation-methods
plan: "02"
subsystem: eval
tags:
  - data-factory
  - augmentation
  - geometric-transforms
  - tdd
  - dispatcher
dependency_graph:
  requires:
    - eval/data_factory.py (DataFactory with scale/rotate/drop_points/sample_new_points from Plan 27-01)
    - eval/config.py (EvalConfig.augmentation_params dict)
  provides:
    - augment() with 6-step dispatch (sigma, n_outliers, scale_factor, rotation_deg, dropout_fraction, n_new_points)
    - TestAugmentExtended (8 test methods covering all four new keys + chaining + regression)
  affects:
    - eval/data_factory.py
    - tests/test_data_factory.py
tech_stack:
  added: []
  patterns:
    - Rodrigues' rotation formula inline in augment() (axis normalisation + K matrix + R = I + sin·K + (1-cos)·K²)
    - 6-step sequential dispatch with fixed ordering (geometric before count-change transforms)
    - TDD RED-first ordering (Task 2 tests committed before Task 1 implementation)
key_files:
  created: []
  modified:
    - eval/data_factory.py
    - tests/test_data_factory.py
decisions:
  - "Dispatch order fixed as sigma → n_outliers → scale_factor → rotation_deg → dropout_fraction → n_new_points — geometric transforms (scale, rotate) before count changes (dropout, add points)"
  - "Rodrigues' rotation formula inlined in augment() (not extracted as helper) — consistent with PATTERNS.md Pattern 5 specification"
  - "rotation_axis defaults to [0.0, 0.0, 1.0] (Z-axis) when not present in params — safe and documented in docstring"
  - "axis normalised with +1e-8 guard (T-27-03 accepted threat) — no crash on near-zero vector from caller config"
metrics:
  duration: "~3 min"
  completed: "2026-06-11"
  tasks_completed: 2
  tests_added: 8
  files_modified: 2
---

# Phase 27 Plan 02: augment() Dispatch Extension Summary

Four new augmentation_params keys wired into augment() in DataFactory using TDD (RED commit before GREEN commit). The 6-step dispatch is the contract that scenario configs and Phase 28 script integration will drive. DF-01 is fully closed.

## What Was Built

**`augment()` extended dispatch** in `eval/data_factory.py`:

Six sequential dispatch blocks in fixed order:
1. `sigma` → `add_gaussian_noise` (pre-existing)
2. `n_outliers` → `add_outliers` (pre-existing)
3. `scale_factor` → `self.scale(result, params["scale_factor"])` (new)
4. `rotation_deg` → Rodrigues' formula → `self.rotate(result, R)` (new)
5. `dropout_fraction` → `self.drop_points(result, params["dropout_fraction"])` (new)
6. `n_new_points` → `self.sample_new_points(result, params["n_new_points"])` (new)

The Rodrigues' rotation formula is inlined directly in the `rotation_deg` dispatch block:
- `axis` tensor normalised with +1e-8 guard
- `K` skew-symmetric matrix constructed element-by-element
- `R = I + sin(theta)*K + (1-cos(theta))*K*K`

The updated docstring enumerates all six recognised keys with their types, descriptions, and the full dispatch order.

**`TestAugmentExtended`** in `tests/test_data_factory.py` — 8 test methods:

| Method | What it tests |
|--------|---------------|
| `test_scale_factor_key` | scale_factor=2.0 → out pos allclose input*2 |
| `test_rotation_deg_zero_noop` | rotation_deg=0.0 → pos unchanged |
| `test_dropout_fraction_key` | dropout_fraction=0.5 on 100pts → 50 points |
| `test_n_new_points_key` | n_new_points=10 on 100pts → 110 points |
| `test_scale_then_dropout_chain` | scale_factor+dropout: count==50 AND pos scaled |
| `test_all_four_new_keys` | all 4 keys: final count == round(100*0.8)+5 == 85 |
| `test_existing_sigma_unaffected` | sigma path still applies noise (non-regression) |
| `test_existing_outliers_unaffected` | n_outliers path still adds 5 pts (non-regression) |

## TDD Gate Compliance

| Gate | Commit | Description |
|------|--------|-------------|
| RED | 5f7e105 | Add failing TestAugmentExtended (8 tests, all fail — augment lacks new dispatch) |
| GREEN | 9663957 | Implement 6-step augment() dispatch — all 8 tests pass |

RED-before-GREEN ordering followed. Task 2 tests were committed first (RED), then Task 1 implementation (GREEN), matching the approach used in Plan 27-01.

## Commits

| Task | Type | Commit | Description |
|------|------|--------|-------------|
| 2 (RED) | test | 5f7e105 | Add failing TestAugmentExtended for augment() dispatch extension |
| 1 (GREEN) | feat | 9663957 | Extend augment() with scale_factor, rotation_deg, dropout_fraction, n_new_points |

## Verification Results

```
pytest tests/test_data_factory.py::TestAugment -x -q
4 passed
```

```
pytest tests/test_data_factory.py::TestAugmentExtended -x -q
8 passed
```

```
pytest tests/test_data_factory.py -x -q
47 passed
```

```
pytest tests/ -q
826 passed, 18 skipped, 0 errors
```

Smoke test (scale_factor + dropout_fraction chain):
```
PASS: scale_factor + dropout_fraction chain works correctly
```

## Deviations from Plan

None — plan executed exactly as written.

The Task 2 (RED) commit preceded the Task 1 (GREEN) commit, following the TDD-first ordering established in Plan 27-01. The plan's task numbering lists implementation first and tests second, but TDD compliance requires tests to be written and committed before implementation.

## Known Stubs

None — augment() dispatch is fully implemented. All six keys are wired and active.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. All augment() operations are pure in-memory tensor transforms. The `rotation_axis` param normalisation (+1e-8 guard) handles the T-27-03 threat (accepted). No new threat flags.

## Self-Check: PASSED

- eval/data_factory.py modified with 6-step augment() dispatch: FOUND
- tests/test_data_factory.py with TestAugmentExtended (8 methods): FOUND
- Commit 5f7e105 (RED): FOUND
- Commit 9663957 (GREEN): FOUND
- 826 passed, 18 skipped full suite: VERIFIED
- grep -c "scale_factor" eval/data_factory.py → 6 (>=2): PASS
- grep -c "rotation_deg" eval/data_factory.py → 4 (>=2): PASS
- grep -c "dropout_fraction" eval/data_factory.py → 4 (>=2): PASS
- grep -c "n_new_points" eval/data_factory.py → 4 (>=2): PASS
- grep -c "self.scale(result" eval/data_factory.py → 1: PASS
- grep -c "self.rotate(result" eval/data_factory.py → 1: PASS
- grep -c "self.drop_points(result" eval/data_factory.py → 1: PASS
- grep -c "self.sample_new_points(result" eval/data_factory.py → 1: PASS
- grep -c "Rodrigues" eval/data_factory.py → 3 (>=1): PASS
</content>
</invoke>