---
phase: 53-gpu-acceleration
plan: "01"
subsystem: eval
tags: [gpu, config, data-factory, validation, tdd]
dependency_graph:
  requires: []
  provides: [EvalConfig.device, DataFactory.device-guard, DataFactory.device-threading]
  affects: [eval/config.py, eval/data_factory.py]
tech_stack:
  added: [pydantic model_validator]
  patterns: [whitelist field_validator, cross-field model_validator, fast-fail availability guard, INFO-level data-load logging]
key_files:
  created: []
  modified:
    - eval/config.py
    - eval/data_factory.py
    - tests/test_eval_config.py
    - tests/test_data_factory.py
decisions:
  - "Use config.device.startswith('cuda') not == 'cuda' in DataFactory guard to cover cuda:0 and cuda:1"
  - "Validators raise ValueError (not EvalConfigError) so pydantic wraps to ValidationError and from_yaml converts to EvalConfigError"
  - "logging.info placed after _standardize() so len(dataset) reflects post-subsample count"
  - "No logging in get_ground_truth() per D-07 specification"
metrics:
  duration: 20min
  completed: "2026-07-16"
  tasks: 3
  files: 4
---

# Phase 53 Plan 01: GPU Device Threading (EvalConfig + DataFactory) Summary

Validated EvalConfig device field with whitelist validator and ICP-incompatibility guard, DataFactory availability fast-fail guard, 6 device="cpu" call-site substitutions, and INFO-level load logging — all TDD-first.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write failing tests for EvalConfig device + DataFactory guard | 61846d0 | tests/test_eval_config.py, tests/test_data_factory.py |
| 2 | Add device field, @field_validator, @model_validator to EvalConfig | 67bfb00 | eval/config.py |
| 3 | Add DataFactory __init__ guard, 6 device substitutions, INFO logging | 8b34b5d | eval/data_factory.py |

## What Was Built

**EvalConfig.device field (eval/config.py):**
- `device: str = Field(default="cpu", ...)` added after `label_transfer_method` field
- `model_validator` added to pydantic import
- `@field_validator("device") validate_device`: whitelist `{"cpu","cuda","cuda:0","cuda:1","mps"}`, raises `ValueError` with message `"device must be one of {...}; got {v!r}"`
- `@model_validator(mode="after") validate_device_icp_compat`: raises `ValueError` when `device != "cpu"` and `alignment_method == "icp"` with message `"device='{device}' is incompatible with alignment_method='icp': Open3D ICP requires CPU-resident tensors (explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"`

**DataFactory (eval/data_factory.py):**
- `__init__` guard (D-03): after `self.config = config`, checks `config.device.startswith("cuda")` → `torch.cuda.is_available()` and `config.device == "mps"` → `torch.backends.mps.is_available()`. Raises `RuntimeError` with descriptive message when unavailable.
- 6 substitutions: `device="cpu"` → `device=self.config.device` in `load_real()` (2 sites), `load_target()` (2 sites), `get_ground_truth()` (2 sites)
- INFO logging after `_standardize()` in `load_real()` and `load_target()` (not in `get_ground_truth()` per D-07)

**Tests:**
- `TestEvalConfigDevice` (9 tests): default cpu, all valid values, yaml round-trip, invalid values
- `TestEvalConfigDeviceICPGuard` (4 tests): cuda+icp fails, mps+icp fails, cuda+cpd succeeds, cpu+icp succeeds
- `TestDataFactoryDeviceGuard` (5 tests): cuda/cuda:0/mps unavailable raises RuntimeError, cpu skips guard, cuda available constructs
- 4 existing dispatch assertions updated from `device="cpu"` literal to `device=cfg.device`

## Verification Results

```
grep -n "def validate_device" eval/config.py     → line 436 (1 match)
grep -n "model_validator" eval/config.py          → line 14 (import) + line 461 (decorator)
grep -c "device=self.config.device" eval/data_factory.py → 6
grep -n "torch.cuda.is_available" eval/data_factory.py   → line 131
grep -n "logging.info" eval/data_factory.py               → lines 185, 254
python -m pytest tests/test_eval_config.py tests/test_data_factory.py -x -q → 150 passed
python -m pytest -x -q → 1330 passed, 19 skipped, 1 xpassed
```

## TDD Gate Compliance

1. RED gate: commit `61846d0` — `test(53-01): add failing tests for EvalConfig device field and DataFactory device guard`
2. GREEN gate: commits `67bfb00` (EvalConfig) + `8b34b5d` (DataFactory)
3. REFACTOR: not needed — implementations were clean on first pass

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary surface introduced. The `device` field is validated at parse time (whitelist) and at runtime (availability check), mitigating T-53-01 and T-53-03 as specified in the plan's threat register.

## Self-Check: PASSED
