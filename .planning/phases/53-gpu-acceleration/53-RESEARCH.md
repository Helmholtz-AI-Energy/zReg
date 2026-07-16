# Phase 53: GPU Acceleration - Research

**Researched:** 2026-07-15
**Domain:** Python/PyTorch device threading, Pydantic v2 validators, YAML configuration
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `EvalConfig.device` is a `str` field with default `"cpu"`. Accepted values: `{"cpu", "cuda", "cuda:0", "cuda:1", "mps"}`. Validated via `@field_validator` at EvalConfig construction (YAML load time), consistent with `alignment_method`, `label_transfer_method`, and `swd_variant` validators already in `eval/config.py`.
- **D-02:** Unknown device strings (e.g., `"gpu"`, `"cuda:7"`) are rejected immediately with a clear `EvalConfigError` message — not passed through to PyTorch where the error would be delayed and opaque.
- **D-03:** `DataFactory.__init__` checks device availability immediately after construction: if `config.device != "cpu"` and the requested backend is unavailable (`torch.cuda.is_available()` for `"cuda"` / `"cuda:N"`; `torch.backends.mps.is_available()` for `"mps"`), raise `RuntimeError` with a message naming the configured device and confirming the backend is not available. Fails fast before any disk I/O or MPI rank setup.
- **D-04:** `dataset.py`'s existing silent CPU fallback (lines 118-120) is never reached — DataFactory's `__init__` raise intercepts the unavailable-device case before any loader is called. Do not modify `dataset.py`.
- **D-05:** `EvalConfig` gains a `@model_validator(mode='after')` that raises `EvalConfigError` when `device != "cpu"` and `alignment_method == "icp"`. Error message: `"device='{device}' is incompatible with alignment_method='icp': Open3D ICP requires CPU-resident tensors (explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"`. Mirrors the existing `validate_swd_variant` cross-field pattern.
- **D-06:** ICP's CPU round-trips are an explicit architectural constraint of Open3D, not a silent fallback. No changes to `icp.py`. The model_validator documents the constraint at parse time rather than burying it in code comments.
- **D-07:** `DataFactory.load_real()` (and `load_target()`) logs the resolved device after the dataset is loaded: `logger.info("Loaded %s dataset on %s (%d frames)", 'source'/'target', self.config.device, len(dataset))`. Runs inside `srun` on compute nodes — appears in SLURM output for operator inspection.
- **D-08:** Cluster configs in `baseline_experiments/configs_horeka/` (all 7 files, plus smoke = 8 total files) are updated to include `device: "cuda"`.
- **D-09:** The existing `[VERIFY ON HOREKA]` convention (from Phase 51/52) is extended to a comment in `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch`: `# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors`.

### Claude's Discretion
- Logger name / level inside DataFactory (use existing logging setup in the file; `INFO` level)
- Whether to add `device` to the `__repr__` / `__str__` of DataFactory (probably not needed — the log line is sufficient)
- Exact wording of `EvalConfigError` messages (be specific: include the configured value and what was expected)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPU-01 | `EvalConfig` exposes a `device` field controlling where tensors are loaded and computed | D-01/D-02/D-05: `@field_validator` + `@model_validator` patterns confirmed in `eval/config.py`; `model_validator` not yet imported — needs adding |
| GPU-02 | `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU | D-03/D-04/D-07: 6 exact `device="cpu"` substitution points confirmed at lines 158, 161, 224, 227, 640, 642; `__init__` at line 117 is the guard insertion point; logging uses bare `logging` module (no named logger) |
| GPU-03 | Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback | D-07/D-08/D-09: 8 YAML files confirmed (no existing `device:` field); sbatch file confirmed; AlignmentStage already follows tensor device automatically at lines 441, 551, 557 |
</phase_requirements>

## Summary

Phase 53 adds GPU device threading through three layers: (1) `EvalConfig` gains a validated `device` field with an ICP-incompatibility guard, (2) `DataFactory` replaces 6 hardcoded `device="cpu"` calls and adds an early `__init__` guard for unavailable devices, and (3) all 8 cluster configs in `baseline_experiments/configs_horeka/` gain `device: "cuda"` plus the sbatch file gains a GPU-03 verification comment. No changes to `icp.py`, `dataset.py`, `AlignmentStage`, or CPD internals — the downstream stages already follow tensor device automatically.

The implementation is highly constrained by existing decisions and confirmed code patterns. All 6 substitution line numbers match exactly (lines 158, 161, 224, 227, 640, 642). The `@field_validator` pattern is confirmed in `eval/config.py` (3 existing examples: `validate_alignment_method`, `validate_swd_variant`, `validate_label_transfer_method`). The `@model_validator(mode='after')` is not yet used in config.py but the import only needs `model_validator` added to the existing pydantic import line. The only CONTEXT.md discrepancy discovered: there are 8 YAML files in `configs_horeka/` (including `smoke/kobitski_ew06_alignment.yaml`), not 7 as stated in D-08 — all 8 should receive `device: "cuda"`.

**Primary recommendation:** Implement in two plans — (1) config.py + data_factory.py code changes with tests, (2) YAML config updates + sbatch annotation.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Device string validation | Config layer (EvalConfig) | — | Parse-time validation consistent with existing `alignment_method` and `swd_variant` validators; fails fast before any I/O |
| Device availability check | Factory layer (DataFactory.__init__) | — | First point of I/O setup; fails before disk reads, before MPI rank work |
| Tensor device placement | Loader layer (load_data_from_tracklets / load_shah_from_csv) | — | Both functions already accept and honor `device` param; threading is pure pass-through |
| ICP+GPU incompatibility guard | Config layer (EvalConfig model_validator) | — | Parse-time enforcement so operators get the error before submitting a job, not mid-run |
| Downstream device propagation | AlignmentStage / CPD | — | Already follows input tensor device via `cloud["pos"].device` — no changes needed |
| Verification logging | Factory layer (DataFactory.load_real / load_target) | SLURM output | INFO log appears in srun stdout; operator greps SLURM log for confirmation |
| Cluster config activation | YAML configs (configs_horeka/*.yaml) | — | `device: "cuda"` in YAML triggers the entire pipeline without SLURM script changes |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic v2 | already installed | Field validation, model validators | Already used for all EvalConfig validators |
| torch | already installed | `torch.cuda.is_available()`, `torch.backends.mps.is_available()` | Standard availability checks |

No new packages to install. This phase is entirely within the existing dependency footprint.

### Supporting

No new supporting libraries.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `@field_validator` for device whitelist | `Literal["cpu","cuda","cuda:0","cuda:1","mps"]` | Literal gives compile-time check but produces a worse error message; `@field_validator` mirrors the existing `alignment_method` pattern and produces a human-readable error — consistent with codebase style |
| `@model_validator(mode='after')` for ICP guard | `@field_validator("device")` with info.data access | `@field_validator` for `device` runs before `alignment_method` is resolved in `info.data` depending on field ordering; `model_validator(mode='after')` runs after all fields are set — safer for cross-field logic |

## Package Legitimacy Audit

No new packages are installed in this phase. This section is not applicable.

## Architecture Patterns

### System Architecture Diagram

```
YAML config (device: "cuda")
        |
        v
EvalConfig.from_yaml()
  -> @field_validator("device")          [D-01/D-02: whitelist check]
  -> @model_validator(mode='after')      [D-05: ICP+GPU guard]
        |
        v
DataFactory.__init__(config)
  -> availability check                  [D-03: torch.cuda/mps.is_available()]
  -> raises RuntimeError if unavailable
        |
        v
DataFactory.load_real() / load_target() / get_ground_truth()
  -> load_data_from_tracklets(path, device=self.config.device)  [lines 158, 224, 640]
  -> load_shah_from_csv(path, device=self.config.device)        [lines 161, 227, 642]
  -> logger.info("Loaded %s dataset on %s (%d frames)", ...)    [D-07]
        |
        v
dict[int, zRegPointCloud] with pos tensors on config.device
        |
        v
AlignmentStage / CPD
  -> device=cloud["pos"].device  (line 441, 551, 557 — no changes needed)
  -> all downstream ops follow tensor device automatically
```

### Recommended Project Structure

No new directories or files beyond existing test files. Changes are:

```
eval/
├── config.py          # ADD: device field + @field_validator + @model_validator import
└── data_factory.py    # ADD: __init__ guard + 6 device="cpu" -> self.config.device + logger.info

baseline_experiments/
├── configs_horeka/
│   ├── baseline_no_hpo/ew06_vs_shah.yaml          # ADD: device: "cuda"
│   ├── baseline_with_selfcal/ew06_vs_shah.yaml     # ADD: device: "cuda"
│   ├── ground_truth/kobitski_ew06.yaml             # ADD: device: "cuda"
│   ├── ground_truth/shah_sample1.yaml              # ADD: device: "cuda"
│   ├── selfcal/kobitski_ew06_alignment.yaml        # ADD: device: "cuda"
│   ├── selfcal/shah_alignment.yaml                 # ADD: device: "cuda"
│   ├── selfcal/shah_label_transfer.yaml            # ADD: device: "cuda"
│   └── smoke/kobitski_ew06_alignment.yaml          # ADD: device: "cuda"
└── scripts/
    └── launch_horeka_multirank_test.sbatch         # ADD: GPU-03 VERIFY ON HOREKA comment
```

### Pattern 1: @field_validator for device whitelist (mirrors validate_alignment_method)

**What:** Single-field validator called at construction time, raises `ValueError` for invalid values (wrapped as `EvalConfigError` by `from_yaml`).
**When to use:** Single-field validation where the constraint is a whitelist of string values.

```python
# Source: eval/config.py lines 347-369 (validate_alignment_method — copy this pattern)
@field_validator("device")
@classmethod
def validate_device(cls, v: str) -> str:
    if v not in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        raise ValueError(
            f"device must be one of {{'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'}}; got {v!r}"
        )
    return v
```

### Pattern 2: @model_validator(mode='after') for cross-field guard (mirrors validate_swd_variant approach but stronger)

**What:** Post-construction validator that sees all fields already resolved. For the ICP guard, `device` and `alignment_method` are both set — a cross-field check is required.
**When to use:** When two fields interact and neither can be validated in isolation.

```python
# Source: Pydantic v2 docs; model_validator must be added to pydantic imports
# Current import line 14: from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
# Required: add model_validator to this import

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

@model_validator(mode='after')
def validate_device_icp_compat(self) -> "EvalConfig":
    if self.device != "cpu" and self.alignment_method == "icp":
        raise ValueError(
            f"device='{self.device}' is incompatible with alignment_method='icp': "
            "Open3D ICP requires CPU-resident tensors "
            "(explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"
        )
    return self
```

Note: `model_validator(mode='after')` raises `ValueError` — pydantic wraps this in `ValidationError`, which `from_yaml` then wraps in `EvalConfigError`. The existing `EvalConfigError` wrapping in `from_yaml` (lines 343-345) handles this automatically.

### Pattern 3: DataFactory.__init__ device availability check

**What:** Fail-fast check immediately after storing config, before any I/O, before any MPI rank setup.
**When to use:** Any time a resource (device, file, service) is required early and failing late is expensive.

```python
# Source: eval/data_factory.py __init__ (line 117); insert after self.config = config
def __init__(self, config: EvalConfig) -> None:
    self.config = config
    # D-03: fail fast if requested device is unavailable
    if config.device != "cpu":
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DataFactory: config.device={config.device!r} but "
                "torch.cuda.is_available() is False"
            )
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"DataFactory: config.device={config.device!r} but "
                "torch.backends.mps.is_available() is False"
            )
    self._real_dataset = ...
```

### Pattern 4: Logging in DataFactory (bare logging module, no named logger)

**What:** The existing data_factory.py uses bare `logging.debug(...)` (line 840) with no named logger instance. The D-07 INFO log should follow the same pattern for consistency.
**When to use:** Wherever a log call is added in data_factory.py.

```python
# Source: eval/data_factory.py line 840 (only existing log call)
# Use bare logging module (no getLogger — module doesn't have a named logger)
import logging  # already imported at line 15

# In load_real(), after dataset = self._standardize(dataset):
logging.info("Loaded %s dataset on %s (%d frames)", "source", self.config.device, len(dataset))

# In load_target(), same pattern with "target"
logging.info("Loaded %s dataset on %s (%d frames)", "target", self.config.device, len(dataset))
```

### Anti-Patterns to Avoid

- **Adding `device` validation inside `DataFactory.load_real()` or `load_target()`**: The guard belongs in `__init__` so it fires once before any I/O, not per-call.
- **Modifying `dataset.py` lines 118-120 (the silent CPU fallback)**: D-04 explicitly forbids this. The `DataFactory.__init__` guard makes those lines unreachable for all managed paths.
- **Using a `@field_validator("device")` with `info.data.get("alignment_method")` for the ICP guard**: `@field_validator` on `device` runs before `alignment_method` may be available in `info.data` (field ordering dependency); use `@model_validator(mode='after')` instead.
- **Raising `EvalConfigError` directly from `@model_validator`**: Raise `ValueError` inside the validator; pydantic converts it to `ValidationError`, which `from_yaml` converts to `EvalConfigError`. Raising `EvalConfigError` directly inside a pydantic validator bypasses pydantic's wrapping and produces inconsistent error handling.
- **Skipping the smoke config**: The smoke config at `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml` is one of the 8 files that needs `device: "cuda"`. CONTEXT.md D-08 says "all 7 files" but the actual count is 8 — include smoke.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Device string validation | Custom regex or manual isinstance check | `@field_validator` whitelist | Pydantic already wraps as EvalConfigError; consistent with 3 existing validators in same file |
| Cross-field validation | Guard inside `load_real()` at runtime | `@model_validator(mode='after')` | Parse-time error before any I/O; consistent with pydantic v2 patterns |
| Device availability detection | Custom os.environ parsing | `torch.cuda.is_available()` / `torch.backends.mps.is_available()` | Standard PyTorch API; already used in conftest.py line 30 |

**Key insight:** All building blocks already exist in this codebase. The entire phase is plumbing and configuration — no new algorithms, no new libraries.

## Common Pitfalls

### Pitfall 1: CONTEXT.md says 7 YAML files but there are 8

**What goes wrong:** Plan targets 7 files, misses `smoke/kobitski_ew06_alignment.yaml`, which then runs with CPU despite submitting to GPU nodes.
**Why it happens:** CONTEXT.md D-08 says "all 7 files" — the smoke config was added in Phase 52 for BUDG-04 and was not counted.
**How to avoid:** Use `find baseline_experiments/configs_horeka/ -name "*.yaml"` to enumerate targets — confirmed 8 files. All 8 must receive `device: "cuda"`.
**Warning signs:** If plan lists exactly 7 YAML edits, it is missing smoke config.

### Pitfall 2: @model_validator raises EvalConfigError instead of ValueError

**What goes wrong:** `EvalConfigError` raised inside a pydantic validator is not caught by pydantic's internal machinery; it escapes `from_yaml` unwrapped and bypasses the EvalConfigError catch in the CLI.
**Why it happens:** Confusing the validator's internal raise convention with the public API error type.
**How to avoid:** Inside `@model_validator` and `@field_validator`, always raise `ValueError`. Pydantic converts it to `ValidationError`. `from_yaml` converts `ValidationError` to `EvalConfigError`. The chain is: `ValueError` → `ValidationError` → `EvalConfigError`.
**Warning signs:** Tests that `pytest.raises(EvalConfigError)` against direct `EvalConfig(...)` construction fail unexpectedly.

### Pitfall 3: model_validator not added to pydantic import

**What goes wrong:** `NameError: name 'model_validator' is not defined` at runtime.
**Why it happens:** The current import line 14 does not include `model_validator`: `from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator`.
**How to avoid:** Add `model_validator` to the import. The full updated import: `from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator`.
**Warning signs:** Any test of the model_validator guard immediately raises NameError.

### Pitfall 4: DataFactory.__init__ guard uses wrong backend string matching

**What goes wrong:** `"cuda:0"` and `"cuda:1"` are not caught by `config.device == "cuda"` equality check.
**Why it happens:** The device string `"cuda:0"` is not equal to `"cuda"`.
**How to avoid:** Use `config.device.startswith("cuda")` for the CUDA backend check, which covers `"cuda"`, `"cuda:0"`, `"cuda:1"`.
**Warning signs:** Test with `device="cuda:0"` on a CPU-only machine passes `__init__` without error, then fails deep inside PyTorch with an obscure error.

### Pitfall 5: D-07 logging placed before self._standardize / self._subsample_to_max

**What goes wrong:** `len(dataset)` reflects the pre-subsample count, misleading the operator about actual frame count processed.
**Why it happens:** Inserting the log call at the wrong position in `load_real()` / `load_target()`.
**How to avoid:** Place `logging.info(...)` after `dataset = self._standardize(dataset)` (the last step in both methods), immediately before `self._real_dataset = dataset` (or equivalent).

### Pitfall 6: Test for DataFactory device check uses an actual torch.cuda.is_available() path

**What goes wrong:** Test that asserts `RuntimeError` is raised for `device="cuda"` passes on GPU machines and fails on CPU-only CI.
**Why it happens:** The test doesn't mock `torch.cuda.is_available()`.
**How to avoid:** Use `unittest.mock.patch("torch.cuda.is_available", return_value=False)` in the test to force the unavailable path regardless of actual hardware.

## Code Examples

### Adding device field to EvalConfig

```python
# Source: eval/config.py — insert near other string fields around line 280
# After label_transfer_method field:
device: str = Field(
    default="cpu",
    description="Compute device: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'"
)
```

### field_validator for device (copy validate_alignment_method pattern exactly)

```python
# Source: modeled on eval/config.py lines 347-369
@field_validator("device")
@classmethod
def validate_device(cls, v: str) -> str:
    if v not in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        raise ValueError(
            f"device must be one of {{'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'}}; got {v!r}"
        )
    return v
```

### model_validator for ICP guard (add after field_validators)

```python
# Source: Pydantic v2 model_validator(mode='after') — add model_validator to pydantic import first
@model_validator(mode='after')
def validate_device_icp_compat(self) -> "EvalConfig":
    if self.device != "cpu" and self.alignment_method == "icp":
        raise ValueError(
            f"device='{self.device}' is incompatible with alignment_method='icp': "
            "Open3D ICP requires CPU-resident tensors "
            "(explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"
        )
    return self
```

### DataFactory.__init__ device availability guard

```python
# Source: eval/data_factory.py — insert after self.config = config (line 126)
if config.device != "cpu":
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DataFactory: config.device={config.device!r} requested but "
            "torch.cuda.is_available() is False"
        )
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"DataFactory: config.device={config.device!r} requested but "
            "torch.backends.mps.is_available() is False"
        )
```

### Substitution in load_real (apply to load_target and get_ground_truth similarly)

```python
# Source: eval/data_factory.py lines 158, 161 (load_real)
# BEFORE:
dataset, _ = load_data_from_tracklets(self.config.data_path, device="cpu")
dataset = load_shah_from_csv(self.config.data_path, device="cpu")

# AFTER:
dataset, _ = load_data_from_tracklets(self.config.data_path, device=self.config.device)
dataset = load_shah_from_csv(self.config.data_path, device=self.config.device)

# Then, after _standardize() call, add D-07 logging:
logging.info("Loaded %s dataset on %s (%d frames)", "source", self.config.device, len(dataset))
```

### sbatch GPU-03 annotation

```bash
# Source: baseline_experiments/scripts/launch_horeka_multirank_test.sbatch
# Add after the existing verification commands block at the end of the file:

# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors
#   grep "Loaded source dataset on cuda" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out
```

### YAML device field addition (same for all 8 files)

```yaml
# Add to each baseline_experiments/configs_horeka/**/*.yaml
# Placement: top of file, after data_path/data_format block, before run_alignment
device: "cuda"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `@field_validator` for cross-field validation using `info.data` | `@model_validator(mode='after')` for cross-field validation | Pydantic v2 | `model_validator(mode='after')` is the canonical v2 way; `info.data` approach still works for fields validated earlier in declaration order but is fragile |

**Deprecated/outdated:**
- Pydantic v1 `@validator` decorator: replaced by `@field_validator` in pydantic v2. This codebase already uses v2 patterns throughout.

## Confirmed Code Details

### Exact line numbers verified against actual source

| Location | Line(s) | What | Confirmed |
|----------|---------|------|-----------|
| `eval/data_factory.py` | 158 | `load_data_from_tracklets(..., device="cpu")` in `load_real()` tracklets path | YES |
| `eval/data_factory.py` | 161 | `load_shah_from_csv(..., device="cpu")` in `load_real()` csv path | YES |
| `eval/data_factory.py` | 224 | `load_data_from_tracklets(..., device="cpu")` in `load_target()` tracklets path | YES |
| `eval/data_factory.py` | 227 | `load_shah_from_csv(..., device="cpu")` in `load_target()` csv path | YES |
| `eval/data_factory.py` | 640 | `load_data_from_tracklets(..., device="cpu")` in `get_ground_truth()` tracklets | YES |
| `eval/data_factory.py` | 642 | `load_shah_from_csv(..., device="cpu")` in `get_ground_truth()` csv path | YES |
| `eval/data_factory.py` | 117 | `DataFactory.__init__` — guard insertion point | YES |
| `src/zreg/registration/icp.py` | 114-115 | `.cpu().numpy()` round-trips for src/tgt normalization | YES |
| `src/zreg/registration/icp.py` | 196-197 | `.cpu().numpy()` round-trips in `_compute_composite_transform` | YES |
| `eval/config.py` | 14 | pydantic import — `model_validator` not yet present | YES |
| `eval/stages/alignment.py` | 441, 551, 557 | `device=cloud["pos"].device` — already device-following | YES |
| `src/zreg/dataset.py` | 118-120 | Silent CPU fallback — do NOT modify (D-04) | YES |

### configs_horeka YAML inventory (8 files, not 7)

| File | Currently has `device:` |
|------|------------------------|
| `baseline_no_hpo/ew06_vs_shah.yaml` | NO |
| `baseline_with_selfcal/ew06_vs_shah.yaml` | NO |
| `ground_truth/kobitski_ew06.yaml` | NO |
| `ground_truth/shah_sample1.yaml` | NO |
| `selfcal/kobitski_ew06_alignment.yaml` | NO |
| `selfcal/shah_alignment.yaml` | NO |
| `selfcal/shah_label_transfer.yaml` | NO |
| `smoke/kobitski_ew06_alignment.yaml` | NO |

### Existing test coverage relevant to this phase

| Test file | Tests to update/add new | Coverage needed |
|-----------|------------------------|-----------------|
| `tests/test_eval_config.py` | ADD: `TestEvalConfigDevice` class | `device` field default, valid values, invalid value raises, `cuda:0`/`mps` accepted |
| `tests/test_eval_config.py` | ADD: `TestEvalConfigDeviceICPGuard` class | `device="cuda" + alignment_method="icp"` raises; `device="cuda" + alignment_method="cpd"` succeeds |
| `tests/test_data_factory.py` | ADD: `TestDataFactoryDeviceGuard` class | CUDA unavailable raises RuntimeError (mocked); `device="cpu"` skips guard; MPS path |
| `tests/test_data_factory.py` | UPDATE: `TestLoadReal.test_dispatches_tracklets`, `test_dispatches_csv` | Change `device="cpu"` to `device=self.config.device` assertion |
| `tests/test_data_factory.py` | UPDATE: `TestLoadTarget.test_dispatches_tracklets`, `test_dispatches_csv` | Same update |
| `tests/test_data_factory.py` | ADD: device threading tests | `config.device="cuda"` threads to loader calls |

### Logging setup in data_factory.py

The file uses bare `logging` module (imported at line 15, no named logger instance). The only existing log call is `logging.debug(...)` at line 840. D-07 should use `logging.info(...)` consistently.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (confirmed, 1332 tests collected) |
| Config file | `pytest.ini` or `setup.cfg` (existing) |
| Quick run command | `python -m pytest tests/test_eval_config.py tests/test_data_factory.py -x -q` |
| Full suite command | `python -m pytest -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-01 | `EvalConfig.device` defaults to `"cpu"` | unit | `pytest tests/test_eval_config.py -k "test_device_default" -x` | ❌ Wave 0 |
| GPU-01 | Valid device strings accepted | unit | `pytest tests/test_eval_config.py -k "TestEvalConfigDevice" -x` | ❌ Wave 0 |
| GPU-01 | Invalid device string raises EvalConfigError | unit | `pytest tests/test_eval_config.py -k "test_device_invalid" -x` | ❌ Wave 0 |
| GPU-01 | ICP+GPU raises at config construction | unit | `pytest tests/test_eval_config.py -k "test_device_icp_guard" -x` | ❌ Wave 0 |
| GPU-02 | DataFactory.__init__ raises RuntimeError when CUDA unavailable | unit | `pytest tests/test_data_factory.py -k "TestDataFactoryDeviceGuard" -x` | ❌ Wave 0 |
| GPU-02 | `load_real()` passes `self.config.device` to loader | unit | `pytest tests/test_data_factory.py -k "test_dispatches_tracklets" -x` | ✅ (needs update) |
| GPU-02 | `load_target()` passes `self.config.device` to loader | unit | `pytest tests/test_data_factory.py -k "TestLoadTarget and test_dispatches" -x` | ✅ (needs update) |
| GPU-02 | `get_ground_truth()` passes `self.config.device` to loader | unit | `pytest tests/test_data_factory.py -k "test_ground_truth_device" -x` | ❌ Wave 0 |
| GPU-03 | YAML configs parseable with `device: "cuda"` field | unit | `pytest tests/test_eval_config.py -k "test_device_yaml_round_trip" -x` | ❌ Wave 0 |
| GPU-03 | `sbatch` comment — manual verification on HoreKa | manual | N/A | N/A |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_eval_config.py tests/test_data_factory.py -x -q`
- **Per wave merge:** `python -m pytest -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] New test class `TestEvalConfigDevice` in `tests/test_eval_config.py` — covers GPU-01 field defaults and validation
- [ ] New test class `TestEvalConfigDeviceICPGuard` in `tests/test_eval_config.py` — covers GPU-01 model_validator
- [ ] New test class `TestDataFactoryDeviceGuard` in `tests/test_data_factory.py` — covers GPU-02 __init__ guard (mock `torch.cuda.is_available`)
- [ ] New test(s) for `get_ground_truth()` device threading — covers GPU-02

## Security Domain

This phase introduces no authentication, session management, access control, cryptography, or user input beyond YAML config parsing. EvalConfig already uses `extra="forbid"` to reject unknown keys. The `device` field is a whitelist-validated string — injection is not a concern. Security domain is not applicable.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | All 8 YAML files should receive `device: "cuda"` (CONTEXT.md says 7) | Confirmed Code Details | Smoke config runs on CPU despite GPU allocation — no functional breakage but GPU-03 verification would fail on the smoke job |

**A1 is LOW risk** — verified by `find` command showing 8 files, all without an existing `device:` field. The smoke config is structurally identical to the selfcal config; adding `device: "cuda"` is correct and safe.

## Open Questions (RESOLVED)

1. **Should `baseline_with_selfcal` configs inherit `best_params.json` device assumptions?**
   - RESOLVED: Not a concern — `device` is not an HPO search parameter and not in `best_params.json`. No action needed.

2. **D-09 sbatch annotation target: only the test sbatch or also launch_horeka.sbatch?**
   - RESOLVED: Both `launch_horeka_multirank_test.sbatch` and `launch_horeka.sbatch` exist (confirmed). Plan 02 Task 2 annotates both files. `launch_horeka.sbatch` is the full-suite script operators submit for production runs — omitting it would leave GPU-03 verification guidance absent for the most common use case.

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies — all required libraries already installed)

## Sources

### Primary (HIGH confidence)
- `eval/config.py` — confirmed all 3 existing `@field_validator` patterns, confirmed `model_validator` not yet imported (line 14), confirmed `EvalConfigError` wrapping in `from_yaml`
- `eval/data_factory.py` — confirmed all 6 `device="cpu"` locations (lines 158, 161, 224, 227, 640, 642), confirmed `__init__` at line 117, confirmed bare `logging` module (no named logger)
- `src/zreg/registration/icp.py` — confirmed `.cpu().numpy()` round-trips at lines 114-115, 196-197
- `src/zreg/dataset.py` — confirmed `load_data_from_tracklets` and `load_shah_from_csv` both accept `device` param; confirmed silent fallback at lines 118-120
- `eval/stages/alignment.py` — confirmed device-following at lines 441, 551, 557
- `baseline_experiments/configs_horeka/` — confirmed 8 YAML files, none have existing `device:` field
- `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` — confirmed sbatch target for D-09
- `tests/test_data_factory.py` — confirmed existing tests assert `device="cpu"` in loader mocks (need updating)
- `tests/conftest.py` — confirmed `@pytest.mark.cuda` skip infrastructure at lines 88-94

### Secondary (MEDIUM confidence)
- Pydantic v2 documentation pattern for `@model_validator(mode='after')` — standard pattern, confirmed consistent with pydantic v2 BaseModel semantics used throughout codebase

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all existing
- Architecture: HIGH — all patterns verified directly in source files
- Pitfalls: HIGH — derived from direct code inspection, not speculation
- Line numbers: HIGH — grep-verified against actual source

**Research date:** 2026-07-15
**Valid until:** 2026-08-15 (stable codebase, no fast-moving dependencies)
