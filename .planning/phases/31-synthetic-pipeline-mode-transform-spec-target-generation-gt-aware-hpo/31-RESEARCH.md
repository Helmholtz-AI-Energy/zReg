# Phase 31: Synthetic Pipeline Mode — Research

**Researched:** 2026-06-13
**Domain:** DataFactory augmentation dispatch, EvalConfig pydantic extension, EvaluationRunner mode branching, HyperparamOptimizer GT-aware objective
**Confidence:** HIGH — all findings derived from direct codebase reads

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** `generate_target(dataset, transform_spec)` delegates to `augment()` internally — reuses existing dispatch and deep-copy semantics. No new generator call paths needed.

**D-02:** `transform_spec` is passed to `augment()` via a temporary override of `self.config.augmentation_params`. The original value is saved and restored after the call. No persistent config mutation.

**D-03:** After calling `augment()`, store results as instance state: `self._synthetic_target = result`, `self._source_dataset = dataset`, `self._transform_spec = transform_spec`. The RuntimeError guard in `get_synthetic_ground_truth()` checks `if self._synthetic_target is None`.

**D-04:** For rigid/affine/noise transforms, correspondence is identity — `source[k][i]` maps to `target[k][i]`. `get_synthetic_ground_truth()` returns `{k: source[k]["id"].to(torch.long) for k in source}` per frame.

**D-05:** When `source[k]["id"]` is `None`, fall back to ordinal indices: `torch.arange(n_points, dtype=torch.long)`. Encodes identity correspondence positionally without requiring actual cell labels.

**D-06:** GT tensor dtype is always `torch.long`, regardless of source `pc["id"]` dtype.

**D-07:** Store `self._transform_spec` as metadata on the DataFactory instance after `generate_target()` is called. Already set as part of D-03.

**D-08:** `_objective()` detects synthetic mode by checking `self._config.pipeline_mode == "synthetic"`.

**D-09:** In synthetic mode, `y_true = self._factory.get_synthetic_ground_truth()[tier_sorted_keys[-1]]` — uses the last frame of the tier slice, mirroring paired-mode behavior.

**D-10:** Dev/full tiers in synthetic mode use `self._factory._synthetic_target` (sliced to tier keys) as `tier_target` instead of calling `load_target()`.

**D-11:** The sanity tier in synthetic mode also applies the transform to its toy dataset WITHOUT calling `generate_target()` on the DataFactory — to avoid overwriting `_synthetic_target` from the real source dataset. Must be isolated via a local helper or scratch DataFactory.

### Claude's Discretion

None declared in CONTEXT.md.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MODE-02 | `EvalConfig` gains `transform_spec: dict | None = None`; `DataFactory.generate_target(dataset, transform_spec)` applies transformation via `augment()` dispatch; `EvaluationRunner` in synthetic mode calls `generate_target()` instead of `load_target()` | EvalConfig pydantic model with `extra="forbid"` — new field must be explicitly declared. `augment()` dispatch table fully understood. `EvaluationRunner.run()` line 189-195 has the placeholder comment. |
| MODE-03 | `DataFactory.get_synthetic_ground_truth()` returns per-frame cell-identity labels; `HyperparamOptimizer._objective()` uses GT F1 as calibration signal in synthetic mode; `synthetic_mode.yaml` scenario config | GT tensor semantics (D-04/D-05/D-06) fully documented. `_objective()` lines 292-405 fully read. Sanity tier isolation (D-11) understood. |
</phase_requirements>

---

## Summary

Phase 31 wires in `pipeline_mode = "synthetic"` end-to-end. The work touches four existing files (`eval/config.py`, `eval/data_factory.py`, `eval/runners/eval_runner.py`, `eval/runners/optimizer.py`) and adds one new file (`configs/synthetic_mode.yaml`). Every design decision was locked in discussion (CONTEXT.md), so this phase is entirely about correct implementation of the locked decisions — not exploration.

The central pattern is: `generate_target()` reuses `augment()` by temporarily swapping `self.config.augmentation_params`, then restoring it. This is the only risky mutability concern. `augment()` reads `self.config.augmentation_params` once at the top of its body (`params = self.config.augmentation_params`), so the temporary override works exactly as D-02 describes. The save/restore must bracket the `augment()` call synchronously — no async exposure.

The optimizer's sanity tier adds the hardest constraint: it must apply the transform to its local toy dataset WITHOUT invoking `DataFactory.generate_target()` to avoid overwriting `_synthetic_target`. The cleanest implementation is a module-level or static `_apply_transform_spec(dataset, transform_spec)` helper that performs the same temp-override-augment pattern on an ad-hoc `DataFactory` instance (or directly on a scratch `EvalConfig` + `DataFactory`). Using a scratch `DataFactory` rather than `self._factory` is the safe choice.

**Primary recommendation:** Implement in three plans — Plan 31-01: `EvalConfig.transform_spec` + `DataFactory.generate_target()` + `DataFactory.get_synthetic_ground_truth()` + tests; Plan 31-02: `EvaluationRunner.run()` synthetic branch + `HyperparamOptimizer._objective()` + `_tier_dataset()` synthetic mode + tests; Plan 31-03: `synthetic_mode.yaml` + scenario smoke test.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `transform_spec` field declaration | Config layer (`eval/config.py`) | — | All EvalConfig fields live here; `extra="forbid"` requires explicit declaration |
| Synthetic target generation | Data layer (`eval/data_factory.py`) | — | DataFactory owns all data production; `augment()` is already here |
| GT label extraction | Data layer (`eval/data_factory.py`) | — | GT derives from the stored `_source_dataset`; DataFactory owns the source |
| Synthetic branch dispatch in run() | Runner layer (`eval/runners/eval_runner.py`) | — | EvaluationRunner.run() already branches on `pipeline_mode` at line 189 |
| GT-aware objective in HPO | Optimizer layer (`eval/runners/optimizer.py`) | — | `_objective()` owns the F1 computation; sanity tier has its own isolated path |
| Scenario config | Config files (`configs/`) | — | Following the pattern of `paired_alignment.yaml` |

---

## Standard Stack

### Core (all already installed, no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.2 [VERIFIED: existing code] | EvalConfig field declaration | Already the config model |
| torch | existing install | GT tensor construction (`torch.arange`, `.to(torch.long)`) | Already the tensor library |
| zreg.generators | existing install | `apply_rigid`, `add_gaussian_noise` used by `augment()` dispatch | Already imported in `data_factory.py` |

No new packages. Phase 31 is a pure code change with no additional dependencies. [VERIFIED: codebase inspection]

---

## Architecture Patterns

### Existing `augment()` Dispatch (fully understood)

`augment()` reads `self.config.augmentation_params` at `params = self.config.augmentation_params` (line 244 of `eval/data_factory.py`) and then dispatches on string keys in a fixed order:

```
sigma → n_outliers → scale_factor → rotation_deg → dropout_fraction → n_new_points
```

Missing keys are silently skipped. Empty dict is a no-op (returns input by reference, not a copy). [VERIFIED: codebase read, eval/data_factory.py:204-281]

**Critical: `augment()` does NOT deep-copy when params is empty.** It only deep-copies via the generator functions it calls (e.g., `add_gaussian_noise` calls `copy.deepcopy`). If `transform_spec` maps only to noise or scale or rotation, the deep-copy is guaranteed because those call `add_gaussian_noise` or `apply_rigid` (both deep-copy). If the transform_spec were somehow empty (no-op), `augment()` returns the input by reference — `generate_target()` must document that it requires a non-empty `transform_spec`.

### `transform_spec` → `augment()` Key Mapping

The `type` key in `transform_spec` is a discriminator for humans/YAML only — `augment()` does not recognise it. Caller must strip `type` before passing to the temp override:

```python
# transform_spec = {"type": "rigid", "rotation_deg": 45.0, "rotation_axis": [0, 0, 1]}
# augment_params  = {"rotation_deg": 45.0, "rotation_axis": [0, 0, 1]}

# transform_spec = {"type": "noise", "sigma": 0.1}
# augment_params = {"sigma": 0.1}
```

[VERIFIED: eval/data_factory.py:244-281 — keys checked are "sigma", "n_outliers", "scale", "scale_factor", "rotation_deg", "rotation_axis", "dropout_fraction", "n_new_points"]

### `generate_target()` Implementation Pattern

```python
def generate_target(
    self,
    dataset: dict[int, zRegPointCloud],
    transform_spec: dict,
) -> dict[int, zRegPointCloud]:
    # Strip "type" discriminator key before passing to augment()
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    # D-02: temporary override of self.config.augmentation_params
    original_params = self.config.augmentation_params
    self.config = self.config.model_copy(update={"augmentation_params": augment_params})
    try:
        result = self.augment(dataset)
    finally:
        self.config = self.config.model_copy(update={"augmentation_params": original_params})
    # D-03: store instance state
    self._synthetic_target = result
    self._source_dataset = dataset
    self._transform_spec = transform_spec
    return result
```

**Critical pitfall — EvalConfig is a pydantic BaseModel, not a frozen dataclass.** It is NOT frozen (no `frozen=True` in `ConfigDict`). Direct assignment `self.config.augmentation_params = augment_params` would mutate the live config. Using `model_copy(update=...)` is the safe immutable-update pattern. However, `model_copy` creates a NEW EvalConfig object — `self.config` reference must be replaced, and restored via `finally`. [VERIFIED: eval/config.py:117 — `ConfigDict(extra="forbid")`, no `frozen=True`]

**Alternative simpler approach:** Since `augmentation_params` is a plain dict and EvalConfig is mutable, direct assignment `self.config.augmentation_params = augment_params` works and the `try/finally` restores it. This avoids creating intermediate EvalConfig objects. Both approaches are correct; the simpler direct-assignment pattern matches how the existing code treats `self.config` (by reference, mutable). The planner may choose either — the critical constraint is that the `try/finally` guarantee restoration on exception.

### `get_synthetic_ground_truth()` Implementation Pattern

```python
def get_synthetic_ground_truth(self) -> dict[int, torch.Tensor]:
    if self._synthetic_target is None:
        raise RuntimeError(
            "DataFactory.get_synthetic_ground_truth: generate_target() must be called first."
        )
    source = self._source_dataset
    result = {}
    for k, pc in source.items():
        if pc["id"] is not None:
            result[k] = pc["id"].to(torch.long)   # D-06: always torch.long
        else:
            result[k] = torch.arange(pc["pos"].shape[0], dtype=torch.long)  # D-05: ordinal fallback
    return result
```

[VERIFIED: D-04, D-05, D-06 from CONTEXT.md; `pc["id"]` field existence verified in eval/data_factory.py:360-366]

### `__init__` Instance State Pattern

Existing pattern (line 69-81 of `eval/data_factory.py`):
```python
self._real_dataset: dict[int, zRegPointCloud] | None = None
self._synthetic_dataset: dict[int, zRegPointCloud] | None = None
self._target_dataset: dict[int, zRegPointCloud] | None = None
```

New state additions (D-03):
```python
self._synthetic_target: dict[int, zRegPointCloud] | None = None
self._source_dataset: dict[int, zRegPointCloud] | None = None
self._transform_spec: dict | None = None
```

[VERIFIED: eval/data_factory.py:69-81]

### `EvaluationRunner.run()` Synthetic Branch

Current code (lines 189-195):
```python
if self.config.pipeline_mode == "paired":
    target = self.factory.load_target()
else:
    # 'synthetic' mode: target generation is wired in Phase 31.
    # For now use source as a no-op placeholder (CR-03).
    target = source
```

Phase 31 replacement for the `else` branch:
```python
else:  # pipeline_mode == "synthetic"
    target = self.factory.generate_target(source, self.config.transform_spec)
```

The GT extraction for `EvaluationRunner` is separate: `_run_single()` calls `self.factory.get_ground_truth(source)` at line 330. In synthetic mode, this should instead use `self.factory.get_synthetic_ground_truth()`. The `get_ground_truth()` call in `_run_single()` is the hook point. [VERIFIED: eval/runners/eval_runner.py:189-195, 329-331]

**Design choice for _run_single:** The planner needs to decide whether `_run_single()` detects `pipeline_mode` and conditionally calls `get_synthetic_ground_truth()`, or whether `EvaluationRunner.run()` passes the GT dict as a parameter to `_run_single()`. The current signature is `_run_single(self, source, target, params)`. Adding a `gt` parameter is a clean option. Alternatively, `_run_single` can check `self.config.pipeline_mode` directly. Both work; the `_run_single` signature change may affect test mocks.

### `_objective()` Synthetic Mode Changes

Current paired-mode structure (lines 332-368 of `eval/runners/optimizer.py`):
```python
if tier_name == "sanity":
    tier_target = tier_dataset
else:
    tier_target = self._factory.load_target()
...
# GT detection (CR-04):
sample_pc = tier_dataset[source_sorted_keys[0]]
gt_key = "id" if sample_pc["id"] is not None else "color"
y_true = tier_dataset[source_sorted_keys[-1]][gt_key]
```

Phase 31 additions (D-08, D-09, D-10, D-11):

1. **tier_target selection** — in synthetic mode, sanity tier must apply transform locally (NOT via `self._factory.generate_target()`); dev/full use `self._factory._synthetic_target` sliced to tier keys.

2. **y_true source** — in synthetic mode, skip the `gt_key` heuristic entirely; use `self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]`.

3. **Sanity tier transform isolation** — the sanity tier's toy dataset is 3 frames of `generate_trajectory` → `generate_labels`. In synthetic mode, a transformed version is needed as `tier_target`. This must NOT call `self._factory.generate_target()` (D-11). The planner must implement this as a local helper. The simplest approach: extract `_apply_transform_spec(dataset, transform_spec, config)` as a module-level or static method that creates a scratch `DataFactory(scratch_config)` and calls `augment()` on it. [VERIFIED: CONTEXT.md §Specific Ideas]

### Sanity Tier Isolation Helper

The sanity tier in synthetic mode needs a transformed toy dataset without touching `self._factory._synthetic_target`. Proposed pattern:

```python
@staticmethod
def _apply_transform_to_dataset(
    dataset: dict,
    transform_spec: dict,
    config: EvalConfig,
) -> dict:
    """Apply transform_spec to dataset using a scratch DataFactory (D-11)."""
    from eval.data_factory import DataFactory
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    scratch_cfg = config.model_copy(update={"augmentation_params": augment_params})
    scratch_factory = DataFactory(scratch_cfg)
    return scratch_factory.augment(dataset)
```

This avoids mutating `self._factory` state. Alternatively, a simpler inline approach directly reassigns `augmentation_params` on a scratch config without creating a static method. The exact factoring is planner discretion.

### `_config` attribute in optimizer

The optimizer uses `self.config` (not `self._config` despite the CONTEXT.md reference at D-08). Line 141: `self.config = config`. The CONTEXT.md D-08 text references `self._config.pipeline_mode` but this should be `self.config.pipeline_mode`. [VERIFIED: eval/runners/optimizer.py:141 — `self.config = config`]

### Scenario Config Structure

`configs/paired_alignment.yaml` is the template:

```yaml
data_path: <path-to-real-data>
data_format: tracklets
pipeline_mode: synthetic
transform_spec:
  type: rigid
  rotation_deg: 30.0
  rotation_axis: [0, 0, 1]
tier: sanity
n_trials: 3
run_alignment: true
run_label_transfer: false
search_space:
  window_size: [3, 5]
output_dir: experiments/runs/synthetic_mode
```

`target_data_path` is NOT required in synthetic mode (EvalConfigError is raised at `load_target()` call time — in synthetic mode, `load_target()` is never called). [VERIFIED: eval/data_factory.py:151-155, CONTEXT.md canonical refs]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Applying transform to dataset | Custom matrix multiplication | `augment()` dispatch | Already handles all 7 key types with deep-copy guarantees |
| Rotation from degrees+axis | Rodrigues' formula from scratch | Already in `augment()` lines 261-274 | Tested and correct |
| Gaussian noise application | Custom `torch.randn` loop | `add_gaussian_noise()` via augment's `sigma` key | Handles seed, deep-copy |
| Deep copy of trajectory | Manual frame copying | Guaranteed by generator functions called inside `augment()` | `apply_rigid` calls `_apply_matrix` which calls `copy.deepcopy`; `add_gaussian_noise` calls `copy.deepcopy` |

---

## Common Pitfalls

### Pitfall 1: `augment()` Returns Input by Reference on No-Op

**What goes wrong:** If `transform_spec` strips `type` and produces an empty dict, `augment()` returns the input unchanged (no deep-copy). `generate_target()` would then store the original source as `_synthetic_target`, making source and target identical.

**Why it happens:** `augment()` line 245: `result = dataset` — no copy on empty params.

**How to avoid:** Validate `transform_spec` contains at least one recognisable augmentation key after stripping `type`. Raise `ValueError` if the resulting `augment_params` dict is empty.

**Warning signs:** Source and target point clouds compare equal after `generate_target()`.

### Pitfall 2: `self.config.augmentation_params` Mutation Without Restoration

**What goes wrong:** If `augment()` raises an exception after setting `self.config.augmentation_params = augment_params`, the original value is lost permanently. Subsequent calls to `augment()` use the transform_spec params instead of the config's intended params.

**Why it happens:** Direct attribute mutation without try/finally.

**How to avoid:** Wrap the `self.config.augmentation_params` assignment and `self.augment()` call in a `try/finally` block that always restores `self.config.augmentation_params = original_params`.

### Pitfall 3: `_objective()` Overwriting `_synthetic_target` in Sanity Tier

**What goes wrong:** If the sanity tier calls `self._factory.generate_target(toy_dataset, transform_spec)`, it overwrites `self._factory._synthetic_target`. Subsequent dev/full tiers that read `self._factory._synthetic_target` get the toy (sanity) target, not the real-data target.

**Why it happens:** D-11 constraint violated — sanity tier must NOT use the main factory's `generate_target()`.

**How to avoid:** Use a scratch DataFactory or direct `augment()` call on a scratch config for the sanity tier transform. [VERIFIED: CONTEXT.md D-11, §Specific Ideas]

### Pitfall 4: GT Key Heuristic in Synthetic Mode

**What goes wrong:** The existing CR-04 block (`gt_key = "id" if sample_pc["id"] is not None else "color"`) will correctly detect the field for the sanity toy dataset (which has `"color"` populated by `generate_labels`). But in synthetic mode this block should be bypassed entirely in favour of `get_synthetic_ground_truth()`.

**Why it happens:** The CR-04 block precedes any `pipeline_mode` branching in the current code.

**How to avoid:** Wrap the CR-04 block in `if self.config.pipeline_mode != "synthetic":` and add a synthetic branch that calls `get_synthetic_ground_truth()`.

### Pitfall 5: Sanity Tier Toy Dataset — Labels in `"color"`, Not `"id"`

**What goes wrong:** Sanity tier generates `generate_trajectory()` → `generate_labels()`. `generate_labels()` puts labels in `pc["color"]` (not `pc["id"]`). `get_synthetic_ground_truth()` reads `pc["id"]`, which is `None` for this toy dataset.

**Why it happens:** The existing sanity tier was built for the CR-04 `gt_key` heuristic, which handles both fields. In synthetic mode the GT path is different.

**How to avoid:** In the sanity tier's synthetic mode path, the GT is derived from the TOY dataset's `pc["color"]` (or positional index), NOT from `get_synthetic_ground_truth()`. `get_synthetic_ground_truth()` is only for the real source dataset stored in `_source_dataset`. The sanity tier computes GT locally: `y_true = tier_dataset[source_sorted_keys[-1]]["color"]` (falling back to `torch.arange(n)` if None). [VERIFIED: eval/runners/optimizer.py:430-433 — generate_labels puts labels in pc["color"]]

### Pitfall 6: `_config` vs `self.config` in optimizer

**What goes wrong:** CONTEXT.md references `self._config.pipeline_mode` but the optimizer uses `self.config` (no underscore). Using `self._config` will raise `AttributeError`.

**Why it happens:** CONTEXT.md uses pseudo-code notation, not exact attribute names.

**How to avoid:** Use `self.config.pipeline_mode` in the implementation.

### Pitfall 7: `_synthetic_target` Slicing for Dev/Full Tiers

**What goes wrong:** Dev/full tier datasets are sliced subsets of the real dataset (via `prepare_split()` or full `load_real()`). The stored `self._factory._synthetic_target` was generated from the full source — its keys must be sliced to match the tier dataset keys.

**How to avoid:** When using `self._factory._synthetic_target` as `tier_target` in dev/full synthetic mode, slice it: `{k: self._factory._synthetic_target[k] for k in tier_dataset.keys() if k in self._factory._synthetic_target}`. Verify key overlap is non-empty before proceeding.

### Pitfall 8: `EvalConfig.transform_spec` and `extra="forbid"`

**What goes wrong:** If `transform_spec` is not declared in `EvalConfig`, loading a YAML with `transform_spec:` raises `EvalConfigError` (extra field rejected by pydantic).

**Why it happens:** `model_config = ConfigDict(extra="forbid")` at `eval/config.py:117`.

**How to avoid:** Add `transform_spec: dict | None = None` explicitly to `EvalConfig`. No validator needed — pydantic v2 accepts `dict` without schema validation of the dict's internal keys. [VERIFIED: eval/config.py:117-148]

---

## Code Examples

### EvalConfig field addition

```python
# Source: eval/config.py, following pattern of augmentation_params / target_data_path
transform_spec: dict | None = None
```

No validator required. Pydantic v2 accepts `dict | None` without further constraints. The dict's internal key structure (`type`, `sigma`, `rotation_deg`, etc.) is validated at use-time in `generate_target()`. [VERIFIED: eval/config.py:124-148 existing dict fields]

### DataFactory `__init__` extension

```python
# Appended after existing None initialisations in __init__ (line 81)
self._synthetic_target: dict[int, zRegPointCloud] | None = None
self._source_dataset: dict[int, zRegPointCloud] | None = None
self._transform_spec: dict | None = None
```

[VERIFIED: eval/data_factory.py:69-81 — pattern]

### `_objective()` synthetic mode structure

```python
# After merged params computed, before align_result assignment:
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # D-11: apply transform locally, NOT via self._factory.generate_target()
        tier_target = _apply_transform_to_dataset(tier_dataset, self.config.transform_spec, self.config)
    else:  # dev / full
        # D-10: use pre-computed _synthetic_target, sliced to tier keys
        tier_target = {
            k: self._factory._synthetic_target[k]
            for k in tier_dataset
            if k in self._factory._synthetic_target
        }
else:  # paired mode (existing code)
    if tier_name == "sanity":
        tier_target = tier_dataset
    else:
        tier_target = self._factory.load_target()
```

Then for y_true:
```python
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # Sanity toy dataset has labels in pc["color"] (generate_labels contract)
        sample_pc = tier_dataset[source_sorted_keys[0]]
        gt_key = "color" if sample_pc["color"] is not None else None
        if gt_key is not None:
            y_true = tier_dataset[source_sorted_keys[-1]]["color"]
        else:
            n = tier_dataset[source_sorted_keys[-1]]["pos"].shape[0]
            y_true = torch.arange(n, dtype=torch.long)
    else:
        # D-09: use get_synthetic_ground_truth() for dev/full
        y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
else:
    # CR-04: existing heuristic for paired mode
    sample_pc = tier_dataset[source_sorted_keys[0]]
    gt_key = "id" if sample_pc["id"] is not None else "color"
    y_true = tier_dataset[source_sorted_keys[-1]][gt_key]
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | setup.cfg `[tool:pytest]` section |
| Quick run command | `python -m pytest tests/test_data_factory.py tests/test_eval_runner.py tests/test_optimizer.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODE-02 | `EvalConfig.transform_spec` field declared | unit | `pytest tests/test_data_factory.py::TestEvalConfigTransformSpec -x` | Wave 0 |
| MODE-02 | `generate_target()` returns distinct dict with no shared tensors | unit | `pytest tests/test_data_factory.py::TestGenerateTarget -x` | Wave 0 |
| MODE-02 | `generate_target()` restores `augmentation_params` after call | unit | `pytest tests/test_data_factory.py::TestGenerateTarget::test_config_restored_after_generate_target -x` | Wave 0 |
| MODE-02 | `EvaluationRunner.run()` in synthetic mode calls `generate_target()` | unit | `pytest tests/test_eval_runner.py::TestEvaluationRunnerSyntheticMode -x` | Wave 0 |
| MODE-03 | `get_synthetic_ground_truth()` returns correct GT tensors | unit | `pytest tests/test_data_factory.py::TestGetSyntheticGroundTruth -x` | Wave 0 |
| MODE-03 | `get_synthetic_ground_truth()` raises RuntimeError before `generate_target()` | unit | `pytest tests/test_data_factory.py::TestGetSyntheticGroundTruth::test_raises_before_generate_target -x` | Wave 0 |
| MODE-03 | `_objective()` uses GT F1 in synthetic mode | unit | `pytest tests/test_optimizer.py::TestOptimizerSyntheticMode -x` | Wave 0 |
| MODE-03 | `synthetic_mode.yaml` loads without error | unit | `pytest tests/test_cli.py::TestScenarioConfigs -x` | existing — new test method |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/test_data_factory.py tests/test_eval_runner.py tests/test_optimizer.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -q`
- **Phase gate:** Full suite green (839 + new tests) before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_data_factory.py::TestEvalConfigTransformSpec` — covers MODE-02 field declaration and YAML round-trip
- [ ] `tests/test_data_factory.py::TestGenerateTarget` — covers generate_target() immutability, augment dispatch, config restoration
- [ ] `tests/test_data_factory.py::TestGetSyntheticGroundTruth` — covers GT id/ordinal fallback, RuntimeError guard, torch.long dtype
- [ ] `tests/test_eval_runner.py::TestEvaluationRunnerSyntheticMode` — covers generate_target() called in synthetic mode
- [ ] `tests/test_optimizer.py::TestOptimizerSyntheticMode` — covers GT-aware y_true, sanity tier isolation

---

## Open Questions

1. **`_run_single()` GT injection mechanism**
   - What we know: `_run_single(source, target, params)` currently calls `self.factory.get_ground_truth(source)` at line 330.
   - What's unclear: Should `_run_single()` check `self.config.pipeline_mode` and branch, or should `run()` pass the GT dict as a parameter?
   - Recommendation: Add `gt_override: dict | None = None` as an optional parameter to `_run_single()`. In `run()`, for synthetic mode, call `get_synthetic_ground_truth()` and pass it in. This keeps the conditional out of `_run_single()`. However, adding a parameter changes the signature which may affect test mocks in `test_eval_runner.py`. The planner should audit existing `_run_single` call sites in tests before choosing approach.

2. **`_synthetic_target` access from `_objective()`**
   - What we know: D-10 states `self._factory._synthetic_target` is used in dev/full synthetic mode.
   - What's unclear: `_synthetic_target` is a "private" attribute (single underscore convention). Accessing it from `optimizer.py` is technically fine in Python but is cross-module private access.
   - Recommendation: Add a `@property synthetic_target` on DataFactory that returns `self._synthetic_target` (raising RuntimeError if None), making the access explicit. Or accept the direct underscore access as pragmatic for a research codebase.

3. **transform_spec validation at config load vs use time**
   - What we know: `extra="forbid"` blocks unknown YAML keys. CONTEXT.md D-05 (from Phase 30) establishes the error-at-use-time pattern.
   - What's unclear: Should `EvalConfig` validate that `transform_spec` is present when `pipeline_mode == "synthetic"`?
   - Recommendation: Follow the established error-at-use-time pattern. `generate_target()` raises `ValueError` if `transform_spec is None` when called. No `model_validator` needed.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 31 has no external dependencies beyond the already-installed project stack. All work is pure Python code changes to existing files.

---

## Sources

### Primary (HIGH confidence)
- `eval/data_factory.py` — augment() dispatch (lines 204-281), __init__ pattern (69-81), load_target pattern (119-170) [VERIFIED: direct read]
- `eval/config.py` — EvalConfig fields, extra="forbid", from_yaml (full file) [VERIFIED: direct read]
- `eval/runners/eval_runner.py` — run() pipeline_mode branch (189-195), _run_single GT extraction (329-331) [VERIFIED: direct read]
- `eval/runners/optimizer.py` — _objective() (292-410), _tier_dataset() (412-442), CR-04 gt_key block (359-368) [VERIFIED: direct read]
- `src/zreg/generators/transforms.py` — apply_rigid() deep-copy via _apply_matrix (92-108) [VERIFIED: direct read]
- `src/zreg/generators/corruption.py` — add_gaussian_noise() deep-copy (53) [VERIFIED: direct read]
- `src/zreg/generators/labels.py` — generate_labels() stores in pc["color"] (77) [VERIFIED: direct read]
- `tests/test_data_factory.py` — mock patterns (patch targets, fixture shapes) [VERIFIED: direct read]
- `tests/test_eval_runner.py` — DataFactory mock pattern (157-170) [VERIFIED: direct read]
- `tests/test_optimizer.py` — optimizer mock pattern (102-122) [VERIFIED: direct read]
- `.planning/phases/31-*/31-CONTEXT.md` — all locked decisions D-01 through D-11 [VERIFIED: direct read]
- `configs/paired_alignment.yaml` — template structure for synthetic_mode.yaml [VERIFIED: direct read]

### Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | EvalConfig is not frozen (no `frozen=True`), so direct attribute mutation is possible for the temp override | Architecture Patterns — generate_target() | If wrong, `model_copy` pattern is required instead of direct assignment; both are safe alternatives |

**One assumption.** Everything else verified by direct codebase reads.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, existing installed packages verified
- Architecture: HIGH — all integration points verified by reading exact line numbers
- Pitfalls: HIGH — derived from reading actual code, not generalised advice
- Test patterns: HIGH — test mocking patterns copied from existing test files

**Research date:** 2026-06-13
**Valid until:** Phase 31 implementation complete (code is stable; no external deps)
