# Phase 31: Synthetic Pipeline Mode — Transform-Spec Target Generation & GT-Aware HPO - Pattern Map

**Mapped:** 2026-06-13
**Files analyzed:** 5 (4 modified + 1 new)
**Analogs found:** 5 / 5

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `eval/config.py` | config | request-response | `eval/config.py` lines 146–148 (existing field additions) | exact |
| `eval/data_factory.py` | service | CRUD | `eval/data_factory.py` — `load_target()`, `get_ground_truth()`, `augment()` | exact |
| `eval/runners/eval_runner.py` | runner | request-response | `eval/runners/eval_runner.py` lines 189–195 (existing `pipeline_mode` branch) | exact |
| `eval/runners/optimizer.py` | runner | request-response | `eval/runners/optimizer.py` lines 332–410 (`_objective()` + `_tier_dataset()`) | exact |
| `configs/synthetic_mode.yaml` | config | — | `configs/paired_alignment.yaml` | exact |

---

## Pattern Assignments

### `eval/config.py` (config, field addition)

**Analog:** `eval/config.py` itself — the three most recently added optional fields at lines 146–148.

**Existing optional field pattern** (`eval/config.py` lines 146–148):
```python
label_names: dict[int, str] | None = None
pipeline_mode: Literal["paired", "synthetic"] = "paired"
target_data_path: str | None = None
```

**Constraint:** `model_config = ConfigDict(extra="forbid")` at line 117 — any YAML key not declared here raises `EvalConfigError`. The new field must be appended after `target_data_path` (line 148) following the `dict | None = None` pattern.

**Addition to make** (append after line 148):
```python
transform_spec: dict | None = None
```

No validator. Pydantic v2 accepts `dict | None` without schema validation of internal keys. Validation occurs at use-time in `generate_target()`.

**Docstring pattern** (follow the `target_data_path` docstring at lines 103–107):
```
transform_spec : dict or None
    Transform specification for synthetic mode.  Required (non-None) when
    ``pipeline_mode='synthetic'`` and ``DataFactory.generate_target()`` is
    invoked; ignored otherwise.  ``ValueError`` is raised at ``generate_target()``
    call time, not at ``EvalConfig`` construction.  Default ``None``.
```

---

### `eval/data_factory.py` (service, CRUD — three additions)

**Analog:** `eval/data_factory.py` itself — `__init__` state pattern, `load_target()` error-at-use-time pattern, `get_ground_truth()` per-frame dict return pattern, `augment()` dispatch pattern.

#### 1. `__init__` — new instance state (analog: lines 79–81)

**Existing pattern** (`eval/data_factory.py` lines 79–81):
```python
self._real_dataset: dict[int, zRegPointCloud] | None = None
self._synthetic_dataset: dict[int, zRegPointCloud] | None = None
self._target_dataset: dict[int, zRegPointCloud] | None = None
```

**Addition** (append after line 81, same pattern):
```python
self._synthetic_target: dict[int, zRegPointCloud] | None = None
self._source_dataset: dict[int, zRegPointCloud] | None = None
self._transform_spec: dict | None = None
```

#### 2. `generate_target(dataset, transform_spec)` — new public method

**Analog:** `augment()` (lines 204–281) for dispatch pattern; `load_target()` (lines 119–170) for method signature and docstring style.

**Import order note:** `math` is already imported at line 15. No new imports needed.

**Error-at-use-time guard** (modeled on `load_target()` lines 151–155):
```python
def load_target(self) -> dict[int, zRegPointCloud]:
    ...
    if self.config.target_data_path is None:
        raise EvalConfigError(
            "DataFactory.load_target: target_data_path is required in paired mode "
            "but was None. Set target_data_path in the YAML config."
        )
```

**generate_target() implementation pattern:**
```python
def generate_target(
    self,
    dataset: dict[int, zRegPointCloud],
    transform_spec: dict,
) -> dict[int, zRegPointCloud]:
    if not transform_spec:
        raise ValueError(
            "DataFactory.generate_target: transform_spec must be non-empty; "
            "an empty spec would return source == target (no-op, Pitfall 1)."
        )
    # Strip "type" discriminator — augment() does not consume it
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    if not augment_params:
        raise ValueError(
            "DataFactory.generate_target: transform_spec contains only 'type' key; "
            "no augmentation parameters remain after stripping 'type'."
        )
    # D-02: temporary override, restored by finally (Pitfall 2)
    original_params = self.config.augmentation_params
    self.config.augmentation_params = augment_params
    try:
        result = self.augment(dataset)
    finally:
        self.config.augmentation_params = original_params
    # D-03: store instance state
    self._synthetic_target = result
    self._source_dataset = dataset
    self._transform_spec = transform_spec
    return result
```

**Key note from RESEARCH.md:** `EvalConfig` is NOT frozen (no `frozen=True` in `ConfigDict` — verified `eval/config.py:117`). Direct attribute assignment `self.config.augmentation_params = augment_params` is valid and simpler than `model_copy`. The `try/finally` is mandatory (Pitfall 2: exception between assignment and restore would corrupt config permanently).

#### 3. `get_synthetic_ground_truth()` — new public method

**Analog:** `get_ground_truth()` (lines 330–366) for per-frame dict return pattern and `torch.Tensor` return type.

**get_ground_truth() pattern to mirror** (`eval/data_factory.py` lines 360–366):
```python
if self.config.ground_truth_path is not None:
    if self.config.data_format == "tracklets":
        gt_ds, _ = load_data_from_tracklets(self.config.ground_truth_path, device="cpu")
    else:
        gt_ds = load_shah_from_csv(self.config.ground_truth_path, device="cpu")
    return {i: pc["id"] for i, pc in gt_ds.items()}
return {i: pc["id"] for i, pc in dataset.items()}
```

**get_synthetic_ground_truth() implementation pattern:**
```python
def get_synthetic_ground_truth(self) -> dict[int, torch.Tensor]:
    if self._synthetic_target is None:
        raise RuntimeError(
            "DataFactory.get_synthetic_ground_truth: generate_target() must be "
            "called before get_synthetic_ground_truth()."
        )
    source = self._source_dataset
    result: dict[int, torch.Tensor] = {}
    for k, pc in source.items():
        if pc["id"] is not None:
            result[k] = pc["id"].to(torch.long)   # D-06: always torch.long
        else:
            result[k] = torch.arange(pc["pos"].shape[0], dtype=torch.long)  # D-05: ordinal
    return result
```

**torch.arange pattern** (already used in `eval/data_factory.py` — same import `torch` is present at line 37).

---

### `eval/runners/eval_runner.py` (runner, request-response — synthetic branch)

**Analog:** `eval/runners/eval_runner.py` lines 189–195 — the existing `pipeline_mode` branch stub that Phase 31 replaces.

**Current code to replace** (lines 189–194):
```python
if self.config.pipeline_mode == "paired":
    target = self.factory.load_target()
else:
    # 'synthetic' mode: target generation is wired in Phase 31.
    # For now use source as a no-op placeholder (CR-03).
    target = source
```

**Replacement pattern** (else branch only):
```python
else:  # pipeline_mode == "synthetic"
    target = self.factory.generate_target(source, self.config.transform_spec)
```

**`_run_single()` GT injection — line 330:**

Current code (line 330):
```python
gt = self.factory.get_ground_truth(source)  # {frame_key: id_tensor}
```

Phase 31 adds a `pipeline_mode` branch here. Pattern from the `pipeline_mode` branch at lines 189–195 (same `if/else` structure):
```python
if self.config.pipeline_mode == "synthetic":
    gt = self.factory.get_synthetic_ground_truth()
else:
    gt = self.factory.get_ground_truth(source)
```

**No new imports needed** — `DataFactory` is already imported at line 56; `get_synthetic_ground_truth()` is a method on the existing `self.factory` instance.

---

### `eval/runners/optimizer.py` (runner, request-response — `_objective()` and `_tier_dataset()`)

**Analog:** `eval/runners/optimizer.py` itself — `_objective()` lines 332–410, `_tier_dataset()` lines 430–442.

#### 1. `_objective()` — tier_target selection (lines 334–337, D-10/D-11)

**Existing paired-mode structure** (lines 334–337):
```python
if tier_name == "sanity":
    tier_target = tier_dataset  # Pitfall 7(a) — sanity reuses same dataset
else:
    tier_target = self._factory.load_target()  # Pitfall 7(b) — dev/full call load_target
```

**Phase 31 replacement** — wrap existing block in pipeline_mode check:
```python
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # D-11: apply transform locally — NOT via self._factory.generate_target()
        # to avoid overwriting _synthetic_target (Pitfall 3)
        tier_target = _apply_transform_to_dataset(
            tier_dataset, self.config.transform_spec, self.config
        )
    else:  # dev / full
        # D-10: use pre-computed _synthetic_target, sliced to tier keys (Pitfall 7)
        tier_target = {
            k: self._factory._synthetic_target[k]
            for k in tier_dataset
            if k in self._factory._synthetic_target
        }
else:  # paired mode — existing code preserved
    if tier_name == "sanity":
        tier_target = tier_dataset
    else:
        tier_target = self._factory.load_target()
```

#### 2. `_objective()` — y_true / GT key block (lines 359–369, D-08/D-09)

**Existing CR-04 block** (lines 359–369):
```python
# CR-04: sanity tier generates labels into pc["color"] (via generate_labels),
# not pc["id"]. get_ground_truth() always reads pc["id"] which is None for
# synthetic data — causing silent all-zero scores. Detect the right field
# directly instead of delegating to get_ground_truth().
sample_pc = tier_dataset[source_sorted_keys[0]]
gt_key = "id" if sample_pc["id"] is not None else "color"
y_true = tier_dataset[source_sorted_keys[-1]][gt_key]
if y_true is None:
    raise ValueError(
        f"No ground-truth labels in field '{gt_key}' for sanity tier dataset."
    )
```

**Phase 31 replacement** — wrap in pipeline_mode check:
```python
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # Sanity toy dataset has labels in pc["color"] (generate_labels contract, Pitfall 5)
        # get_synthetic_ground_truth() reads _source_dataset which is the real dataset —
        # not the toy dataset. Use pc["color"] directly from the toy tier_dataset.
        sample_pc = tier_dataset[source_sorted_keys[0]]
        if sample_pc["color"] is not None:
            y_true = tier_dataset[source_sorted_keys[-1]]["color"]
        else:
            n = tier_dataset[source_sorted_keys[-1]]["pos"].shape[0]
            y_true = torch.arange(n, dtype=torch.long)
    else:  # dev / full in synthetic mode — D-09
        y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
else:  # paired mode — existing CR-04 block preserved
    sample_pc = tier_dataset[source_sorted_keys[0]]
    gt_key = "id" if sample_pc["id"] is not None else "color"
    y_true = tier_dataset[source_sorted_keys[-1]][gt_key]
    if y_true is None:
        raise ValueError(
            f"No ground-truth labels in field '{gt_key}' for sanity tier dataset."
        )
```

#### 3. `_tier_dataset()` — no change required for synthetic mode

The sanity tier in `_tier_dataset()` (lines 430–433) still generates the same toy dataset regardless of pipeline_mode:
```python
if tier == "sanity":
    traj = generate_trajectory(n_points=50, n_frames=3, seed=42)
    return generate_labels(traj, n_classes=4, seed=42)
```

The transform is applied inside `_objective()` (not inside `_tier_dataset()`) via `_apply_transform_to_dataset()`. No change to `_tier_dataset()`.

#### 4. Module-level helper — `_apply_transform_to_dataset()` (D-11)

Add as a module-level function (not a method) above the `HyperparamOptimizer` class. Pattern: same save/restore as `generate_target()` but on a scratch `DataFactory`.

**Scratch DataFactory pattern** (mirrors `__init__` at line 143: `self._factory = DataFactory(config)`):
```python
def _apply_transform_to_dataset(
    dataset: dict,
    transform_spec: dict,
    config: "EvalConfig",
) -> dict:
    """Apply transform_spec to dataset using a scratch DataFactory (D-11).

    Does NOT touch the caller's DataFactory instance — avoids overwriting
    _synthetic_target on the main factory (Pitfall 3).
    """
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    scratch_cfg = config.model_copy(update={"augmentation_params": augment_params})
    scratch_factory = DataFactory(scratch_cfg)
    return scratch_factory.augment(dataset)
```

**Import note:** `DataFactory` is already imported at line 90 (`from eval.data_factory import DataFactory`). `EvalConfig` is at line 89. No new imports.

**Attribute access note (Pitfall 6):** CONTEXT.md references `self._config` but the optimizer uses `self.config` (no underscore — verified at `eval/runners/optimizer.py:141`). Always use `self.config.pipeline_mode`.

---

### `configs/synthetic_mode.yaml` (config file — new)

**Analog:** `configs/paired_alignment.yaml` (full file, lines 1–26).

**paired_alignment.yaml structure** (`configs/paired_alignment.yaml`):
```yaml
# Phase 30 D-03 smoke-test scenario config: paired alignment using Kobitski tracklets
# as BOTH source (data_path) and target (target_data_path).
#
# ...
data_path: data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets
target_data_path: data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets
data_format: tracklets
pipeline_mode: paired
tier: sanity
n_trials: 3
n_synthetic: 0
run_alignment: true
run_label_transfer: false
default_params:
  window_size: 10
  step: 1
  cpd_penalty: null
  dtw_dist_fn: euclidean
  n_breakpoints: 5
output_dir: experiments/runs/paired_alignment
```

**synthetic_mode.yaml pattern** — key differences from analog:
- `pipeline_mode: synthetic`
- `transform_spec:` block replaces `target_data_path` (no `target_data_path` in synthetic mode)
- `output_dir: experiments/runs/synthetic_mode`
- Same `data_path`, `data_format`, `tier`, `n_trials`, `run_alignment`, `run_label_transfer` structure

**Concrete output:**
```yaml
# Phase 31 synthetic pipeline mode scenario config.
# Generates a transformed target from the source via augment() dispatch.
# target_data_path is not required — DataFactory.generate_target() is called instead.
#
# Run with: python run_eval.py --config configs/synthetic_mode.yaml --mode eval
data_path: data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets
data_format: tracklets
pipeline_mode: synthetic
transform_spec:
  type: rigid
  rotation_deg: 30.0
  rotation_axis: [0, 0, 1]
tier: sanity
n_trials: 3
n_synthetic: 0
run_alignment: true
run_label_transfer: false
default_params:
  window_size: 10
  step: 1
  cpd_penalty: null
  dtw_dist_fn: euclidean
  n_breakpoints: 5
output_dir: experiments/runs/synthetic_mode
```

---

## Shared Patterns

### Error-at-Use-Time Pattern
**Source:** `eval/data_factory.py` lines 151–155 (`load_target()` EvalConfigError guard)
**Apply to:** `generate_target()` (raise `ValueError` when `transform_spec` is None or empty), `get_synthetic_ground_truth()` (raise `RuntimeError` when `_synthetic_target is None`)
```python
if self.config.target_data_path is None:
    raise EvalConfigError(
        "DataFactory.load_target: target_data_path is required in paired mode "
        "but was None. Set target_data_path in the YAML config."
    )
```

### pipeline_mode Branch Pattern
**Source:** `eval/runners/eval_runner.py` lines 189–194 (existing `if pipeline_mode == "paired"` / `else`)
**Apply to:** `eval_runner.py` lines 189–195, `optimizer.py` `_objective()` lines 332–368
```python
if self.config.pipeline_mode == "paired":
    target = self.factory.load_target()
else:
    # synthetic mode
    target = ...
```

### try/finally Config-Restore Pattern
**Source:** RESEARCH.md §generate_target() Implementation Pattern (no existing analog — new pattern for this phase)
**Apply to:** `DataFactory.generate_target()` augmentation_params temp-override
```python
original_params = self.config.augmentation_params
self.config.augmentation_params = augment_params
try:
    result = self.augment(dataset)
finally:
    self.config.augmentation_params = original_params
```

### Per-Frame Dict Return Pattern
**Source:** `eval/data_factory.py` lines 364–366 (`get_ground_truth()` return)
**Apply to:** `get_synthetic_ground_truth()` return value
```python
return {i: pc["id"] for i, pc in dataset.items()}
```

### model_copy Pattern (frozen pydantic model update)
**Source:** `eval/runners/eval_runner.py` lines 242–244
**Apply to:** `_apply_transform_to_dataset()` scratch config construction
```python
scratch_cfg = config.model_copy(update={"augmentation_params": augment_params})
```

### Test — `@patch("eval.data_factory.*")` + mock dataset fixture
**Source:** `tests/test_data_factory.py` lines 200–242 (`TestLoadTarget`)
**Apply to:** `TestGenerateTarget`, `TestGetSyntheticGroundTruth`
```python
def _make_mock_ds(self):
    return {0: zRegPointCloud(pos=torch.zeros(3, 3), color=None, id=torch.arange(3))}

def test_dispatches_tracklets(self):
    cfg = EvalConfig(data_path="x.mat", target_data_path="y.mat", data_format="tracklets")
    factory = DataFactory(cfg)
    mock_ds = self._make_mock_ds()
    with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
        result = factory.load_target()
    m.assert_called_once_with("y.mat", device="cpu")
    assert result is mock_ds
```

### Test — `@patch("eval.runners.optimizer.DataFactory")` + `.side_effect` lambda
**Source:** `tests/test_optimizer.py` lines 102–115
**Apply to:** `TestOptimizerSyntheticMode`
```python
@patch("eval.runners.optimizer.DataFactory")
def test_sanity_tier_completes_under_two_minutes(self, mock_factory_cls, optimizer_config, synthetic_dataset):
    mock_factory = mock_factory_cls.return_value
    mock_factory.load_real.return_value = synthetic_dataset
    mock_factory.load_target.return_value = synthetic_dataset
    mock_factory.get_ground_truth.side_effect = (
        lambda ds: {k: ds[k]["color"] for k in ds}
    )
```

### Test — `EvalConfig.from_yaml` scenario config smoke test
**Source:** `tests/test_cli.py` lines 274–311 (`TestScenarioConfigs`)
**Apply to:** New test method in `TestScenarioConfigs` for `synthetic_mode.yaml`
```python
def test_synthetic_mode_yaml_loads_and_declares_synthetic_mode(self) -> None:
    cfg = EvalConfig.from_yaml(_REPO_ROOT / "configs" / "synthetic_mode.yaml")
    assert cfg.pipeline_mode == "synthetic"
    assert cfg.transform_spec is not None
    assert cfg.target_data_path is None
    assert cfg.run_alignment is True
```

---

## No Analog Found

None — all 5 files have direct analogs in the codebase.

---

## Critical Pitfalls for Planner

| # | Pitfall | Guard |
|---|---------|-------|
| 1 | `augment()` returns input by reference when `augment_params` is empty (no deep-copy) | Validate `augment_params` is non-empty before calling; raise `ValueError` if empty |
| 2 | `augmentation_params` mutated without `try/finally` — lost on exception | Wrap assignment + `augment()` call in `try/finally` that always restores |
| 3 | Sanity tier calls `self._factory.generate_target()` — overwrites `_synthetic_target` | Use `_apply_transform_to_dataset()` module-level helper with scratch `DataFactory` |
| 4 | CR-04 gt_key heuristic runs in synthetic mode — must be bypassed | Wrap entire CR-04 block in `if self.config.pipeline_mode != "synthetic":` |
| 5 | Sanity toy dataset has labels in `pc["color"]`, not `pc["id"]` | Sanity tier synthetic path reads `"color"`, dev/full reads `get_synthetic_ground_truth()` |
| 6 | CONTEXT.md uses `self._config` — optimizer uses `self.config` (no underscore) | Always use `self.config.pipeline_mode` (verified `optimizer.py:141`) |
| 7 | `_synthetic_target` keys are for full dataset; tier slice has subset of keys | Slice: `{k: self._factory._synthetic_target[k] for k in tier_dataset if k in ...}` |
| 8 | `transform_spec` not declared in `EvalConfig` → `extra="forbid"` rejects YAML | Add `transform_spec: dict | None = None` explicitly to `EvalConfig` |

---

## Metadata

**Analog search scope:** `eval/`, `tests/`, `configs/`
**Files scanned:** 7 (eval/config.py, eval/data_factory.py, eval/runners/eval_runner.py, eval/runners/optimizer.py, configs/paired_alignment.yaml, tests/test_data_factory.py, tests/test_optimizer.py, tests/test_eval_runner.py, tests/test_cli.py)
**Pattern extraction date:** 2026-06-13
