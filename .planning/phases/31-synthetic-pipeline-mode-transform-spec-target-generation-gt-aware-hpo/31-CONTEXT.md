# Phase 31: Synthetic Pipeline Mode — Transform-Spec Target Generation & GT-Aware HPO - Context

**Gathered:** 2026-06-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `pipeline_mode = "synthetic"` end-to-end: `EvalConfig` gains `transform_spec: dict | None = None`; `DataFactory.generate_target(dataset, transform_spec)` applies the specified transform (rigid/noise via existing `augment()` dispatch) and returns a distinct target trajectory; `DataFactory.get_synthetic_ground_truth()` returns per-frame cell-identity labels derived from the known deterministic correspondence; `EvaluationRunner.run()` in synthetic mode calls `generate_target()` to produce the target instead of `load_target()`; `HyperparamOptimizer._objective()` uses GT F1 from `get_synthetic_ground_truth()` as the calibration signal when `pipeline_mode == "synthetic"`, including the sanity tier which also applies the transform to its toy dataset; add `configs/synthetic_mode.yaml` scenario config.

</domain>

<decisions>
## Implementation Decisions

### generate_target() implementation

- **D-01:** `generate_target(dataset, transform_spec)` delegates to `augment()` internally — reuses existing dispatch and deep-copy semantics. No new generator call paths needed.
- **D-02:** `transform_spec` is passed to `augment()` via a temporary override of `self.config.augmentation_params`. The original value is saved and restored after the call. No persistent config mutation.
- **D-03:** After calling `augment()`, store results as instance state: `self._synthetic_target = result`, `self._source_dataset = dataset`, `self._transform_spec = transform_spec`. The RuntimeError guard in `get_synthetic_ground_truth()` checks `if self._synthetic_target is None`.

### GT label semantics

- **D-04:** For rigid/affine/noise transforms, correspondence is identity — `source[k][i]` maps to `target[k][i]`. `get_synthetic_ground_truth()` returns `{k: source[k]["id"].to(torch.long) for k in source}` per frame.
- **D-05:** When `source[k]["id"]` is `None` (generated trajectory without cell labels), fall back to ordinal indices: `torch.arange(n_points, dtype=torch.long)`. Encodes the identity correspondence positionally without requiring actual cell labels.
- **D-06:** GT tensor dtype is always `torch.long`, regardless of source `pc["id"]` dtype. Consistent with `compute_f1` expectations.
- **D-07:** Store `self._transform_spec` as metadata on the DataFactory instance after `generate_target()` is called. `get_synthetic_ground_truth()` has it available for debugging/reproducibility; it is already set as part of D-03.

### _objective() synthetic mode

- **D-08:** `_objective()` detects synthetic mode by checking `self._config.pipeline_mode == "synthetic"` — direct, readable, consistent with how `EvaluationRunner.run()` already branches on `pipeline_mode`.
- **D-09:** In synthetic mode, `y_true = self._factory.get_synthetic_ground_truth()[tier_sorted_keys[-1]]` — uses the last frame of the tier slice, mirroring the paired-mode behavior. Only the source of `y_true` changes; the rest of the F1 computation is unchanged.
- **D-10:** Dev/full tiers in synthetic mode use `self._factory._synthetic_target` (sliced to tier keys) as `tier_target` instead of calling `load_target()`.
- **D-11:** The sanity tier in synthetic mode also applies the transform to its toy dataset (3-frame `generate_trajectory` output) to produce a local sanity target. This must be done WITHOUT calling `generate_target()` on the DataFactory (to avoid overwriting `_synthetic_target` from the real source dataset). The planner must isolate this transform application — e.g., via a local helper or by calling `augment()` directly with a temporary config override on a scratch DataFactory instance.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### DataFactory — primary change area

- `eval/data_factory.py` — Add `generate_target(dataset, transform_spec)` and `get_synthetic_ground_truth()`. Instance state to add: `_synthetic_target`, `_source_dataset`, `_transform_spec` (all initialised to `None` in `__init__`). `augment()` is the internal dispatch delegate for D-01/D-02.

### EvalConfig — field addition

- `eval/config.py` — Add `transform_spec: dict | None = None`. Note `extra="forbid"` is in effect — field must be explicitly declared. `pipeline_mode` already exists from Phase 30.

### EvaluationRunner — synthetic mode branch

- `eval/runners/eval_runner.py` — `run()` line 189–192 already has the paired-mode branch and a `# 'synthetic' mode: target generation is wired in Phase 31` comment. Wire the synthetic branch here: call `generate_target()`, store result, pass to `_run_single()`.

### HyperparamOptimizer — _objective() and _tier_dataset() changes

- `eval/runners/optimizer.py`:
  - `_objective()` lines 292–405: add `pipeline_mode == "synthetic"` branch for `y_true` source (D-08/D-09) and `tier_target` selection (D-10).
  - `_tier_dataset()` lines 412–442: extend sanity tier branch to handle synthetic mode (D-11).
  - Pitfall 7 comment (lines 332–337): update to reflect synthetic mode dispatch.
  - CR-04 comment (lines 359–368): update `gt_key` detection block — in synthetic mode, skip the `pc["id"]`/`pc["color"]` heuristic entirely.

### Scenario config

- `configs/synthetic_mode.yaml` — new file. `pipeline_mode: synthetic`. `transform_spec` with at least one of `{"type": "rigid", "rotation_deg": float, "rotation_axis": [x,y,z]}` or `{"type": "noise", "sigma": float}`. `target_data_path` not required in synthetic mode.

### Requirements

- `.planning/REQUIREMENTS.md` §MODE-02, §MODE-03 — authoritative requirement text for this phase.

### Prior phase context

- `.planning/phases/30-two-dataset-paired-alignment-architecture/30-CONTEXT.md` — D-05 (EvalConfig validation pattern: error at use-time, not model_validator), D-04 (LabelTransferStage sequential frame transfer), D-06 (AlignResult structure unchanged).
- `.planning/phases/27-datafactory-geometric-augmentation-methods/27-CONTEXT.md` — `augment()` dispatch design (Phase 27), rotation/scale/noise dispatch order.

### Existing generator functions (read-only reference)

- `src/zreg/generators/transforms.py` — `apply_rigid()`, `apply_affine()`
- `src/zreg/generators/corruption.py` — `add_gaussian_noise()`, `add_outliers()`
- `src/zreg/generators/generators.py` — `generate_trajectory()` (used by sanity tier)
- `src/zreg/generators/labels.py` — `generate_labels()` (used by sanity tier)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `DataFactory.augment()` (`eval/data_factory.py:204`): dispatch table for `sigma → add_gaussian_noise`, `rotation_deg/rotation_axis → self.rotate`, `scale_factor → self.scale`, `dropout_fraction → self.drop_points`, `n_new_points → self.sample_new_points`. `generate_target()` reuses this by temporarily overriding `self.config.augmentation_params` (D-02). augment() already deep-copies its input.
- `EvalConfig.pipeline_mode` (`eval/config.py:147`): already `Literal["paired", "synthetic"]` — no change needed, just read it in new branches.
- `_objective()` GT heuristic (`eval/runners/optimizer.py:363–368`): `gt_key = "id" if sample_pc["id"] is not None else "color"`. In synthetic mode this block is bypassed in favour of `get_synthetic_ground_truth()`.

### Established Patterns

- **Error at use-time (not at config load):** `EvalConfigError` is raised when the required field is missing at call time (e.g., `load_target()` raises when `target_data_path is None`). Same pattern for `get_synthetic_ground_truth()` RuntimeError when `_synthetic_target is None`.
- **Instance state initialised to None in `__init__`:** DataFactory currently initialises lazy-loaded data to `None`. `_synthetic_target`, `_source_dataset`, `_transform_spec` follow the same pattern.
- **Tier branching in `_objective()`:** currently `if tier_name == "sanity": tier_target = tier_dataset` / `else: tier_target = self._factory.load_target()`. In synthetic mode, the `else` branch changes to use `_synthetic_target`; sanity gets its own synthetic transform logic.

### Integration Points

- `EvaluationRunner.run()` line 192: synthetic branch wires in between existing paired-mode `load_target()` call and `_run_single()`.
- `HyperparamOptimizer.search()` line 189–213: tier loop calls `_tier_dataset()` and `make_objective()`. No signature changes needed — `pipeline_mode` is already readable via `self._config`.
- `configs/` directory: existing scenario configs (`paired_alignment.yaml`, `selfcal_kobitski.yaml`) serve as structural templates for `synthetic_mode.yaml`.

</code_context>

<specifics>
## Specific Ideas

- The sanity tier in synthetic mode must apply the transform to its own toy dataset locally — NOT via `DataFactory.generate_target()` — to avoid overwriting the real-data `_synthetic_target`. Planner should isolate this, e.g., by extracting a stateless `_apply_transform(dataset, transform_spec)` helper or using a scratch config override inline.
- `transform_spec` dict keys map directly to `augment()` dispatch keys: `{"type": "rigid", "rotation_deg": 45.0, "rotation_axis": [0, 0, 1]}` → `augment_params = {"rotation_deg": 45.0, "rotation_axis": [0, 0, 1]}`. `{"type": "noise", "sigma": 0.1}` → `{"sigma": 0.1}`. The `type` key itself is not consumed by `augment()` and should be stripped before the temp override.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 31-Synthetic Pipeline Mode — Transform-Spec Target Generation & GT-Aware HPO*
*Context gathered: 2026-06-13*
