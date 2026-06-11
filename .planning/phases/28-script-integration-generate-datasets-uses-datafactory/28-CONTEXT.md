# Phase 28: Script Integration — generate_datasets uses DataFactory - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor `scripts/generate_datasets.py` to import `DataFactory` from `eval.data_factory` and replace the local `_augment_scaling`, `_augment_dropout`, `apply_augmentation`, and `_augment_noise` functions with `DataFactory.augment()` calls. Geometry helpers (`in_bowl`, `sample_ball_shell`, `sample_bowl_frame`, `sample_bowl_shell`, `_make_trajectory`) and CSV I/O (`_frame_to_df`, `save_as_csv`) remain unchanged. The script's external behaviour (generated files, output paths) is preserved; only the augmentation implementation is replaced.

</domain>

<decisions>
## Implementation Decisions

### DataFactory wiring

- **D-01:** Use `DataFactory(EvalConfig(data_path="", augmentation_params={...})).augment(dataset)` per augmentation call. `data_path=""` is safe — DataFactory performs no I/O on construction (D-08 in Phase 17). One `EvalConfig` + `DataFactory` object is created per call (9 calls total across the augmentation grid). The overhead is negligible.
- **D-02:** All three augmentation types (noise, scaling, dropout) route through `DataFactory.augment()` with the appropriate single-key `augmentation_params` dict: `{"sigma": value}` for noise, `{"scale_factor": value}` for scaling, `{"dropout_fraction": value}` for dropout. This uses the canonical DF-02 dispatch path.
- **D-03:** `apply_augmentation` dispatcher, `_augment_scaling`, `_augment_dropout`, and `_augment_noise` are all removed from the script. No local augmentation logic remains.

### Dropout RNG parity

- **D-04:** The dropout RNG changes from NumPy (`np.random.default_rng(42)` → `rng.choice()`) to PyTorch (`torch.manual_seed(42)` → `torch.randperm()`). This produces different random subsets, so **dropout CSV outputs will not be bit-identical** to the pre-refactor version. This divergence is accepted. Success criterion 3 ("bit-identical output") is relaxed to: scaling and noise outputs are bit-identical; dropout outputs differ in which specific points are removed but retain the same statistical properties (same fraction removed, same seed intent, fully reproducible).
- **D-05:** The plan must document the RNG divergence explicitly so reviewers understand dropout CSV content changes are expected, not a bug.

### Import order

- **D-06:** The script's existing import order guard (`# zreg imports must precede torch`) must be respected. Add `from eval.data_factory import DataFactory` and `from eval.config import EvalConfig` after the existing `zreg` imports and before `import torch`. The `eval/data_factory.py` module itself handles the libomp SIGABRT workaround internally, but the calling script must not import `torch` before `eval.data_factory`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Script under refactor
- `scripts/generate_datasets.py` — the file being modified. Functions to remove: `_augment_noise`, `_augment_scaling`, `_augment_dropout`, `apply_augmentation`. Functions to keep: `in_bowl`, `sample_ball_shell`, `sample_bowl_frame`, `sample_bowl_shell`, `_make_trajectory`, `_frame_to_df`, `save_as_csv`. Entry point (`generate_fully_synthetic`, `generate_semi_synthetic`, `__main__`) stays structurally unchanged.

### DataFactory (augmentation target)
- `eval/data_factory.py` — `DataFactory.augment()` at line ~150: 6-step dispatch via `config.augmentation_params`. Keys used in this phase: `"sigma"` (noise), `"scale_factor"` (scaling), `"dropout_fraction"` (dropout). Also: `DataFactory.__init__` at line ~74 — no I/O on construction (D-08).
- `eval/config.py` — `EvalConfig`: only `data_path` is required; all other fields default. `augmentation_params` defaults to `{}`. `extra="forbid"` means no unknown keys allowed.

### Requirements
- `REQUIREMENTS.md` §DF-02 — Script Integration requirement. The refactored script must import DataFactory and use it for scaling, dropout, and noise; `_augment_scaling`, `_augment_dropout`, and `apply_augmentation` must be removed; output must be as-specified above.

### Prior phase (augmentation methods delivered)
- `.planning/phases/27-datafactory-geometric-augmentation-methods/` — Phase 27 delivered `scale()`, `rotate()`, `drop_points()`, `sample_new_points()` as DataFactory instance methods and extended `augment()` with 6-step dispatch. DF-01 validated.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/data_factory.py:DataFactory.augment()` — canonical dispatch method. Accepts `dataset: dict[int, zRegPointCloud]`, reads `self.config.augmentation_params`. Returns augmented dataset; input never mutated.
- `eval/config.py:EvalConfig` — pydantic v2 BaseModel. Minimal construction: `EvalConfig(data_path="", augmentation_params={"key": value})`. All 14 other fields have defaults.

### Established Patterns
- `DataFactory` construction is cheap (no I/O). Creating one per call in a tight loop (9 augmentations across the grid) is the intended lightweight use per D-08.
- The libomp SIGABRT workaround requires `zreg.*` imports before `torch`. `eval.data_factory` repeats this guard internally; the script must honour the same order.

### Integration Points
- `generate_semi_synthetic()` calls `apply_augmentation()` on line ~389. This is the call site that switches to `DataFactory(EvalConfig(...)).augment(dataset)`.
- The augmentation grid (`AUGMENTATION_GRID`) keys must map to `augmentation_params` dict keys: `"noise"` → `"sigma"`, `"scaling"` → `"scale_factor"`, `"dropout"` → `"dropout_fraction"`.

</code_context>

<specifics>
## Specific Ideas

- The ROADMAP explicitly notes `_augment_noise` as: "replaced by a direct `DataFactory.augment(dataset, {"sigma": value})` call (or equivalent)". Decision D-01/D-02 satisfies this via the minimal-EvalConfig pattern.
- The AUGMENTATION_GRID key names (`"noise"`, `"scaling"`, `"dropout"`) and `param_name` values (`"sigma"`, `"factor"`, `"fraction"`) do not match the `augmentation_params` keys DataFactory expects. The plan must include a mapping: `"factor"` → `"scale_factor"`, `"fraction"` → `"dropout_fraction"`. `"sigma"` already matches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 28-script-integration-generate-datasets-uses-datafactory*
*Context gathered: 2026-06-11*
