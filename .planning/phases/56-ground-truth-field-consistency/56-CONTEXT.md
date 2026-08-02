# Phase 56: Ground-Truth Field Consistency - Context

**Gathered:** 2026-08-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Label-transfer F1 scoring is computed against the correct ground-truth field (`pc["label"]`, the field `LabelTransferStage` actually transfers), correct per-point correspondence is preserved even when `transform_spec`'s `dropout_fraction`/`n_new_points` change point counts between source and target, and existing ground-truth configs are audited/corrected for the new convention. This phase fixes the existing GT mechanism; it does not add new GT-generation capabilities (that's Phase 57).

</domain>

<decisions>
## Implementation Decisions

### GT Field Convention
- **D-01:** Add `ground_truth_field: Literal["id", "label"] = "label"` to `EvalConfig`. `DataFactory.get_ground_truth()` and `get_synthetic_ground_truth()` (`eval/data_factory.py`) read whichever field this names, defaulting to `"label"` — matching `generate_labels()`'s documented intent ("directly compatible with `zreg.metrics.compute_f1` without conversion") and what `LabelTransferStage` (`eval/stages/label_transfer.py:434`) actually transfers.
- **D-02:** This is a default-plus-override, not a fully free-form field name — the only two valid values are `"id"` and `"label"` (the only integer-label-shaped fields on `zRegPointCloud`). No dataset identified during this discussion needs the override (see D-05/D-06 below), but the escape hatch stays in the design per explicit user request.

### Dropout/New-Points Correspondence Tracking
- **D-03:** Full correspondence-tracking fix (not a detect-and-guard fallback). Thread the actual retained/appended original-index map through `DataFactory.drop_points()` and `DataFactory.sample_new_points()` (`eval/data_factory.py:918-1023`) so `get_synthetic_ground_truth()` can gather `y_true` correctly by tracked index instead of `eval_runner._run_single`'s current `min(len(y_true), len(y_pred))` positional truncation (`eval/runners/eval_runner.py:346-349`).
- **D-04:** This correspondence-tracking mechanism is deliberately built to be reusable by Phase 57 (subsample-pair generation has the same "index subset with tracked correspondence" need) — do not build something Phase 56-specific that Phase 57 has to redo.

### Shah/Kobitski Config Audit
- **D-05:** Shah's real cell-type classes live in `pc["label"]` (populated from the CSV `layer` column by `load_shah_from_csv`), **not** `pc["id"]` — confirmed directly by the user, correcting a stale/wrong comment in `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` (lines 16-18: "...unlike Shah's 'id' which carries real small-integer class labels") that claims Shah's `id` carries real classes. (Corrected citation — an earlier draft of this document mis-attributed this comment to `configs/experiments/stage1_alignment/rigid_cpd/synthetic.yaml`, which does not contain it; verified directly.) That comment must be corrected during this phase's audit. No `ground_truth_field` override is needed for Shah — the new `"label"` default already reads the right field.
- **D-06:** Kobitski has no field suitable as label-transfer ground truth as a *source* — its `id` is a unique per-point tracking identity (not classes) and its `label` is continuous RGB-like noise. `run_label_transfer: false` in `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` is already correct and needs no change; the audit target IS this file's own header comment (lines 10-19), which needs correcting per D-05 (it currently frames the Shah/Kobitski contrast around `id`, which is backwards for Shah).
- **D-07 (environment constraint):** Neither Shah's CSV nor Kobitski's tracklets exist in this worktree (`data/` is fully gitignored, confirmed empty via `git ls-files data/`). The audit (GT-03) must proceed from code/config review and the user-confirmed facts in D-05/D-06 — it cannot include an empirical re-verification pass against the real files from this environment. The fully-synthetic ball/bowl CSVs (`data/synthetic/fully_synthetic/*.csv`), by contrast, are regeneratable locally via `scripts/generate_datasets.py` (self-contained, not real external data) and CAN be empirically checked/fixed if their degenerate placeholder `label` (constant `1`, per `scripts/generate_datasets.py:239`) needs addressing as part of this phase's config audit.

### Claude's Discretion
- Exact implementation shape of the correspondence-tracking data structure (e.g. index tensor returned alongside the dataset dict, vs. an attribute stored on `DataFactory`) — left to the planner/executor, constrained only by D-04's reusability requirement.
- Whether the fully-synthetic ball/bowl `stage2_label_transfer/*/synthetic.yaml` configs get a real label source (e.g. wiring in `generate_labels()`) as part of this phase's audit, or are left flagged as still-degenerate with a documented reason — planner's call, informed by whether it fits phase scope without creeping into Phase 57's generation-mechanism territory.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Ground-truth extraction (the bug)
- `eval/data_factory.py` (`get_ground_truth()`, `get_synthetic_ground_truth()`) — currently reads `pc["id"]`; must be changed per D-01
- `eval/stages/label_transfer.py` (around line 432) — `LabelTransferStage.run()` transfers `pc["label"]`; this is the field GT extraction must match
- `src/zreg/data_generation/labels.py` (`generate_labels()`) — docstring establishes `label` as the canonical F1-compatible field

### Correspondence tracking (the truncation bug)
- `eval/runners/eval_runner.py` (`_run_single`, lines ~327-349) — current positional `min(len)` truncation (WR-01) to replace
- `eval/data_factory.py` (`drop_points()`, `sample_new_points()`, lines ~918-1023) — augmentation functions that need to expose retained/appended index correspondence

### Existing configs to audit (GT-03)
- `baseline_experiments/configs/ground_truth/shah_sample1.yaml` — real-data, both stages enabled; should work correctly once D-01 lands (label was always correct here per D-05)
- `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` — real-data, alignment-only; header comment (lines 10-19) contains the stale "Shah's id carries real classes" claim to correct per D-05/D-06
- `configs/experiments/stage2_label_transfer/*/synthetic.yaml` (knn, hybrid_knn_cpd, pointnet2, cpd_weighted, egnn) — fully-synthetic ball/bowl configs; degenerate `label` placeholder per D-07, audit scope left to planner discretion
- `scripts/generate_datasets.py` (line ~239) — source of the fully-synthetic `label` placeholder, relevant if the planner chooses to fix it during this phase's audit

No external specs/ADRs beyond the above code files — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DataFactory.augment()`'s existing dispatch pattern (`eval/data_factory.py:492-580`) already deep-copies and threads `pos`/`label`/`id`/`fps-idx` through every transform step — the correspondence-tracking fix extends this established pattern rather than replacing it.
- `MetricsEngine.sanity_check()` (`eval/metrics.py`) already has degenerate-label detection (all-same, all-sentinel) — a natural place to extend if a defensive check for mismatched GT sources is wanted, though D-03's fix should make this less necessary for the dropout case specifically.

### Established Patterns
- Error-at-use-time pattern (D-05/D-08 elsewhere in the codebase, e.g. `target_data_path`, `transform_spec` validation) — any new `ground_truth_field` validation should follow this convention: valid at `EvalConfig` construction (it's a `Literal`, pydantic validates automatically), consumed/used at `get_ground_truth()`/`get_synthetic_ground_truth()` call time.

### Integration Points
- `eval/runners/eval_runner.py::_run_single` is the single call site that assembles `y_true`/`y_pred` for `compute_f1` — both D-01 (field read) and D-03 (correspondence) changes converge here.

</code_context>

<specifics>
## Specific Ideas

No specific UI/behavioral requests beyond the decisions above — this is an internal correctness fix, not user-facing.

</specifics>

<deferred>
## Deferred Ideas

- **Kobitski as label-transfer target (not source):** User noted Kobitski, while unsuited as a label-transfer *source* (no real class field), could serve as a *target* in a heterogeneous paired label-transfer scenario (receiving labels transferred from a real-labeled source like Shah). This is a new capability/config, not an audit fix — belongs in a future phase, potentially building on the existing `HETERO-01` heterogeneous paired evaluation support (Phase 32) rather than this phase's scope.

</deferred>

---

*Phase: 56-ground-truth-field-consistency*
*Context gathered: 2026-08-01*
