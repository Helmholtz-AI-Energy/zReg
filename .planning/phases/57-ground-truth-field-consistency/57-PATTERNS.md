# Phase 57: Ground-Truth Field Consistency - Pattern Map

**Mapped:** 2026-08-01
**Files analyzed:** 8 (2 core modified, 6 config-audit)
**Analogs found:** 8 / 8 (all self-referential — this phase modifies existing files, it does not create new ones)

**Scope note:** Unlike most phases, Phase 57 has NO new files — every file in
scope is an in-place edit to code/config that already exists. "Closest analog"
therefore means: the closest *sibling pattern within the same file or
codebase* that shows how to make the edit correctly, not a different file to
scaffold from. Confirmed via CONTEXT.md `<canonical_refs>` — no RESEARCH.md
exists for this phase.

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|----------------|------|-----------|-----------------|---------------|
| `eval/config.py` | config (pydantic model) | transform (validate-on-construct) | `pipeline_mode: Literal["paired", "synthetic"]` field, same file lines 275 | exact (same file, same pattern: 2-value `Literal` field with default, no custom validator needed) |
| `eval/data_factory.py` (`get_ground_truth`, `get_synthetic_ground_truth`) | service | CRUD (read/extract) | same methods, same file, lines 629-724 | exact (in-place field-name parameterization) |
| `eval/data_factory.py` (`drop_points`, `sample_new_points`) | service | transform (index-subset augmentation) | same methods, same file, lines 918-1023; index-threading precedent in `_subsample_to_max` (lines 885-916) and `AlignResult.estep_results` (`eval/types.py:102-137`) | exact (extend existing per-frame `idx` computation to also be exposed, not just consumed) |
| `eval/runners/eval_runner.py` (`_run_single`) | controller/orchestrator | request-response (assembles y_true/y_pred for metrics) | same method, same file, lines 327-349 (WR-01 truncation block) | exact (replace positional truncation with index-based gather once D-03 correspondence exists) |
| `baseline_experiments/configs/ground_truth/shah_sample1.yaml` | config (YAML) | n/a | itself — no code change expected, D-05 confirms `label` already correct here | exact (verification-only) |
| `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` | config (YAML, comments only) | n/a | itself — comment correction per D-06 | exact (comment edit only) |
| `configs/experiments/stage1_alignment/rigid_cpd/synthetic.yaml` | config (YAML, comments only) | n/a | itself — stale comment at ~line 19 per D-05 (comment not present in current file body; verify exact line at edit time) | exact (comment edit only) |
| `configs/experiments/stage2_label_transfer/{knn,hybrid_knn_cpd,pointnet2,cpd_weighted,egnn}/synthetic.yaml` | config (YAML) | n/a | `configs/experiments/stage2_label_transfer/knn/synthetic.yaml` (read in full below) as the template for the other 4 — they are structural siblings | exact (all 5 share the same `data_path: data/synthetic/fully_synthetic/ball_realistic_kobitski.csv` degenerate-label source) |
| `scripts/generate_datasets.py` (~line 239 `_make_trajectory`) | data-generation script | batch/transform | `generate_labels()` (`src/zreg/data_generation/labels.py:25-80`) — the canonical Voronoi label generator already used elsewhere in the codebase; NOT currently imported into `generate_datasets.py` | role-match (planner's discretion whether to wire this in per CONTEXT.md "Claude's Discretion") |

## Pattern Assignments

### `eval/config.py` — add `ground_truth_field: Literal["id", "label"] = "label"`

**Analog:** same file, `pipeline_mode` field (line 275) and `tier` field (line 256) — both are small closed-vocabulary `Literal` fields with a default, requiring **no** `@field_validator` (pydantic validates `Literal` membership automatically at construction). This is the correct pattern to copy — do NOT add a manual validator like `alignment_method`'s (that pattern is only used for plain `str` fields, not `Literal`).

**Field declaration pattern to copy** (`eval/config.py:275`):
```python
pipeline_mode: Literal["paired", "synthetic"] = "paired"
```
becomes, by direct analogy:
```python
ground_truth_field: Literal["id", "label"] = "label"
```

**Docstring pattern to copy** (`eval/config.py:180-183`, the `pipeline_mode` Attributes entry — same structure: name, type, one-paragraph purpose, default called out):
```python
    pipeline_mode : str
        Pipeline mode: ``'paired'`` loads ``target_data_path`` via
        ``DataFactory.load_target()``; ``'synthetic'`` is reserved for Phase 31
        and is currently a no-op.  Default ``'paired'``.
```

**Where NOT to copy from:** `label_transfer_method` (line 285-288) and `alignment_method` (line 280) use `Field(default=..., description=...)` + a separate `@field_validator` because they are plain `str` fields (not `Literal`) — this is more code than necessary for a 2-value closed field. `ground_truth_field` should use the simpler `Literal` pattern, consistent with `pipeline_mode`/`tier`/`search_strategy`/`data_format` dispatch fields elsewhere in the same class.

**`__all__` / class docstring integration:** add a new `Attributes` entry (alphabetically near `ground_truth_path`, the existing related field at line 118-120) documenting D-01/D-02's design: default `"label"` matches `generate_labels()`'s documented F1-compatible convention and what `LabelTransferStage` actually transfers; the only two valid values are `"id"` and `"label"` (D-02).

---

### `eval/data_factory.py` — `get_ground_truth()` / `get_synthetic_ground_truth()` field-name fix (D-01)

**Analog:** the methods themselves — this is a parameterization of an existing hardcoded string, not a structural rewrite.

**Current pattern to change** (`eval/data_factory.py:659-665`, `get_ground_truth`):
```python
        if self.config.ground_truth_path is not None:
            if self.config.data_format == "tracklets":
                gt_ds, _ = load_data_from_tracklets(self.config.ground_truth_path, device=self.config.device)
            else:
                gt_ds = load_shah_from_csv(self.config.ground_truth_path, device=self.config.device)
            return {i: pc["id"] for i, pc in gt_ds.items()}
        return {i: pc["id"] for i, pc in dataset.items()}
```
Both `pc["id"]` occurrences must become `pc[self.config.ground_truth_field]` (D-01). Docstring at lines 644-646 ("This method is intended for **real data**... Synthetic data flows should use `pc["label"]`...") needs updating too — it currently documents the OLD hardcoded split that D-01 supersedes.

**Current pattern to change** (`eval/data_factory.py:713-724`, `get_synthetic_ground_truth`):
```python
        result: dict[int, torch.Tensor] = {}
        for k, pc in self._source_dataset.items():
            if pc["id"] is not None:
                result[k] = pc["id"].to(torch.long)  # D-04 + D-06: id cast to torch.long
            else:
                result[k] = torch.arange(pc["pos"].shape[0], dtype=torch.long)  # D-05 + D-06: ordinal fallback
        return result
```
`pc["id"]` (both occurrences, condition and value) become `pc[self.config.ground_truth_field]`. The ordinal fallback (`pc[field] is None` → `torch.arange(...)`) stays structurally identical — it is field-name-agnostic already.

**Field access idiom used throughout `eval/data_factory.py`:** `zRegPointCloud` is a dict subclass; existing code always does `pc["label"]` / `pc["id"]` as plain dict-key lookups (see `scale()` line 755-756, `_standardize()` line 878-880, `drop_points()` line 957-960) — so `pc[self.config.ground_truth_field]` is idiomatically consistent, no new accessor method needed.

---

### `eval/data_factory.py` — `drop_points()` / `sample_new_points()` correspondence tracking (D-03/D-04)

**Analog:** the existing `idx` computation inside these same two methods, plus `_subsample_to_max`'s identical `idx` pattern (lines 885-916), plus `AlignResult.estep_results` in `eval/types.py` as the precedent for "expose a per-frame diagnostic dict alongside the primary return value, keyed like the dataset, default empty."

**Current pattern — the index IS already computed but silently discarded after use** (`eval/data_factory.py:950-962`, `drop_points`):
```python
        torch.manual_seed(seed)
        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            n = pc["pos"].shape[0]
            keep = max(1, round(n * (1.0 - fraction)))
            idx = torch.randperm(n, device=pc["pos"].device)[:keep].sort().values
            result[i] = zRegPointCloud(
                pos=pc["pos"][idx],
                label=pc["label"][idx] if pc["label"] is not None else None,
                id=pc["id"][idx] if pc["id"] is not None else None,
            )
            result[i]["fps-idx"] = pc["fps-idx"][idx] if pc["fps-idx"] is not None else None
        return result
```
`idx` here IS the retained-original-index correspondence D-03 needs (it maps new-position `j` → original-position `idx[j]`). The fix is to **capture and expose this `idx` per-frame** rather than discard it once the loop moves to the next frame — e.g. accumulate into a `dict[int, torch.Tensor]` returned alongside the dataset, or stored as `self._correspondence_idx` (Claude's Discretion per CONTEXT.md, constrained by D-04 reusability for Phase 58).

**Same discard-after-use shape in `sample_new_points()`** (`eval/data_factory.py:999-1023`) — here new points get sentinel `-1` (see `_extend` helper, lines 1009-1015), so the reusable structure needs to represent "this new point has NO original-index correspondence" as well as "this retained point maps to original index N." A sentinel value (e.g. `-1`, mirroring the existing `id`/`fps-idx` sentinel convention already used in this exact method) is the idiomatic choice — do not invent a different sentinel convention.

**Reusable-structure precedent** (`eval/types.py:133-137`, `AlignResult.estep_results` — copy this *shape*, not this literal field):
```python
    velocity_landmarks: list[int] = Field(default_factory=list)
    # CPD E-step posterior (Phase 44, D-09), keyed the same way as aligned_cloud
    # (by target frame key). Empty unless alignment_method="cpd" and
    # cpd_penalty is not None (D-01/D-03).
    estep_results: dict[int, EstepResult] = Field(default_factory=dict)
```
This shows the established codebase idiom for "attach an optional per-frame diagnostic/tracking dict to a result, keyed by frame index, empty by default" — the same shape (`dict[int, torch.Tensor]`, keyed by frame index) is the natural fit for a correspondence-index map that `get_synthetic_ground_truth()` (D-03) needs to consume, and that Phase 58 needs to reuse (D-04).

**`_subsample_to_max` shares the identical `idx`-then-discard shape** (`eval/data_factory.py:899-916`) — NOT in the D-03 canonical_refs list, but structurally identical to `drop_points`/`sample_new_points`. Flagging in case the planner decides subsample-driven correspondence should also be tracked for consistency (out of explicit D-03 scope but worth a scoping note in the plan).

---

### `eval/runners/eval_runner.py` — `_run_single` truncation → correspondence-based gather (D-03)

**Analog:** the block being replaced, same method, same file.

**Current pattern to replace** (`eval/runners/eval_runner.py:344-349`, WR-01):
```python
        # WR-01: truncate to min length when source and target have different point counts
        # (heterogeneous paired datasets). compute_f1 validates shape equality strictly.
        if y_true.shape[0] != y_pred.shape[0]:
            min_len = min(y_true.shape[0], y_pred.shape[0])
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
```
This is the exact call site both D-01 and D-03 converge on (per CONTEXT.md `<code_context>` Integration Points). The fix: once `DataFactory` exposes a correspondence-index map (per the `drop_points`/`sample_new_points` fix above), `_run_single` should gather `y_true` by that tracked index rather than positionally truncating — e.g. `y_true = y_true[correspondence_idx]` (exact shape/plumbing left to planner per Claude's Discretion) instead of blind `[:min_len]` slicing. The surrounding `gt = self.factory.get_synthetic_ground_truth()` call (line 329) is the natural place for the factory to also return/attach the correspondence map, mirroring how `label_result.transferred_labels` (line 337-339) is already consumed from a stage-result object in this same method.

**Immediately-preceding pattern in the same method for "prefer a stage result's own keys over positional guessing"** (`eval/runners/eval_runner.py:333-339`, showing the established idiom of deriving the correct key/index from the actual result object rather than assuming positional alignment — same spirit as the D-03 fix):
```python
        if label_result is not None:
            # Label keys are TARGET frames per Plan 30-01 LabelTransferStage contract
            # Use last key actually present in transferred_labels (= last paired target
            # frame) — guards against KeyError when |source| < |target| (CR-01).
            transferred_keys = sorted(label_result.transferred_labels.keys())
            knn_target_key = transferred_keys[-1]
            y_pred = label_result.transferred_labels[knn_target_key]
```

---

### `baseline_experiments/configs/ground_truth/shah_sample1.yaml` (GT-03 audit — verification only)

**Analog:** itself. Full file already read; D-05 confirms no change needed — `pc["label"]` (populated from the CSV `layer` column by `load_shah_from_csv`) already holds Shah's real cell-type classes, and the new `ground_truth_field="label"` default reads exactly that field. **Action:** verify only, no edit expected unless audit finds something new.

---

### `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` (GT-03 audit — comment fix, D-06)

**Analog:** itself, existing header comment block (full file already read above).

**Current stale comment** (top-of-file header, ~lines 10-19):
```
# Alignment only, same as selfcal/kobitski_ew06_alignment.yaml: Kobitski has
# no usable ground-truth labels for label transfer. Its "label" field is
# continuous RGB-like noise (N,3 floats, not discrete classes) and its "id"
# field is a unique per-point tracking identity (every point its own
# singleton — verified directly against the real tracklets file: 6930/6930
# unique ids in frame 0), unlike Shah's "id" which carries real small-integer
# class labels. Running label transfer against Kobitski would score against
# a meaningless "ground truth". ground_truth/shah_sample1.yaml is the one
# that exercises both stages (Shah's label field is genuine classes).
```
The phrase **"unlike Shah's `id` which carries real small-integer class labels"** is the stale claim D-05 corrects — it is Shah's `label` field (not `id`) that carries the real classes. Rewrite that clause to say Shah's real classes live in `pc["label"]` (populated from the CSV `layer` column), not `pc["id"]`, so the Kobitski/Shah contrast is framed correctly. `run_label_transfer: false` (line 34) is already correct and must NOT change.

---

### `configs/experiments/stage1_alignment/rigid_cpd/synthetic.yaml` (GT-03 audit — stale comment fix, D-05)

**Analog:** itself. Full current file content already read above — note the stale comment referenced in CONTEXT.md (`"id" carries real small-integer class labels`, originally at "~line 19") is **not present verbatim in the current file body** as read in this pass; the file's only comments are the 3-line header (`# Stage 1 — Alignment...`) and `# Run: ...`. **Action for planner/executor:** re-verify at implementation time whether this comment already migrated/was removed, or whether CONTEXT.md's line reference (~19) refers to a different revision — do not assume the comment is still there without re-checking at edit time. If found, correct using the same D-05 language as the kobitski_ew06.yaml fix above (Shah's real classes live in `label`, not `id`).

---

### `configs/experiments/stage2_label_transfer/{knn,hybrid_knn_cpd,pointnet2,cpd_weighted,egnn}/synthetic.yaml` (GT-03 audit — degenerate placeholder, discretion)

**Analog:** `configs/experiments/stage2_label_transfer/knn/synthetic.yaml` (full file read above) as the structural template — all 5 files share `data_path: data/synthetic/fully_synthetic/ball_realistic_kobitski.csv`, `pipeline_mode: synthetic`, and `run_label_transfer: true`. All 5 read `pc["label"]` (post-D-01 default) from a CSV whose `label` column is a constant `1` for every point (per `scripts/generate_datasets.py:239`, `"label": np.ones(len(pts), dtype=np.int32)`), i.e. all points share ONE class — F1 against a single-class ground truth is degenerate (trivially maximizable).

**Two paths, planner's call per CONTEXT.md "Claude's Discretion":**
1. **Leave degenerate, document why** — add a comment to each of the 5 configs (or a shared note) stating the `label` field is currently a placeholder constant, so the F1 metric on these `synthetic.yaml` configs is not meaningful, without touching `generate_datasets.py`.
2. **Fix the generator** — wire `generate_labels()` (`src/zreg/data_generation/labels.py:25`) into `scripts/generate_datasets.py`'s `_make_trajectory` (or a post-processing step) so `data/synthetic/fully_synthetic/*.csv` gets real Voronoi multi-class labels instead of the constant-`1` placeholder. This is regeneratable locally (D-07) so it CAN be empirically checked in this environment, unlike Shah/Kobitski.

---

### `scripts/generate_datasets.py` (if planner selects path 2 above)

**Analog:** `generate_labels()` itself (`src/zreg/data_generation/labels.py:25-80`, read in full above) — the canonical Voronoi label generator, already the established pattern for producing F1-compatible categorical labels elsewhere in the codebase (`DataFactory.generate_training_triple`, `eval/data_factory.py:420`, calls `generate_labels(base, n_classes=n_classes, seed=seed)` on a `dict[int, zRegPointCloud]`).

**Current degenerate placeholder** (`scripts/generate_datasets.py:236-240`, inside `_make_trajectory`):
```python
        traj[i] = {
            "pos":   pts.copy(),
            "id":    ids.copy(),
            "label": np.ones(len(pts), dtype=np.int32),
        }
```
This builds a plain-dict (numpy-backed) trajectory, NOT a `zRegPointCloud` — `generate_labels()` expects/returns `dict[int, zRegPointCloud]` with `torch.Tensor` fields, so direct reuse requires either (a) converting `traj` to `zRegPointCloud`/torch before calling `generate_labels`, then back to numpy for CSV export (see `_frame_to_df`, lines 252-279, which already branches on `isinstance(pos_raw, np.ndarray)` vs `zRegPointCloud`+tensor — this dual-path handling is the existing precedent to extend), or (b) reimplementing the same Voronoi-nearest-seed assignment directly in numpy at the `_make_trajectory` call site to avoid a torch round-trip inside a script whose surrounding code is numpy-first. Given `_frame_to_df` already tolerates both representations, path (a) is more consistent with existing conventions — call `generate_labels` once per generated trajectory dict (converting to/from `zRegPointCloud`) rather than inlining new label logic.

**Import pattern to copy if wiring `generate_labels` in** (mirrors `eval/data_factory.py:27-36`'s import block, which already imports from `zreg.data_generation`):
```python
from zreg.data_generation import generate_labels
```

## Shared Patterns

### Literal-typed EvalConfig field with default (no custom validator)
**Source:** `eval/config.py:275` (`pipeline_mode`), `eval/config.py:256` (`tier`)
**Apply to:** `ground_truth_field: Literal["id", "label"] = "label"` (D-01/D-02)
```python
pipeline_mode: Literal["paired", "synthetic"] = "paired"
```
No `@field_validator` needed — pydantic enforces `Literal` membership at construction automatically. This differs from `alignment_method`/`label_transfer_method`/`device`, which are plain `str` fields requiring manual `@field_validator` methods; do not copy that heavier pattern for a `Literal` field.

### `pc[field]` dict-subscript field access
**Source:** `eval/data_factory.py` throughout (`scale()` line 755, `_standardize()` line 878-880, `drop_points()` line 957-960, `get_ground_truth()` line 664-665)
**Apply to:** All `get_ground_truth`/`get_synthetic_ground_truth` edits — replace the hardcoded `pc["id"]` string key with `pc[self.config.ground_truth_field]`. `zRegPointCloud` is a plain dict subclass, so this is a drop-in parameterization, not a new accessor pattern.

### Per-frame `torch.Tensor` dict keyed by frame index, exposed as diagnostic/tracking state
**Source:** `eval/types.py:133-137` (`AlignResult.estep_results: dict[int, EstepResult] = Field(default_factory=dict)`); `DataFactory.get_ground_truth`/`get_synthetic_ground_truth` return type `dict[int, torch.Tensor]` (`eval/data_factory.py:632`, `eval/data_factory.py` docstring at 685)
**Apply to:** The D-03/D-04 correspondence-tracking structure — whatever shape the planner chooses (returned dict alongside the dataset, or a `DataFactory` instance attribute like the existing `self._preprocessing_stats`/`self._synthetic_target` pattern at `eval/data_factory.py:142-148`), it should be `dict[int, torch.Tensor]` keyed by frame index to match every other per-frame tracking structure in this codebase.

### Existing per-frame index computed via `torch.randperm`, then applied to every field in lockstep
**Source:** `eval/data_factory.py:909` (`_subsample_to_max`), `:955` (`drop_points`), pattern repeats 3 times in the file
**Apply to:** `drop_points`/`sample_new_points` D-03 fix — the `idx` variable already computed inside the existing per-frame loop is the exact correspondence data D-03 needs; the fix is exposure, not recomputation.

### Instance-attribute caching for cross-method state on `DataFactory`
**Source:** `eval/data_factory.py:142-148` (`__init__`: `self._synthetic_target`, `self._source_dataset`, `self._transform_spec`, `self._preprocessing_stats`, all `None`-initialized and populated by a producing method, consumed by a later method call)
**Apply to:** If the planner chooses "attribute stored on DataFactory" (one of the two options CONTEXT.md's Claude's Discretion explicitly names) for the correspondence map, follow this exact `__init__`-declared/producer-sets/consumer-reads convention rather than inventing a new state-management style.

## No Analog Found

None — every file in this phase's scope is an edit to pre-existing code/config; there are no genuinely new files being scaffolded, so there is no "missing analog" case in the usual sense. (`scripts/generate_datasets.py` wiring `generate_labels()` is the closest to a "new capability," and its analog — `DataFactory.generate_training_triple`'s existing `generate_labels()` call — is a role-match, not an exact match, flagged above.)

## Metadata

**Analog search scope:** `eval/`, `eval/stages/`, `eval/runners/`, `src/zreg/data_generation/`, `scripts/`, `baseline_experiments/configs/ground_truth/`, `configs/experiments/stage1_alignment/`, `configs/experiments/stage2_label_transfer/`, `tests/test_data_factory.py`, `eval/types.py`
**Files scanned:** `eval/config.py` (full, 485 lines), `eval/data_factory.py` (full, 1023 lines), `eval/runners/eval_runner.py` (full, 420 lines), `eval/stages/label_transfer.py` (lines 400-494), `eval/types.py` (lines 1-140, 330-430), `src/zreg/data_generation/labels.py` (lines 1-80), `scripts/generate_datasets.py` (lines 200-280), `tests/test_data_factory.py` (lines 638-1023), all 8 target YAML configs (full content)
**Pattern extraction date:** 2026-08-01
