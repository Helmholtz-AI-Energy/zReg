# Phase 30: Two-Dataset Paired Alignment Architecture - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix `AlignmentStage` to always align two distinct trajectories — source and target — so DTW is never run on a single trajectory against itself. Change `AlignmentStage.run(dataset, params)` to `AlignmentStage.run(source, target, params)` where both arguments are `dict[int, zRegPointCloud]`. Change `LabelTransferStage.run(dataset, params)` to `LabelTransferStage.run(source, target, params)` for the same reason. Add `pipeline_mode: Literal["paired", "synthetic"] = "paired"` and `target_data_path: str | None = None` to `EvalConfig`. Add `DataFactory.load_target()` that loads the second dataset from `config.target_data_path` using the same tracklets/CSV dispatch as `load_real()`. Update `EvaluationRunner` to load source and target separately in paired mode and pass both to `AlignmentStage.run()` and `LabelTransferStage.run()`. Add a `paired_alignment.yaml` scenario config that uses the Kobitski dataset as both source and target (smoke-test). Update existing `EvaluationRunner` tests to mock `DataFactory.load_target()`.

</domain>

<decisions>
## Implementation Decisions

### LabelTransferStage signature

- **D-01:** `LabelTransferStage.run()` changes to `run(source, target, params)` where `source` and `target` are both `dict[int, zRegPointCloud]`. Matches the `AlignmentStage` change pattern — explicit rather than hiding the two-dataset distinction inside the runner.
- **D-02:** Label transfer logic stays **sequential**: for each time step `k`, transfer labels from `source[k]` to `target[k]` using the existing nearest-neighbor/KNN dispatch. No warping-path-aware logic is introduced in this phase; the transfer operates on corresponding time indices.

### paired_alignment.yaml scenario config

- **D-03:** `paired_alignment.yaml` uses the **same dataset as both source and target** — Kobitski tracklets (`data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets`) as both `data_path` and `target_data_path`. This is a smoke-test to prove the two-dataset pipeline runs end-to-end; results show self-alignment but mechanics are validated. A true heterogeneous paired scenario is deferred to a future phase or config.

### Backward compatibility for existing tests

- **D-04:** Existing `EvaluationRunner` tests are updated to **mock `DataFactory.load_target()`** returning an appropriate dataset fixture alongside the existing `load_real()` mock. No production-path fallback or `target_data_path`-is-None guard is added to `EvaluationRunner.run()`; the paired-mode contract is enforced cleanly and tests are updated to match.

### EvalConfig validation

- **D-05:** `EvalConfig` gains `pipeline_mode` and `target_data_path` as optional fields (defaults: `"paired"` and `None`). No pydantic `model_validator` is added. The `EvalConfigError` when `target_data_path is None` in paired mode is raised **at `DataFactory.load_target()` call time** only — consistent with the pattern in REQUIREMENTS.md MODE-01 and the existing error-at-use-time convention in this codebase.

### AlignResult continuity

- **D-06:** `AlignResult` structure is **unchanged** — it carries `aligned_cloud` (source dataset reference, unchanged), `warping_path`, `transforms`, etc. `AlignmentStage` receives both source and target, uses them to build the DTW distance matrix (`DynamicTimeWarping(x=source_sub, y=target_sub)`), and returns the same `AlignResult` shape. No new source/target fields are added to `AlignResult` in this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Stages being modified

- `eval/stages/alignment.py` — `AlignmentStage`: change `run(dataset, params)` → `run(source, target, params)`. Internal DTW call changes from `DynamicTimeWarping(x=sub, y=sub)` self-alignment to `DynamicTimeWarping(x=source_sub, y=target_sub)`.
- `eval/stages/label_transfer.py` — `LabelTransferStage`: change `run(dataset, params)` → `run(source, target, params)`. Transfer logic: `source[k]` → `target[k]` sequentially.
- `eval/stages/base.py` — `PipelineStage` ABC: check whether the base `run()` signature needs updating.

### Runner being modified

- `eval/runners/eval_runner.py` — `EvaluationRunner`: `run()` calls `DataFactory.load_target()` in paired mode; `_run_single(source, target, params)` updated to pass both to stages. Lines 185–300 are the primary change area.

### Config and DataFactory

- `eval/config.py` — `EvalConfig`: add `pipeline_mode: Literal["paired", "synthetic"] = "paired"` and `target_data_path: str | None = None`. Note `extra="forbid"` — fields must be explicitly declared.
- `eval/data_factory.py` — `DataFactory`: add `load_target()` method. Follows same tracklets/CSV dispatch as `load_real()` (line ~82). `EvalConfigError` raised when `config.target_data_path is None` in paired mode.

### Tests to update

- `tests/test_eval_runner.py` — patch `DataFactory.load_target` in existing tests.
- `tests/test_alignment_stage.py` — new tests for `run(source, target, params)` two-argument form.
- `tests/test_label_transfer_stage.py` — update tests for new `run(source, target, params)` signature.
- `tests/test_data_factory.py` — new tests for `DataFactory.load_target()`.

### Requirements

- `REQUIREMENTS.md` §MODE-01 — full requirement text for this phase. Success criteria numbered 1–6 are authoritative.

### Prior phase context (EvaluationRunner, DataFactory, AlignmentStage)

- `.planning/phases/21-evaluationrunner-visualisation/21-CONTEXT.md` — EvaluationRunner design decisions (FRAME-07).
- `.planning/phases/19-alignmentstage/` — AlignmentStage D-09: validate_params() called as first line of run().
- `.planning/phases/17-framework-config-datafactory/` — DataFactory D-08: no I/O on construction; EvalConfig extra="forbid".

### New scenario config

- `configs/paired_alignment.yaml` — new file. `data_path` and `target_data_path` both point to Kobitski tracklets. `pipeline_mode: paired`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `eval/data_factory.py:DataFactory.load_real()` (line ~82) — canonical tracklets/CSV dispatch. `load_target()` should replicate this dispatch reading from `config.target_data_path` instead of `config.data_path`. Reuse the same private dispatch logic.
- `eval/stages/alignment.py:AlignmentStage.validate_params()` — unchanged; params validation contract stays the same.
- `zreg/dtw.py:DynamicTimeWarping` — takes `x` and `y` kwargs. Currently called with `x=sub, y=sub` (self). Phase 30 changes to `x=source_sub, y=target_sub`.
- `configs/alignment_dev.yaml` — reference for `data_path`, `data_format`, `default_params` structure to follow for `paired_alignment.yaml`.

### Established Patterns

- `EvalConfigError` raised at call time (not at construction) for invalid config states — consistent with `load_real()` raising when `data_path` is invalid.
- `extra="forbid"` on EvalConfig — all new fields need explicit declaration or pydantic will reject unknown keys in YAML configs.
- `pipeline_mode` is a `Literal` type — pydantic validates the string at parse time.
- `AlignResult.aligned_cloud` is the input dataset reference unchanged (not a copy) — callers treat source as read-only after calling `run()`.

### Integration Points

- `eval/runners/eval_runner.py:_run_single()` — primary integration point. Currently `(dataset, params)` → becomes `(source, target, params)`. All callers of `_run_single` are inside `EvaluationRunner.run()`.
- `eval/runners/eval_runner.py:run()` lines 185–187 — currently calls `load_real()` only. After Phase 30: calls both `load_real()` and `load_target()` in paired mode.
- `run_eval.py` CLI — passes config to EvaluationRunner. No changes needed as long as EvalConfig parsing handles the new optional fields transparently.

</code_context>

<specifics>
## Specific Ideas

- Smoke-test approach for `paired_alignment.yaml`: same file for source and target is intentional — validates the plumbing without requiring a second real dataset. Downstream phases (31+) will add real heterogeneous pairs.
- The `load_target()` implementation should be a near-copy of `load_real()` with `data_path` replaced by `target_data_path`. If a shared private `_load_from_path(path, fmt)` helper can be extracted cleanly, that's a bonus — but not required if it complicates the diff.

</specifics>

<deferred>
## Deferred Ideas

- **Heterogeneous paired scenario config** (Kobitski source + Shah target) — deferred until `load_target()` can handle a separate `target_data_format` field or format inference from extension. Out of scope for Phase 30.
- **Warping-path-aware label transfer** — using `AlignResult.warping_path` to map source frames to target frames in LabelTransferStage. Deferred to a future phase; sequential `source[k]→target[k]` is sufficient for Phase 30.

</deferred>

---

*Phase: 30-Two-Dataset-Paired-Alignment-Architecture*
*Context gathered: 2026-06-12*
