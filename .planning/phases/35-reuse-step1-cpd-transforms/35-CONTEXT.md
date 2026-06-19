# Phase 35: Reuse Step-1 CPD Transforms in Aligned-Cloud Construction - Context

**Gathered:** 2026-06-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Thread CPD transforms (and normalisation parameters) computed inside `pairwise_distance_matrix` (Step 1, on normalised clouds) through `DynamicTimeWarping` into `AlignmentStage._build_aligned_cloud` (Step 3), so Step 3 reuses the stored transform for the `(src_sub_idx, tgt_sub_idx)` pair selected by the DTW path — instead of re-running CPD from identity on raw unnormalised data. This fixes convergence failure when source and target datasets have an 8× scale difference (Shah ~50 units, Kobitski ~400 units).

</domain>

<decisions>
## Implementation Decisions

### New shared types module

- **D-01:** Create `src/zreg/types.py` — a new thin module (imports only `dataclasses` and `torch`) containing both `StoredTransform` and `PairwiseResult`. This avoids circular imports: `pairwise_distance_matrix.py` imports `StoredTransform` and `PairwiseResult` from `types.py`; `dtw/result.py` imports `StoredTransform` from `types.py`. Neither file imports from the other.

### StoredTransform dataclass

- **D-02:** `StoredTransform` dataclass fields: `transform` (the CPD transform object — `RigidCPDTransformation`, `AffineCPDTransformation`, or `NonRigidCPDTransformation`), `src_mean: torch.Tensor`, `src_scale: torch.Tensor`, `tgt_mean: torch.Tensor`, `tgt_scale: torch.Tensor`.
- **D-03:** `StoredTransform` is keyed by `(src_sub_idx, tgt_sub_idx)` in the dict — matching the indices produced by the DTW warping path. The dict is empty (`{}`) when `cpd_type is None`.
- **D-04:** In `pairwise_distance_matrix`, the normalisation params are captured from `utils.normalize_point_cloud` — currently the second return value is discarded (`_`). Change to capture `(xi_norm, (src_mean, src_scale))` and `(yj_norm, (tgt_mean, tgt_scale))` and store in `StoredTransform`.

### PairwiseResult return type

- **D-05:** `create_pairwise_distance_matrix` returns a `PairwiseResult` dataclass (defined in `types.py`) with fields: `cost_matrix: torch.Tensor`, `rotations: torch.Tensor | None`, `stored_transforms: dict[tuple[int, int], StoredTransform]`. Replaces the current plain 2-tuple `(matrix, rotations)`.
- **D-06:** `DynamicTimeWarping.compute_cost_matrix()` switches from tuple unpacking to field access: `result = create_pairwise_distance_matrix(...); self._cost_matrix = result.cost_matrix; self._rotations = result.rotations; self._stored_transforms = result.stored_transforms`.

### DTWResult threading

- **D-07:** Add `stored_transforms: dict[tuple[int, int], StoredTransform]` field to `DTWResult` (default `{}`). `DynamicTimeWarping.compute()` passes `self._stored_transforms` when constructing `DTWResult`.
- **D-08:** `AlignmentStage.run()` passes `dtw_result.stored_transforms` to `_build_aligned_cloud` as a new parameter `stored_transforms: dict[tuple[int, int], StoredTransform]` (default `{}`).

### _build_aligned_cloud reuse path

- **D-09:** When `stored_transforms[(src_sub_idx, tgt_sub_idx)]` exists and `cpd_penalty is not None`: normalise raw source frame using stored `src_mean`/`src_scale` → apply stored `transform.transform()` → denormalise using `tgt_mean`/`tgt_scale` to place points in target coordinate space. **No outlier removal** in the reuse path — `remove_outliers_knn` is not replicated. The small difference in normalisation input (outliers still present) introduces negligible error compared to the 8× scale problem being fixed.
- **D-10:** Fallback to fresh CPD (current Step-3 logic) when the key is absent — this covers edge frames outside the DTW window that were never computed in Step 1.

### Claude's Discretion

- Field name for stored transforms in `DTWResult`: `stored_transforms` (consistent with the dict type name)
- Whether `create_pairwise_distance_matrix_given_rigid_rot` is updated: **leave unchanged** — it is a separate code path not called by `DynamicTimeWarping`, so updating it is out of scope for Phase 35
- `utils.normalize_point_cloud` return signature: assumed to return `(normalized_tensor, (mean, scale))` — confirm at implementation time and adapt if the actual signature differs

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core files to modify

- `src/zreg/pairwise_distance_matrix.py` — current return signature `(distance_matrix, rotations_tensor)`; CPD transform computed at line ~194; normalisation params discarded at lines 168–169. Primary change target.
- `src/zreg/dtw/core.py` — `DynamicTimeWarping.compute_cost_matrix()` at line ~157; unpacks 2-tuple from `create_pairwise_distance_matrix` at line ~173; `DynamicTimeWarping.compute()` builds `DTWResult` at line ~145. Threading point for stored transforms.
- `src/zreg/dtw/result.py` — `DTWResult` dataclass; needs new `stored_transforms` field with default `{}`.
- `eval/stages/alignment.py` — `_build_aligned_cloud()` at line 288; current fresh-CPD implementation lines 348–381; receives `warp_path` and `cpd_penalty`; needs new `stored_transforms` parameter.

### New file

- `src/zreg/types.py` — to be created; contains `StoredTransform` and `PairwiseResult` dataclasses.

### Phase specification

- `.planning/ROADMAP.md` Phase 35 entry — goal, success criteria (5 items), and dependency on Phase 33.
- `.planning/phases/35-reuse-step1-cpd-transforms/.continue-here.md` — architecture notes, blocking patterns, decisions made in prior session, list of uncommitted files.

### Prior phase for context

- `eval/stages/alignment.py` `_build_aligned_cloud()` — Phase 33 delivered this method (ALIGN-01); Phase 35 extends it. Read Phase 33 context at `.planning/phases/33-cpd-aligned-trajectory/33-CONTEXT.md` if the aligned-cloud construction logic is unclear.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `utils.normalize_point_cloud(pos)` — returns `(normalized_pos, params)` where params contain mean/scale; currently called at lines 168–169 of `pairwise_distance_matrix.py` with params discarded. D-04 captures these params instead.
- `DTWResult` dataclass (`dtw/result.py`) — already holds `rotations: torch.Tensor | None = None` with a None default; adding `stored_transforms` with default `{}` follows the same pattern.
- `AlignmentStage._build_aligned_cloud()` — `@staticmethod`; currently takes `source, target, source_sub, target_sub, warp_path, cpd_penalty`. Adding `stored_transforms` as a keyword argument with default `{}` is backward compatible.

### Established Patterns

- CPD objects: `RigidCPD`, `AffineCPD`, `NonRigidCPD` — instantiated with `source=pos`, `use_color=False`, `log_freq=-1`. Transform applied via `reg_result.transformation.transform(pos)` (Phase 33 CR-01 pattern — use `reg_result.transformation`, not `cpd_obj.transformation`, to handle NonRigidCPD's override).
- Deepcopy before mutation: `src_frame = deepcopy(source_sub[src_sub_idx])` before any in-place transform — keep this pattern in the reuse path too.
- DTWResult construction in `DynamicTimeWarping.compute()` at line ~145: keyword args only — adding `stored_transforms=self._stored_transforms` follows the existing style.

### Integration Points

- `DynamicTimeWarping.compute_cost_matrix()` line ~173: the single call site of `create_pairwise_distance_matrix` — this is where the return type changes from 2-tuple to `PairwiseResult`.
- `AlignmentStage.run()` calls `_build_aligned_cloud()` and passes `result.warping_path` from `DTWResult` — this is where `result.stored_transforms` needs to be threaded through.
- `src/zreg/__init__.py` — check whether `create_pairwise_distance_matrix` is re-exported; if so, the `PairwiseResult` type should also be exported so external callers can type-check the result.

</code_context>

<specifics>
## Specific Ideas

- The Shah/Kobitski sanity config (`configs/shah_vs_kobitski_rigid_sanity.yaml`) is the canonical end-to-end validation: run with `cpd_penalty=rigid` and verify that `aligned_cloud` points fall within the Kobitski bounding box. This is success criterion 4 from ROADMAP.md.
- The `TestStoredTransformReuse` test class (success criterion 5) should cover the normalise→transform→denormalise round-trip using a synthetic pair of scaled point clouds (no real data required) — rigid CPD gives a deterministic transform that is easy to verify.
- Several files were modified in the prior session and are uncommitted (see `.continue-here.md` §Uncommitted Files): `eval/config.py`, `eval/data_factory.py`, `eval/viz.py`, `eval/tracking/trajectory.py`, and 4 new configs. The planner should note these exist but are not Phase 35 work — they should be committed before or after planning, not confused with Phase 35 changes.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 35-reuse-step1-cpd-transforms*
*Context gathered: 2026-06-19*
