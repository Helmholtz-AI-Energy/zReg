# Roadmap: zReg

## Milestones

- 🚧 **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–34 (in progress)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1-5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

### v1.2 Evaluation Framework & Debt Resolution (in progress)

- [x] **Phase 27: DataFactory Geometric Augmentation Methods** (2/2 plans) — completed 2026-06-11
  **Goal:** Add four standalone augmentation methods to `DataFactory` in `eval/data_factory.py`: `rotate(dataset, rotation_matrix)`, `drop_points(dataset, fraction, seed)`, `sample_new_points(dataset, n_extra, seed)`, and `scale(dataset, factor)`. Port `_augment_scaling` and `_augment_dropout` logic from `scripts/generate_datasets.py`; rotation delegates to the already-imported `apply_rigid` from `zreg.generators`; `sample_new_points` adds uniformly-random points in the per-frame bounding box. Extend `augment()` dict dispatch to recognise new keys: `"scale_factor"`, `"dropout_fraction"`, `"rotation_deg"` (+ optional `"rotation_axis"`), `"n_new_points"` — applied in a fixed order after existing noise/outlier steps. Extend `tests/test_data_factory.py` with a new test class per method.
  **Requirements:** DF-01
  **Depends on:** Phase 17
  **Success criteria:**
  1. `DataFactory.rotate(dataset, R)` applies rotation matrix R to every frame's `pos` tensor
  2. `DataFactory.drop_points(dataset, 0.3)` removes ≈30 % of points per frame, preserving `id` and `color`
  3. `DataFactory.sample_new_points(dataset, n_extra)` appends `n_extra` uniform-in-bbox points to each frame
  4. `DataFactory.scale(dataset, factor)` multiplies all `pos` tensors by `factor`
  5. `augment({"scale_factor": 1.2, "dropout_fraction": 0.1})` chains scale then dropout correctly
  6. All new methods deep-copy their inputs (no in-place mutation)
  7. Tests green; existing 789 tests unaffected
  **Plans:** 2 plans
  Plans:
  - [x] 27-01-PLAN.md — Implement `rotate`, `drop_points`, `sample_new_points`, `scale` as `DataFactory` instance methods + unit tests for each
  - [x] 27-02-PLAN.md — Extend `augment()` dispatch to handle `"scale_factor"`, `"dropout_fraction"`, `"rotation_deg"`/`"rotation_axis"`, `"n_new_points"` keys + integration tests covering multi-key compositions

- [x] **Phase 28: Script Integration — generate_datasets uses DataFactory** (1/1 plans) — completed 2026-06-11
  **Goal:** Refactor `scripts/generate_datasets.py` to import `DataFactory` from `eval.data_factory` and replace the local `_augment_scaling`, `_augment_dropout`, and `apply_augmentation` dispatcher with calls to `DataFactory` methods. The geometry helpers (`in_bowl`, `sample_ball_shell`, `sample_bowl_frame`, `sample_bowl_shell`, `_make_trajectory`) and CSV I/O (`_frame_to_df`, `save_as_csv`) remain unchanged. `_augment_noise` is replaced by a direct `DataFactory.augment(dataset, {"sigma": value})` call (or equivalent). The script's external behaviour (generated files, output paths) must be identical before and after.
  **Requirements:** DF-02
  **Depends on:** Phase 27
  **Success criteria:**
  1. `scripts/generate_datasets.py` imports `DataFactory` and uses it for scaling, dropout, and noise augmentations
  2. `_augment_scaling`, `_augment_dropout`, and `apply_augmentation` functions removed from the script
  3. Running the script produces bit-identical output to the pre-refactor version (same CSV content, same augmentation values)
  4. No new duplicate augmentation logic introduced
  **Plans:** 1 plan
  Plans:
  - [x] 28-01-PLAN.md — Replace local augmentation functions in `scripts/generate_datasets.py` with `DataFactory` method calls; verify output parity

- [x] **Phase 29: Viz Unification** (2/2 plans) — completed 2026-06-12
  **Goal:** (1) Update the point cloud scatter style in `eval/viz.py` to match `scripts/visualize_datasets.py`: `s=1.5`, `alpha=0.45`, subsampling at 4 000 pts, DPI=150 for PNG saves, smaller tick/label fonts (`fontsize=6` ticks, `fontsize=7` axis labels, `labelpad=2`), `ax.{x,y,z}axis.pane.fill = False`. All existing public API (`plot_trajectory`, `plot_metrics`) and PDF+PNG output are preserved. (2) Add a new public function `render_dataset_triptych(csv_path, name, output_dir, dpi=150)` to `eval/viz.py` that chunk-reads a CSV and renders a 1×3 3D triptych PNG — consolidating the logic currently duplicated in `scripts/visualize_datasets.py`. (3) Refactor `scripts/visualize_datasets.py` to import `render_dataset_triptych` from `eval.viz` and remove the local `_scatter3` and `render_dataset` functions.
  **Requirements:** VIZ-01
  **Depends on:** Phase 25
  **Success criteria:**
  1. `plot_trajectory` PNG output uses `s=1.5, alpha=0.45` scatter style and DPI=150
  2. `render_dataset_triptych` in `eval/viz.py` produces the same triptych as the old `render_dataset` in the script
  3. `scripts/visualize_datasets.py` delegates entirely to `eval.viz.render_dataset_triptych`; `_scatter3` and `render_dataset` are removed
  4. PDF+PNG output from `plot_trajectory` and `plot_metrics` still written correctly
  5. Existing `tests/test_viz.py` tests pass; new smoke test for `render_dataset_triptych` added
  **Plans:** 2 plans
  Plans:
  **Wave 1**
  - [x] 29-01-PLAN.md — Update scatter style in `plot_trajectory` + add `render_dataset_triptych` to `eval/viz.py` + smoke test
  **Wave 2** *(blocked on Wave 1 completion)*
  - [x] 29-02-PLAN.md — Refactor `scripts/visualize_datasets.py` to use `render_dataset_triptych`; remove `_scatter3`, `render_dataset`

- [x] **Phase 30: Two-Dataset Paired Alignment Architecture** — completed 2026-06-12 (3/3 plans complete)
  **Goal:** Fix `AlignmentStage` to always align two distinct trajectories — source and target — so DTW is never run on a single trajectory against itself. Change `AlignmentStage.run(source, target, params)` to accept two `dict[int, zRegPointCloud]` arguments. Add `pipeline_mode: Literal["paired", "synthetic"] = "paired"` and `target_data_path: str | None = None` to `EvalConfig`. Add `DataFactory.load_target()` that loads the second dataset using `target_data_path` following the same tracklets/CSV dispatch as `load_real()`. Update `EvaluationRunner` to load source and target separately in paired mode and pass both to `AlignmentStage.run()`. Add a `paired_alignment.yaml` scenario config. Update `LabelTransferStage` and `EvaluationRunner._run_single()` signatures where needed so source/target propagate cleanly through the pipeline.
  **Requirements:** MODE-01
  **Depends on:** Phase 21 (EvaluationRunner), Phase 19 (AlignmentStage), Phase 17 (EvalConfig + DataFactory)
  **Success criteria:**
  1. `AlignmentStage.run(source, target, params)` calls `DynamicTimeWarping(x=source_sub, y=target_sub)` — source and target are different datasets; `x=y` self-alignment is removed
  2. `EvalConfig` accepts `pipeline_mode: "paired" | "synthetic"` (default `"paired"`) and `target_data_path: str | None`
  3. `DataFactory.load_target()` loads the target dataset from `config.target_data_path` using the same tracklets/CSV dispatch as `load_real()`; raises `EvalConfigError` when `target_data_path` is None in paired mode
  4. `EvaluationRunner.run()` in paired mode loads source via `DataFactory.load_real()` and target via `DataFactory.load_target()`, then passes both to `AlignmentStage.run(source, target, params)`
  5. `paired_alignment.yaml` scenario config runs end-to-end without errors
  6. All existing 789 tests pass; new tests for two-argument `AlignmentStage.run()` and `DataFactory.load_target()` added
  **Plans:** 3 plans
  Plans:
  **Wave 1**
  - [x] 30-01-PLAN.md — Add pipeline_mode + target_data_path to EvalConfig; add DataFactory.load_target(); change PipelineStage/AlignmentStage/LabelTransferStage to run(source, target, params); update stage + DataFactory tests
  **Wave 2** *(blocked on Wave 1 completion)*
  - [x] 30-02-PLAN.md — Update EvaluationRunner.run() + _run_single + HyperparamOptimizer._objective for source/target propagation; resolve variable shadowing (source_pos/target_pos); update test_eval_runner and test_optimizer mocks
  **Wave 3** *(blocked on Wave 2 completion)*
  - [x] 30-03-PLAN.md — Add configs/paired_alignment.yaml (Kobitski as both source and target); update tests/test_trajectory_export.py mocks; add tests/test_cli.py smoke-test for paired_alignment.yaml

- [ ] **Phase 36: plot_trajectory Alignment Figure Refactor — 4 separate 1×3 figures** (0/1 plans)
  **Goal:** Refactor the alignment branch of `plot_trajectory` in `eval/viz.py` to produce four independent 1×3 figure pairs (PDF + PNG each = 8 files total) instead of the current single superimposed figure: (1) `alignment_source_trajectory` — source cloud only (blue); (2) `alignment_target_trajectory` — target cloud only (green, omitted when `target` is None); (3) `alignment_aligned_trajectory` — aligned source only (orange); (4) `alignment_superposed_trajectory` — all available clouds superposed with legend. The label branch is not touched.
  **Requirements:** VIZ-02
  **Depends on:** Phase 29 (viz.py scatter style), Phase 35 (aligned_cloud keyed by target indices)
  **Success criteria:**
  1. `plot_trajectory(align_result, None, dataset, None, tmp_path, target=target_ds)` writes exactly 8 files with the 4 new stems (PDF + PNG each)
  2. `plot_trajectory(align_result, None, dataset, None, tmp_path)` (no target) writes 6 files — skips target figure
  3. Old stem `alignment_trajectory` is no longer produced
  4. Label branch behaviour unchanged; all 969 existing tests pass after updates; ≥3 new tests added
  **Plans:** 1 plan
  Plans:
  - [x] 36-01-PLAN.md — Refactor alignment branch to 4 helpers; update TestPlotTrajectory; add 3 new tests

- [x] **Phase 37: plot_trajectory Label Figure Refactor — 2 separate 1×3 figures** (1/1 plans) — completed 2026-06-22
  **Goal:** Refactor the label branch of `plot_trajectory` in `eval/viz.py` to produce two independent 1×3 figure pairs (PDF + PNG each = 4 files) instead of the current single figure: (1) `label_source_trajectory` — source point cloud coloured by source labels (`dataset[fk]["id"]`, falling back to `color`); (2) `label_target_trajectory` — target cloud coloured by transferred labels (`label_result.transferred_labels`). The alignment branch is not touched.
  **Requirements:** VIZ-03
  **Depends on:** Phase 36 (alignment refactor complete; shared helpers `_deduplicate_frames`, `_subsample` from Phase 36 reused)
  **Success criteria:**
  1. `plot_trajectory(None, label_result, dataset, None, tmp_path)` writes exactly 4 files: `label_source_trajectory.pdf/.png` and `label_target_trajectory.pdf/.png`
  2. Old stem `label_trajectory` is no longer produced
  3. Source figure colours points by `dataset[fk]["id"]` when present; falls back to `dataset[fk]["color"]` when `id` is None
  4. Target figure colours points by `label_result.transferred_labels[fk]` using `target[fk]["pos"]` when available
  5. All tests pass after updates; ≥2 new tests added
  **Plans:** 1 plan
  Plans:
  - [x] 37-01-PLAN.md — Refactor label branch to 2 figure helpers; add source-labels figure; update TestPlotTrajectory

- [x] **Phase 38: zRegPointCloud color→label Field Rename** (2/2 plans) — completed 2026-06-24
  **Goal:** Rename the `color` field in `zRegPointCloud` to `label` throughout the entire codebase — in the class definition, all data loaders, all consumers in `src/`, `eval/`, and `scripts/`, and all test fixtures. Simultaneously fix two label-source heuristics that silently fall back to `"id"` when labels are absent: (1) `LabelTransferStage.run()` must raise `ValueError` loudly when `src_frame["label"]` is None; (2) `_get_source_labels` in `eval/viz.py` must return `None` when `label` is absent (grey render is acceptable in viz).
  **Requirements:** CLN-01, CLN-02
  **Depends on:** Phase 37 complete
  **Success criteria:**
  1. `zRegPointCloud(pos=..., label=..., id=...)["label"]` returns the tensor; `["color"]` returns `None`
  2. All data loaders (`load_data_from_tracklets`, `load_shah_from_csv`, `open3d_to_zreg`) populate `pc["label"]`
  3. `LabelTransferStage.run()` raises `ValueError("Source frame {sk} has no 'label' field...")` when `label=None`
  4. `_get_source_labels(pc)` returns `pc["label"].long()` or `None`; never reads `pc["id"]`
  5. No occurrence of `pc["color"]`, `frame["color"]`, or `color=` as a `zRegPointCloud` field remains in the repo
  6. Full test suite green; ≥3 new CLN-02 regression tests added
  **Plans:** 2 plans
  Plans:
  - [x] 38-01-PLAN.md — Rename `color` → `label` in all production files; fix `_get_source_labels` and `LabelTransferStage` label-source logic
  - [x] 38-02-PLAN.md — Update all test fixtures, mocks, and assertions; add CLN-02 regression tests

- [x] **Phase 35: Reuse Step-1 CPD Transforms in Aligned-Cloud Construction** (2/2 plans) — completed 2026-06-22
  **Goal:** Persist the CPD transforms computed inside `pairwise_distance_matrix` (Step 1, on normalised clouds) so that `_build_aligned_cloud` (Step 3) can reuse the transform for the (src_sub_idx, tgt_sub_idx) pair selected by the DTW path, instead of re-running CPD from scratch on raw unnormalised data. Currently Step 3 starts CPD from identity on clouds that may have an 8× scale difference, causing convergence failure. The fix threads a `dict[(i,j) → (transform, src_norm_params, tgt_norm_params)]` out of `pairwise_distance_matrix`, through `DynamicTimeWarping`, into `AlignmentStage._build_aligned_cloud`, which then: normalises the raw source frame with the stored params, applies the stored transform, and denormalises into target coordinate space.
  **Requirements:** ALIGN-03
  **Depends on:** Phase 33 (CPD-aligned cloud construction in `_build_aligned_cloud`)
  **Success criteria:**
  1. `create_pairwise_distance_matrix` returns a third element: `dict[tuple[int,int], StoredTransform]` where `StoredTransform` holds the CPD transform object and per-cloud normalisation params (mean, scale) — only populated when `cpd_type is not None`
  2. `DynamicTimeWarping` threads the stored-transform dict from `compute_cost_matrix` through to the result object so `AlignmentStage` can access it
  3. `_build_aligned_cloud` uses `stored_transforms[(src_sub_idx, tgt_sub_idx)]` when available: normalise raw source frame → apply stored transform → denormalise to target space; falls back to fresh CPD when the key is absent
  4. When `cpd_penalty=rigid` and both datasets are at different scales, the aligned_cloud points fall within the target dataset's bounding box (verifiable on the Shah/Kobitski sanity config)
  5. All existing tests pass; new `TestStoredTransformReuse` test class covers the normalise→transform→denormalise round-trip
  **Plans:** 2 plans
  Plans:
  **Wave 1**
  - [x] 35-01-PLAN.md — Create src/zreg/types.py (StoredTransform + PairwiseResult); update pairwise_distance_matrix.py to capture norm params and return PairwiseResult; extend DTWResult with stored_transforms; update dtw/core.py to use field access; fix 2-tuple unpacking in test_pairwise_distance_matrix.py and test_dtw.py
  **Wave 2**
  - [x] 35-02-PLAN.md — Add stored_transforms reuse path to _build_aligned_cloud in alignment.py; thread stored_transforms from DTWResult through AlignmentStage.run(); add TestStoredTransformReuse class to test_alignment_stage.py

- [x] **Phase 34: Alignment Quality Guard in LabelTransferStage** (1/1 plans) — completed 2026-06-19
  **Goal:** Add a pre-transfer alignment check to `LabelTransferStage.run()` that computes mean per-frame Chamfer distance between the received source and target, stores it in `LabelResult.pre_transfer_alignment`, and emits a `warnings.warn()` when `config.run_alignment=False` and the distance exceeds `ALIGNMENT_WARN_THRESHOLD` (default 1.0). When alignment was run upstream (`run_alignment=True`), log the metric at INFO level instead. This surfaces poor alignment before label transfer and warns users who skip `AlignmentStage` when their inputs are not pre-aligned.
  **Requirements:** ALIGN-02
  **Depends on:** Phase 33 (CPD-aligned cloud, so aligned_cloud is the actual registered input)
  **Success criteria:**
  1. `LabelResult` has a `pre_transfer_alignment: float` field (default 0.0) recording mean Chamfer distance
  2. `LabelTransferStage._check_alignment()` computes mean Chamfer distance across sequential frame pairs
  3. `warnings.warn()` is issued when `run_alignment=False` and mean Chamfer > 1.0
  4. No warning is raised when `run_alignment=True` (even if alignment is poor — user chose to run alignment)
  5. No warning is raised when `run_alignment=False` but input is pre-aligned (Chamfer < threshold)
  6. All existing tests pass; new `TestLabelTransferAlignmentGuard` class added
  **Plans:** 1 plan
  Plans:
  **Wave 1**
  - [x] 34-01-PLAN.md — Add `pre_transfer_alignment` to `LabelResult`; add `_check_alignment()` + warning logic to `LabelTransferStage.run()`; add `TestLabelTransferAlignmentGuard` tests

- [x] **Phase 33: CPD-Aligned Trajectory Output from AlignmentStage** (2/2 plans) — completed 2026-06-16
  **Goal:** Replace the current `AlignResult.aligned_cloud = source` pass-through (D-06, Phase 30) with a spatially-registered trajectory. After DTW computes the warp path, `AlignmentStage._build_aligned_cloud()` maps each full target frame to its DTW-corresponding source frame and — when `cpd_penalty` is set — applies CPD registration to produce spatially-transformed coordinates. The result is keyed by full target keys so `LabelTransferStage` can pair by position without any runner changes. When `cpd_penalty=None`, the cloud is DTW-temporally-resampled only (no spatial shift).
  **Requirements:** ALIGN-01
  **Depends on:** Phase 30 (two-dataset architecture), Phase 32 (heterogeneous paired evaluation)
  **Success criteria:**
  1. `AlignResult.aligned_cloud` keys equal `set(target.keys())` in all cases
  2. `AlignResult.aligned_cloud` length equals `len(target)`
  3. When `cpd_penalty=None`, each frame is a deep copy of the DTW-matched source frame (no spatial shift)
  4. When `cpd_penalty` is set, each frame's `pos` tensor differs from the raw source frame (CPD was applied)
  5. Original `source` dict is never mutated — all frames are deep copies
  6. `step > 1` still produces aligned cloud covering all full target frames (nearest-neighbour interpolation)
  7. All existing 879 tests pass; new `TestAlignedCloudSemantics` class added
  **Plans:** 2 plans
  Plans:
  **Wave 1**
  - [x] 33-01-PLAN.md — Add `_build_aligned_cloud()` to `AlignmentStage`; change `run()` to return CPD-transformed cloud; update `AlignResult` docstring in `eval/types.py`
  **Wave 2** *(blocked on Wave 1 completion)*
  - [x] 33-02-PLAN.md — Add `TestAlignedCloudSemantics` tests; update any existing assertions about `aligned_cloud` identity

- [x] **Phase 32: Heterogeneous Paired Evaluation — `target_data_format`** — completed 2026-06-14
  **Goal:** Enable paired evaluation across datasets with different file formats (e.g. Kobitski `.tracklets` as source, Shah `.csv` as target). Add `target_data_format: str | None = None` to `EvalConfig` — `None` falls back to `data_format` so all existing configs remain valid. Update `DataFactory.load_target()` to resolve the effective format as `target_data_format or data_format`. Add `configs/kobitski_vs_shah.yaml` and `configs/shah_vs_kobitski.yaml` scenario configs, plus a same-format cross-embryo config `configs/kobitski_vs_kobitski_cross.yaml`. No orchestration code changes — `EvaluationRunner`, `AlignmentStage`, `LabelTransferStage`, `MetricsEngine`, and `run_eval.py` are untouched.
  **Requirements:** HETERO-01
  **Depends on:** Phase 30 (two-dataset architecture)
  **Success criteria:**
  1. `EvalConfig` accepts `target_data_format: "tracklets" | "csv" | None` without raising `EvalConfigError`; `None` (default) preserves backward compatibility
  2. `DataFactory.load_target()` resolves effective format as `self.config.target_data_format or self.config.data_format` and routes to the correct loader
  3. `python run_eval.py --config configs/kobitski_vs_shah.yaml --mode eval` completes without error and writes `experiments/runs/kobitski_vs_shah/eval_report.json`
  4. `configs/paired_alignment.yaml` (no `target_data_format` key) still works — backward compatibility confirmed
  5. All existing 869 tests pass; new tests for `target_data_format` field and `load_target()` format dispatch added
  **Plans:** 2 plans
  Plans:
  **Wave 1**
  - [ ] 32-01-PLAN.md — Add `target_data_format: str | None = None` to `EvalConfig`; update `DataFactory.load_target()` format dispatch; add `TestEvalConfigTargetDataFormat` and `TestLoadTargetFormatDispatch` test classes
  **Wave 2** *(blocked on Wave 1 completion)*
  - [ ] 32-02-PLAN.md — Add `configs/kobitski_vs_shah.yaml`, `configs/shah_vs_kobitski.yaml`, `configs/kobitski_vs_kobitski_cross.yaml`; add `TestScenarioConfigs` smoke tests for the three new configs; verify backward compatibility of `paired_alignment.yaml`

- [x] **Phase 31: Synthetic Pipeline Mode — Transform-Spec Target Generation & GT-Aware HPO** — completed 2026-06-14
  **Goal:** Implement `pipeline_mode = "synthetic"` where a second dataset is generated by applying a known transformation to the first, enabling ground-truth-supervised hyperparameter calibration. Add `transform_spec: dict | None = None` to `EvalConfig`. Add `DataFactory.generate_target(dataset, transform_spec)` that applies the specified transformation (rigid, affine, or noise via the existing `zreg.generators` and `DataFactory.augment()` API) and returns the transformed trajectory as the target. Add `DataFactory.get_synthetic_ground_truth()` that returns per-frame cell-identity labels derived directly from the known transform (since the correspondence is deterministic). Update `EvaluationRunner` to support synthetic mode: generate the target and extract GT automatically. Update `HyperparamOptimizer._objective()` to use GT F1 score as the primary calibration signal when `pipeline_mode == "synthetic"`. Add a `synthetic_mode.yaml` scenario config.
  **Requirements:** MODE-02, MODE-03
  **Depends on:** Phase 30 (two-dataset architecture), Phase 22 (HyperparamOptimizer)
  **Success criteria:**
  1. `EvalConfig` accepts `transform_spec: dict | None` with at least `{"type": "rigid", "rotation_deg": float, "rotation_axis": [x,y,z]}` and `{"type": "noise", "sigma": float}` specs
  2. `DataFactory.generate_target(dataset, transform_spec)` applies the specified transform and returns a distinct `dict[int, zRegPointCloud]` (no shared tensors with the input)
  3. `DataFactory.get_synthetic_ground_truth()` returns a `dict[int, Tensor]` of per-frame cell IDs that reflect the deterministic transform correspondence; raises `RuntimeError` when called outside synthetic mode
  4. `EvaluationRunner.run()` in synthetic mode calls `generate_target()` to produce the target and `get_synthetic_ground_truth()` for GT labels; no `target_data_path` required
  5. `HyperparamOptimizer._objective()` uses GT F1 (from `get_synthetic_ground_truth()`) as the optimisation objective in synthetic mode
  6. `synthetic_mode.yaml` scenario config runs end-to-end including HPO calibration without errors
  7. All existing tests pass; new tests cover `generate_target()`, `get_synthetic_ground_truth()`, GT-aware objective function
  **Plans:** 3 plans
  Plans:
  **Wave 1**
  - [ ] 31-01-PLAN.md — Add `EvalConfig.transform_spec` field; add `DataFactory.generate_target()` (try/finally augment dispatch + state) and `DataFactory.get_synthetic_ground_truth()` (id-or-ordinal torch.long); new `TestEvalConfigTransformSpec`, `TestGenerateTarget`, `TestGetSyntheticGroundTruth`
  **Wave 2** *(blocked on Wave 1 completion)*
  - [ ] 31-02-PLAN.md — Wire `EvaluationRunner.run()` + `_run_single()` synthetic branches (`generate_target`, `get_synthetic_ground_truth`); add `_apply_transform_to_dataset()` helper + synthetic branches in `HyperparamOptimizer._objective()` (sanity tier isolation per D-11, dev/full tier slice `_synthetic_target` + GT F1); new `TestEvaluationRunnerSyntheticMode`, `TestOptimizerSyntheticMode`
  **Wave 3** *(blocked on Wave 2 completion)*
  - [ ] 31-03-PLAN.md — Add `configs/synthetic_mode.yaml` (rigid rotation 30 deg on Kobitski source, no target_data_path) + `TestScenarioConfigs.test_synthetic_mode_yaml_loads_and_declares_synthetic_mode`; full pytest suite regression sweep

- [x] Phase 12: Carry-Forward Debt Closure (3/3 plans) — completed 2026-05-14
- [x] Phase 13: Core Metrics Library (3/3 plans) — completed 2026-05-15
- [x] Phase 14: Synthetic Data Generators (3/3 plans) — completed 2026-05-18
  **Goal:** Deliver the `src/zreg/generators/` package with a from-scratch trajectory factory and immutable corruption wrappers (rigid/affine transforms, Gaussian noise, outlier injection, Voronoi label generation, label removal) — all seed-deterministic, immutable, and `dict[int, zRegPointCloud]`-shaped per EVAL-03.
  **Requirements:** EVAL-03
  **Plans:** 3 plans
  Plans:
  - [x] 14-01-PLAN.md — Package skeleton + `generate_trajectory` factory + rigid/affine transform wrappers
  - [x] 14-02-PLAN.md — Corruption wrappers (Gaussian noise, outliers) + label utilities (Voronoi generate, remove) + extend `__init__.py`
  - [x] 14-03-PLAN.md — `tests/conftest.py` sys.path extension + `tests/test_generators.py` (4 test classes covering all 7 public symbols)

- [x] Phase 15: Experiment Tracking & Run Management (2/2 plans) — completed 2026-05-18
  **Goal:** Deliver the `eval/tracking/` package at the repo root with a single `log_run()` function that writes local JSON + CSV per run using stdlib only — capturing all 9 EVAL-04 required fields (6 caller-supplied + 3 auto-captured: git_hash, zreg_version, timestamp) — plus unit tests in `tests/test_tracking.py` with all auto-captured fields mocked.
  **Requirements:** EVAL-04
  **Depends on:** Phase 14
  Plans:
  - [x] 15-01-PLAN.md — `eval/tracking/__init__.py` + `eval/tracking/tracking.py` (`log_run()` implementation, stdlib-only)
  - [x] 15-02-PLAN.md — `tests/test_tracking.py` (TestLogRun class: 13 tests covering all 9 fields, fallbacks, file output)

- [x] Phase 16: Runner Scripts (2/2 plans) — completed 2026-05-19
  **Goal:** Deliver `eval/run_synthetic.py` and `eval/run_real.py` as standalone (non-importable) scripts at the repo root that run noise/corruption sweeps and scale/density sweeps respectively — accepting `dict[int, zRegPointCloud]` inputs, importing metrics from `zreg.metrics` (installed package), and writing outputs to `evaluation/runs/` via `log_run()`.
  **Requirements:** EVAL-05
  **Depends on:** Phase 15
  Plans:
  - [x] 16-01-PLAN.md — `eval/run_synthetic.py` (noise + outlier sweep) + `eval/run_real.py` (scale/density sweep with missing-data guard)
  - [x] 16-02-PLAN.md — `tests/test_runners.py` (TestRunSynthetic + TestRunReal: file creation, field presence, missing-data skip, import path verification)

- [x] **Phase 17: Framework Config & DataFactory** (2/2 plans) — completed 2026-05-27
  **Goal:** Deliver `eval/config.py` (`EvalConfig` dataclass with YAML loading/validation and `EvalConfigError`) and `eval/data_factory.py` (`DataFactory` wrapping existing `load_data_from_tracklets`, `load_shah_from_csv`, generators, and corruption functions) — the data foundation all downstream phases depend on.
  **Requirements:** FRAME-01, FRAME-02
  **Depends on:** Phase 16
  Plans:
  - [x] 17-01-PLAN.md — `eval/config.py` (EvalConfig pydantic BaseModel + EvalConfigError + from_yaml) + setup.cfg pydantic/pyyaml deps + tests/test_data_factory.py TestEvalConfigFromYAML class (FRAME-01)
  - [x] 17-02-PLAN.md — `eval/data_factory.py` (DataFactory class: load_real, generate_synthetic, augment, prepare_split, get_ground_truth) + 6 populated test classes in tests/test_data_factory.py (FRAME-02)

- [x] **Phase 18: MetricsEngine & Result Types** (2/2 plans) — Complete 2026-05-28
  **Goal:** Deliver `eval/types.py` (six dataclasses: `AlignResult`, `LabelResult`, `StageMetrics`, `Trial`, `SearchResult`, `EvalReport`) and `eval/metrics.py` (`MetricsEngine` wrapping all existing `zreg.metrics.*` with normalization, aggregation, scoring, and sanity checking).
  **Requirements:** FRAME-03, FRAME-04
  **Depends on:** Phase 17
  Plans:
  - [x] 18-01-PLAN.md — `eval/types.py` (6 frozen pydantic result models per FRAME-04) + `tests/test_metrics.py` scaffold (2 populated + 4 stubbed test classes)
  - [x] 18-02-PLAN.md — `EvalConfig.metric_weights` extension + `eval/metrics.py` `MetricsEngine` + populate remaining 4 test classes (FRAME-03)
  **Success criteria:**
  1. All six dataclasses importable from `eval.types`
  2. `MetricsEngine.normalize()` maps every metric to [0,1] with correct direction (lower/higher is better)
  3. `MetricsEngine.sanity_check()` fires warnings on degenerate inputs (empty cloud, all-same labels)
  4. `MetricsEngine.compute_score()` returns a scalar in [0,1]
  5. `tests/test_metrics.py` passes all gate criteria on handcrafted fixtures

- [x] **Phase 19: AlignmentStage** (2/2 plans) — Complete 2026-05-28
  **Goal:** Deliver `eval/stages/base.py` (`PipelineStage` ABC) and `eval/stages/alignment.py` (`AlignmentStage`) wrapping existing DTW + CPD code — standalone-runnable, `validate_params`-gated, with full test coverage.
  **Requirements:** FRAME-05
  **Depends on:** Phase 18
  **Success criteria:**
  1. `AlignmentStage.run()` completes without `LabelTransferStage` present
  2. DTW distance after alignment is measurably smaller than before on ≥2 synthetic datasets
  3. `validate_params()` raises on missing/invalid params
  4. `tests/test_alignment_stage.py` passes all gate criteria
  5. No reimplementation of DTW or CPD — all calls delegate to `zreg.dtw.*` and `zreg.cpd.*`
  **Plans:** 2 plans
  Plans:
  - [x] 19-01-PLAN.md — `StageResult` TypeAlias in `eval/types.py` + `eval/stages/base.py` (`PipelineStage` ABC) + `eval/stages/__init__.py` + test scaffold (2 populated + 4 stubbed)
  - [x] 19-02-PLAN.md — `eval/stages/alignment.py` (`AlignmentStage`) + update `eval/stages/__init__.py` + populate 4 test stubs

- [x] **Phase 20: LabelTransferStage** (2/2 plans) — completed 2026-05-29
  **Goal:** Deliver `eval/stages/label_transfer.py` (`LabelTransferStage`) wrapping existing `color_transfer` code — accepts raw or aligned clouds, standalone-runnable, with tests proving accuracy beats random baseline and correct chaining with `AlignmentStage`.
  **Requirements:** FRAME-06
  **Depends on:** Phase 19
  **Plans:** 2 plans
  Plans:
  - [x] 20-01-PLAN.md — `eval/stages/label_transfer.py` (`LabelTransferStage`) + update `eval/stages/__init__.py`
  - [x] 20-02-PLAN.md — `tests/test_label_transfer_stage.py` (5 test classes covering all FRAME-06 gate criteria)
  **Success criteria:**
  1. `LabelTransferStage.run()` completes without `AlignmentStage` present (raw cloud input)
  2. Label accuracy on synthetic data beats random baseline
  3. Stage accepts `AlignResult.aligned_cloud` as input (correct chaining)
  4. `tests/test_label_transfer_stage.py` passes all gate criteria
  5. No reimplementation of color transfer — delegates to `zreg.color_transfer`

- [x] **Phase 21: EvaluationRunner & Visualisation** (2/2 plans) — completed 2026-05-29
  **Goal:** Deliver `eval/runners/eval_runner.py` (`EvaluationRunner`) orchestrating DataFactory → stages → MetricsEngine → aggregation → sanity checks → plots → `eval_report.json`, plus `eval/viz.py` with matplotlib Agg backend point-cloud and metric-summary plots.
  **Requirements:** FRAME-07, FRAME-08
  **Depends on:** Phase 20
  **Plans:** 2 plans
  Plans:
  - [x] 21-01-PLAN.md — `eval/runners/__init__.py` + `eval/runners/eval_runner.py` (EvaluationRunner orchestrator) + setup.cfg matplotlib in install_requires + `tests/test_eval_runner.py` (5 FRAME-07 gate test classes)
  - [x] 21-02-PLAN.md — `eval/viz.py` (plot_point_cloud + plot_metrics_summary, FRAME-08 Agg backend) + wire save_plots branch in EvaluationRunner.run() + `tests/test_viz.py` + TestEvaluationRunnerSavePlots integration class
  **Success criteria:**
  1. `EvaluationRunner.run()` with fixed params produces complete report (JSON + plots) in `output_dir`
  2. `eval_report.json` contains both per-dataset metrics and aggregated overview
  3. `sanity_flags` list is non-empty when intentionally bad inputs are provided
  4. All figure code runs inside `matplotlib.rc_context`; `plt.close(fig)` called; Agg backend used
  5. `tests/test_eval_runner.py` passes all gate criteria

- [x] **Phase 22: HyperparamOptimizer & Search Strategies** (2/2 plans) — completed 2026-05-29
  **Goal:** Deliver `eval/search_strategies.py` (GridSearch, RandomSearch, Optuna Bayesian via Optuna 4.x with SQLite storage) and `eval/runners/optimizer.py` (`HyperparamOptimizer`) with sanity/dev/full tier logic, candidate pruning, and JSON output.
  **Requirements:** FRAME-09, FRAME-10
  **Depends on:** Phase 21
  **Success criteria:**
  1. Sanity tier completes on laptop in under 2 minutes
  2. `best_params.json` and `search_history.json` written to `output_dir`
  3. Best params from optimizer improve score vs default params (verified via EvaluationRunner)
  4. `prune_candidates()` demonstrably reduces candidate count between tiers
  5. `tests/test_optimizer.py` passes all gate criteria; Optuna ≥4.0,<5 with TPE sampler
  **Plans:** 2 plans
  Plans:
  - [x] 22-01-PLAN.md — `optuna` dep + test scaffold (Wave 0) + `eval/search_strategies.py` (GridSearch, RandomSearch, BayesianSearch) + `eval/runners/optimizer.py` (HyperparamOptimizer)
  - [x] 22-02-PLAN.md — Populate `tests/test_optimizer.py` with all 5 FRAME-09+10 gate test classes

- [x] **Phase 23: CLI Entrypoint & Scenario Configs** (2/2 plans) — completed 2026-06-02
  **Goal:** Deliver `run_eval.py` CLI (`--config`, `--mode optimize|eval|full`) and 5 scenario YAML configs in `configs/`; `full` mode chains Optimizer → EvaluationRunner; all 5 configs run end-to-end without errors.
  **Requirements:** FRAME-11, FRAME-12
  **Depends on:** Phase 22
  **Success criteria:**
  1. All 5 scenario configs (`alignment_sanity`, `alignment_dev`, `label_transfer_sanity`, `label_transfer_dev`, `combined_full`) run without errors
  2. `--mode optimize`, `--mode eval`, `--mode full` all work correctly
  3. Config validation errors produce readable messages — no raw stacktraces
  4. `run_config.yaml` copy written to `output_dir` for each run (reproducibility)
  5. `run_eval.py --help` shows all flags with descriptions
  **Plans:** 2 plans
  Plans:
  - [x] 23-01-PLAN.md — `run_eval.py` CLI entrypoint (argparse, 4 flags, EvalConfigError clean message, run_config.yaml exact-copy before pipeline, mode dispatch) + `tests/test_cli.py` with FRAME-11 unit tests populated (8 tests)
  - [x] 23-02-PLAN.md — 5 scenario YAML configs in `configs/` (alignment_sanity, alignment_dev, label_transfer_sanity, label_transfer_dev, combined_full) + populate `TestScenarioConfigs` (FRAME-12)

- [x] **Phase 24: Trajectory Export** (2/2 plans) — completed 2026-06-04
  **Goal:** Export `AlignResult` and/or `LabelResult` as point-per-row CSV files (`align_trajectory.csv`, `label_trajectory.csv`) with accompanying `metadata.json`, written automatically from `EvaluationRunner.run()` conditional on which stages ran. CSV columns: `frame_idx, point_idx, x, y, z, label` — optimised for LaTeX/pgfplots reuse.
  **Requirements:** EXT-01
  **Depends on:** Phase 21
  **Success criteria:**
  1. `align_trajectory.csv` written when `run_alignment=True`; absent otherwise
  2. `label_trajectory.csv` written when `run_label_transfer=True`; absent otherwise
  3. CSV has correct header and one row per point per frame
  4. `align_metadata.json` / `label_metadata.json` contain all required fields (run_id, frame_count, frame_indices, data_path, params_used, tier, n_trials, n_synthetic, git_hash, zreg_version, timestamp)
  5. `EvalReport.trajectory_paths` list reflects written files
  **Plans:** 2 plans
  Plans:
  - [x] 24-01-PLAN.md — `eval/tracking/trajectory.py` (`export_trajectory` + CSV/JSON helpers) + update `eval/tracking/__init__.py`
  - [x] 24-02-PLAN.md — Wire into `EvaluationRunner.run()` + `EvalReport.trajectory_paths` field + `tests/test_trajectory_export.py`

- [x] **Phase 25: Visualisation Refactor** (completed 2026-06-04)
  **Goal:** Replace `plot_point_cloud` with `plot_trajectory` — two independent 1×3 figures (first/middle/last frame), each saved as PDF + PNG. `alignment_trajectory.pdf/png` shows source (blue) vs. aligned (orange) point clouds per frame. `label_trajectory.pdf/png` shows label-coloured clouds with a legend using `EvalConfig.label_names` when provided. Rename `plot_metrics_summary` → `plot_metrics`. Each file is written only when the corresponding stage ran (D-03).
  **Requirements:** EXT-02
  **Depends on:** Phase 21, Phase 17
  **Success criteria:**
  1. `plot_trajectory` produces `alignment_trajectory.pdf/png` and/or `label_trajectory.pdf/png` — filenames depend on which stages ran
  2. Each figure is a 1×3 grid; columns are first, middle, last frame of the trajectory
  3. `alignment_trajectory` subplots show source (blue) and aligned (orange) clouds superimposed; shared legend identifies "Source" and "Aligned"
  4. `label_trajectory` subplots show label-coloured clouds; `label_names` values used in legend when provided, raw int-as-string otherwise
  5. `plot_point_cloud` no longer importable from `eval.viz`
  6. All four stage-combination cases (align-only, label-only, both, neither) produce correct output (2 files, 2 files, 4 files, [])
  **Plans:** 2 plans
  Plans:
  - [x] 25-01-PLAN.md — Wave 0 TDD stubs in `tests/test_viz.py` + implement `plot_trajectory` + rename `plot_metrics_summary` → `plot_metrics` in `eval/viz.py` + add `label_names` to `EvalConfig`
  - [x] 25-02-PLAN.md — Update `EvaluationRunner.run()` save_plots branch (D-10) + update `TestEvaluationRunnerSavePlots` assertions

- [x] **Phase 26: Propulate Optimizer** (2/2 plans) — completed 2026-06-06
  **Goal:** Add `PropulateSearch` as a selectable optimizer backend alongside Optuna. `HyperparamOptimizer` auto-selects Propulate when `SLURM_JOB_ID` is set or MPI world size > 1; defaults to Optuna otherwise. Users override via `optimiser: auto|optuna|propulate` in config.
  **Requirements:** EXT-03
  **Depends on:** Phase 22, Phase 17
  **Success criteria:**
  1. `optimiser: propulate` selects `PropulateSearch` unconditionally
  2. `optimiser: auto` selects Propulate when `SLURM_JOB_ID` set or MPI world size > 1
  3. Missing propulate install raises `ImportError` with install instructions
  4. `PropulateSearch.search` produces valid `Trial` objects (integration test via `mpirun -n 2`)
  5. Existing Optuna path unaffected — all Phase 22 tests still pass
  6. Docstrings explain when to prefer each backend
  **Plans:** 2 plans
  Plans:
  - [x] 26-01-PLAN.md — Test scaffold (`tests/test_propulate.py` + `tests/_propulate_mwe.py`) + `search_strategy` docstring extension in `eval/config.py` (D-01: no new field) + `PropulateSearch` in `eval/search_strategies.py` (D-12 lazy import) + `_detect_backend()` + auto/propulate dispatch in `eval/runners/optimizer.py` + propulate optional extra in `setup.cfg` (`propulate>=1.0,<2` per Pitfall 3)
  - [x] 26-02-PLAN.md — Populate 5 test classes in `tests/test_propulate.py` (TestPropulateImportGuard, TestEvalConfigAcceptsPropulate, TestDetectBackend, TestRunDispatchPropulate, TestPropulateMPIIntegration) + verify Phase 22 regression + full suite green

<details>
<summary>✅ v1.1 Code Quality & Refactoring (Phases 6–11.1) — SHIPPED 2026-05-13</summary>

- [x] Phase 6: Python 3.12 Migration (2/2 plans) — completed 2026-04-13
- [x] Phase 7: CPD Deep Restructure (3/3 plans) — completed 2026-04-20
- [x] Phase 8: DTW Deep Restructure (2/2 plans) — completed 2026-04-23
- [x] Phase 9: Distance & Transform Restructure (2/2 plans) — completed 2026-04-27
- [x] Phase 10: Code Quality & Verification (3/3 plans) — completed 2026-04-29
- [x] Phase 11: Validate Refactoring and Fix Coverage (1/1 plan) — completed 2026-05-12
- [x] Phase 11.1: Close DTW-02 — Consistent Metric Variant Interface (1/1 plan, INSERTED) — completed 2026-05-13

Full details: [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1.0 Consolidation (Phases 1-5) — SHIPPED 2026-04-09</summary>

- [x] Phase 1: Validation Foundation & Quick Wins (2/2 plans) — completed 2026-04-09
- [x] Phase 2: Distance Metric & CPD Bug Fixes (3/3 plans) — completed 2026-04-09
- [x] Phase 3: DTW, Transform & CPD Enhancements (3/3 plans) — completed 2026-04-09
- [x] Phase 4: Infrastructure & Color Transfer Quality (2/2 plans) — completed 2026-04-09
- [x] Phase 5: Test Coverage (3/3 plans) — completed 2026-04-09

Full details: [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Validation Foundation & Quick Wins | v1.0 | 2/2 | Complete | 2026-04-09 |
| 2. Distance Metric & CPD Bug Fixes | v1.0 | 3/3 | Complete | 2026-04-09 |
| 3. DTW, Transform & CPD Enhancements | v1.0 | 3/3 | Complete | 2026-04-09 |
| 4. Infrastructure & Color Transfer Quality | v1.0 | 2/2 | Complete | 2026-04-09 |
| 5. Test Coverage | v1.0 | 3/3 | Complete | 2026-04-09 |
| 6. Python 3.12 Migration | v1.1 | 2/2 | Complete | 2026-04-13 |
| 7. CPD Deep Restructure | v1.1 | 3/3 | Complete | 2026-04-20 |
| 8. DTW Deep Restructure | v1.1 | 2/2 | Complete | 2026-04-23 |
| 9. Distance & Transform Restructure | v1.1 | 2/2 | Complete | 2026-04-27 |
| 10. Code Quality & Verification | v1.1 | 3/3 | Complete | 2026-04-29 |
| 11. Validate Refactoring and Fix Coverage | v1.1 | 1/1 | Complete | 2026-05-12 |
| 11.1. Close DTW-02: consistent metric variant interface | v1.1 | 1/1 | Complete | 2026-05-13 |
| 12. Carry-Forward Debt Closure | v1.2 | 3/3 | Complete | 2026-05-14 |
| 13. Core Metrics Library | v1.2 | 3/3 | Complete | 2026-05-15 |
| 14. Synthetic Data Generators | v1.2 | 3/3 | Complete | 2026-05-18 |
| 15. Experiment Tracking & Run Management | v1.2 | 2/2 | Complete | 2026-05-18 |
| 16. Runner Scripts | v1.2 | 2/2 | Complete | 2026-05-19 |
| 17. Framework Config & DataFactory | v1.2 | 2/2 | Complete | 2026-05-27 |
| 18. MetricsEngine & Result Types | v1.2 | 2/2 | Complete | 2026-05-28 |
| 19. AlignmentStage | v1.2 | 2/2 | Complete | 2026-05-28 |
| 20. LabelTransferStage | v1.2 | 2/2 | Complete | 2026-05-29 |
| 21. EvaluationRunner & Visualisation | v1.2 | 2/2 | Complete | 2026-05-29 |
| 22. HyperparamOptimizer & Search Strategies | v1.2 | 2/2 | Complete | 2026-05-29 |
| 23. CLI Entrypoint & Scenario Configs | v1.2 | 2/2 | Complete | 2026-06-02 |
| 24. Trajectory Export | v1.2 | 2/2 | Complete | 2026-06-04 |
| 25. Visualisation Refactor | v1.2 | 2/2 | Complete    | 2026-06-04 |
| 26. Propulate Optimizer | v1.2 | 2/2 | Complete | 2026-06-06 |
| 27. DataFactory Geometric Augmentation Methods | v1.2 | 2/2 | Complete | 2026-06-11 |
| 28. Script Integration — generate_datasets uses DataFactory | v1.2 | 0/1 | Not started | — |
| 29. Viz Unification | v1.2 | 2/2 | Complete | 2026-06-12 |
| 30. Two-Dataset Paired Alignment Architecture | v1.2 | 3/3 | Complete | 2026-06-12 |
| 31. Synthetic Pipeline Mode — Transform-Spec Target & GT-Aware HPO | v1.2 | 3/3 | Complete | 2026-06-14 |
| 32. Heterogeneous Paired Evaluation — target_data_format | v1.2 | 2/2 | Complete | 2026-06-14 |
| 35. Reuse Step-1 CPD Transforms in Aligned-Cloud Construction | v1.2 | 2/2 | Complete | 2026-06-22 |
| 38. zRegPointCloud color→label Field Rename | v1.2 | 2/2 | Complete | 2026-06-24 |
