---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Evaluation Framework & Debt Resolution
status: complete
stopped_at: Phase 26 complete
last_updated: "2026-06-06T12:00:00.000Z"
last_activity: 2026-06-06 -- Phase 26 complete (Propulate Optimizer)
progress:
  total_phases: 22
  completed_phases: 22
  total_plans: 47
  completed_plans: 47
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** v1.2 milestone complete

## Current Position

Phase: 25
Plan: Not started
Status: Ready to execute
Last activity: 2026-06-06 -- Phase 26 planning complete

## Phase Overview

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 16 | Runner Scripts | EVAL-05 | Complete 2026-05-19 |
| 17 | Framework Config & DataFactory | FRAME-01, FRAME-02 | Complete 2026-05-27 |
| 18 | MetricsEngine & Result Types | FRAME-03, FRAME-04 | Complete 2026-05-28 |
| 19 | AlignmentStage | FRAME-05 | Complete 2026-05-28 |
| 20 | LabelTransferStage | FRAME-06 | Complete 2026-05-29 |
| 21 | EvaluationRunner & Visualisation | FRAME-07, FRAME-08 | Complete 2026-05-29 |
| 22 | HyperparamOptimizer & Search Strategies | FRAME-09, FRAME-10 | Pending |
| 23 | CLI Entrypoint & Scenario Configs | FRAME-11, FRAME-12 | Pending |

## Accumulated Context

### Roadmap Evolution

- Phase 15 added: Experiment Tracking & Run Management
- Phase 16 added: Runner Scripts
- Phases 17–23 added 2026-05-27: Full eval framework (config-driven, modular stages, optimizer, CLI) — EVAL-06 and EVAL-07 superseded by FRAME-09/10 and FRAME-07/08

### Decisions (v1.1)

- [Phase 08]: compose_constraints uses variadic args with AND semantics for flexible constraint composition
- [Phase 08]: DTW package restructure complete — DynamicTimeWarping, DTWResult, compose_constraints as public API
- [Phase 11-01]: torch.zeros(5,3) with ratio=1.0 is the canonical trigger for _fps_numpy break condition
- [Phase 11-01]: No pragma: no cover annotation needed — 93% coverage achieved without suppression
- [Phase 11-01]: _preserve_labels had zero callers and was safely removed from downsampling.py
- [Phase 11.1-01]: Callable pass-through in _sanitize_pairwise_distance_matrix — any Callable[[Tensor, Tensor], Tensor] accepted without special-casing
- [Phase 11.1-01]: DistanceMetric Protocol promoted to public API from zreg.distances

### Phase 12 Decisions (v1.2)

- CARRY-01 closed: `collections.abc.Callable` in cpd/base.py and _registration.py
- CARRY-02 closed: DistanceMetric used as annotation in pairwise_distance_matrix.py and dtw/core.py (Phase 11.1)
- CARRY-03 closed: color_transfer.py now uses `from .cpd import EstepResult`
- CARRY-04 closed: `from . import config as config` added to __init__.py
- CARRY-05 closed: VALIDATION.md at repo root with 7 v1.1 phase records
- open3d imports made lazy throughout (dataset, downsampling, homogeneous, _registration) — no SIGABRT
- All open3d imports are optional; HAS_OPEN3D computed via importlib.util.find_spec

### Phase 13 Decisions (v1.2)

- EVAL-01 closed: `src/zreg/metrics/alignment.py` — chamfer/hausdorff via torch.cdist+quantile (GPU-safe), knn_consistency via sklearn KDTree with .detach().cpu().numpy(), temporal_stability via manual 4×4 matrix construction from .rot/.t/.scale and .b/.t
- EVAL-02 closed: `src/zreg/metrics/label_transfer.py` — compute_f1 with sentinel masking (y_true != -1), average="weighted" default (proto stub bug fixed), zero_division=0
- temporal_stability([]) and temporal_stability([tf]) return torch.tensor(0.0) — no exception
- path_smoothness returns variance of slope changes; short paths (< 4 points) return 0.0
- `src/zreg/metrics/__init__.py` is the new package init — re-exports all 6 functions; src/zreg/eval/ is never created
- Proto stubs (alignment_metrics.py, label_transfer_metrics.py) left as orphaned reference code per D-03
- 480 tests pass after phase 13 (40 new in test_eval_metrics.py, 32 in test_alignment_metrics.py, 17 in test_label_transfer_metrics.py)

### Phase 14 Decisions (v1.2)

- EVAL-03 closed: `eval/generators/` package at repo root — 5 files, 7 public symbols, no eval/__init__.py (namespace dir)
- generate_trajectory uses independent Gaussian draws per frame (torch.randn per frame, not incremental perturbations)
- AffineTransformation() default has t=[1,1,1] (NOT identity) — tests must use AffineTransformation(t=torch.zeros(3)) for identity verification
- apply_rigid/apply_affine use shared _apply_matrix helper with defensive w-divide (matches homogeneous.py pattern; unreachable for rigid/affine but retained for consistency)
- deepcopy precedes manual_seed in stochastic wrappers (add_gaussian_noise, add_outliers, generate_labels) — reproducible but values offset from naive seed expectation; documented in REVIEW.md WR-03
- generate_labels uses torch.cdist Voronoi assignment with N(0,I) seed points — produces spatially coherent clusters compatible with compute_f1 without conversion
- 514 tests pass after phase 14 (34 new in test_generators.py; 480 pre-existing unaffected)

### Phase 15 Decisions (v1.2)

- log_run() plain function (stateless) — returns run_id str, auto-captures git_hash/zreg_version/timestamp internally
- run_id is caller-provided required str (no auto-generation)
- Per-run output: evaluation/runs/{run_id}.json + evaluation/runs/{run_id}.csv (single-row CSV)
- output_dir defaults to "evaluation/runs/"; auto-created via Path.mkdir(parents=True, exist_ok=True)
- Unit tests in tests/test_tracking.py; auto-captured fields tested via unittest.mock.patch

### Phase 17 Decisions (v1.2)

- FRAME-01 closed: `eval/config.py` — EvalConfig pydantic v2 BaseModel (16 fields, only data_path required, extra="forbid"), EvalConfigError(ValueError), from_yaml with safe_load + exception-handler MRO order (FileNotFoundError → YAMLError → ValidationError)
- FRAME-02 closed: `eval/data_factory.py` — DataFactory with lazy init (D-08), by-reference cache (D-09), tracklets/CSV dispatch (Pitfalls 4+5), noise-then-outliers augment, sorted-key random split with single-frame guard (D-07), pc["id"] ground-truth extraction (D-10/D-11)
- eval/ is a namespace directory (no __init__.py); conftest.py:16 inserts repo root for test discovery
- pydantic 2.12.2 + pyyaml 6.0.3 installed; declared in setup.cfg install_requires
- macOS ARM import order: zreg.dataset → zreg.generators → torch → eval.config (enforced in data_factory.py)
- prepare_split non-reproducible by design (T-17-07 accepted; D-05 requires random.sample)
- 579 tests pass after phase 17 (21 new in test_data_factory.py; 558 pre-existing unaffected)
- CR-01 open: eval/ not pip-discoverable beyond conftest.py sys.path insertion — requires attention before Phase 23 CLI

### Phase 18 Decisions (v1.2)

- [Phase 18-01]: FRAME-04 (result types) closed — `eval/types.py` defines 6 frozen pydantic v2 models (AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport) with `ConfigDict(frozen=True, arbitrary_types_allowed=True)` per D-02
- [Phase 18-01]: StageMetrics carries 6 raw float fields per D-03 plus `normalized: dict[str, float] = Field(default_factory=dict)`; canonical short-name key set documented in class docstring per Pitfall 4
- [Phase 18-01]: Pitfall 1 (shallow frozen — `sm.normalized["x"] = 1.0` silently succeeds) and Pitfall 2 (`model_dump_json()` cannot serialise torch.Tensor) documented in `eval/types.py` module docstring so Plan 18-02 / Phase 21 consumers do not re-discover them
- [Phase 18-01]: Trial.metrics typed as StageMetrics; SearchResult.history typed as list[Trial] (RESEARCH Q5/Q6 recommendations adopted)
- [Phase 18-01]: tests/test_metrics.py scaffolded with 6 test classes — TestResultTypesImportable + TestStageMetricsFrozen populated (4 active tests), TestNormalize/TestComputeScore/TestSanityCheck/TestAggregate stubbed with `pytest.skip("populated in Plan 18-02")` so eval.metrics import is deferred and suite remains green
- [Phase 18-01]: 583 tests pass after Plan 18-01 (+4 active vs Phase 17 baseline 579), 21 skipped (+4 stub placeholders)
- [Phase 18 gap closure]: CR-01: Field(ge=0) added to StageMetrics lower-is-better fields; CR-02: compute_score() normalises over present keys only; CR-03: empty tensor guard before all-sentinel check

### Phase 19 Decisions (v1.2)

- FRAME-05 closed: `eval/stages/base.py` — PipelineStage ABC with @abstractmethod run() and concrete no-op validate_params(); `eval/stages/__init__.py` — regular package exporting PipelineStage + AlignmentStage; `eval/stages/alignment.py` — AlignmentStage delegates entirely to DynamicTimeWarping(...).compute()
- StageResult TypeAlias (D-01): `StageResult: TypeAlias = Union[AlignResult, LabelResult]` added to eval/types.py; in __all__
- eval/stages/ is a regular package (not namespace dir) — required for `from eval.stages import PipelineStage` to work (D-11)
- D-09: run() calls validate_params() as first line; confirmed via test_run_calls_validate_first (empty params raises ValueError not KeyError)
- D-04/D-05: _count_jumps staticmethod counts diagonal-to-non-diagonal transitions; n_breakpoints cap applied in run(), not in _count_jumps
- Pydantic v2 validates typed dicts into new objects — `aligned_cloud is dataset` identity test replaced with `==` equality (shallow dict comparison); pass-through documented in run() docstring
- DynamicTimeWarping called with `downsample_method=None` to avoid crash on `color=None` in generate_trajectory fixtures (pre-existing upstreamissue in pairwise_distance_matrix.py — out of scope for Phase 19)
- Known upstream limitation documented in alignment.py module docstring: pairwise_distance_matrix.py:182-194 hardcodes RigidCPD whenever cpd_type is not None; cpd_penalty string is a binary toggle, not a dispatcher
- 628 tests pass after Phase 19 (24 new in test_alignment_stage.py; 604 pre-existing unaffected)
- WR-01 open (CR): bool subclass of int allows window_size=True/False — fix: add `not isinstance(x, bool)` guards in validate_params
- WR-02 open (CR): aligned_cloud=dataset stores caller's dict by reference; shallow frozen hazard not documented on AlignResult field

### Phase 20 Decisions (v1.2)

- FRAME-06 closed: `eval/stages/label_transfer.py` — LabelTransferStage(PipelineStage) with KNN_VOTING delegation, 4-param validation + bool exclusion guards, frame-0 pass-through (D-02), source_colors.unsqueeze(-1) / [:, 0] squeeze (D-12)
- `eval/stages/__init__.py` — __all__ extended to ["PipelineStage", "AlignmentStage", "LabelTransferStage"]
- 5 FRAME-06 gate test classes: TestLabelTransferStageRunStandalone, TestLabelTransferStageLabelAccuracy, TestLabelTransferStageChainedRun, TestLabelTransferStageValidateParams, TestLabelTransferStageOutputShape — all passing
- D-09 fixture pattern: 1-frame generate_labels + add_gaussian_noise for frame 1 (not 2-frame generate_labels — avoids independent RNG pitfall)
- compute_f1(y_true, y_pred) positional order enforced in test: ground_truth_labels first (Pitfall 5 from RESEARCH.md)
- Manual AlignResult construction for chained-run test (D-11) — no DTW end-to-end needed
- 690 tests pass after Phase 20 (17 new in Plan 20-02; 673 pre-existing unaffected); 17 skipped

### Phase 21 Decisions (v1.2)

- FRAME-07 closed: `eval/runners/eval_runner.py` — EvaluationRunner orchestrates DataFactory → AlignmentStage → LabelTransferStage → MetricsEngine → EvalReport; D-06 fail-fast guard, D-12 mkdir-first, D-10 model_dump+json.dump, transforms=[] for compute_stage_metrics
- FRAME-08 closed: `eval/viz.py` — plot_point_cloud (3D scatter, ≤4 frames, AlignResult-only) + plot_metrics_summary (6 horizontal bars, [0,1] x-axis); all figure code inside matplotlib.rc_context({"backend":"Agg"}); plt.close(fig) after every savefig
- Ground truth in tests: dataset[k]["color"] (torch.long, populated by generate_labels) — id=None in generate_trajectory output
- per_dataset: flat metric→mean-float dict extracted from aggregate() output, not nested agg itself
- preliminary_report + model_copy(update={"plot_paths": ...}) pattern to mutate frozen EvalReport (Pitfall 6)
- point_cloud.pdf only when run_alignment=True (align result available); metrics_summary.pdf always when save_plots=True
- WR-01 open (CR): matplotlib.rc_context backend Agg leaks permanently — interactive callers get broken display after calling viz functions
- WR-05 open (CR): eval/ excluded from --cov in setup.cfg; coverage of eval/runners/ and eval/viz.py is untracked
- 718 tests pass after Phase 21 (+28 vs Phase 20 baseline 690); 17 skipped unchanged

### Phase 24 Decisions (v1.2)

- EXT-01 closed: `eval/tracking/trajectory.py` — export_trajectory writes align_trajectory.csv (5 cols) + align_metadata.json and label_trajectory.csv (6 cols) + label_metadata.json; stdlib-only (csv, json, uuid, subprocess, importlib.metadata)
- aligned_cloud uses zRegPointCloud instances (not plain dicts) — pydantic AlignResult validates; test fixtures must use zRegPointCloud
- D-07/D-08: label_trajectory.csv uses aligned_cloud["pos"] when align ran, else dataset["pos"]
- Single run_id UUID generated once per export_trajectory call; shared across both metadata files when both stages ran
- trajectory_paths: list[str] = Field(default_factory=list) added to EvalReport between plot_paths and sanity_flags
- EvaluationRunner.run(): unconditional export (D-13), single model_copy for both plot_paths + trajectory_paths (D-14)
- 787 tests pass after Phase 24 (+46 vs Phase 23 baseline 741); 17 skipped unchanged
- WR-01 open (CR): subprocess.run missing timeout= in trajectory.py git hash capture
- WR-02 open (CR): no length guard for labels vs pos mismatch in label CSV loop
- WR-03 open (CR): bare dict access result["align"]/result["label"] raises KeyError on malformed input
- WR-04 open (CR): three test classes make live git subprocess calls (should patch)

### Open Blockers

None.

## Session Continuity

Last session: 2026-06-06T10:06:10.647Z
Stopped at: Phase 26 context gathered
Resume file: .planning/phases/26-propulate-optimizer/26-CONTEXT.md
Next action: `/gsd-plan-phase 22` — HyperparamOptimizer & Search Strategies (FRAME-09, FRAME-10)
