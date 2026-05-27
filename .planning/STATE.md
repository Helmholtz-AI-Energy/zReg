---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Evaluation Framework & Debt Resolution
status: in_progress
stopped_at: Phase 17 complete 2026-05-27
last_updated: "2026-05-27T12:30:00.000Z"
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 14
  completed_plans: 2
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Current focus:** Phase 18 — MetricsEngine & Result Types (next)

## Current Position

Phase: 18 — MetricsEngine & Result Types
Plan: —
Status: Pending (phase 17 complete 2026-05-27)
Last activity: 2026-05-27 — Phase 17 executed (FRAME-01: EvalConfig, FRAME-02: DataFactory — 21 tests green, 579 suite green)

## Phase Overview

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 16 | Runner Scripts | EVAL-05 | Complete 2026-05-19 |
| 17 | Framework Config & DataFactory | FRAME-01, FRAME-02 | Complete 2026-05-27 |
| 18 | MetricsEngine & Result Types | FRAME-03, FRAME-04 | Pending |
| 19 | AlignmentStage | FRAME-05 | Pending |
| 20 | LabelTransferStage | FRAME-06 | Pending |
| 21 | EvaluationRunner & Visualisation | FRAME-07, FRAME-08 | Pending |
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

### Open Blockers

None.

## Session Continuity

Last session: 2026-05-27
Stopped at: Phase 17 complete — FRAME-01 (EvalConfig) + FRAME-02 (DataFactory), 16/16 verification passed
Next action: `/gsd-discuss-phase 18` or `/gsd-plan-phase 18`
