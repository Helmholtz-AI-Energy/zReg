---
phase: 21-evaluationrunner-visualisation
plan: "01"
subsystem: eval/runners
tags: [evaluation, orchestration, frame-07, runner, metrics]
dependency_graph:
  requires:
    - eval/stages/alignment.py        # AlignmentStage
    - eval/stages/label_transfer.py   # LabelTransferStage
    - eval/data_factory.py            # DataFactory
    - eval/metrics.py                 # MetricsEngine
    - eval/types.py                   # EvalReport, StageMetrics, AlignResult, LabelResult
    - eval/config.py                  # EvalConfig
  provides:
    - eval/runners/__init__.py        # EvaluationRunner package
    - eval/runners/eval_runner.py     # EvaluationRunner class
  affects:
    - tests/test_eval_runner.py       # new test file
    - setup.cfg                       # matplotlib in install_requires
tech_stack:
  added:
    - matplotlib (moved from extras_require to install_requires)
  patterns:
    - pydantic v2 frozen model mutation via model_copy(update={...})
    - unittest.mock.patch on DataFactory to avoid real I/O in unit tests
    - libomp SIGABRT import order: zreg.dataset → torch → eval.*
key_files:
  created:
    - eval/runners/__init__.py
    - eval/runners/eval_runner.py
    - tests/test_eval_runner.py
  modified:
    - setup.cfg
decisions:
  - "Use color tensor (not id) as ground truth in tests — generate_labels populates color (torch.long) but leaves id=None"
  - "per_dataset constructed as flat metric→mean-float dict extracted from agg, not nested agg itself — satisfies EvalReport.per_dataset: dict[str, dict[str, float]]"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-29T09:21:59Z"
  tasks_completed: 3
  tasks_total: 3
  files_created: 3
  files_modified: 1
  tests_before: 690
  tests_after: 706
  tests_added: 16
---

# Phase 21 Plan 01: EvaluationRunner Package Summary

**One-liner:** EvaluationRunner orchestrating DataFactory → AlignmentStage → LabelTransferStage → MetricsEngine → EvalReport with eval_report.json via model_dump + json.dump, plus matplotlib moved to install_requires.

## What Was Created

### 3 New Files

**`eval/runners/__init__.py`** — Regular package init exporting `EvaluationRunner` via `__all__`. Follows `eval/stages/__init__.py` pattern exactly (module docstring + single import + `__all__`).

**`eval/runners/eval_runner.py`** (~320 lines) — `EvaluationRunner` class implementing:
- `__init__(config, params)`: D-06 fail-fast guard, stores config, shallow-copy params, `MetricsEngine(config)`
- `run() → EvalReport`: D-12 mkdir first, DataFactory.load_real(), `_run_single()`, builds EvalReport, calls `save_report()`
- `_run_single(dataset, params) → dict`: argument assembly recipe — AlignmentStage/LabelTransferStage conditional execution, 8-positional `compute_stage_metrics` call, D-04 zero-fill via `model_copy(update={...})`
- `save_report(report, output_dir) → Path`: D-10 `model_dump()` + `json.dump(indent=2)`, D-11 fixed filename `eval_report.json`

**`tests/test_eval_runner.py`** (442 lines) — 5 test classes, 16 tests covering FRAME-07 gates G1-G3:
- `TestEvaluationRunnerInit` — constructor, D-06 ValueError, signature, shallow-copy identity
- `TestEvaluationRunnerRunFixedParams` (FRAME-07-G1) — EvalReport returned, 6 float metric fields, non-empty aggregated_metrics, per_dataset["dataset"] key
- `TestEvaluationRunnerSanityFlags` (FRAME-07-G2) — 1-frame dataset triggers "single-frame" sanity flag
- `TestEvaluationRunnerSaveReport` (FRAME-07-G3) — eval_report.json exists, valid JSON, expected keys, save_report returns path
- `TestEvaluationRunnerConditionalStages` — stage-skip mocking, both-disabled ValueError

### 1 Modified File

**`setup.cfg`** — `matplotlib` added to `install_requires` block (was only in `[extras_require] viz` and `all`). Addresses RESEARCH Finding 4 / Pitfall 5.

## Key Implementation Decisions Traced to D-01..D-12

| Decision | Implementation |
|----------|---------------|
| D-01 (flat params) | `self.params = dict(params)`; passed to both stages in `_run_single` |
| D-02 (constructor-only params) | `run()` uses `self.params` with no override; Phase 22 creates new runner per trial |
| D-03 (DataFactory internally) | `self.factory = DataFactory(self.config)` created inside `run()` |
| D-04 (zero-fill skipped stages) | `metrics.model_copy(update={...})` for alignment and label-transfer skipped paths |
| D-05 (raw dict to LT when no alignment) | `stage_input = dataset` when `run_alignment=False` |
| D-06 (fail-fast on both-off) | First line of `__init__` raises `ValueError("At least one stage must be enabled")` |
| D-10 (model_dump + json.dump) | `data = report.model_dump(); json.dump(data, f, indent=2)` — never `model_dump_json()` |
| D-11 (fixed filename) | `Path(output_dir) / "eval_report.json"` — caller cannot override |
| D-12 (mkdir first) | `Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)` is first line of `run()` |

## FRAME-07 Gate Proof

| Gate | Test class | Behavioral assertion |
|------|-----------|----------------------|
| G1 (fixed-params full-report) | `TestEvaluationRunnerRunFixedParams` | EvalReport with 6 float fields, non-empty aggregated_metrics, per_dataset["dataset"] |
| G2 (sanity-flag-on-bad-input) | `TestEvaluationRunnerSanityFlags` | 1-frame dataset → `len(report.sanity_flags) > 0` and "single-frame" substring present |
| G3 (eval_report.json in output_dir) | `TestEvaluationRunnerSaveReport` | File exists, valid JSON, all 6 top-level keys |

## Test Count Delta

- **Baseline (Phase 20 complete):** 690 passed, 17 skipped
- **After Plan 21-01:** 706 passed (+16), 17 skipped

## _run_single Argument Assembly Recipe

The 8-positional `compute_stage_metrics` call assembles arguments as follows:
- `source = dataset[sorted_keys[0]]["pos"]` — first frame positions
- `target = dataset[sorted_keys[-1]]["pos"]` — last frame positions (max temporal span, RESEARCH A2)
- `warp_path = align_result.warp_path if align_result else []`
- `transforms = []` — AlignResult has no transforms field; `temporal_stability([])` returns 0.0 (RESEARCH Pitfall 3)
- `y_true = gt[sorted_keys[-1]]` — ground truth labels for target frame from DataFactory.get_ground_truth()
- `y_pred = label_result.transferred_labels[sorted_keys[-1]]` or `torch.zeros_like(y_true)` (D-04)
- `points_for_knn = target`, `labels_for_knn = y_pred`
- `k_neighbours = params.get("k_neighbours", 10)` — uses .get() to handle run_label_transfer=False case

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed per_dataset construction — nested dict violated EvalReport type**
- **Found during:** Task 3 test execution
- **Issue:** Plan specified `per_dataset={"dataset": agg}` but `agg` is `dict[str, dict[str, float]]`, making per_dataset 3 levels deep. `EvalReport.per_dataset` type is `dict[str, dict[str, float]]` — pydantic raised `ValidationError` trying to assign `{"mean": float, ...}` to a `float` field in StageMetrics.
- **Fix:** Extract mean values: `per_dataset_flat = {metric: stats["mean"] for metric, stats in agg.items()}` then `per_dataset={"dataset": per_dataset_flat}`.
- **Files modified:** `eval/runners/eval_runner.py`
- **Commit:** 8e1e9f4

**2. [Rule 1 - Bug] Fixed ground truth fixture — generate_labels leaves id=None**
- **Found during:** Task 3 test execution
- **Issue:** Mock `get_ground_truth.return_value = {k: synthetic_dataset[k]["id"] ...}` returned `{k: None}` because `generate_labels()` populates `color` (torch.long labels) but sets `id=None`. Passing `None` to `compute_f1` raised `AttributeError: 'NoneType' object has no attribute 'ndim'`.
- **Fix:** Use `synthetic_dataset[k]["color"]` instead of `["id"]` in all mock ground-truth return values.
- **Files modified:** `tests/test_eval_runner.py`
- **Commit:** 8e1e9f4

## Known Stubs

None — `plot_paths=[]` is intentional per plan ("populated in Plan 21-02 integration; left empty here"). Plan 21-02 wires `eval/viz.py` and populates this field when `config.save_plots=True`.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary changes introduced.

## Self-Check: PASSED

- FOUND: eval/runners/__init__.py
- FOUND: eval/runners/eval_runner.py
- FOUND: tests/test_eval_runner.py
- FOUND commit: 1cd4e70 (Task 1)
- FOUND commit: 9ac7d18 (Task 2)
- FOUND commit: 8e1e9f4 (Task 3)
