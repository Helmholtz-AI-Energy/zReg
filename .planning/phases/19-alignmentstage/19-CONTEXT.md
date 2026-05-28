# Phase 19: AlignmentStage - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `eval/stages/base.py` (`PipelineStage` ABC) and `eval/stages/alignment.py` (`AlignmentStage`) wrapping existing `zreg.dtw.*` and `zreg.cpd.*` — standalone-runnable, `validate_params`-gated (auto-called inside `run()`), with full test coverage in `tests/test_alignment_stage.py`. Also adds `StageResult` TypeAlias to `eval/types.py` and creates `eval/stages/__init__.py` as a regular package.

All downstream phases (20–23) are out of scope for this phase.

</domain>

<decisions>
## Implementation Decisions

### StageResult Return Type
- **D-01:** `StageResult = Union[AlignResult, LabelResult]` added as a TypeAlias to `eval/types.py`, alongside the existing 6 result models. Single source of truth for all result types. Export via `__all__`.
- **D-02:** `PipelineStage.run()` signature: `run(self, dataset: dict[int, zRegPointCloud], params: dict[str, Any]) -> StageResult`. `dataset` matches the canonical shape returned by `DataFactory`; `params` is a loose dict validated by `validate_params`.
- **D-03:** `eval/stages/base.py` imports `StageResult` from `eval.types`. No circular imports — stages depend on types, types don't depend on stages.

### n_changepoints Computation
- **D-04:** `n_changepoints` is a **heuristic derived from `DTWResult.warping_path`** — no `ruptures` dependency, no external library needed.
- **D-05:** Path jump = a transition from a diagonal run (both i and j advance) to a non-diagonal run (only i or only j advances), or vice versa. `raw_jump_count` = number of such transitions in the full warping path.
- **D-06:** `n_changepoints = min(raw_jump_count, params["n_breakpoints"])`. `n_breakpoints` caps the reported count — HPO can tune it to suppress noisy transitions. This makes `n_breakpoints` a live HPO parameter.
- **D-07:** Hyperparam mapping to existing API:
  - `window_size` → `DynamicTimeWarping(window=window_size)` (Sakoe-Chiba band width)
  - `dtw_dist_fn` → `DynamicTimeWarping(distance_metric=dtw_dist_fn)` (e.g., `"swd"`, `"euclidean"`)
  - `cpd_penalty` → `DynamicTimeWarping(cpd_type=cpd_penalty)` (CPD registration type: `"rigid"`, `"affine"`, `"nonrigid"`, or `None`)
  - `step` → temporal stride — step over the dataset frames when building the DTW input trajectories
  - `n_breakpoints` → cap on `n_changepoints` (D-06 above)

### validate_params Contract
- **D-08:** `validate_params(self, params: dict[str, Any]) -> None` raises `ValueError` with a descriptive message on any missing required param or invalid value. Returns `None` implicitly on success. Roadmap success criterion ("raises on missing/invalid params") takes precedence over the `→ bool` annotation in REQUIREMENTS.md.
- **D-09:** `run()` calls `self.validate_params(params)` as its **first line** — guaranteed guard. Callers cannot bypass validation by forgetting to call it explicitly.
- **D-10:** Required params for `AlignmentStage`: `window_size` (int, > 0), `step` (int, ≥ 1), `cpd_penalty` (str or None), `dtw_dist_fn` (str), `n_breakpoints` (int, ≥ 0). Missing keys raise `ValueError("Missing required param: {key}")`.

### eval/stages/ Package Structure
- **D-11:** `eval/stages/` is a **regular package** with `eval/stages/__init__.py` — follows the `eval/tracking/` pattern, not the top-level `eval/` namespace pattern.
- **D-12:** `eval/stages/__init__.py` exports `PipelineStage` and `AlignmentStage` via `__all__`. Import pattern: `from eval.stages import PipelineStage, AlignmentStage`. Phase 20 will add `LabelTransferStage` to this `__init__.py`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FRAME-05 — Full `AlignmentStage` spec: `PipelineStage` ABC signature, hyperparam list (`window_size`, `step`, `cpd_penalty`, `dtw_dist_fn`, `n_breakpoints`), standalone-run gate, `validate_params` gate, `tests/test_alignment_stage.py` acceptance criteria

### Result Types (Phase 18 — already implemented, must extend)
- `eval/types.py` — Add `StageResult = Union[AlignResult, LabelResult]` TypeAlias and export via `__all__`. `AlignResult` fields already defined here (`aligned_cloud`, `warp_path`, `dtw_distance`, `n_changepoints`, `params_used`).

### Existing DTW API (wrap, do not reimplement)
- `src/zreg/dtw/__init__.py` — Public API: `DynamicTimeWarping`, `DTWResult`, `compose_constraints`
- `src/zreg/dtw/core.py` — `DynamicTimeWarping.__init__(x, y, distance_metric, cpd_type, window, ...)` and `.compute() → DTWResult`
- `src/zreg/dtw/result.py` — `DTWResult.warping_path: list[tuple[int, int]]`, `DTWResult.distance: float`, `DTWResult.rotations`

### Existing CPD API (used via cpd_type in DynamicTimeWarping)
- `src/zreg/cpd/__init__.py` — CPD registration types: `RigidCPD`, `AffineCPD`, `NonRigidCPD`; `cpd_type` string accepted by `DynamicTimeWarping` maps to these internally

### eval/ Patterns (prior phases)
- `eval/tracking/__init__.py` — Pattern for `eval/stages/__init__.py`: regular package, `__all__` with exported symbols, `from eval.stages import ...` works after path setup
- `eval/metrics.py` — Pattern: module with `__all__`, zreg.* imports before torch, NumPy-style docstrings, `MetricsEngine.__init__(config: EvalConfig)` as constructor model
- `eval/config.py` — `EvalConfigError(ValueError)` wrapper pattern — `validate_params` raises plain `ValueError`, not a wrapped error (no new exception class needed)

### Testing
- `tests/conftest.py` — `sys.path.insert(0, repo_root)` already present; `from eval.stages import ...` works without changes
- `tests/test_eval_metrics.py` — Fixture pattern and `torch.allclose(..., atol=1e-5)` assert style reference

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `zreg.dtw.DynamicTimeWarping` — Main class `AlignmentStage.run()` delegates to. Constructor accepts `x`, `y` as `dict[int, zRegPointCloud]`, `distance_metric`, `cpd_type`, `window`. Call `.compute()` to get `DTWResult`.
- `DTWResult.warping_path` — `list[tuple[int, int]]` — iterate to count diagonal/non-diagonal transitions for `n_changepoints` heuristic
- `eval/types.py` `AlignResult` — already fully defined; `AlignmentStage.run()` constructs and returns one

### Established Patterns
- `eval/` namespace: `eval/stages/` is a **subpackage**, not a top-level module — needs `__init__.py` per D-11
- libomp SIGABRT ordering: `from zreg.dtw import DynamicTimeWarping` MUST precede `import torch` in `eval/stages/alignment.py` (enforced in conftest.py:20-24)
- `__all__` defined in every new module (`base.py`, `alignment.py`, `__init__.py`)
- NumPy-style docstrings on all public classes and methods

### Integration Points
- `EvaluationRunner` (Phase 21) will call `AlignmentStage(config).run(dataset, params)` — constructor takes `EvalConfig` per DataFactory/MetricsEngine precedent
- `MetricsEngine` (Phase 18) is called by `EvaluationRunner` after `AlignmentStage.run()` returns `AlignResult` — `AlignResult` fields map to `StageMetrics` inputs
- `eval/stages/__init__.py` exports must be stable for Phase 20 `LabelTransferStage` to be added without breaking Phase 21 imports

</code_context>

<specifics>
## Specific Ideas

- `n_changepoints` heuristic: iterate `DTWResult.warping_path`, detect transitions between diagonal moves `(+1,+1)` and non-diagonal moves `(+1,0)` or `(0,+1)`, cap at `params["n_breakpoints"]`
- `cpd_penalty` is passed as `cpd_type` string to `DynamicTimeWarping` — valid values are `"rigid"`, `"affine"`, `"nonrigid"`, `None`
- `step` controls temporal stride: when building the `x` and `y` dicts for `DynamicTimeWarping`, only include every `step`-th frame

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 19-AlignmentStage*
*Context gathered: 2026-05-28*
