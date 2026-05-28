# Phase 19: AlignmentStage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 19-AlignmentStage
**Areas discussed:** StageResult return type, n_changepoints computation, validate_params contract, eval/stages/ package type

---

## StageResult Return Type

| Option | Description | Selected |
|--------|-------------|----------|
| Union TypeAlias | `StageResult = Union[AlignResult, LabelResult]` in `eval/types.py`. Simple, no new abstractions. | ✓ |
| Generic ABC | `PipelineStage[T]` with `run() → T`. Clean typing but complex with pydantic frozen models. | |
| Plain BaseModel | `run() → BaseModel` in ABC. Minimal abstraction, no static enforcement. | |

**User's choice:** Union TypeAlias in `eval/types.py`
**Notes:** TypeAlias location confirmed as `eval/types.py` (not `base.py`). ABC run() signature: `run(dataset: dict[int, zRegPointCloud], params: dict[str, Any]) → StageResult`.

---

## n_changepoints Computation

| Option | Description | Selected |
|--------|-------------|----------|
| Change point detection (ruptures) | Add ruptures dep; `cpd_penalty` = PELT penalty; real CPD step. | |
| Pass-through from params | `n_changepoints = params["n_breakpoints"]`. No computation. | |
| Warping path jumps | Count diagonal/non-diagonal run transitions in `DTWResult.warping_path`. | ✓ |

**User's choice:** Warping path jumps (heuristic, no external dep)
**Notes:** User confirmed it's a simpler heuristic — no ruptures needed. Elaboration requested on `n_breakpoints` relationship. Final decision: `n_changepoints = min(raw_jump_count, params["n_breakpoints"])` — n_breakpoints caps the count, making it a live HPO param. Hyperparam mapping confirmed: `cpd_penalty → cpd_type`, `step → window stride`.

---

## validate_params Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Raises ValueError | Raises with descriptive message on missing/invalid param. Roadmap wins over REQUIREMENTS.md `→ bool`. | ✓ |
| Returns bool | Returns True/False. REQUIREMENTS.md wins. Callers must check return value. | |
| Dual semantics | Raises for missing params, returns False for invalid values. | |

**User's choice:** Raises ValueError
**Notes:** Auto-called inside `run()` as first line — callers cannot bypass it. No new exception class needed; plain `ValueError` per `eval/config.py` pattern.

---

## eval/stages/ Package Type

| Option | Description | Selected |
|--------|-------------|----------|
| Regular package with `__init__.py` | Follows `eval/tracking/` pattern. `from eval.stages import PipelineStage, AlignmentStage`. | ✓ |
| Namespace, no `__init__.py` | Follows top-level `eval/` pattern. Per-module imports required. | |

**User's choice:** Regular package with `__init__.py`
**Notes:** Phase 20 will add `LabelTransferStage` to the same `__init__.py`.

---

## Claude's Discretion

None — all gray areas were resolved by the user.

## Deferred Ideas

None — discussion stayed within phase scope.
