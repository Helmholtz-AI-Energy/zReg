# Phase 18: MetricsEngine & Result Types - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 18-MetricsEngine & Result Types
**Areas discussed:** Result type flavor, Normalization strategy, Score weights

---

## Result type flavor

| Option | Description | Selected |
|--------|-------------|----------|
| `@dataclass` | Stdlib, zero extra overhead. EvalReport serialized via `dataclasses.asdict()` in Phase 21. | |
| pydantic `BaseModel` | Consistent with EvalConfig (Phase 17 pattern). `.model_dump()` gives free JSON serialization in Phase 21. Requires `arbitrary_types_allowed=True` for tensor fields. | ✓ |

**User's choice:** pydantic `BaseModel`
**Notes:** Consistency with Phase 17 and free `.model_dump()` serialization for `EvalReport → eval_report.json` drove the decision.

### Follow-up: Frozen vs mutable

| Option | Description | Selected |
|--------|-------------|----------|
| `frozen=True` | Signals read-only output objects. `ConfigDict(frozen=True, arbitrary_types_allowed=True)`. | ✓ |
| Mutable | Allows patching `plot_paths` into `EvalReport` after plot generation in Phase 21. | |

**User's choice:** Frozen
**Notes:** Phase 21 runner must construct `EvalReport` with all fields including `plot_paths` at creation time (collect plots first, then build report).

### Follow-up: StageMetrics structure

| Option | Description | Selected |
|--------|-------------|----------|
| Both raw + normalized dict | Raw typed fields + `normalized: dict[str, float]`. Matches FRAME-03. | ✓ |
| Raw values only | normalize() returns a separate dict; downstream code always calls it explicitly. | |

**User's choice:** Both raw + normalized dict

---

## Normalization strategy

| Option | Description | Selected |
|--------|-------------|----------|
| `1/(1+x)` asymptotic | Always in (0,1], single-sample safe, no dataset context needed. Works in HPO. | ✓ |
| Min-max over batch | Requires seeing multiple trials at once; breaks single-sample use in Phase 22 HPO. | |

**User's choice:** `1/(1+x)` asymptotic

### Follow-up: normalize() input type

| Option | Description | Selected |
|--------|-------------|----------|
| `StageMetrics` object | Strongly typed; knows direction per field without a lookup table. | ✓ |
| `dict[str, float]` | Flexible; requires hardcoded direction table keyed by string name. | |

**User's choice:** `StageMetrics` object

---

## Score weights

| Option | Description | Selected |
|--------|-------------|----------|
| `EvalConfig` field with defaults | `metric_weights: dict[str, float]` with sensible defaults. Phase 22 can override via YAML. | ✓ |
| Hardcoded defaults only | Weights baked into `MetricsEngine` as class constant. Simpler but not configurable. | |

**User's choice:** `EvalConfig` field with defaults

### Follow-up: Weight normalization behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-normalize | Divide each weight by sum before computing score. User sets relative weights. | ✓ |
| Strict validation | Raise `EvalConfigError` if weights don't sum to 1.0 ± 1e-6. | |

**User's choice:** Auto-normalize
**Notes:** Default weights `{chamfer: 0.35, hausdorff: 0.15, path_smoothness: 0.10, temporal_stability: 0.10, f1: 0.20, knn_consistency: 0.10}` were accepted as-is.

---

## Claude's Discretion

None — user made explicit choices on all presented options.

## Deferred Ideas

None — discussion stayed within phase scope.
