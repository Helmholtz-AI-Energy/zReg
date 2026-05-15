# Phase 13: Core Metrics Library - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 13-core-metrics-library
**Areas discussed:** Temporal stability, Migration scope, eval package API, Test expectations

---

## Temporal Stability

| Option | Description | Selected |
|--------|-------------|----------|
| Transformation smoothness | Mean Frobenius norm of matrix differences between consecutive transforms | ✓ |
| Point cloud drift | Mean chamfer distance between consecutive registered clouds | |
| Label consistency | Stability of label assignments across frames | |

**Q: Input type?**
| Option | Selected |
|--------|----------|
| List of zreg transform objects (RigidTransformation \| AffineTransformation) | ✓ |
| List of 4×4 tensors | |

**Q: Delta computation?**
| Option | Selected |
|--------|----------|
| Frobenius norm of matrix difference | ✓ |
| Separate rotation angle + translation magnitude namedtuple | |

**User's choice:** Frobenius norm of differences between consecutive transform matrices (4×4 homogeneous).

---

## Migration Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Delete src/zreg/metrics/ in phase 13 | Create new, delete proto stubs | |
| Leave src/zreg/metrics/ in place | Create src/zreg/eval/metrics/, leave proto stubs | |

**User's choice (freeform):** "leave it and implement metrics in src which are then used in eval" — leave `src/zreg/metrics/` as-is, create correct implementations in `src/zreg/eval/metrics/`, eval runner scripts import from there.

**Notes:** Nothing imports `src/zreg/metrics/` — it's orphaned proto code. No risk from leaving it.

---

## eval Package API

| Option | Description | Selected |
|--------|-------------|----------|
| Nothing — internal library (empty __init__.py) | Import from submodules directly | |
| All metrics at top level | from zreg.eval import chamfer | |
| Submodules only | from zreg.eval import metrics | |

**User's choice (freeform):** "eval should live at root level and import metrics and transformation functions from src" — the `eval/` repo-root scripts are the consumer; `src/zreg/eval/__init__.py` is minimal.

**Q: Re-export from zreg/__init__.py?**
| Option | Selected |
|--------|----------|
| No — keep eval separate | ✓ |
| Yes — add to zreg namespace | |

---

## Test Expectations

| Option | Description | Selected |
|--------|-------------|----------|
| Full unit tests for all metrics | Known-distance inputs, edge cases, GPU handling, sentinel masking | ✓ |
| Smoke tests only | One test per metric, deferred to phase 16 | |

**Q: GPU test strategy?**
| Option | Selected |
|--------|----------|
| Skip GPU tests if CUDA unavailable | ✓ |
| CPU-only, no GPU test infrastructure | |

---

## Claude's Discretion

- `src/zreg/eval/metrics/__init__.py` re-export strategy — can expose key functions at subpackage level for ergonomic imports
- `temporal_stability` behavior on single-element list — return 0.0 or raise ValueError

## Deferred Ideas

None — discussion stayed within phase scope.
