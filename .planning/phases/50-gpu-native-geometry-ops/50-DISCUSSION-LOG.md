# Phase 50: GPU-Native Geometry Ops for Cluster Deployment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-15
**Phase:** 50-gpu-native-geometry-ops-for-cluster-deployment
**Areas discussed:** Scope of replacement, Fallback strategy, Phase outcome if cluster check fails, Interface adaptation

---

## Scope of Replacement

| Option | Description | Selected |
|--------|-------------|----------|
| PointNet++ only (FPS + ball_query) | build_radius_graph stays Open3D. eGNN's radius graph construction is untouched. | |
| All three ops | torch_cluster.radius_graph replaces build_radius_graph too. | ✓ |

**User's choice:** All three ops — include `build_radius_graph`
**Notes:** User asked "does it matter at a larger scale i.e. 1k points?" — investigation showed the baseline suite runs at `max_points_per_frame: 1000`, not the 100–300 point training regime from Phase 47. At 1k points, `build_radius_graph` (all N points as centers) is the most expensive op and benefits most from GPU acceleration. The ROADMAP's eGNN exclusion was clarified to apply to the equivariant conv math (EGNNConv), not to the shared geometry indexing layer.

Sub-question — Interface adaptation location:

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt inside _ops.py (keep public signatures) | Callers unchanged; conversion/reshaping inside each function. | ✓ |
| Adapt callers (update SetAbstractionLayer + EGNNConv) | Clean GPU interface but wider code change. | |
| You decide | | |

**User's choice:** Adapt inside _ops.py
**Notes:** Recommended for minimum blast radius and consistency with downsampling.py's approach (public signatures stable, internals switch paths).

---

## Fallback Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Follow downsampling.py exactly (Recommended) | try/except ImportError, TORCH_CLUSTER_AVAILABLE flag, GPU auto-detect, Open3D fallback. | ✓ |
| Clean swap, no fallback | Remove Open3D, torch_cluster hard required. Simpler but dev machine broken. | |
| You decide | | |

**User's choice:** Follow downsampling.py exactly

Sub-question — TORCH_CLUSTER_AVAILABLE location:

| Option | Description | Selected |
|--------|-------------|----------|
| Duplicate in _ops.py | Self-contained, matches existing pattern. | ✓ |
| Extract to shared zreg._compat module | Single source of truth, avoids duplication. | |
| You decide | | |

**User's choice:** Duplicate in _ops.py

---

## Phase Outcome if Cluster Check Fails

| Option | Description | Selected |
|--------|-------------|----------|
| Document the finding and close the phase | Write finding, no code changes, phase marked done. | ✓ |
| Assume success — no failure path planned | torch_cluster installs on all Linux/CUDA; only plan the success path. | |
| Treat failure as a blocker — stop and escalate | Halt; user decides next steps. | |

**User's choice:** Document the finding and close the phase

Sub-question — Where install check happens:

| Option | Description | Selected |
|--------|-------------|----------|
| During Phase 51 smoke job | Piggyback on existing HoreKa access. | |
| As Phase 50's own first task | Plan 1 = install check; Plan 2 = code changes (conditional). | ✓ |

**User's choice:** Phase 50's own first task (two-plan structure)

---

## Interface Adaptation

Sub-question — Isolated-point guard in torch_cluster path:

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, preserve the guard | Re-implement after grouping edge_index; any empty group → [center]. | ✓ |
| You decide | | |

**User's choice:** Preserve the guard

Sub-question — FPS n_samples clipping:

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, clip to n_samples (strict contract) | Return idx[:n_samples]; handles ceil() rounding. | ✓ |
| You decide | | |

**User's choice:** Clip to n_samples

---

## Claude's Discretion

- Exact `torch_cluster` wheel URL / version to pin in the install command
- Whether to add `torch_cluster` as a new optional extra in `setup.cfg`
- `# pragma: no cover` marks for torch_cluster execution branch
- `pytest.mark.skipif(not TORCH_CLUSTER_AVAILABLE, ...)` markers for smoke tests

## Deferred Ideas

None — discussion stayed within phase scope.
