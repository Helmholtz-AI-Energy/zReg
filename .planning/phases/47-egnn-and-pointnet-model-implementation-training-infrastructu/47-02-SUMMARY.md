---
phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu
plan: 02
subsystem: ml-models
tags: [pointnet2, open3d, torch, set-abstraction, feature-propagation, joint-cloud, label-transfer]

# Dependency graph
requires:
  - phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu (plan 01)
    provides: src/zreg/models/_ops.py — farthest_point_sample, ball_query (isolated-point self-fallback guard), device-agnostic
  - phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-
    provides: 45-DESIGN.md joint-cloud conditioning pattern (one-hot label on source, unknown-flag bit on target)
provides:
  - "src/zreg/models/pointnet2.py — SetAbstraction, FeaturePropagation, PointNet2LabelTransfer nn.Module classes"
  - "PointNet2LabelTransfer.forward(joint_pos, joint_feat) -> (n_joint, n_classes) logits; caller slices logits[n_source:] for target subset"
  - "tests/test_zreg_models_pointnet2.py — 6 tests: SA ratio-downsampling + backward, slicing contract, ratio-based sizing at 2 scales, bowl-sparse robustness"
affects: [47-03, 47-04, 48, 49]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SetAbstraction: FPS -> ball-query -> per-group [relative_pos|neighbour_feat] MLP -> max-pool, ratio-based n_samples=max(1,int(n*ratio)) never fixed-count"
    - "FeaturePropagation: KDTreeFlann.search_knn_vector_3d(p,3) 3-NN inverse-distance interpolation + U-Net skip concat + MLP (mirrors icp.py's Open3D wrapper precedent)"
    - "Joint-cloud slicing contract: model has no notion of n_source; forward returns per-point logits over the FULL joint cloud, caller slices logits[n_source:] since target rows are always last by torch.cat([source, target]) construction"

key-files:
  created:
    - src/zreg/models/pointnet2.py
    - tests/test_zreg_models_pointnet2.py
  modified: []

key-decisions:
  - "2 SA layers (ratio 0.5 each, hidden 32->64) + 2 symmetric FP layers with matching skip dims (fp2: in=64,skip=32,out=32; fp1: in=32,skip=in_dim,out=32) + final per-point MLP head to n_classes — Assumption A1 defaults from 47-RESEARCH.md Pattern 3, not empirically tuned"
  - "SetAbstraction's second layer uses 2x the base radius (coarser resolution after first downsample) — a reasonable default not specified explicitly in RESEARCH.md, documented in the constructor docstring"
  - "FeaturePropagation clamps k=min(3, n_sparse) to handle degenerate cases where the sparse cloud has fewer than 3 points, and clamps NN distance to 1e-10 before inverse-distance weighting to avoid divide-by-zero on exact-match points"

patterns-established:
  - "Pattern: Open3D KDTreeFlann-based interpolation ops live inline in the consuming nn.Module (FeaturePropagation), following icp.py's wrapper precedent rather than adding a new _ops.py function, since 3-NN interpolation is PointNet++-specific (not shared with eGNN)."

requirements-completed: [MODEL-02, MODEL-04]

# Metrics
duration: 20min
completed: 2026-07-13
---

# Phase 47 Plan 02: PointNet++ Joint-Cloud Model Summary

**Hand-rolled joint-cloud PointNet++ (`SetAbstraction` + `FeaturePropagation` + `PointNet2LabelTransfer`) implementing 2 SA + 2 symmetric FP layers over a source+target-concatenated cloud, with ratio-based FPS sampling and a documented `logits[n_source:]` slicing contract for the target-only supervised subset.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-07-13T10:40:00Z
- **Completed:** 2026-07-13T10:50:01Z
- **Tasks:** 2
- **Files modified:** 2 (both created)

## Accomplishments
- `SetAbstraction` layer: FPS (ratio-based, `n_samples = max(1, int(n*ratio))`, never a fixed absolute count) -> `_ops.ball_query` grouping -> shared per-group `[relative_pos | neighbour_feat]` MLP -> max-pool; forward+backward both verified
- `FeaturePropagation` layer: 3-NN inverse-distance interpolation via `open3d.geometry.KDTreeFlann.search_knn_vector_3d(p, 3)` (weights `1/dist`, normalized) + U-Net skip-connection concat + MLP
- `PointNet2LabelTransfer` composes 2 SA (hidden 32->64) + 2 symmetric FP + a final per-point MLP head, returning `(n_joint, n_classes)` logits over the FULL joint cloud — model has no notion of `n_source`, caller slices `logits[n_source:]` (documented explicitly in the class docstring)
- `_ops.ball_query`'s isolated-point self-fallback guard is consumed as-is (no reimplementation), which is what keeps `sample_bowl`'s sparser rim regions from crashing `SetAbstraction`'s max-pool (T-47-03)
- 6 new tests in `tests/test_zreg_models_pointnet2.py`, including a real bowl-shaped (odd-seed) `DataFactory.generate_training_triple` robustness test — full repo suite at 1265 passed (was 1259), 18 skipped, 1 xpassed, zero regressions, 100% coverage on `pointnet2.py`

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SetAbstraction + FeaturePropagation + PointNet2LabelTransfer** - `e2165aa` (feat)
2. **Task 2: Create tests/test_zreg_models_pointnet2.py** - `20f9de9` (test)

**Plan metadata:** (this commit) (docs: complete plan)

_Note: Task 1's own inline `<verify>` command (a live smoke test asserting forward shape + backward) served as the RED->GREEN gate for the model implementation itself; Task 2 then added the full pytest suite, which passed immediately against the already-correct Task 1 implementation (no further implementation changes were needed)._

## Files Created/Modified
- `src/zreg/models/pointnet2.py` - `SetAbstraction`, `FeaturePropagation`, `PointNet2LabelTransfer` nn.Module classes (287 lines)
- `tests/test_zreg_models_pointnet2.py` - 6 tests: SA ratio-downsampling shape + backward gradients, joint-cloud slicing contract, ratio-based sizing at 2 joint-cloud sizes (200, 600), bowl-sparse robustness via real `DataFactory` odd-seed triple

## Decisions Made
- FP layer channel dims chosen to keep a consistent `hidden_dim` (32) throughout the U-Net skip path rather than doubling further, since `45-DESIGN.md`/`47-RESEARCH.md` Pattern 3 specified the SA-side dims (32->64) but left FP-side dims as an implementation detail — kept symmetric and simple per the "Assumption A1 defaults" framing in the plan
- Used `torch.randint`/`F.one_hot` to build synthetic joint-cloud features directly in shape/gradient tests (no `DataFactory` dependency needed there), reserving the real `DataFactory.generate_training_triple` call for the one test that specifically needs authentic bowl-sparse geometry — keeps the fast unit tests independent of the eval-framework import surface where possible
- See key-decisions in frontmatter for FP layer dimension and radius-scaling defaults

## Deviations from Plan

None - plan executed exactly as written. Task 1's implementation passed its own verify command on first attempt; Task 2's test suite passed immediately with zero implementation changes required.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `PointNet2LabelTransfer` is ready for Phase 47's remaining plans (training loop, checkpoint I/O) to consume via its `forward(joint_pos, joint_feat) -> (n_joint, n_classes)` interface
- The `logits[n_source:]` slicing contract is documented in the class docstring and exercised by tests — downstream training-loop code can rely on it directly
- No blockers for the next plan in this phase (`egnn.py`, per 45-DESIGN.md's target module layout)

## Self-Check: PASSED

- FOUND: src/zreg/models/pointnet2.py
- FOUND: tests/test_zreg_models_pointnet2.py
- FOUND: .planning/phases/47-egnn-and-pointnet-model-implementation-training-infrastructu/47-02-SUMMARY.md
- FOUND commit: e2165aa
- FOUND commit: 20f9de9

---
*Phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu*
*Completed: 2026-07-13*
