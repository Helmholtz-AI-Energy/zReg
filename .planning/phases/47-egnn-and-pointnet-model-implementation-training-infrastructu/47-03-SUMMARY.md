---
phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu
plan: 03
subsystem: ml-models
tags: [egnn, torch_geometric, message-passing, equivariance, pytorch, label-transfer]

# Dependency graph
requires:
  - phase: 47-01
    provides: "src/zreg/models/_ops.py — build_radius_graph, farthest_point_sample, ball_query"
  - phase: 45
    provides: "45-DESIGN.md MessagePassing-subclass mandate, joint-cloud conditioning scheme, label-vs-id discipline"
provides:
  - "EGNNConv(MessagePassing): single-propagate() E(3)-equivariant convolution layer (Satorras et al. 2021), numerically verified equivariant/invariant at atol=1e-4"
  - "EGNNLabelTransfer: joint-cloud eGNN label-transfer model (embed -> radius graph -> 4x EGNNConv -> linear readout), forward(joint_pos, joint_feat) -> (n_joint, n_classes) logits"
  - "tests/test_egnn_equivariance.py: E(3) equivariance, feature invariance, gradient flow, and end-to-end joint-cloud tests"
affects: [phase-48-inference-integration, phase-47-training-loop]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "torch_geometric.nn.MessagePassing subclass with aggr=None + custom aggregate() splitting a concatenated [h-message|coord-message] tensor into independently-reduced (sum/mean) halves — single propagate() call, no manual scatter bookkeeping"
    - "EGNNLabelTransfer mirrors PointNet2LabelTransfer's slicing contract: forward() returns logits for the full joint cloud, no n_source stored on the model, caller slices logits[n_source:]"

key-files:
  created:
    - src/zreg/models/egnn.py
    - tests/test_egnn_equivariance.py
  modified: []

key-decisions:
  - "Used RESEARCH.md Pattern 4's exact verified EGNNConv code unmodified (constructor aggr=None flow=source_to_target, message() returns cat([m_ij, coord_msg]), custom aggregate() splits [hidden_dim, 3] and scatters sum/mean independently) — equivariance was numerically pre-verified against this exact structure, so no deviation was risked"
  - "EGNNLabelTransfer builds the radius graph once from the initial joint_pos and reuses the same edge_index across all 4 EGNNConv layers (positions update per-layer via residual, but graph topology does not get rebuilt) — matches typical EGNN reference implementations and keeps the module simple; not explicitly specified in the plan but consistent with Pattern 4's composition notes"

patterns-established:
  - "Pattern: EGNN equivariance test asserts torch.allclose(rotate(forward(pos)), forward(rotate(pos)), atol=1e-4) using a single fixed radius graph built once and reused for both rotated/unrotated forward calls (graph topology is itself rotation-invariant since it depends only on pairwise distances)"

requirements-completed: [MODEL-03, MODEL-04]

# Metrics
duration: 8min
completed: 2026-07-13
---

# Phase 47 Plan 03: EGNN Equivariant Convolution Summary

**Hand-rolled `EGNNConv(MessagePassing)` (single-propagate, custom split-aggregate E(3)-equivariant layer) plus `EGNNLabelTransfer` stacking 4 layers over a joint-cloud radius graph, numerically verified equivariant/invariant at atol=1e-4.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-13T08:52:22Z
- **Completed:** 2026-07-13T09:00:25Z
- **Tasks:** 2 completed
- **Files modified:** 2 (both new)

## Accomplishments
- `EGNNConv` implements Satorras, Hoogeboom, Welling (2021) E(n)-equivariant update equations as a single `propagate()` call with a custom `aggregate()` override — copied exactly from 47-RESEARCH.md Pattern 4's session-verified code, no restructuring
- `EGNNLabelTransfer` composes the joint-cloud cross-cloud message-passing mechanism: `_ops.build_radius_graph` over the concatenated source+target cloud, 4 stacked `EGNNConv` layers, linear readout to per-point class logits
- E(3) coordinate equivariance and feature invariance both verified numerically at `atol=1e-4` in an automated pytest regression (mirroring the session's manual `~1e-6` check)
- Forward + backward gradient flow verified on both `h` and `pos`
- Zero regressions: 1269 tests pass (was 1265, +4 new), 18 skipped, 1 xpassed; `egnn.py` at 100% coverage

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement EGNNConv + EGNNLabelTransfer** - `cca5b1f` (feat)
2. **Task 2: Create tests/test_egnn_equivariance.py** - `777e158` (test)

_Note: Task 1's own inline verify command (forward+backward shape/gradient check) already gated correctness before Task 2's test file was written, so no RED-phase failure was needed — mirrors 47-02's precedent._

**Plan metadata:** (pending — this commit)

## Files Created/Modified
- `src/zreg/models/egnn.py` - `EGNNConv(MessagePassing)` (single-propagate, custom aggregate split-reduce) + `EGNNLabelTransfer` (embed -> radius graph -> 4x EGNNConv -> linear readout)
- `tests/test_egnn_equivariance.py` - E(3) equivariance (`atol=1e-4`), feature invariance, forward/backward gradient, and `EGNNLabelTransfer` end-to-end shape/backward tests

## Decisions Made
- Followed RESEARCH.md Pattern 4's EGNNConv code verbatim (mandated by the plan — a subtle change silently breaks equivariance with no error raised); no deviation
- `EGNNLabelTransfer` builds the radius graph once from the input `joint_pos` and reuses the same `edge_index` across all layers rather than rebuilding it per-layer from updated positions — standard EGNN practice, keeps the module simple, and is consistent with (though not explicitly dictated by) Pattern 4's composition notes
- Default `hidden_dim=32`, `n_layers=4`, `radius=0.3`, `max_neighbors=16` mirror `PointNet2LabelTransfer`'s defaults from 47-02 for consistency across the two model families (Assumption A1, not empirically tuned)

## Deviations from Plan

None - plan executed exactly as written. RESEARCH.md Pattern 4's EGNNConv code was used unmodified per the plan's explicit mandate.

## Issues Encountered

None. The verify command from the plan passed on first attempt; the equivariance test suite passed on first attempt (no RED-phase debugging needed since the underlying `EGNNConv` implementation was already the session-verified Pattern 4 code).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both eGNN and PointNet++ (47-02) model implementations are now complete, sharing the same `forward(joint_pos, joint_feat) -> (n_joint, n_classes)` interface and slicing contract (`logits[n_source:]`)
- Ready for 47-04/47-05: shared training loop (Pattern 5), checkpoint save/load (Pattern 6), and `train_label_transfer.py` entry point
- No blockers

---
*Phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu*
*Completed: 2026-07-13*

## Self-Check: PASSED

- FOUND: src/zreg/models/egnn.py
- FOUND: tests/test_egnn_equivariance.py
- FOUND: cca5b1f
- FOUND: 777e158
