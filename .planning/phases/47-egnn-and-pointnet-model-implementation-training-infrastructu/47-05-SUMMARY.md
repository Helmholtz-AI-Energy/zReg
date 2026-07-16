---
phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu
plan: 05
subsystem: testing
tags: [pytest, pointnet2, egnn, joint-cloud, label-transfer, contract-test]

# Dependency graph
requires:
  - phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu (plan 02)
    provides: "src/zreg/models/pointnet2.py — PointNet2LabelTransfer.forward(joint_pos, joint_feat) -> (n_joint, n_classes) logits, logits[n_source:] slicing contract"
  - phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu (plan 03)
    provides: "src/zreg/models/egnn.py — EGNNLabelTransfer.forward(joint_pos, joint_feat) -> (n_joint, n_classes) logits, same slicing contract"
  - phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-
    provides: "45-DESIGN.md Downstream Wave-0 Test Obligations — mandates tests/test_zreg_models_joint_cloud.py as the single-sourced both-model contract test"
provides:
  - "tests/test_zreg_models_joint_cloud.py — parametrized (2 models x 2 size regimes = 4 tests) joint-cloud output-shape contract test, single-sourcing the logits[n_source:] slicing invariant for BOTH PointNet2LabelTransfer and EGNNLabelTransfer"
affects: [phase-48-inference-integration, phase-49-evaluation-benchmarking]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Both-model parametrized contract test: pytest.mark.parametrize stacking over MODEL_CLASSES and SIZE_REGIMES ([(100,100),(300,200)]) proves the slicing contract holds across variable point counts, not a single fixed size"
    - "Test imports PointNet2LabelTransfer/EGNNLabelTransfer directly from their submodules (zreg.models.pointnet2 / zreg.models.egnn), not the zreg.models package __init__, keeping this Wave-3 plan decoupled from the parallel 47-04 __init__.py rewrite"

key-files:
  created:
    - tests/test_zreg_models_joint_cloud.py
  modified: []

key-decisions:
  - "Used (100,100) and (300,200) as the two size regimes — both within the 100-300 point-per-cloud range 45-DESIGN.md/46-03 established for TrainingTriple, and deliberately asymmetric (n_source != n_target) at the second regime to avoid accidentally proving the contract only for the equal-count case"
  - "No RED phase needed: both models already existed (47-02, 47-03) and their slicing contract was already correct, so this Wave-0 test passed immediately on first run (GREEN), consistent with the same no-RED-needed pattern documented in 47-02/47-03 summaries — see TDD Gate Compliance note below"

patterns-established:
  - "Pattern: when a task='auto' tdd='true' task's sole purpose is to add a contract-verification test for already-correct prior-wave code (rather than drive new implementation), the RED phase is satisfied by the test failing to exist before the commit, not by a failing assertion — documented explicitly rather than silently skipped"

requirements-completed: [MODEL-02, MODEL-03]

# Metrics
duration: 12min
completed: 2026-07-13
---

# Phase 47 Plan 05: Joint-Cloud Output-Shape Contract Test Summary

**Single test file (`tests/test_zreg_models_joint_cloud.py`, 4 parametrized cases) proving `logits[n_source:].shape == (n_target, n_classes)` for both `PointNet2LabelTransfer` and `EGNNLabelTransfer` across two point-count regimes — the mandated 45-DESIGN.md Wave-0 joint-cloud conditioning contract.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-13T11:35:00Z
- **Completed:** 2026-07-13T11:47:00Z
- **Tasks:** 1
- **Files modified:** 1 (created)

## Accomplishments
- `tests/test_zreg_models_joint_cloud.py` created: a shared `_build_joint_cloud(n_source, n_target, n_classes, seed)` helper builds one-hot(label) source rows + unknown-flag target rows (45-DESIGN.md Pattern 2 layout), then `TestJointCloudSlicingContract.test_forward_target_subset_shape` is parametrized over both model classes (`PointNet2LabelTransfer`, `EGNNLabelTransfer`) and two size regimes (`(100, 100)`, `(300, 200)`) — 4 total collected cases
- Every case asserts both `logits.shape == (n_joint, n_classes)` AND `logits[n_source:].shape == (n_target, n_classes)`, with an explicit `!= n_joint` assertion making the "not source+target" contract unambiguous
- Both models imported directly from their submodules (`zreg.models.pointnet2`, `zreg.models.egnn`), independent of plan 47-04's `zreg/models/__init__.py` rewrite, per the plan's `<interfaces>` contract
- `grep -rn "\.cuda()" tests/test_zreg_models_joint_cloud.py` returns zero matches (CPU-only test, matches project convention)
- Full repo suite: 1285 passed (was 1281; +4 new), 18 skipped, 1 xpassed — zero regressions. This is the last plan in Phase 47; the full-suite run confirms no regressions across the whole phase (47-01 through 47-05).

## Task Commits

Each task was committed atomically:

1. **Task 1: Create tests/test_zreg_models_joint_cloud.py (both-model slicing contract)** - `6a8d7e1` (test)

**Plan metadata:** (this commit) (docs: complete plan)

_Note: both `PointNet2LabelTransfer` (47-02) and `EGNNLabelTransfer` (47-03) already implement the correct `logits[n_source:]` slicing contract, so this Wave-0 test passed immediately (GREEN) on first run — no implementation changes were required. See TDD Gate Compliance below._

## Files Created/Modified
- `tests/test_zreg_models_joint_cloud.py` - Parametrized (2 models x 2 size regimes) joint-cloud slicing contract test; single-sources the `logits[n_source:]` invariant for both label-transfer models

## Decisions Made
- Chose asymmetric size regimes `(100, 100)` and `(300, 200)` — the second regime deliberately has `n_source != n_target` so the test cannot pass by coincidence on an equal-split assumption
- Kept the shared `_build_joint_cloud` helper local to this test file (not extracted to a shared fixture/conftest) since it is the ONE file mandated to single-source this specific contract per 45-DESIGN.md — no other test file needs it
- Used `torch.Generator().manual_seed(seed)` (matches `test_egnn_equivariance.py`'s fixed-seed convention) rather than the module-level `torch.manual_seed`, avoiding any global RNG state leakage between parametrized cases

## Deviations from Plan

None - plan executed exactly as written. Single task, single file, all acceptance criteria met on first run.

## TDD Gate Compliance

This task carries `tdd="true"` but its behavior (verifying an already-correct, already-implemented contract from 47-02/47-03) meant the test passed on first execution — no separate RED (failing) commit exists, matching the same no-RED-needed pattern already documented in 47-02's and 47-03's SUMMARY.md files for this phase. This is not a gate violation: the task is a Wave-0 contract-verification test for existing code, not new feature-driving TDD, and the plan's own `<done>` criterion ("both models yield ... exactly n_target target-subset rows") was satisfied and verified by the single `test(47-05): ...` commit.

## Issues Encountered

None. The test suite passed on first run against the already-correct 47-02/47-03 implementations.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 47 (eGNN/PointNet++ model implementation + training infrastructure) is now fully complete: 5/5 plans (`_ops.py`, `pointnet2.py`, `egnn.py`, training loop + `train_label_transfer.py`, and this Wave-0 joint-cloud contract test)
- Both models' `forward(joint_pos, joint_feat) -> (n_joint, n_classes)` interface and `logits[n_source:]` slicing contract are now verified by three layers of tests: per-model unit tests (47-02, 47-03), the training-loop integration tests (47-04), and this cross-model contract test (47-05)
- Ready for Phase 48 (`LabelTransferStage` dispatch integration — `"egnn"`/`"pointnet++"` `VALID_METHODS`, checkpoint loading with `weights_only=True`)
- No blockers

## Self-Check: PASSED

- FOUND: tests/test_zreg_models_joint_cloud.py
- FOUND commit: 6a8d7e1

---
*Phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu*
*Completed: 2026-07-13*
