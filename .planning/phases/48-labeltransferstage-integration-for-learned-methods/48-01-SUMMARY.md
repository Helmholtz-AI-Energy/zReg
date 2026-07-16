---
phase: 48-labeltransferstage-integration-for-learned-methods
plan: 01
subsystem: config
tags: [pydantic, eval-config, label-transfer, egnn, pointnet2]

# Dependency graph
requires:
  - phase: 47
    provides: PointNet2LabelTransfer / EGNNLabelTransfer models + checkpoint save format (train_label_transfer.py)
provides:
  - EvalConfig.egnn_checkpoint_path and EvalConfig.pointnet2_checkpoint_path fields (str | None, default None, independently settable)
  - label_transfer_method validator extended to accept 'pointnet2' and 'egnn' alongside 'knn_voting' and 'cpd_weighted'
affects: [48-02, LabelTransferStage]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Use-time (not construction-time) filesystem validation for path fields — mirrors target_data_path/D-05"]

key-files:
  created: []
  modified:
    - eval/config.py
    - tests/test_eval_config.py

key-decisions:
  - "Checkpoint paths are two separate fields (egnn_checkpoint_path, pointnet2_checkpoint_path), not one shared path, per D-02, so both learned models can be configured simultaneously"
  - "No field_validator checks checkpoint-file existence at EvalConfig construction time — existence is deferred to LabelTransferStage.run() in Plan 02 (Pitfall 4)"

patterns-established:
  - "New optional str|None config fields for learned-method artifacts follow the target_data_path precedent: default None, no construction-time filesystem check, declared explicitly under extra='forbid'"

requirements-completed: [D-02]

# Metrics
duration: ~15min (active work across a session resumed after a connection interruption mid-Task-2)
completed: 2026-07-14
---

# Phase 48 Plan 01: EvalConfig checkpoint-path fields + 4-value label_transfer_method validator Summary

**Added independently-settable `egnn_checkpoint_path` / `pointnet2_checkpoint_path` fields and extended `label_transfer_method`'s validator to a 4-value allowlist, giving Plan 02's `LabelTransferStage` dispatcher config surface to read from.**

## Performance

- **Duration:** ~15min active work (session spanned a connection interruption; wall-clock gap between Task 1 and Task 2 commits is not representative of effort)
- **Completed:** 2026-07-14
- **Tasks:** 2/2 completed
- **Files modified:** 2

## Accomplishments
- `EvalConfig` now accepts `label_transfer_method='pointnet2'` and `='egnn'` without validation error, alongside the pre-existing `'knn_voting'`/`'cpd_weighted'`
- Two new `str | None = None` fields (`egnn_checkpoint_path`, `pointnet2_checkpoint_path`) added, independently settable, no construction-time filesystem validation
- Test suite fully updated: 3 pre-existing validator-message assertions fixed for the new 4-value message, plus 5 new tests covering the new accepted values and the new fields' defaults/settability/use-time-validation contract

## Task Commits

Each task was committed atomically:

1. **Task 1: Add checkpoint-path fields + extend label_transfer_method validator** - `40a2c3b` (feat)
2. **Task 2: Update test_eval_config.py for new fields and 4-value validator message** - `911387f` (test)

**Plan metadata:** commit pending (docs: complete plan)

_Note: Task 1 was `tdd="true"` but was completed as a single feat commit (its own inline verify command gated correctness); Task 2 supplied the broader regression-test coverage as its own atomic `test` commit._

## Files Created/Modified
- `eval/config.py` - Added `egnn_checkpoint_path` and `pointnet2_checkpoint_path` `Field(default=None)` declarations immediately after `label_transfer_method`; updated `label_transfer_method`'s description string; extended `validate_label_transfer_method`'s membership tuple and docstring to the four values `('knn_voting', 'cpd_weighted', 'pointnet2', 'egnn')`; updated the raised `ValueError` message to name all four values
- `tests/test_eval_config.py` - Updated `TestEvalConfigLabelTransferMethodValidation`'s three `pytest.raises(match=...)` assertions to the new 4-value message (via a shared `LABEL_TRANSFER_METHOD_MESSAGE` class constant); added `test_label_transfer_method_pointnet2_valid` and `test_label_transfer_method_egnn_valid`; added new `TestEvalConfigCheckpointPathFields` class with 3 tests (defaults-None, independently-settable, construction-succeeds-with-nonexistent-paths)

## Decisions Made
- Reused a class-level `LABEL_TRANSFER_METHOD_MESSAGE` constant in the test file instead of repeating the 4-value regex string in each of the three updated assertions, reducing drift risk if the validator message changes again
- Added the nonexistent-checkpoint-path test explicitly asserting `not Path(...).exists()` before construction, to make the use-time-vs-construction-time validation contract unambiguous and regression-proof

## Deviations from Plan

None - plan executed exactly as written. Task 1 had already been completed and committed (`40a2c3b`) in a prior session before a connection interruption; this session resumed at Task 2 exactly as directed, with no redo of Task 1.

## Known Stubs

None - no stub patterns introduced. The new checkpoint-path fields are inert configuration surface (no consumer yet); Plan 02 wires `LabelTransferStage` to read them.

## Threat Flags

None - `T-48-01` and `T-48-02` from the plan's threat model are fully addressed by the validator allowlist and the existing path-field precedent; no new threat surface introduced beyond what the plan's threat_model already registers.

## Verification

- `pytest tests/test_eval_config.py -q` → 29 passed
- `grep -n "knn_voting' or 'cpd_weighted'" tests/test_eval_config.py` → zero matches (old 2-value message fully removed)
- Full suite: `pytest -q` → 1290 passed, 18 skipped, 1 xpassed (was 1285 before this plan; +5 new tests), zero regressions
