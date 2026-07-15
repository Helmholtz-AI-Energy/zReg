---
phase: 48-labeltransferstage-integration-for-learned-methods
plan: 02
subsystem: eval-stages
tags: [pytorch, label-transfer, egnn, pointnet2, checkpoint-loading, tdd]

# Dependency graph
requires:
  - phase: 47
    provides: PointNet2LabelTransfer / EGNNLabelTransfer models + train_label_transfer.py checkpoint format (save_checkpoint, train_step)
  - phase: 48-01
    provides: EvalConfig.egnn_checkpoint_path / pointnet2_checkpoint_path fields + 4-value label_transfer_method validator
provides:
  - LabelTransferStage.VALID_METHODS 4-tuple ("knn_voting", "cpd_weighted", "pointnet2", "egnn")
  - LabelTransferStage._load_learned_model staticmethod (checkpoint load + construct + load_state_dict, 3 ValueError guards)
  - LabelTransferStage.run() per-frame learned-method branch (joint-cloud construction mirroring train_step exactly)
  - MODEL_REGISTRY module-level dict in eval/stages/label_transfer.py
affects: [Phase 49]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Load-once-per-run() pattern for learned models — local variable inside run(), not an instance attribute (Phase 47's per-trial EvaluationRunner instantiation means cross-instance caching never pays off)"
    - "n_classes read from the loaded model (model.n_classes), never re-derived from data — prevents silent joint_feat under-sizing"

key-files:
  created: []
  modified:
    - eval/stages/label_transfer.py
    - tests/test_label_transfer_stage.py

key-decisions:
  - "Direct-construction smoke-test checkpoints (fresh untrained model + save_checkpoint) instead of invoking train_label_transfer.py as a subprocess — faster, exercises the identical load path (D-01 only requires wiring correctness, not model quality)"
  - "n_classes=4 in test fixtures to match synthetic_dataset's generate_labels(n_classes=4) vocabulary, avoiding an unrelated one_hot RuntimeError (48-RESEARCH.md Pitfall 1)"
  - "No point-count-regime guard added anywhere (D-03 explicitly defers scale/generalization concerns to Phase 49)"

patterns-established:
  - "MODEL_REGISTRY dict pattern (mirrors train_label_transfer.py:80) as the single source of truth for method-string -> model-class dispatch"

requirements-completed: [D-01, D-02, D-03]

# Metrics
duration: ~25min
completed: 2026-07-14
---

# Phase 48 Plan 02: LabelTransferStage learned-method wiring (pointnet2/egnn) Summary

**Wired Phase 47's `PointNet2LabelTransfer`/`EGNNLabelTransfer` into `LabelTransferStage` as two new `method` values, with checkpoint loading, per-frame joint-cloud construction that byte-for-byte mirrors `train_step`'s encoding, and a `torch.equal`-asserted parity test proving it.**

## Performance

- **Duration:** ~25min
- **Completed:** 2026-07-14
- **Tasks:** 2/2 completed
- **Files modified:** 2

## Accomplishments

- `LabelTransferStage.VALID_METHODS` extended from 2 to 4 values (`"knn_voting"`, `"cpd_weighted"`, `"pointnet2"`, `"egnn"`); the invalid-method `ValueError` now names all four.
- New `_load_learned_model` staticmethod: `torch.load(..., map_location="cpu", weights_only=True)` + `MODEL_REGISTRY[method](**ckpt["hyperparams"])` + `load_state_dict` + `.eval()`, with three explicit `ValueError` guards (missing checkpoint-path config, file not found, `model_class` mismatch).
- New per-frame-loop `elif` branch for `pointnet2`/`egnn`: builds the joint cloud exactly as `train_label_transfer.py:train_step` does (`torch.cat([src_pos, tgt_pos])` source-then-target, one-hot over `[:n_src, :n_classes]`, unknown-flag at `[n_src:, -1] = 1.0`), runs `model(joint_pos, joint_feat)` under `torch.no_grad()`, and slices `logits[n_src:].argmax(dim=1)`.
- Checkpoint is loaded exactly once per `run()` call, as a local variable before the `for k in range(n_pairs)` loop — never inside it.
- `n_classes` is read from `learned_model.n_classes` (the loaded model), never re-derived from `labels_tensor.max()`, avoiding silent `joint_feat` under-sizing if a frame's labels don't include the highest class index.
- `tests/test_label_transfer_stage.py`: new `learned_smoke_checkpoint` factory fixture (direct model construction + `save_checkpoint`, no subprocess) and `TestLabelTransferStageLearnedMethods` (10 tests: happy path, encoding parity, missing-path, not-found, model_class-mismatch — parametrized over both methods).
- `test_learned_encoding_parity` independently reconstructs the reference joint cloud and asserts `torch.equal(result.transferred_labels[tk], ref)` for both `pointnet2` and `egnn` — the single highest-risk item in this phase, now regression-proof.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing smoke-test fixture + TestLabelTransferStageLearnedMethods (RED)** - `4936a80` (test)
2. **Task 2: Implement learned-method wiring in LabelTransferStage (GREEN)** - `acfbde4` (feat)

## TDD Gate Compliance

Task 2 was marked `tdd="true"`. Gate sequence confirmed in git log:
- RED gate: `4936a80` `test(48-02): add failing smoke-test fixture + TestLabelTransferStageLearnedMethods` — 11 assertions failed as expected (VALID_METHODS was still 2-valued).
- GREEN gate: `acfbde4` `feat(48-02): wire pointnet2/egnn learned methods into LabelTransferStage` — all 11 assertions pass.
- No REFACTOR commit needed (implementation was clean on first pass).

## Files Created/Modified

- `eval/stages/label_transfer.py` — Added `from pathlib import Path` and `from zreg.models import PointNet2LabelTransfer, EGNNLabelTransfer` to the existing zreg-before-torch import block; added module-level `MODEL_REGISTRY`; extended `VALID_METHODS` to 4 values; added `_load_learned_model` staticmethod (placed alongside `_check_alignment`, before `run`); added the load-once-per-`run()` call before the per-frame loop; added the `elif params["method"] in ("pointnet2", "egnn")` branch in the loop; updated the module docstring's hyperparam table and added a "Learned label transfer (Phase 48)" notes section.
- `tests/test_label_transfer_stage.py` — Added `from zreg.models import PointNet2LabelTransfer, EGNNLabelTransfer` (zreg.* block, before `torch`) and `from train_label_transfer import save_checkpoint`; added `LEARNED_METHOD_CASES`, `learned_smoke_checkpoint` fixture, and `TestLabelTransferStageLearnedMethods` (parametrized, 5 tests × 2 methods = 10 collected); strengthened `test_invalid_method_value_raises` to assert `"pointnet2"`/`"egnn"` appear in the error message.

## Decisions Made

- Used the direct-construction approach for smoke-test checkpoints (fresh `model_cls(**hyperparams)` + `save_checkpoint`) rather than invoking `train_label_transfer.py`'s CLI as a subprocess — matches 48-RESEARCH.md's Wave 0 Gaps recommendation, avoids a slow training-loop dependency in the test suite, and still exercises every line this phase adds.
- Kept `n_classes=4` consistent across both `LEARNED_METHOD_CASES` hyperparams and `synthetic_dataset`'s `generate_labels(n_classes=4)` to avoid an unrelated `one_hot` `RuntimeError` (48-RESEARCH.md Pitfall 1) — not a wiring bug, a fixture-design requirement.
- Did not add any point-count-regime guard or class-count-mismatch guard — both are explicitly out of scope per D-03 and 48-RESEARCH.md's Assumption A3 (treated the same way as D-03's "no guard" philosophy).

## Deviations from Plan

None — plan executed exactly as written. Both tasks completed as specified: Task 1 produced a genuine RED state (11 failures, all due to `VALID_METHODS` not yet including the new methods), Task 2 turned all of them GREEN with no additional fixes needed.

## Known Stubs

None — every code path added is fully wired and exercised by a passing test, including the highest-risk joint-cloud encoding (parity-tested via `torch.equal` against an independently-reconstructed reference for both model classes).

## Threat Flags

None — this plan's changes fall entirely within `48-01-PLAN.md`'s already-registered threat model (T-48-03 checkpoint deserialization via `weights_only=True`, T-48-04 input validation via `_load_learned_model`'s three `ValueError` guards, T-48-05 same-project-checkpoint trust boundary accepted per 45-DESIGN.md, T-48-SC no new packages). No new network endpoints, auth paths, or schema changes were introduced.

## Verification

- `pytest tests/test_label_transfer_stage.py --collect-only -q` → 85 tests collected, exit code 0 (no SIGABRT / exit 134 — import-order rule satisfied for both the new `zreg.models` imports in the stage and the test file).
- `pytest tests/test_label_transfer_stage.py tests/test_eval_config.py -q` → 114 passed.
- `pytest tests/test_label_transfer_stage.py -k "Learned or invalid_method" -q` → 11/11 pass (was 11 failures pre-Task-2, confirming genuine RED→GREEN).
- Full suite: `pytest -q` → **1300 passed, 18 skipped, 1 xpassed** (was 1290 before this plan; +10 new tests), zero regressions.
- `grep -n "from zreg.models import" eval/stages/label_transfer.py` appears before `import torch` (line 78 vs. line 80).
- `grep -n "weights_only=True"` confirms the mandated safe-deserialization flag on the `torch.load` call.
- `grep -n "learned_model.n_classes"` confirms `n_classes` is read from the model, not re-derived from data.
- `_load_learned_model` call (line 421) confirmed before `for k in range(n_pairs)` (line 423) — load-once-per-run contract satisfied.

## Self-Check: PASSED

- FOUND: eval/stages/label_transfer.py
- FOUND: tests/test_label_transfer_stage.py
- FOUND commit 4936a80 (test)
- FOUND commit acfbde4 (feat)
