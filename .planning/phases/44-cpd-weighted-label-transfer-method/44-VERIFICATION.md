---
phase: 44-cpd-weighted-label-transfer-method
verified: 2026-07-10T12:45:45Z
status: passed
score: 9/9 must-haves verified (D-01 through D-09)
overrides_applied: 0
---

# Phase 44: CPD-Weighted Label Transfer Method Verification Report

**Phase Goal:** Wire the existing `zreg.color_transfer` `CPD_WEIGHTED` method into
`eval.stages.label_transfer.LabelTransferStage` as an alternative to the hardcoded `KNN_VOTING`
(Phase 20 FRAME-06 constraint), following the `alignment_method` optional-param precedent from
Phase 39. Pure plumbing/wiring — no new algorithms. eGNN/PointNet++ explicitly out of scope.

**Verified:** 2026-07-10T12:45:45Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

This phase has no formal REQUIREMENTS.md entries (added ad hoc; confirmed `.planning/REQUIREMENTS.md`
does not exist in this repo state). Verification is against the 9 decisions (D-01 through D-09) in
`44-CONTEXT.md`, independently re-derived from the current source, not from SUMMARY.md self-reports.

### Observable Truths (D-01 through D-09)

| # | Truth (Decision) | Status | Evidence |
|---|---------|------------|----------|
| 1 | D-01/D-02: `_build_aligned_cloud`'s CPD branch computes the posterior at a single fork-free insertion point shared by the `StoredTransform` reuse sub-path and the fresh-CPD fallback sub-path, via a throwaway side-effect-free `RigidCPD` + public `.expectation_step()` call | ✓ VERIFIED | `eval/stages/alignment.py:472-486` — insertion point sits immediately before `aligned[tk] = matched_source_frame` (line 488), physically after both the reuse branch (lines 426-436) and fallback branch (lines 437-470) converge. Uses `utils.squared_kernel_sum` + `RigidCPD(source=..., use_color=False).expectation_step(...)` exactly as specified. |
| 2 | D-03: `estep_results` populated only for frames that ran CPD registration (`alignment_method="cpd"` and `cpd_penalty is not None`); empty for icp/swd/no-cpd-penalty branches | ✓ VERIFIED | Read full branch structure `eval/stages/alignment.py:416-524`: `estep_results[tk] = estep_result` (line 486) appears exactly once, strictly inside the `elif cpd_penalty is not None and alignment_method == "cpd":` block. The `cpd_penalty is None` branch (418), `icp` branch (489-501), `swd` branch (502-519), and final `else` (520-522) never touch `estep_results`. Confirmed by 4 passing tests in `TestEstepResultsCapture` (`test_cpd_rigid_populates_estep_results`, `test_no_cpd_penalty_estep_results_empty`, `test_icp_estep_results_empty`, `test_swd_estep_results_empty`), independently executed and all PASSED. |
| 3 | D-04: pmat orientation transpose bug fixed at the `LabelTransferStage` call site (not in `zreg.color_transfer`) | ✓ VERIFIED | `eval/stages/label_transfer.py:358` — `transposed = estep_result._replace(pmat=estep_result.pmat.T)` applied unconditionally before every `cpd_weighted` call. `src/zreg/color_transfer.py` confirmed untouched throughout Phase 44 (`git diff acf9886 HEAD --stat -- src/zreg/` returns empty). Test fixture (`cpd_weighted_source_target`) deliberately uses differing point counts (12 source / 9 target) so a degenerate square-matrix pass would not mask the bug — independently re-read and confirmed at `tests/test_label_transfer_stage.py:544-568`. |
| 4 | D-05: categorical label averaging fixed via one-hot encode + argmax, entirely inside `LabelTransferStage` | ✓ VERIFIED | `eval/stages/label_transfer.py:359-367` — `one_hot = torch.nn.functional.one_hot(labels_tensor.long()).float()`, `transfer_colors(..., source_colors=one_hot, ...)`, `transferred[tk] = soft_scores.argmax(dim=1)`. `zreg.color_transfer` untouched (same diff-empty check as above). |
| 5 | D-06: `EvalConfig.label_transfer_method` field + validator restricted to `{"knn_voting", "cpd_weighted"}`, mirroring `validate_alignment_method` | ✓ VERIFIED | `eval/config.py:285` (field), `eval/config.py:386-409` (`validate_label_transfer_method` validator, raises `ValueError` on invalid value). Independently ran `EvalConfig(data_path='x')` → `label_transfer_method == 'knn_voting'`; `EvalConfig(data_path='x', label_transfer_method='bogus')` → raises with matching message. 5/5 tests in `TestEvalConfigLabelTransferMethodValidation` independently executed, all PASSED. |
| 6 | D-07: `LabelTransferStage.OPTIONAL_PARAMS`/`VALID_METHODS` added; `params["method"]` populated from `self.config.label_transfer_method` when absent; none of the existing YAML configs need a `method` key | ✓ VERIFIED | `eval/stages/label_transfer.py:101-104` (`OPTIONAL_PARAMS`, `VALID_METHODS`), `:158-160` (config-default population, placed after required-key loop, before value-range checks, matching `AlignmentStage`'s ordering), `:186-187` (final validation). Independently loaded all 16 `configs/*.yaml` files via `EvalConfig.from_yaml()` — all succeed with `label_transfer_method == 'knn_voting'` default; none contain a `method` key in `params`. |
| 7 | D-08: `LabelTransferStage.run()` gains `align_result: AlignResult \| None = None`; raises clear `ValueError` when `cpd_weighted` requested but `align_result` is `None` or missing the frame's `estep_results` entry; `EvaluationRunner` threads `align_result` through automatically | ✓ VERIFIED | `eval/stages/label_transfer.py:224` (signature), `:345-356` (both `ValueError` branches, matching D-08's specified wording). `eval/runners/eval_runner.py:314` — `LabelTransferStage(self.config).run(stage_input, target, params, align_result=align_result)` confirmed by direct grep and read; `align_result` is `None` when `run_alignment=False` (line 303), satisfying the "existing callers keep working" requirement. `TestAlignResultThreadedToLabelTransfer::test_align_result_passed_as_kwarg` independently executed, PASSED (asserts `call_args.kwargs["align_result"] is fake_align_result`, an identity check, not just no-exception). |
| 8 | D-09: `AlignResult.estep_results: dict[int, EstepResult] = Field(default_factory=dict)` added, `EstepResult` imported from `zreg.cpd` (public export), not `zreg.cpd._types` | ✓ VERIFIED | `eval/types.py:51` (`from zreg.cpd import EstepResult`), `:128` (field declaration), docstring `Parameters`/`Attributes` sections both updated (lines 93-99, 108-110). Confirmed `zreg.cpd.__init__.py:28,56` publicly exports `EstepResult`. `eval/stages/alignment.py:46` also imports from `zreg.cpd` (not `._types`). |
| 9 | Phase-goal-level: `zreg.color_transfer` and `zreg.cpd` math reused as-is throughout — no new algorithms, no `zreg/` changes | ✓ VERIFIED | `git diff acf9886 HEAD --stat -- src/zreg/` returns empty (zero files changed under `src/zreg/` across all 4 phase-44 commits). All new logic lives in `eval/` (stage/config/runner layer). |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/types.py` | `AlignResult.estep_results: dict[int, EstepResult]` field | ✓ VERIFIED | Present at line 128; `EstepResult` imported from `zreg.cpd` at line 51; docstring updated. |
| `eval/config.py` | `label_transfer_method` field + `validate_label_transfer_method` validator | ✓ VERIFIED | Field at line 285, validator at lines 386-409. |
| `eval/stages/alignment.py` | `_build_aligned_cloud` returns `(aligned, estep_results)`; `run()` threads it | ✓ VERIFIED | Return type `tuple[dict[int, zRegPointCloud], dict[int, EstepResult]]` (line 344); `return aligned, estep_results` (line 524); `run()` unpacks at line 272 and passes `estep_results=estep_results` at line 290. |
| `eval/stages/label_transfer.py` | `OPTIONAL_PARAMS`, `VALID_METHODS`, `run(align_result=...)`, `cpd_weighted` branch | ✓ VERIFIED | All present and wired (see truths 3, 4, 6, 7 above). |
| `eval/runners/eval_runner.py` | `align_result=align_result` passed to `LabelTransferStage.run()` | ✓ VERIFIED | Line 314, confirmed by direct read. |
| `tests/test_eval_config.py` | `TestEvalConfigLabelTransferMethodValidation` (5 tests) | ✓ VERIFIED | Collected 5, all PASSED (independently executed). |
| `tests/test_alignment_stage.py` | `TestEstepResultsCapture` (4 tests) + 8 call-site tuple-unpack fixes | ✓ VERIFIED | Collected 4, all PASSED. `grep -c 'result = AlignmentStage._build_aligned_cloud'` returns 0; `grep -c 'result, estep_results = AlignmentStage._build_aligned_cloud'` returns matching count. |
| `tests/test_label_transfer_stage.py` | `TestLabelTransferStageCpdWeighted` (5 tests) | ✓ VERIFIED | Collected 5, all PASSED. |
| `tests/test_eval_runner.py` | Regression test proving `align_result` passed as kwarg | ✓ VERIFIED | `test_align_result_passed_as_kwarg` PASSED, asserts identity on `call_args.kwargs`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `eval/stages/alignment.py::_build_aligned_cloud` (CPD branch) | `eval/types.py::AlignResult.estep_results` | `return aligned, estep_results` → `AlignResult(..., estep_results=estep_results)` | ✓ WIRED | Confirmed at lines 272, 290 (`run()`). |
| `eval/stages/alignment.py::_build_aligned_cloud` (CPD branch) | `zreg.cpd.RigidCPD.expectation_step` / `zreg.utils.squared_kernel_sum` | posterior-only E-step call after registration | ✓ WIRED | Lines 475-486. |
| `eval/config.py::EvalConfig.label_transfer_method` | `eval/stages/label_transfer.py::validate_params` | `self.config.label_transfer_method` read when `"method"` absent | ✓ WIRED | Line 160. |
| `eval/stages/label_transfer.py::run` | `src/zreg/color_transfer.py::transfer_colors(method=CPD_WEIGHTED)` | transposed `estep_result.pmat` + one-hot `source_colors` | ✓ WIRED | Lines 358-367. |
| `eval/runners/eval_runner.py::_run_single` | `eval/stages/label_transfer.py::LabelTransferStage.run(align_result=...)` | `align_result=align_result` kwarg | ✓ WIRED | Line 314; regression test asserts identity, not just call success. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `LabelTransferStage.run` (`cpd_weighted` branch) | `estep_result.pmat` | `AlignmentStage._build_aligned_cloud`'s `RigidCPD.expectation_step()` call on real registered point-cloud positions | Yes — real per-frame CPD posterior tensor, not a stub/static value | ✓ FLOWING |
| `EvaluationRunner._run_single` | `align_result` | `AlignmentStage(self.config).run(...)` (real stage execution, not mocked in production path) | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `EvalConfig` accepts/rejects `label_transfer_method` | `python -c "from eval.config import EvalConfig; c = EvalConfig(data_path='x'); assert c.label_transfer_method == 'knn_voting'"` | Exit 0 | ✓ PASS |
| `AlignResult.estep_results` field exists with correct default | `python -c "from eval.types import AlignResult; assert 'estep_results' in AlignResult.model_fields"` | Exit 0 | ✓ PASS |
| All 16 YAML scenario configs load with `knn_voting` default (no `method` key needed) | `EvalConfig.from_yaml(f)` for each of `configs/*.yaml` | "all configs load with knn_voting default" | ✓ PASS |
| `EstepResult` publicly exported from `zreg.cpd`, imported correctly in both `eval/types.py` and `eval/stages/alignment.py` | grep on import lines | Both import `from zreg.cpd import ... EstepResult` (not `._types`) | ✓ PASS |
| `src/zreg/` unmodified across all Phase 44 commits | `git diff acf9886 HEAD --stat -- src/zreg/` | Empty output | ✓ PASS |

### Probe Execution

Not applicable — this phase has no `scripts/*/tests/probe-*.sh` conventional probes and none are referenced in the plans/summaries. Skipped.

### Requirements Coverage

No `.planning/REQUIREMENTS.md` file exists in this repo state (confirmed via `ls`/`grep`) — this phase was added ad hoc per ROADMAP.md line 105 ("phase added ad hoc, no formal REQUIREMENTS.md IDs"). Coverage assessed against the 9 D-IDs in `44-CONTEXT.md` instead (see Observable Truths table above — all 9 SATISFIED).

### Anti-Patterns Found

None. Scanned all 5 modified production files (`eval/types.py`, `eval/config.py`, `eval/stages/alignment.py`, `eval/stages/label_transfer.py`, `eval/runners/eval_runner.py`) for `TBD|FIXME|XXX|TODO|HACK|PLACEHOLDER` and placeholder-language patterns — zero matches.

### Independent Test Execution

All test claims from SUMMARY.md files were independently re-run (not trusted from self-report):

- `tests/test_eval_config.py -k TestEvalConfigLabelTransferMethodValidation`: 5/5 PASSED
- `tests/test_alignment_stage.py -k TestEstepResultsCapture`: 4/4 PASSED
- `tests/test_label_transfer_stage.py -k TestLabelTransferStageCpdWeighted`: 5/5 PASSED
- `tests/test_eval_runner.py -k test_align_result_passed_as_kwarg`: 1/1 PASSED
- Combined Phase 44 regression sweep (`test_eval_config.py`, `test_alignment_stage.py`, `test_label_transfer_stage.py`, `test_eval_runner.py`): 190 passed (matches SUMMARY claim exactly)
- Full repo suite: 1205 passed, 18 skipped, 1 xpassed, **100% coverage** (matches SUMMARY claim exactly)

### Human Verification Required

None. This is a pure backend/library pipeline change (config → stage → runner wiring) with no UI, no external service integration, and no behavior that requires human judgment beyond what automated tests already cover. All 9 decisions in CONTEXT.md are independently verifiable via source inspection and test execution, both of which were performed above.

### Gaps Summary

No gaps found. All 9 CONTEXT.md decisions (D-01 through D-09) are correctly implemented and independently confirmed against the current source tree, not merely claimed in SUMMARY.md. All grep-based acceptance criteria from all 4 plans were independently re-executed and passed. All test classes named in the plans exist with the exact test counts claimed, and all pass. `zreg.color_transfer`/`zreg.cpd` were confirmed unmodified (zero-diff check), satisfying the phase's "pure plumbing, no new algorithms" boundary constraint. Backward compatibility was independently verified by loading all 16 existing YAML scenario configs through `EvalConfig.from_yaml()`.

---

_Verified: 2026-07-10T12:45:45Z_
_Verifier: Claude (gsd-verifier)_
