---
phase: 57-ground-truth-field-consistency
verified: 2026-08-04T08:18:08Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Phase 57: Ground-Truth Field Consistency Verification Report

**Phase Goal:** Label-transfer F1 scoring is computed against the correct ground-truth field (`pc["label"]`, the field `LabelTransferStage` actually transfers) with correct per-point correspondence preserved even when `transform_spec`'s `dropout_fraction`/`n_new_points` change point counts between source and target, and existing ground-truth configs are audited/updated to the corrected convention.
**Verified:** 2026-08-04T08:18:08Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `EvalConfig.ground_truth_field` exists as `Literal["id","label"]="label"`, validated at construction | VERIFIED | `eval/config.py:260` — exact literal declaration; confirmed via direct `python -c` execution: default `"label"`, override to `"id"` works, `ground_truth_field="bogus"` raises `pydantic.ValidationError` |
| 2 | `get_ground_truth()`/`get_synthetic_ground_truth()` read `pc[config.ground_truth_field]`, not hardcoded `pc["id"]`, for both `pipeline_mode="paired"` and `"synthetic"` | VERIFIED | `eval/data_factory.py:669-676` (`get_ground_truth`) and `:747-754` (`get_synthetic_ground_truth`) both use `self.config.ground_truth_field`; no remaining hardcoded `pc["id"]` GT-extraction path |
| 3 | `DataFactory.drop_points()`/`sample_new_points()` thread a per-frame original-index correspondence map (`self._correspondence_idx`), composing correctly across chained calls, reset at the start of every `generate_target()` | VERIFIED | `eval/data_factory.py:154` (init), `:336` (reset in `generate_target`), `:1008-1024` (`drop_points`), `:1071-1103` (`sample_new_points`); dispatch order in `augment()` (`:520-527`) applies `dropout_fraction` before `n_new_points`, matching the chaining logic |
| 4 | `get_synthetic_ground_truth()` gathers `y_true` by tracked correspondence instead of truncating positionally — output length matches the target's actual post-dropout/new-points point count | VERIFIED | `eval/data_factory.py:756-772 (gather block); `tests/test_data_factory.py::test_gather_matches_target_length_after_dropout`, `::test_gather_appends_sentinel_for_new_points`, `::test_gather_combined_dropout_and_new_points` — all pass, and assert length/value correctness directly (not just non-crash) |
| 5 | Newly-added points (no source correspondence) receive a `-1` ground-truth sentinel, excluded from scoring by `compute_f1`'s existing mask | VERIFIED | `eval/data_factory.py` `sample_new_points` appends `-1` sentinel (`:1100-1102`); `compute_f1` sentinel mask confirmed pre-existing in `src/zreg/evaluation/label_transfer.py`; `tests/test_data_factory.py::test_compute_f1_accepts_gathered_length_without_shape_error` passes |
| 6 | `eval_runner._run_single`'s WR-01 truncation is documented as scoped to paired/heterogeneous mode, not relied on for synthetic-mode correctness | VERIFIED | `eval/runners/eval_runner.py:344-357` — comment re-scoped exactly as planned; code itself unchanged (still a valid paired-mode fallback); `tests/test_eval_runner.py::test_run_single_synthetic_mode_no_truncation_with_sentinel` proves the block is a no-op when correspondence-gathered lengths already match |
| 7 | `kobitski_ew06.yaml`'s header comment correctly attributes Shah's real classes to `pc["label"]`, not `pc["id"]` (D-05/D-06) | VERIFIED | File content directly read — stale "unlike Shah's 'id' which carries real small-integer class labels" claim removed, replaced with corrected D-05/D-06-cited text; `run_label_transfer: false` unchanged |
| 8 | `shah_sample1.yaml` carries a confirming D-05 note; no `ground_truth_field` override needed | VERIFIED | File content directly read — D-05 note present verbatim; `git diff` on the introducing commit shows comment-only addition, non-comment YAML byte-identical |
| 9 | All 5 `stage2_label_transfer/*/synthetic.yaml` configs document the constant `label=1` placeholder (`scripts/generate_datasets.py:239`) so F1 there isn't mistaken for real accuracy | VERIFIED | `grep -c "GT-03 audit note"` returns `1` for each of knn/hybrid_knn_cpd/pointnet2/cpd_weighted/egnn; `scripts/generate_datasets.py:239` confirmed to contain `"label": np.ones(len(pts), dtype=np.int32)`; `stage1_alignment/rigid_cpd/synthetic.yaml` confirmed to NOT contain the stale phrase (0 matches), consistent with plan's re-verification requirement; `run_label_transfer: true` unchanged in all 5 |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/config.py` | `ground_truth_field: Literal["id","label"]="label"` | VERIFIED | Present at line 260, plus Attributes docstring entry lines 121-132 documenting D-01/D-02 rationale |
| `eval/data_factory.py` | `_correspondence_idx` correspondence tracking + field-parameterized GT extraction | VERIFIED | `_correspondence_idx` present (init, reset, drop_points, sample_new_points, get_synthetic_ground_truth — 10 occurrences); `get_ground_truth`/`get_synthetic_ground_truth` both field-parameterized |
| `eval/runners/eval_runner.py` | WR-01 truncation block re-scoped/documented for paired-mode fallback only | VERIFIED | Comment block at lines 344-353 explicitly documents the Phase 57 D-03/GT-02 re-scoping; underlying truncation code (`if y_true.shape[0] != y_pred.shape[0]`) intentionally unchanged |
| `baseline_experiments/configs/ground_truth/kobitski_ew06.yaml` | corrected GT-field header comment | VERIFIED | D-05/D-06 correction present, stale claim removed |
| `baseline_experiments/configs/ground_truth/shah_sample1.yaml` | confirming GT-field header comment | VERIFIED | D-05 note present, comment-only diff confirmed |
| `configs/experiments/stage2_label_transfer/{knn,hybrid_knn_cpd,pointnet2,cpd_weighted,egnn}/synthetic.yaml` | degenerate-label audit note | VERIFIED | "GT-03 audit note" present in all 5 (grep count 1 each); comment-only diff confirmed for knn (representative) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `drop_points`/`sample_new_points` | `get_synthetic_ground_truth` | `self._correspondence_idx` | WIRED | `get_synthetic_ground_truth` (line 756) checks `self._correspondence_idx is not None and k in self._correspondence_idx` and gathers by it — same instance-attribute producer/consumer pattern verified by reading both sides |
| `get_synthetic_ground_truth` | `eval_runner._run_single` | `gt = self.factory.get_synthetic_ground_truth()` | WIRED | `eval/runners/eval_runner.py:329` calls this exactly when `pipeline_mode == "synthetic"` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GT-01 | 57-01 | F1 ground truth drawn from `pc["label"]` for both pipeline modes | SATISFIED | `eval/config.py` + `eval/data_factory.py` field-parameterization, verified above |
| GT-02 | 57-01 | Correct per-point correspondence preserved under dropout/new-points | SATISFIED | `_correspondence_idx` mechanism + gather logic, verified with passing tests including value-level correctness assertions |
| GT-03 | 57-02 | Existing GT configs audited/updated for corrected convention | SATISFIED | Both real-data configs corrected/confirmed; all 5 synthetic stage2 configs documented; stage1_alignment file re-verified clean |

No orphaned requirements — REQUIREMENTS.md maps exactly GT-01/02/03 to Phase 57, and all three are declared in the two plans' frontmatter and satisfied.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers introduced in any of the 12 modified files (5 code/test files from 57-01, 7 config files from 57-02). The word "placeholder" appears only in intentional documentation prose (describing the known-degenerate `label=1` constant in the 5 stage2 configs), not as a code stub marker.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `ground_truth_field` Literal validation | `EvalConfig(data_path="x", ground_truth_field="bogus")` | raises `pydantic.ValidationError` | PASS |
| `ground_truth_field` default/override | `EvalConfig(data_path="x").ground_truth_field`, `...ground_truth_field="id")...` | `"label"`, `"id"` respectively | PASS |
| Targeted test suite (data_factory, must-haves subset) | `pytest tests/test_data_factory.py -k "TestDropPoints or TestSampleNewPoints or TestGenerateTarget or TestEvalConfigFromYAML or TestGetGroundTruth or TestGetSyntheticGroundTruth" -q` | 53 passed | PASS |
| Targeted test suite (eval_runner, must-haves subset) | `pytest tests/test_eval_runner.py -k "Correspondence or truncat or Synthetic" -q` | 6 passed | PASS |
| Full data_factory + eval_runner regression suite | `pytest tests/test_data_factory.py tests/test_eval_runner.py -q` | 149 passed, 0 failed | PASS |
| Pre-existing unrelated failure re-check | `pytest tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery -q` | 1 passed (in isolation, consistent with claimed test-order-dependent flakiness, unrelated to Phase 57 files) | PASS |
| Config comment-only diff verification | `git diff <commit>^ <commit> -- <config file>` | comment-only additions, non-comment YAML unchanged | PASS |
| `generate_datasets.py:239` citation accuracy | direct file read | `"label": np.ones(len(pts), dtype=np.int32)` confirmed at cited line | PASS |

### Human Verification Required

None. This phase is an internal correctness fix (no UI, no user-facing behavior) — all success criteria are verifiable programmatically via code inspection and test execution, and all have been verified directly.

### Gaps Summary

No gaps found. All 9 derived observable truths (covering ROADMAP.md's 4 success criteria plus PLAN-frontmatter-level detail) are verified directly against current file contents and passing test executions — not inferred from SUMMARY.md claims. Both plans' `must_haves` (truths, artifacts, key_links) hold. All three requirement IDs (GT-01, GT-02, GT-03) are satisfied with concrete evidence. No anti-patterns, no orphaned requirements, no regressions in the targeted or full data_factory/eval_runner suites.

---

*Verified: 2026-08-04T08:18:08Z*
*Verifier: Claude (gsd-verifier)*
