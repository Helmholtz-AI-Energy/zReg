---
phase: 37-plot-trajectory-label-refactor
verified: 2026-06-22T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Confirm VIZ-03 is added to the REQUIREMENTS.md traceability table"
    expected: "A row `| VIZ-03 | Phase 37 | Complete ... |` appears in the Traceability section of .planning/REQUIREMENTS.md"
    why_human: "VIZ-03 is declared in PLAN frontmatter and is the governing requirement ID for this phase, but it does not appear anywhere in .planning/REQUIREMENTS.md (neither the category definition list nor the traceability table). All other phase requirements (ALIGN-01, ALIGN-02, MODE-01…03, etc.) have matching rows. Automated grep on .planning/REQUIREMENTS.md returns no matches for VIZ-03. This is a bookkeeping gap that cannot be auto-fixed; a human must decide whether to add a definition + traceability row, or whether VIZ-03 is intentionally an informal sub-requirement of FRAME-08."
---

# Phase 37: plot_trajectory Label Figure Refactor — Verification Report

**Phase Goal:** Refactor the label branch of `plot_trajectory` in `eval/viz.py` to produce two independent 1×3 figure pairs (PDF + PNG each = 4 files) instead of the current single figure: (1) `label_source_trajectory` — source point cloud coloured by source labels; (2) `label_target_trajectory` — target cloud coloured by transferred labels. Alignment branch not touched.
**Verified:** 2026-06-22
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `plot_trajectory(None, label_result, dataset, None, tmp_path)` writes exactly 4 files: `label_source_trajectory.pdf/.png` and `label_target_trajectory.pdf/.png` | VERIFIED | `test_label_only_writes_label_files` and `test_label_writes_two_figure_pairs` both assert `len(result)==4` and check all 4 stems — 6 tests passed |
| 2 | Old stem `label_trajectory` is no longer produced | VERIFIED | No string `"label_trajectory"` appears as a stem in `eval/viz.py`; `test_label_only_writes_label_files` asserts `assert not (tmp_path / "label_trajectory.pdf").exists()` — passed |
| 3 | Source figure uses `dataset[fk]["id"]` when not None; falls back to `dataset[fk]["color"]`; graceful single-colour scatter when both are None | VERIFIED | `_get_source_labels` at line 116 implements priority exactly (`id` → `color` → None); `_write_label_figure` renders `"#aaaaaa"` when `c_vals is None` (line 281); `test_label_source_uses_id_when_available` passes with `color=None, id=ones` — does not crash |
| 4 | Target figure uses `target[fk]["pos"]` when target is provided; falls back to `align_result.aligned_cloud[fk]` or `dataset[fk]` otherwise (D-07 logic) | VERIFIED | Lines 525–535 in `viz.py`: `if target is not None and fk in target` → `elif align_result is not None and fk in align_result.aligned_cloud` → `else dataset[fk]` — full D-07 precedence preserved |
| 5 | No matplotlib figure leaks | VERIFIED | `_save_fig` uses a `finally: plt.close(fig)` block (line 150–154); `test_label_no_figure_leak` and `test_no_figure_leak` both assert `after == before` — passed |
| 6 | All tests pass after updates; ≥3 new tests added for label 2-figure behaviour | VERIFIED | `tests/test_viz.py` contains exactly 3 new tests: `test_label_writes_two_figure_pairs`, `test_label_source_uses_id_when_available`, `test_label_no_figure_leak`; full suite: 26 passed, 0 failed |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/viz.py` — `_get_source_labels` | Module-level private helper, id>color priority | VERIFIED | Lines 116–125; substantive (not a stub); called at lines 487, 504 |
| `eval/viz.py` — `_write_label_figure` | Module-level private helper for 1×3 label figure | VERIFIED | Lines 242–295; substantive; called at lines 555 and 563 |
| `eval/viz.py` — label branch refactor | `label_source_trajectory` and `label_target_trajectory` stems | VERIFIED | Lines 467–568; replaces old single-stem code; no `label_trajectory` stem present |
| `tests/test_viz.py` — 3 new label tests | `test_label_writes_two_figure_pairs`, `test_label_source_uses_id_when_available`, `test_label_no_figure_leak` | VERIFIED | Lines 315–348; all three present and passing |
| `tests/test_viz.py` — 5 updated existing tests | Updated counts and stems | VERIFIED | `test_label_only_writes_label_files` (len==4), `test_both_stages_writes_ten_files` (len==10), `test_label_names_used_when_provided` (len==10), `test_no_figure_leak`, `test_plot_trajectory_large_label_dataset_subsamples` (len==4) — all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `plot_trajectory` label branch | `_get_source_labels` | called at lines 487, 504 | WIRED | Both union-label computation and per-frame source pre-computation call the helper |
| `plot_trajectory` label branch | `_write_label_figure` | called at lines 555, 563 | WIRED | Source figure (stem `label_source_trajectory`) and target figure (stem `label_target_trajectory`) |
| `_write_label_figure` | `_save_fig` | called at line 295 | WIRED | Save+close path is the shared helper, not inline |
| `_get_source_labels` | `zRegPointCloud.get("id")` / `.get("color")` | lines 121–124 | WIRED | Uses `.get()` guard — safe when key is absent |
| Label branch | `_deduplicate_frames` | line 469 | WIRED | Reuses Phase 36 helper as required |
| Label branch | `_subsample` | lines 518, 539 | WIRED | Used for grey-fallback and target pre-computation |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `_write_label_figure` (label_source_trajectory) | `source_colors_map[fk]` | `_get_source_labels(dataset[fk])` → torch.Tensor from `pc["id"]` or `pc["color"]` | Yes — reads actual tensor fields from dataset | FLOWING |
| `_write_label_figure` (label_target_trajectory) | `target_colors_map[fk]` | `label_result.transferred_labels[fk]` — real tensor from LabelTransferStage | Yes | FLOWING |
| `source_pos_map[fk]` | `dataset[fk]["pos"].detach().cpu().numpy()` | Dataset pos tensors | Yes | FLOWING |
| `target_pos_map[fk]` | `target/aligned_cloud/dataset[fk]["pos"]` | D-07 precedence chain | Yes | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| label-only → 4 files, new stems | `pytest tests/test_viz.py::TestPlotTrajectory::test_label_writes_two_figure_pairs -q` | 1 passed | PASS |
| id-priority over color | `pytest tests/test_viz.py::TestPlotTrajectory::test_label_source_uses_id_when_available -q` | 1 passed | PASS |
| no figure leak | `pytest tests/test_viz.py::TestPlotTrajectory::test_label_no_figure_leak -q` | 1 passed | PASS |
| label-only count==4 | `pytest tests/test_viz.py::TestPlotTrajectory::test_label_only_writes_label_files -q` | 1 passed | PASS |
| align+label no target = 10 files | `pytest tests/test_viz.py::TestPlotTrajectory::test_both_stages_writes_ten_files -q` | 1 passed | PASS |
| large dataset subsampling (4 files) | `pytest tests/test_viz.py::TestVizCoverageGaps::test_plot_trajectory_large_label_dataset_subsamples -q` | 1 passed | PASS |
| full test_viz.py suite | `pytest tests/test_viz.py -q` | 26 passed | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| VIZ-03 | 37-01-PLAN.md | Label branch produces two 1×3 figure pairs (4 files) with `label_source_trajectory` and `label_target_trajectory` stems | SATISFIED — implementation complete | `eval/viz.py` lines 467–568; 6 new/updated tests pass |
| VIZ-03 traceability entry | — | VIZ-03 must appear in `.planning/REQUIREMENTS.md` Category listing and traceability table | NOT RECORDED | Grep of `.planning/REQUIREMENTS.md` returns zero matches for "VIZ-03". All sibling requirements (ALIGN-01, ALIGN-02, MODE-01–03, etc.) have explicit definitions and traceability rows. This is an open bookkeeping gap — flagged for human action below. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | No TBD/FIXME/XXX/HACK/PLACEHOLDER markers detected in `eval/viz.py` or `tests/test_viz.py` |

---

### Human Verification Required

#### 1. Add VIZ-03 to REQUIREMENTS.md

**Test:** Open `.planning/REQUIREMENTS.md`. Search for "VIZ-03". Confirm it is absent. Add:
- A definition entry in Category 9 (Runners & Visualisation) alongside FRAME-07 and FRAME-08:
  `- [x] **VIZ-03**: label branch of plot_trajectory produces two independent 1×3 figure pairs (4 files): label_source_trajectory.{pdf,png} (source labels) and label_target_trajectory.{pdf,png} (transferred labels) — Complete Phase 37`
- A row in the traceability table:
  `| VIZ-03 | Phase 37 | Complete 2026-06-22 | Two-figure label output — label_source_trajectory + label_target_trajectory |`

**Expected:** After edit, `grep "VIZ-03" .planning/REQUIREMENTS.md` returns 2+ lines.

**Why human:** Requirements bookkeeping is a human editorial decision (whether VIZ-03 is a first-class requirement or a sub-item of FRAME-08). Cannot auto-patch REQUIREMENTS.md under this verification mandate.

---

### Gaps Summary

No implementation gaps found. All 6 success criteria are satisfied in the codebase. The single open item is a requirements bookkeeping gap — VIZ-03 is not recorded in `.planning/REQUIREMENTS.md` — which does not block the code from working but is a traceability audit concern. Status is `human_needed` because this editorial decision requires a human.

---

_Verified: 2026-06-22_
_Verifier: Claude (gsd-verifier)_
