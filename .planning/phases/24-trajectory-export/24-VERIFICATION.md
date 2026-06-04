---
phase: 24-trajectory-export
verified: 2026-06-04T12:00:00Z
status: passed
score: 5/5
overrides_applied: 0
re_verification: false
---

# Phase 24: Trajectory Export Verification Report

**Phase Goal:** Export AlignResult and/or LabelResult as point-per-row CSV files (align_trajectory.csv, label_trajectory.csv) with accompanying metadata.json, written automatically from EvaluationRunner.run() conditional on which stages ran. CSV columns: frame_idx, point_idx, x, y, z, label — optimised for LaTeX/pgfplots reuse.
**Verified:** 2026-06-04T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                         | Status     | Evidence                                                                                                                                                      |
|----|---------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | align_trajectory.csv written when run_alignment=True; absent otherwise                                        | VERIFIED   | `trajectory.py` lines 118–153: `if result["align"] is not None` guards CSV write. `TestExportTrajectoryAlignOnly::test_label_csv_absent` and `TestExportTrajectoryLabelOnly::test_align_csv_absent` both PASS (21/21 in test_trajectory_export.py). |
| 2  | label_trajectory.csv written when run_label_transfer=True; absent otherwise                                   | VERIFIED   | `trajectory.py` lines 156–197: `if result["label"] is not None` guards write. Covered by `TestExportTrajectoryLabelOnly::test_label_csv_created` and `TestExportTrajectoryAlignOnly::test_label_csv_absent` PASS. |
| 3  | CSV has correct header and one row per point per frame                                                        | VERIFIED   | align header `["frame_idx", "point_idx", "x", "y", "z"]` (line 125); label header adds `"label"` (line 163). Row-count test asserts 3×20=60 rows for 3 frames × 20 points — PASSES. x/y/z are floats via `.tolist()`; label is int via `int(labels[i].item())`. |
| 4  | align_metadata.json / label_metadata.json contain all 11 required fields (run_id, frame_count, frame_indices, data_path, params_used, tier, n_trials, n_synthetic, git_hash, zreg_version, timestamp) | VERIFIED   | `trajectory.py` lines 133–145 (align) and 177–189 (label): dict literals contain all 11 keys. `TestExportTrajectoryMetadata::test_metadata_all_required_fields` asserts `EXPECTED_META_FIELDS.issubset(meta.keys())` — PASSES. |
| 5  | EvalReport.trajectory_paths list reflects written files                                                       | VERIFIED   | `eval/types.py` line 314: `trajectory_paths: list[str] = Field(default_factory=list)`. `eval_runner.py` line 221 calls `export_trajectory(...)`, line 227 passes result to `model_copy`. `TestExportTrajectoryIntegration::test_eval_runner_report_has_trajectory_paths` asserts `len(report.trajectory_paths) >= 2` — PASSES. `test_trajectory_paths_in_json` confirms key present in `eval_report.json` — PASSES. |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact                              | Expected                                          | Status     | Details                                                                                      |
|---------------------------------------|---------------------------------------------------|------------|----------------------------------------------------------------------------------------------|
| `eval/tracking/trajectory.py`         | export_trajectory function — CSV and JSON writers | VERIFIED   | 201-line stdlib-only implementation. No third-party imports at module level. Exports `export_trajectory` in `__all__`. |
| `eval/tracking/__init__.py`           | Public re-export of export_trajectory alongside log_run | VERIFIED | Lines 12–14: `from .trajectory import export_trajectory`; `__all__ = ["log_run", "export_trajectory"]`. |
| `eval/types.py`                       | trajectory_paths field on EvalReport              | VERIFIED   | Line 314: `trajectory_paths: list[str] = Field(default_factory=list)`. Positioned after `plot_paths` (line 313) and before `sanity_flags` (line 315). |
| `eval/runners/eval_runner.py`         | export_trajectory wiring in run()                 | VERIFIED   | 6 trajectory-related lines: import (57), comment (197), field init (204), call assignment (221), comment (223), model_copy key (227). |
| `tests/test_trajectory_export.py`     | EXT-01 gate test suite                            | VERIFIED   | 21 tests across 5 classes. All pass. 787 total tests pass, 17 skipped (no regressions). |

---

## Key Link Verification

| From                            | To                                         | Via                                           | Status   | Details                                                                                  |
|---------------------------------|--------------------------------------------|-----------------------------------------------|----------|------------------------------------------------------------------------------------------|
| `eval/tracking/trajectory.py`   | `eval/tracking/__init__.py`                | `from .trajectory import export_trajectory`   | WIRED    | Line 12 of `__init__.py` confirmed.                                                      |
| `export_trajectory`             | `align_trajectory.csv / label_trajectory.csv` | `csv.writer` loop over `sorted(dataset.keys())` | WIRED | `csv.writer` calls at lines 123–130 and 161–174. Files produced and verified by tests.  |
| `eval/runners/eval_runner.py`   | `eval/tracking/trajectory.py`              | `from eval.tracking import export_trajectory` | WIRED    | Line 57 imports function; line 221 calls it with `(result, dataset, self.config, output_dir_path)`. |
| `eval/runners/eval_runner.py`   | `eval/types.py EvalReport`                 | `model_copy(update={'plot_paths': ..., 'trajectory_paths': ...})` | WIRED | Single `model_copy` call at lines 226–228 sets both fields atomically per D-14.         |
| `eval/types.py EvalReport`      | `eval_report.json`                         | `report.model_dump()` in `save_report()`      | WIRED    | `save_report` at line 367 calls `report.model_dump()`. Integration test confirms `trajectory_paths` key in written JSON. |

---

## Data-Flow Trace (Level 4)

| Artifact                      | Data Variable       | Source                                                   | Produces Real Data | Status    |
|-------------------------------|---------------------|----------------------------------------------------------|--------------------|-----------|
| `align_trajectory.csv`        | pos tensor          | `AlignResult.aligned_cloud[frame_idx]["pos"]`            | Yes — tensor from stage result, not hardcoded | FLOWING |
| `label_trajectory.csv`        | pos tensor + labels | `aligned_cloud["pos"]` or `dataset["pos"]`; `LabelResult.transferred_labels[frame_idx]` | Yes — live stage results | FLOWING |
| `EvalReport.trajectory_paths` | list[str]           | Return value of `export_trajectory(result, ...)` in `run()` | Yes — file paths written to disk | FLOWING |

---

## Behavioral Spot-Checks

| Behavior                                               | Command                                                                                                 | Result              | Status |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------|--------|
| export_trajectory importable from eval.tracking        | `python -c "from eval.tracking import export_trajectory, log_run; print('both importable')"`            | "both importable"   | PASS   |
| align CSV header correct                               | `pytest tests/test_trajectory_export.py::TestExportTrajectoryAlignOnly::test_align_csv_header`          | PASSED              | PASS   |
| label CSV absent when label=None                       | `pytest tests/test_trajectory_export.py::TestExportTrajectoryAlignOnly::test_label_csv_absent`          | PASSED              | PASS   |
| 21 EXT-01 tests all pass                               | `pytest tests/test_trajectory_export.py -v`                                                             | 21/21 passed        | PASS   |
| No regressions across full suite                       | `pytest tests/ -q`                                                                                      | 787 passed, 17 skipped | PASS |

---

## Probe Execution

No explicit probe scripts declared for this phase. Step 7c skipped — behavioral spot-checks above cover the equivalent gate criteria.

---

## Requirements Coverage

| Requirement | Source Plan    | Description                                                         | Status    | Evidence                                                              |
|-------------|----------------|---------------------------------------------------------------------|-----------|-----------------------------------------------------------------------|
| EXT-01      | 24-01, 24-02   | Export per-point trajectory CSVs with metadata JSON from EvalRunner | SATISFIED | align/label CSV files written conditionally; metadata contains all 11 fields; trajectory_paths in EvalReport and eval_report.json; 21 EXT-01 gate tests pass. |

---

## Anti-Patterns Found

| File                              | Line | Pattern          | Severity | Impact |
|-----------------------------------|------|------------------|----------|--------|
| No debt markers found             | —    | —                | —        | —      |

Scanned `eval/tracking/trajectory.py`, `eval/tracking/__init__.py`, `eval/types.py`, `eval/runners/eval_runner.py`, `tests/test_trajectory_export.py`. No TBD, FIXME, XXX, TODO, HACK, or placeholder strings found. No stub `return null` / `return []` patterns. All returns produce live written paths.

---

## Human Verification Required

None. All success criteria are machine-verifiable and the full test suite confirms them.

---

## Gaps Summary

No gaps. All 5 must-have truths are VERIFIED with direct code evidence and passing test coverage. The test suite (787 tests, 17 skipped) shows no regressions.

---

_Verified: 2026-06-04T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
