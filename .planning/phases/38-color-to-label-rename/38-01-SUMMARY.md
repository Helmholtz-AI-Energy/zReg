---
phase: 38
plan: 38-01
subsystem: dataset, downsampling, generators, cpd, color_transfer, eval
tags: [rename, refactor, CLN-01, CLN-02]
dependency_graph:
  requires: [phase-37]
  provides: [label-field-in-zRegPointCloud]
  affects: [dataset.py, downsampling.py, generators/labels.py, generators/corruption.py, cpd/_registration.py, color_transfer.py, eval/data_factory.py, eval/viz.py, eval/stages/label_transfer.py, eval/runners/optimizer.py, scripts/generate_datasets.py, scripts/color_transfer_example.py]
tech_stack:
  added: []
  patterns: [pure-rename, CLN-02-heuristic-removal]
key_files:
  created: []
  modified:
    - src/zreg/dataset.py
    - src/zreg/downsampling.py
    - src/zreg/generators/labels.py
    - src/zreg/generators/corruption.py
    - src/zreg/cpd/_registration.py
    - src/zreg/color_transfer.py
    - eval/data_factory.py
    - eval/viz.py
    - eval/stages/label_transfer.py
    - eval/runners/optimizer.py
    - scripts/generate_datasets.py
    - scripts/color_transfer_example.py
decisions:
  - "Renamed zRegPointCloud field 'color' to 'label' across all 12 production files"
  - "CLN-02: Removed id/color heuristic from LabelTransferStage; reads label directly with ValueError"
  - "CLN-02: Rewrote _get_source_labels in viz.py to read label only, never falls back to id"
  - "use_color= parameter in CPD classes left unchanged (controls registration behavior, not field name)"
  - "External .mat field name 'color' and Open3D 'colors' map key left unchanged (external APIs)"
metrics:
  duration: 604s
  completed: "2026-06-23T13:40:43Z"
  tasks_completed: 12
  files_modified: 12
---

# Phase 38 Plan 01: color→label field rename in production code — Summary

Renamed `zRegPointCloud["color"]` to `zRegPointCloud["label"]` across all 12 production
source files. Also replaced the `id`/`color` heuristic in `LabelTransferStage` and
`_get_source_labels` with direct `label` field reads (CLN-02).

## What Was Built

`zRegPointCloud` previously used `color` as the field name for semantic class labels
(integer IDs from synthetic data, or RGB triples from real tracklets). This was semantically
wrong because `color` implies display/appearance, while the field stores cell-type identity.
All 12 production files that read or wrote this field were updated to use `label`.

Additionally, two CLN-02 behavioral changes:
1. `LabelTransferStage.run()` no longer uses a multi-channel RGB dimension heuristic to
   decide between `id` and `color`. It reads `src_frame.get("label")` directly and raises
   `ValueError("Source frame {sk} has no 'label' field.")` when absent.
2. `_get_source_labels` in `eval/viz.py` no longer has an `id` priority fallback. It reads
   `pc["label"]` and returns `None` when absent (grey render fallback).

## Commits

| Task | File | Commit | Change |
|------|------|--------|--------|
| 1 | src/zreg/dataset.py | 218f639 | __init__, load_data_from_tracklets (3 sites), zreg_to_open3d (2 sites + docstring), open3d_to_zreg (2 sites + docstring), load_shah_from_csv |
| 2 | src/zreg/downsampling.py | 3278019 | 4 target["color"] → target["label"] |
| 3 | src/zreg/generators/labels.py | 23eb830 | 2 code sites + module/function docstrings |
| 4 | src/zreg/generators/corruption.py | 172048f | 7 code sites + docstring |
| 5 | src/zreg/cpd/_registration.py | 96e7ad8 | 3 sites in cpd_registration + init_cpd_from_existing |
| 6 | src/zreg/color_transfer.py | 5289bb8 | 1 site in transfer_colors |
| 7 | eval/data_factory.py | 542cbd7 | 4 constructor args + 1 docstring |
| 8 | eval/viz.py | 7b8b059 | _get_source_labels rewritten (CLN-02) + docstring |
| 9 | eval/stages/label_transfer.py | c6c3b9c | heuristic removed; direct label read + ValueError (CLN-02) |
| 10 | eval/runners/optimizer.py | 04f0450 | 2 field accesses + gt_key fallback + comments |
| 11 | scripts/generate_datasets.py | 7039e96 | dict literal key + 2 frame accesses |
| 12 | scripts/color_transfer_example.py | cb1a5fb | constructor arg |

## Deviations from Plan

None — plan executed exactly as written.

The fast-forward merge from `feature/evaluation_framework` at plan start (b76a576 → ada3de5)
was required because the worktree was spawned from an intermediate commit that predated the
eval/ directory. This is an execution setup artifact, not a plan deviation.

## Known Stubs

None. All renames are mechanical; no stubs introduced.

## Threat Flags

None. Pure rename with no new network endpoints, auth paths, or schema changes.

## Self-Check: PASSED

All 12 modified files confirmed present on disk. All 12 task commits confirmed in git log.
