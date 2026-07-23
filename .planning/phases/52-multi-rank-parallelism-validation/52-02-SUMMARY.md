---
phase: 52-multi-rank-parallelism-validation
plan: "02"
subsystem: baseline_experiments/configs_horeka
tags: [cluster-configs, hpo, search-strategy, para-03, budg-01]
dependency_graph:
  requires: []
  provides:
    - baseline_experiments/configs_horeka/ (7 YAML configs for cluster runs)
  affects:
    - baseline_experiments/scripts/run_all.py (reads --configs-dir configs_horeka at cluster runtime)
tech_stack:
  added: []
  patterns:
    - "YAML config mirroring: source configs/ copied verbatim to configs_horeka/ with search_strategy sobol→auto"
key_files:
  created:
    - baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_label_transfer.yaml
    - baseline_experiments/configs_horeka/ground_truth/kobitski_ew06.yaml
    - baseline_experiments/configs_horeka/ground_truth/shah_sample1.yaml
    - baseline_experiments/configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml
    - baseline_experiments/configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml
  modified: []
decisions:
  - "search_strategy: auto used in HPO configs so cluster runs resolve to propulate when SLURM_JOB_ID is set (auto-detect in eval/runners/optimizer.py _detect_backend)"
  - "Eval-only configs (baseline_no_hpo, baseline_with_selfcal) receive header comment only — no field changes since they have no search_strategy field"
  - "output_dir values kept identical to configs/ originals — cluster runs write into the same experiment directory tree"
metrics:
  duration: 2min
  completed: "2026-07-15"
  tasks_completed: 1
  tasks_total: 1
  files_created: 7
  files_modified: 0
---

# Phase 52 Plan 02: configs_horeka/ Cluster Config Tree Summary

7 cluster-targeted YAML configs created under baseline_experiments/configs_horeka/ by mirroring configs/ verbatim with search_strategy sobol→auto in the 5 HPO configs and header comment additions only in the 2 eval-only copies.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create configs_horeka/ with 7 mirrored configs | d3f32d7 | 7 new YAML files |

## What Was Built

Created `baseline_experiments/configs_horeka/` with the same 4-subdirectory layout as `configs/`:

- `selfcal/` — 3 HPO configs (kobitski_ew06_alignment, shah_alignment, shah_label_transfer)
- `ground_truth/` — 2 HPO configs (kobitski_ew06, shah_sample1)
- `baseline_no_hpo/` — 1 eval-only config (ew06_vs_shah)
- `baseline_with_selfcal/` — 1 eval-only config (ew06_vs_shah)

**5 HPO configs:** `search_strategy: sobol` changed to `search_strategy: auto` (all other fields verbatim). Header comment added explaining cluster variant and auto→propulate resolution path.

**2 eval-only configs:** Exact field copies of configs/ originals. Header comment added noting the cluster variant and that no search_strategy field exists (eval-only, no HPO).

**BUDG-01 preserved:** `max_points_per_frame: 1000` in all 7 files, `step: 8` in default_params across all HPO configs and both eval-only configs.

**PARA-03 satisfied:** configs_horeka/ exists as an independent cluster-targeted config tree. `run_all.py --configs-dir baseline_experiments/configs_horeka` will invoke propulate-backed HPO on the cluster without touching local sobol configs.

## Verification Results

All acceptance criteria passed:

- `find baseline_experiments/configs_horeka -name "*.yaml" | wc -l` → 7
- `grep -r "^search_strategy: auto" baseline_experiments/configs_horeka/` → 5 lines (all HPO configs)
- `grep -r "search_strategy: sobol" baseline_experiments/configs_horeka/` → 0 results
- `grep -r "max_points_per_frame: 1000" baseline_experiments/configs_horeka/` → 7 lines
- `grep -rn "step: 8" baseline_experiments/configs_horeka/` → 7 lines (across all files)
- `diff configs/baseline_no_hpo/ew06_vs_shah.yaml configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml` → header comment addition only
- `diff configs/baseline_with_selfcal/ew06_vs_shah.yaml configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml` → header comment addition only
- `git diff baseline_experiments/configs/` → no changes (local configs/ untouched)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. All 7 configs are fully wired with real data paths, output directories, and field values copied verbatim from audited configs/ sources.

## Threat Flags

None. No new network endpoints, auth paths, or trust boundaries introduced. YAML files are version-controlled developer artifacts with relative output_dir paths inside the repo tree (matches T-52-04 and T-52-05 accepted dispositions in the plan's threat model).

## Self-Check: PASSED

- [x] baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml — FOUND
- [x] baseline_experiments/configs_horeka/selfcal/shah_alignment.yaml — FOUND
- [x] baseline_experiments/configs_horeka/selfcal/shah_label_transfer.yaml — FOUND
- [x] baseline_experiments/configs_horeka/ground_truth/kobitski_ew06.yaml — FOUND
- [x] baseline_experiments/configs_horeka/ground_truth/shah_sample1.yaml — FOUND
- [x] baseline_experiments/configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml — FOUND
- [x] baseline_experiments/configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml — FOUND
- [x] Commit d3f32d7 — FOUND
