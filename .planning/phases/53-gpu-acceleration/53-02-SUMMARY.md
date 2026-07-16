---
phase: 53-gpu-acceleration
plan: "02"
subsystem: baseline_experiments
tags: [gpu, config, cluster, horeka, slurm, verification]
dependency_graph:
  requires: [53-01]
  provides: [GPU-03]
  affects:
    - baseline_experiments/configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml
    - baseline_experiments/configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml
    - baseline_experiments/configs_horeka/ground_truth/kobitski_ew06.yaml
    - baseline_experiments/configs_horeka/ground_truth/shah_sample1.yaml
    - baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_label_transfer.yaml
    - baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml
    - baseline_experiments/scripts/launch_horeka_multirank_test.sbatch
    - baseline_experiments/scripts/launch_horeka.sbatch
tech_stack:
  added: []
  patterns: [operator-facing VERIFY ON HOREKA comment convention, EvalConfig.device field (from Plan 01)]
key_files:
  created: []
  modified:
    - baseline_experiments/configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml
    - baseline_experiments/configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml
    - baseline_experiments/configs_horeka/ground_truth/kobitski_ew06.yaml
    - baseline_experiments/configs_horeka/ground_truth/shah_sample1.yaml
    - baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_alignment.yaml
    - baseline_experiments/configs_horeka/selfcal/shah_label_transfer.yaml
    - baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml
    - baseline_experiments/scripts/launch_horeka_multirank_test.sbatch
    - baseline_experiments/scripts/launch_horeka.sbatch
decisions:
  - "device: cuda placed after max_points_per_frame (end of source data block), before run_alignment — consistent across all 8 file structures"
  - "GPU-03 comment appended as new block after existing verify section (multirank) and after srun command (full-suite) to preserve existing file conventions"
metrics:
  duration: 4min
  completed: "2026-07-16"
  tasks: 2
  files: 10
---

# Phase 53 Plan 02: HoreKa Cluster GPU Config Activation Summary

Activated GPU tensor placement on HoreKa by adding `device: "cuda"` to all 8 cluster YAML configs, and annotated both launch sbatch files with the GPU-03 operator verification comment — completing the end-to-end verification path for GPU-03.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add device: "cuda" to all 8 cluster YAML configs | 61817af | 8 configs_horeka/**/*.yaml files |
| 2 | Append GPU-03 verification comment to launch_horeka_multirank_test.sbatch and launch_horeka.sbatch | 65c2d0f | launch_horeka_multirank_test.sbatch, launch_horeka.sbatch |

## What Was Built

**8 cluster YAML configs (baseline_experiments/configs_horeka/):**
- `device: "cuda"  # GPU-03: GPU tensor placement for HoreKa runs` added to all 8 files
- Placement: after `max_points_per_frame` (end of source data block), before `run_alignment` (first execution-control key)
- Files: `baseline_no_hpo/ew06_vs_shah.yaml`, `baseline_with_selfcal/ew06_vs_shah.yaml`, `ground_truth/kobitski_ew06.yaml`, `ground_truth/shah_sample1.yaml`, `selfcal/kobitski_ew06_alignment.yaml`, `selfcal/shah_alignment.yaml`, `selfcal/shah_label_transfer.yaml`, `smoke/kobitski_ew06_alignment.yaml`
- All 8 configs parse correctly via `EvalConfig.from_yaml()` with `cfg.device == "cuda"` verified

**2 sbatch scripts (baseline_experiments/scripts/):**
- `launch_horeka_multirank_test.sbatch`: GPU-03 comment block appended after existing VERIFY ON HOREKA section (after line with exception grep)
- `launch_horeka.sbatch`: GPU-03 comment block appended after `srun` command at end of file
- Comment text: `# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors`

## Verification Results

```
grep -rl 'device:' baseline_experiments/configs_horeka/ | wc -l  → 8
grep -r "device:" baseline_experiments/configs_horeka/            → device: "cuda" in all 8 files
EvalConfig.from_yaml() on all 8 files                            → All OK, cfg.device == "cuda"
grep "VERIFY ON HOREKA.*GPU-03" launch_horeka_multirank_test.sbatch → 1 match
grep "VERIFY ON HOREKA.*GPU-03" launch_horeka.sbatch              → 1 match
python -m pytest -x -q                                           → 1330 passed, 19 skipped, 1 xpassed
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary surface introduced. The `device: "cuda"` value is validated at parse time by `validate_device` (whitelist + ICP-compat guard, added in Plan 01), mitigating T-53-05 as specified in the plan's threat register. The `[VERIFY ON HOREKA] GPU-03` SLURM log grep line (T-53-04) is intentional operator-facing information, not sensitive data.

## Self-Check: PASSED
