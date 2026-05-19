---
plan: "16-01"
phase: "16-runner-scripts"
status: complete
self_check: PASSED
---

# Summary: Plan 16-01 — Runner Scripts (run_synthetic.py + run_real.py)

## What Was Built

Created two standalone sweep orchestrators that wire together the upstream packages
(eval/generators/, eval/tracking/, zreg.metrics) into logged sweep loops per EVAL-05.

## Key Files Created

### key-files.created
- path: eval/run_synthetic.py
  description: Noise/outlier sweep — iterates SIGMAS × N_OUTLIERS_LIST (12 cells), computes chamfer/hausdorff via zreg.metrics, logs every run via log_run()
- path: eval/run_real.py
  description: Scale/density sweep — iterates SCALES × DENSITY_FRACTIONS, logs via log_run(); exits gracefully with "not found" when DATASET_PATH is absent

## Verification Results

- run_synthetic.py: run_sweep(output_dir=tmp) created exactly 12 .json files (4 SIGMAS × 3 N_OUTLIERS_LIST) ✓
- run_real.py: run_sweep(output_dir=tmp) prints "not found", creates 0 files when DATASET_PATH absent ✓
- Both scripts have `if __name__ == "__main__": run_sweep()` guard ✓
- No eval/__init__.py created — eval/ remains namespace directory ✓
- run_synthetic.py imports from zreg.metrics (installed package) ✓
- run_id construction embeds all sweep-varying params (sigma/n_out/seed, scale/density/seed) ✓

## Commits

- feat(16-01): create eval/run_synthetic.py — noise/outlier sweep orchestrator
- feat(16-01): create eval/run_real.py — scale/density sweep with missing-data guard

## Deviations

None. Implementation matches plan specification exactly.

## Self-Check: PASSED
