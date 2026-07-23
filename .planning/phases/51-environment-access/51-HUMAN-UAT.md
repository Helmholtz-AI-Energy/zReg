---
status: partial
phase: 51-environment-access
source: [51-VERIFICATION.md]
started: 2026-07-15T00:00:00Z
updated: 2026-07-15T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Submit smoke SLURM job on HoreKa
expected: Fill in <name> and <your-project> placeholders in launch_horeka_smoke.sbatch and setup_horeka.sh. Run setup_horeka.sh, transfer data via rsync + symlink, then submit: sbatch baseline_experiments/scripts/launch_horeka_smoke.sbatch. Job must show COMPLETED (not FAILED/CANCELLED) in squeue, SLURM output file must contain no Python tracebacks, eval_report.json must exist under baseline_experiments/experiments/smoke/kobitski_ew06_alignment/<timestamp>/, and GPU must be available (torch.cuda.is_available() = True on compute node).
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
