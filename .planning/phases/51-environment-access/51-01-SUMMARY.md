---
phase: 51
plan: "01"
subsystem: hpc-environment
tags: [slurm, hpc, horeka, environment-setup, mpi4py, sbatch, smoke-config]
dependency_graph:
  requires: []
  provides:
    - scripts/setup_horeka.sh
    - baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml
    - baseline_experiments/scripts/launch_horeka_smoke.sbatch
    - baseline_experiments/scripts/launch_horeka.sbatch
  affects:
    - Phase 52 (PARA-01/PARA-02/PARA-03) — multi-rank job activated via launch_horeka.sbatch
    - baseline_experiments/scripts/run_all.py — launch_horeka.sbatch calls it (rank-aware fix deferred to Phase 52)
tech_stack:
  added: []
  patterns:
    - Self-contained SLURM sbatch (no separate launch_srun.sh; D-04)
    - Absolute symlink for HPC data path resolution (D-06)
    - mpi4py source build against system MPI (pip install --no-binary --force-reinstall)
    - [VERIFY ON HOREKA] inline comment convention (HOREKA_PLAN.md pattern)
key_files:
  created:
    - scripts/setup_horeka.sh
    - baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml
    - baseline_experiments/scripts/launch_horeka_smoke.sbatch
    - baseline_experiments/scripts/launch_horeka.sbatch
  modified: []
decisions:
  - "D-04: Self-contained sbatch scripts (no separate launcher shell); inline module load + venv activation + Python invocation"
  - "D-06: Absolute symlink for data path (ln -s /hkfs/work/workspace/<name>/...) — relative symlinks break when srun changes cwd"
  - "D-12: Smoke config in new configs/smoke/ subdirectory; smoke sbatch points directly at it via --config"
  - "Smoke sbatch uses run_eval.py (not run_all.py): run_all.py hardcodes CONFIGS path (Pitfall 6); run_eval.py avoids code changes"
  - "search_strategy: sobol in smoke config (not auto): auto resolves to propulate when SLURM_JOB_ID set, deadlocks single-rank job"
  - "tier: sanity in smoke config: 5 fixed trials (SANITY_N_TRIALS constant) — fastest proof-of-env path"
metrics:
  duration: "~3 minutes"
  completed: "2026-07-15"
  tasks: 2
  files: 4
---

# Phase 51 Plan 01: HoreKa Environment & Access Artifacts Summary

Four committed HoreKa environment artifacts: setup script with venv + mpi4py source build, smoke YAML config (tier:sanity/sobol), single-rank smoke SLURM job calling run_eval.py, and multi-rank target template for Phase 52.

## What Was Built

### Task 1 — Environment Setup Script + Smoke Config

**scripts/setup_horeka.sh** — One-time login-node setup script. Structure (top to bottom):
- `set -euo pipefail` immediately after header comment
- 4 `module load` lines each with `[VERIFY ON HOREKA]` inline comment
- `python3 -m venv ~/regvenv_horeka` + `source ~/regvenv_horeka/bin/activate`
- `pip install -e ".[propulate]"` (propulate>=1.0,<2 + mpi4py>=3.1)
- `pip install mpi4py --no-binary mpi4py --force-reinstall` (source build against loaded MPI)
- Echo instructions for rsync + absolute symlink data transfer (framed as operator instructions, not executable commands on the cluster)
- Verification one-liners: mpi4py rank print, torch.cuda.is_available()

**baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml** — Verbatim copy of `configs/selfcal/kobitski_ew06_alignment.yaml` with exactly four overrides:
- `tier: sanity` (was: dev) — 5 fixed trials via SANITY_N_TRIALS constant
- `n_trials: 2` (was: 50) — inert at tier=sanity, carried to signal smoke intent
- `search_strategy: sobol` (was: sobol, unchanged) — with deadlock warning comment added
- `output_dir: baseline_experiments/experiments/smoke/kobitski_ew06_alignment` (was: selfcal/)

### Task 2 — SLURM Job Scripts

**baseline_experiments/scripts/launch_horeka_smoke.sbatch** — Single-rank Phase 51 validation job:
- `--ntasks=1`, `--gres=gpu:1`, `--time=00:30:00`
- `[VERIFY ON HOREKA]` on partition, account, output path, module names, workspace cd path (9 occurrences)
- Calls `run_eval.py --config ... --mode full --output-dir ...` directly (not run_all.py — avoids Pitfall 6)
- Trailing operator validation checklist (4 post-job checks)

**baseline_experiments/scripts/launch_horeka.sbatch** — Multi-rank Phase 52+ target template:
- `--nodes=4`, `--ntasks-per-node=4`, `--gres=gpu:4`, `--time=04:00:00`
- Header: "DO NOT SUBMIT until run_all.py has been made rank-aware in Phase 52 (PARA-01/PARA-02)"
- `srun --mpi=pspmix --label python -u baseline_experiments/scripts/run_all.py --phase all --verbose`
- `[VERIFY ON HOREKA]` on all cluster-specific values (11 occurrences)

## Requirements Addressed

| Req ID | How Addressed |
|--------|---------------|
| ENV-01 | setup_horeka.sh: module load + venv + pip install -e "[propulate]" + mpi4py source rebuild + verification one-liners |
| ENV-02 | setup_horeka.sh: rsync + absolute symlink instructions; all configs need zero edits (data_path resolves via symlink) |
| ENV-03 | launch_horeka_smoke.sbatch: single-rank SLURM job; launch_horeka.sbatch: multi-rank template for Phase 52 |
| OUT-01 | launch_horeka.sbatch cd's to workspace checkout; run_all.py REPO_ROOT resolves output_dir relative to that path automatically |

## Deviations from Plan

None — plan executed exactly as written.

The plan note that `search_strategy: sobol` is "already sobol in source" was correct; the smoke config simply carries a safety comment explaining why it must not be changed to `auto`.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. All four files are committed shell/YAML artifacts with `<your-project>` and `<name>` placeholders (non-functional until operator fills them in — T-44-02 accepted in plan threat model).

## Known Stubs

The SLURM scripts contain intentional `[VERIFY ON HOREKA]` placeholder values (`<name>`, `<your-project>`, specific module names) that must be substituted before submitting. These are documented operational placeholders, not code stubs — they do not prevent the plan's goal from being achieved (the scripts are committed for operator use, not local execution).

## Self-Check

- [x] scripts/setup_horeka.sh created and committed (437763f)
- [x] baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml created and committed (437763f)
- [x] baseline_experiments/scripts/launch_horeka_smoke.sbatch created and committed (18aa1e3)
- [x] baseline_experiments/scripts/launch_horeka.sbatch created and committed (18aa1e3)
- [x] bash -n scripts/setup_horeka.sh exits 0
- [x] bash -n baseline_experiments/scripts/launch_horeka_smoke.sbatch exits 0
- [x] bash -n baseline_experiments/scripts/launch_horeka.sbatch exits 0
- [x] python yaml.safe_load on smoke config exits 0
- [x] grep "set -euo pipefail" scripts/setup_horeka.sh — PASS
- [x] grep "pip install mpi4py --no-binary mpi4py --force-reinstall" scripts/setup_horeka.sh — PASS
- [x] grep "^tier: sanity" smoke config — PASS
- [x] grep "^search_strategy: sobol" smoke config — PASS
- [x] grep "#SBATCH --ntasks=1" smoke.sbatch — PASS
- [x] grep "DO NOT SUBMIT" horeka.sbatch — PASS
- [x] grep -c "run_all.py" smoke.sbatch outputs 0 — PASS
- [x] Zero existing Python files modified

## Self-Check: PASSED
