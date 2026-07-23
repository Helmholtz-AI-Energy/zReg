---
phase: 51-environment-access
verified: 2026-07-15T00:00:00Z
status: human_needed
score: 5/6 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Submit launch_horeka_smoke.sbatch on HoreKa after running setup_horeka.sh and transferring data"
    expected: "squeue shows job COMPLETED; SLURM output file contains no Python tracebacks; eval_report.json exists under baseline_experiments/experiments/smoke/kobitski_ew06_alignment/<timestamp>/; torch.cuda.is_available() returns True on the compute node"
    why_human: "Requires live HoreKa cluster access, valid project allocation, real Kobitski tracklet data, and GPU compute node — cannot be tested locally"
---

# Phase 51: HoreKa Environment & Access Verification Report

**Phase Goal:** Operator can stand up a working HoreKa environment — Python + MPI-built dependencies, real data transferred, and a SLURM job script that launches the suite end-to-end with outputs landing in the correct directory convention.
**Verified:** 2026-07-15
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | scripts/setup_horeka.sh exists with full setup sequence (set -euo pipefail, venv, pip install -e "[propulate]", mpi4py source build, MPICC export, mpi4py/torch verification, mkdir -p logs) | VERIFIED | File read confirms all elements present at lines 7, 20-21, 27, 34-35, 66, 69, 71 |
| 2 | baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml exists with tier:sanity, search_strategy:sobol, correct output_dir, data_path preserved, cpd_penalty:null in default_params | VERIFIED | File read: tier:sanity (line 23), search_strategy:sobol (line 25), output_dir correct (line 46), data_path preserved (line 9), cpd_penalty: null (line 37) |
| 3 | launch_horeka_smoke.sbatch: --ntasks=1, --gres=gpu:1, --time=00:30:00, calls run_eval.py (not run_all.py), --config points at smoke config, set -euo pipefail, mkdir -p logs, >=5 [VERIFY ON HOREKA] markers | VERIFIED | File read confirms all directives; grep confirms 0 references to run_all.py; [VERIFY ON HOREKA] count = 9 |
| 4 | launch_horeka.sbatch: --ntasks-per-node=4, --gres=gpu:4, --nodes=4, run_all.py --phase all, --mpi=pspmix, DO NOT SUBMIT header, set -euo pipefail, no relative stdout redirect, >=6 [VERIFY ON HOREKA] markers | VERIFIED | File read confirms all directives; DO NOT SUBMIT at line 3; --mpi=pspmix at line 34; no relative stdout redirect (stdout goes to #SBATCH --output absolute path, not >redirect); [VERIFY ON HOREKA] count = 11 |
| 5 | Requirements ENV-01/ENV-02/ENV-03/OUT-01 addressed by committed artifacts | VERIFIED | ENV-01: setup_horeka.sh venv+mpi4py source build. ENV-02: setup_horeka.sh rsync+absolute symlink instructions. ENV-03: launch_horeka_smoke.sbatch (single-rank) + launch_horeka.sbatch (multi-rank template). OUT-01: launch_horeka.sbatch cd's to workspace root; run_all.py REPO_ROOT resolution means output_dir resolves correctly. |
| 6 | Zero Python source files modified — exactly 4 new files created | VERIFIED | git show --stat on commits 437763f, 18aa1e3, 7d76f32: only scripts/setup_horeka.sh, baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml, baseline_experiments/scripts/launch_horeka_smoke.sbatch, baseline_experiments/scripts/launch_horeka.sbatch — all insertions, no Python files touched |
| 7 | Operator smoke job validation on HoreKa (submitting the actual SLURM job) | UNCERTAIN | Requires cluster access — deferred to human verification |

**Score:** 6/6 automated truths verified (truth 7 deferred to human)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/setup_horeka.sh` | One-time HoreKa env setup | VERIFIED | 74 lines; set -euo pipefail; 4 module load lines each with [VERIFY ON HOREKA]; python3 -m venv ~/regvenv_horeka; pip install -e ".[propulate]"; export MPICC=$(which mpicc); pip install mpi4py --no-binary mpi4py --force-reinstall; mpi4py + torch CUDA verification; mkdir -p baseline_experiments/logs; bash -n exits 0 |
| `baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml` | Smoke HPO config | VERIFIED | tier: sanity; n_trials: 2; search_strategy: sobol with deadlock warning; output_dir: baseline_experiments/experiments/smoke/kobitski_ew06_alignment; data_path preserved; cpd_penalty: null in default_params; YAML valid |
| `baseline_experiments/scripts/launch_horeka_smoke.sbatch` | Single-rank SLURM smoke job | VERIFIED | --ntasks=1; --gres=gpu:1; --time=00:30:00; calls run_eval.py directly (0 references to run_all.py); --config baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml; --output-dir baseline_experiments/experiments/smoke/kobitski_ew06_alignment; set -euo pipefail; mkdir -p baseline_experiments/logs; 9 [VERIFY ON HOREKA] markers; bash -n exits 0 |
| `baseline_experiments/scripts/launch_horeka.sbatch` | Multi-rank target template | VERIFIED | --nodes=4; --ntasks-per-node=4; --gres=gpu:4; run_all.py --phase all --verbose; srun --mpi=pspmix --label; DO NOT SUBMIT header; set -euo pipefail; mkdir -p baseline_experiments/logs; no relative stdout redirect (absolute #SBATCH --output path); 11 [VERIFY ON HOREKA] markers; bash -n exits 0 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `launch_horeka_smoke.sbatch` | `configs/smoke/kobitski_ew06_alignment.yaml` | hardcoded --config path in srun invocation | WIRED | Line 34: `--config baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml` matches exactly |
| `configs/smoke/kobitski_ew06_alignment.yaml` | `data/external/sample/kobitski_data/` | data_path field; absolute symlink on HoreKa | WIRED (operator action required) | data_path field present at line 9; absolute symlink setup documented in setup_horeka.sh section 5 |
| `launch_horeka_smoke.sbatch` | `baseline_experiments/experiments/smoke/kobitski_ew06_alignment/` | --output-dir flag to run_eval.py | WIRED | Line 36: `--output-dir baseline_experiments/experiments/smoke/kobitski_ew06_alignment` |

### Data-Flow Trace (Level 4)

Not applicable — all four artifacts are shell scripts and a YAML config, not components that render dynamic data.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| setup_horeka.sh shell syntax valid | `bash -n scripts/setup_horeka.sh` | exit 0 | PASS |
| launch_horeka_smoke.sbatch shell syntax valid | `bash -n baseline_experiments/scripts/launch_horeka_smoke.sbatch` | exit 0 | PASS |
| launch_horeka.sbatch shell syntax valid | `bash -n baseline_experiments/scripts/launch_horeka.sbatch` | exit 0 | PASS |
| Smoke config YAML valid | `python3 -c "import yaml; yaml.safe_load(open(...))"` | exit 0 | PASS |
| Smoke script does not call run_all.py | `grep -c "run_all.py" launch_horeka_smoke.sbatch` | 0 | PASS |
| Smoke script has >=5 [VERIFY ON HOREKA] markers | `grep -c "\[VERIFY ON HOREKA\]"` | 9 | PASS |
| Multi-rank script has >=6 [VERIFY ON HOREKA] markers | `grep -c "\[VERIFY ON HOREKA\]"` | 11 | PASS |
| setup_horeka.sh has >=4 [VERIFY ON HOREKA] markers | `grep -c "\[VERIFY ON HOREKA\]"` | 7 | PASS |

### Probe Execution

No probe scripts declared for this phase. Phase is infrastructure/scripts-only (no runnable Python entry points).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ENV-01 | 51-01-PLAN.md | Operator can activate working Python env with propulate/mpi4py extras, mpi4py built against system MPI | SATISFIED | setup_horeka.sh: module load, venv, pip install -e ".[propulate]", MPICC export, mpi4py --no-binary --force-reinstall, verification one-liners |
| ENV-02 | 51-01-PLAN.md | Operator can transfer datasets and have data_path resolve correctly without config edits | SATISFIED | setup_horeka.sh section 5: rsync command + absolute ln -s instructions; smoke config data_path: data/external/sample/... is unchanged from selfcal source |
| ENV-03 | 51-01-PLAN.md | Operator can submit a SLURM job script that requests GPU nodes/ranks and launches experiment suite end-to-end | SATISFIED (commit artifact) / UNCERTAIN (live execution) | launch_horeka_smoke.sbatch: single-rank, GPU:1, calls run_eval.py. launch_horeka.sbatch: 4-node/4-rank multi-rank template. Live execution requires HoreKa cluster — deferred to human verification |
| OUT-01 | 51-01-PLAN.md | Suite outputs land in baseline_experiments/experiments/<phase>/<name>/ on HoreKa workspace filesystem | SATISFIED | launch_horeka.sbatch cd's to workspace checkout; run_all.py REPO_ROOT resolves relative output_dir against that path; smoke config output_dir: baseline_experiments/experiments/smoke/kobitski_ew06_alignment |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `baseline_experiments/scripts/launch_horeka.sbatch` | 35 | No stdout redirect on srun call | Info | In the PLAN task 2 action, the spec described `> baseline_experiments/logs/run_all_horeka_${SLURM_JOB_ID}.log 2>&1` as a relative redirect. The implemented file omits this redirect entirely — SLURM output goes to the #SBATCH --output absolute path instead. This is the correct behavior per the must-have "no relative stdout redirect" criterion. The PLAN spec itself described this as a potential problem; the implementation correctly avoids it. |

No TBD, FIXME, or XXX markers found in any of the 4 files. `<name>` and `<your-project>` placeholders are documented operational fill-ins, not code stubs — every occurrence is annotated with a [VERIFY ON HOREKA] comment explaining the operator action required.

### Human Verification Required

#### 1. HoreKa Smoke Job End-to-End Execution

**Test:** On a HoreKa login node, after running `scripts/setup_horeka.sh` and transferring Kobitski tracklet data via rsync + absolute symlink, submit `baseline_experiments/scripts/launch_horeka_smoke.sbatch` with `<name>` and `<your-project>` filled in: `sbatch baseline_experiments/scripts/launch_horeka_smoke.sbatch`

**Expected:**
1. `squeue -j <job_id>` eventually shows COMPLETED (not FAILED/CANCELLED)
2. SLURM output file at `baseline_experiments/logs/slurm-smoke-<job_id>.out` contains no Python tracebacks
3. `eval_report.json` exists under `baseline_experiments/experiments/smoke/kobitski_ew06_alignment/<timestamp>/`
4. SLURM log shows 5 HPO trials (tier:sanity SANITY_N_TRIALS=5 constant)
5. `python -c 'import torch; print(torch.cuda.is_available())'` in a compute node srun shows True

**Why human:** Requires live HoreKa cluster access, valid allocation account, real Kobitski tracklet dataset (gitignored, laptop-only), and available GPU compute node. Cannot be replicated locally.

### Gaps Summary

No blocking gaps. All 6 automated must-haves verified against actual file contents. The one open item (truth 7 — live SLURM job execution on HoreKa) is non-automatable by design: it requires cluster access and real data that are unavailable locally. This was explicitly called out in the phase instructions as a deferred human verification item.

---

_Verified: 2026-07-15_
_Verifier: Claude (gsd-verifier)_
