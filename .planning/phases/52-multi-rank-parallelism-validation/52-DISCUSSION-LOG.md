# Phase 52: Multi-Rank Parallelism & Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-15
**Phase:** 52-multi-rank-parallelism-validation
**Areas discussed:** Configs directory switch, Configs scope, BUDG-04 test job artifact

---

## Configs Directory Switch

| Option | Description | Selected |
|--------|-------------|----------|
| `--configs-dir` CLI arg | Add to existing argparse; sbatch passes it explicitly | ✓ |
| Env var (`HOREKA_CONFIGS_DIR`) | Set in sbatch header; run_all.py reads os.environ | |
| Auto-detect via `SLURM_JOB_ID` | Mirror `_detect_backend()` pattern; risk: can't use sobol on cluster | |

**User's choice:** `--configs-dir` CLI arg
**Notes:** None — decided on first pass.

---

## Configs Scope

| Option | Description | Selected |
|--------|-------------|----------|
| All 7 configs (complete mirror) | `--configs-dir` replaces the whole CONFIGS root; clean substitution in main() | ✓ |
| 5 HPO configs only | Eval-only phases still read from configs/; requires per-phase branching in run_all.py | |

**User's choice:** All 7 configs (complete mirror)
**Notes:** User requested elaboration on options before deciding. After clarification, chose the simpler code path (single CONFIGS root replacement).

---

## BUDG-04 Test Job Artifact

| Option | Description | Selected |
|--------|-------------|----------|
| New sbatch + test config | `launch_horeka_multirank_test.sbatch` (2×2 ranks) + `configs_horeka/smoke/kobitski_ew06_alignment.yaml` | ✓ |
| Test config only, reuse `launch_horeka.sbatch` | Fewer files; operator adjusts sbatch manually | |
| Validation checklist only | Lightest footprint; doesn't satisfy BUDG-04's "short test job" requirement | |

**User's choice:** New sbatch + test config
**Notes:** User requested elaboration before deciding.

---

## Claude's Discretion

- Exact wording/count of `[VERIFY ON HOREKA]` markers in `launch_horeka_multirank_test.sbatch`
- Whether RANK guard wraps `run_phase()` in `main()` or is distributed into each helper function
- Log line format for non-rank-0 skip messages

## Deferred Ideas

None — discussion stayed within phase scope.
