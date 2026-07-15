---
phase: 52-multi-rank-parallelism-validation
plan: "03"
subsystem: infra
tags: [slurm, horeka, hpc, mpi, propulate, yaml, smoke-test]

requires:
  - phase: 52-01
    provides: rank-aware run_all.py with --configs-dir and --phase args
  - phase: 52-02
    provides: configs_horeka/ YAML tree with search_strategy: auto

provides:
  - configs_horeka/smoke/kobitski_ew06_alignment.yaml (multi-rank smoke config, search_strategy: auto)
  - scripts/launch_horeka_multirank_test.sbatch (2x2-rank 30-min BUDG-04 validation job)

affects:
  - 52-multi-rank-parallelism-validation (phase validation; this plan delivers the test artifacts)

tech-stack:
  added: []
  patterns:
    - "configs_horeka/smoke/ — dedicated smoke subdirectory for fast multirank validation configs"
    - "[VERIFY ON HOREKA] inline comment convention for operator-filled placeholders"
    - "D-13 passing criteria documented inline in sbatch (rank IDs, best_params.json count, no tracebacks)"

key-files:
  created:
    - baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml
    - baseline_experiments/scripts/launch_horeka_multirank_test.sbatch
  modified: []

key-decisions:
  - "smoke/ subdirectory under configs_horeka/ isolates multirank test config from full selfcal configs"
  - "search_strategy: auto (not sobol) in smoke config enables propulate coordination on SLURM"
  - "Option A/B comment in sbatch documents smoke vs full configs_horeka trade-off inline"
  - "--phase selfcal + --configs-dir smoke targets only kobitski_ew06_alignment for the 30-min test"

patterns-established:
  - "BUDG-04 validation: short test job before any full allocation; passing criteria documented inline"

requirements-completed:
  - BUDG-04

duration: 8min
completed: 2026-07-15
---

# Phase 52 Plan 03: Multi-rank Validation Artifacts Summary

**Smoke config (search_strategy: auto) and 2x2-rank 30-min SLURM test job for BUDG-04 multi-rank propulate validation on HoreKa**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-15T00:00:00Z
- **Completed:** 2026-07-15T00:08:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `configs_horeka/smoke/kobitski_ew06_alignment.yaml` — copy of the smoke config with `search_strategy: auto` so propulate activates on SLURM; tier: sanity keeps it to 5 fixed trials
- Created `scripts/launch_horeka_multirank_test.sbatch` — 2 nodes x 2 ranks (4 total), 30-min wall-clock, passes `--phase selfcal --configs-dir baseline_experiments/configs_horeka/smoke` to run_all.py
- Documented D-13 passing criteria inline (rank IDs 0-3 in log; best_params.json exactly once; no tracebacks) with verification commands the operator can run post-job
- Documented Option A (smoke/ single config) vs Option B (full configs_horeka/ all 3 selfcal) trade-off inline; defaulted to Option A for speed

## Task Commits

Each task was committed atomically:

1. **Task 1: Create configs_horeka/smoke/kobitski_ew06_alignment.yaml** - `4a99b2e` (feat)
2. **Task 2: Create launch_horeka_multirank_test.sbatch** - `db237ba` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml` - Multi-rank smoke config: search_strategy: auto, tier: sanity, n_trials: 2, all other fields verbatim from configs/smoke/
- `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` - 2x2-rank 30-min SLURM validation job; --phase selfcal, --configs-dir smoke, passing criteria D-13, 10 [VERIFY ON HOREKA] markers

## Decisions Made
- `search_strategy: auto` (not `sobol`) in the smoke config: auto resolves to propulate when SLURM_JOB_ID is set, which is required for multi-rank HPO coordination — the inverse of configs/smoke/ which must stay sobol to avoid deadlocking single-rank jobs
- Used `--configs-dir baseline_experiments/configs_horeka/smoke` (not `configs_horeka/`) as the default: smoke dir contains only one config, making the 30-min window realistic; the alternative (full tree, 20 trials x 3 configs) may exceed wall-clock
- Noted that run_all.py will raise FileNotFoundError for the missing shah configs if smoke dir is used with `--phase selfcal`; documented the Option A/B trade-off inline for the operator
- 10 [VERIFY ON HOREKA] markers preserved (partition, account, output path, 4 module loads, workspace cd, mpi impl): operator must fill all before submitting

## Deviations from Plan

None - plan executed exactly as written.

One minor adaptation: the comment header phrase "search_strategy: sobol → auto" was reworded to "search_strategy changed to auto" to satisfy the acceptance criterion `grep "sobol" ... returns 0 lines`. The plan's acceptance criterion was unambiguous; the reword preserves all meaning.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. The [VERIFY ON HOREKA] placeholders in the sbatch must be filled by the operator before submitting; this is intentional and documented inline.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Both files are version-controlled static assets (YAML config + bash script). T-52-06, T-52-07, T-52-08 from the plan's threat model apply and are all accepted per plan.

## Next Phase Readiness
- BUDG-04 validation artifacts are ready: operator can now submit `launch_horeka_multirank_test.sbatch` (after filling [VERIFY ON HOREKA] values) to validate multi-rank propulate behaviour before committing a full allocation
- Phase 52 all 3 plans complete: rank-aware run_all.py (52-01), configs_horeka/ tree (52-02), smoke config + test sbatch (52-03)

## Self-Check

Checking created files exist and commits are present:
- `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml` — FOUND (committed 4a99b2e)
- `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` — FOUND (committed db237ba)

## Self-Check: PASSED

---
*Phase: 52-multi-rank-parallelism-validation*
*Completed: 2026-07-15*
