# Phase 52: Multi-Rank Parallelism & Validation - Context

**Gathered:** 2026-07-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 52 delivers three concrete things before any full HoreKa allocation is committed:

1. **Rank-aware `run_all.py`** — PARA-02: the orchestration loop (file writes, `EvalConfig` loads, `EvaluationRunner` calls, per-run bookkeeping) executes exactly once per run regardless of MPI world size. `HyperparamOptimizer.run()` remains collective across all ranks. Eval-only phases (baseline_no_hpo, baseline_with_selfcal) are skipped entirely on non-rank-0.

2. **`baseline_experiments/configs_horeka/`** — PARA-03/BUDG-01: a complete mirror of all 7 run configs, independently runnable without touching `configs/`. The 5 HPO configs (selfcal x3, ground_truth x2) switch to `search_strategy: auto`; the 2 eval-only configs are identical to their `configs/` originals. Subsampling stays `max_points_per_frame: 1000` / `step: 8` (same as calibrated on laptop).

3. **Multi-rank test job** — BUDG-04: `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` (2 nodes × 2 ranks, `--time=00:30:00`) + `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml` (`search_strategy: auto`, `n_trials: 2`). Operator submits this short job to confirm correct multi-rank behaviour before the full allocation.

No changes to `eval/` Python source, `EvalConfig`, or any existing `configs/` files in this phase.

</domain>

<decisions>
## Implementation Decisions

### Configs Directory Switch (PARA-03)
- **D-01:** `run_all.py` gets a `--configs-dir` CLI arg added to its existing argparse. Default: `SUITE_ROOT / "configs"`. Cluster sbatch scripts pass `--configs-dir baseline_experiments/configs_horeka` explicitly.
- **D-02:** `--configs-dir` path must be resolved to absolute before the `os.chdir(REPO_ROOT)` call in `main()` — the chdir would break a relative path passed on the CLI. Resolve it immediately after `argparse.parse_args()`.
- **D-03:** All 7 run configs are mirrored in `configs_horeka/` (selfcal x3, ground_truth x2, baseline_no_hpo x1, baseline_with_selfcal x1). The `--configs-dir` flag replaces the whole `CONFIGS` root — no per-phase branching in `run_all.py`.

### Rank-Gating in `run_all.py` (PARA-02)
- **D-04:** MPI import at module top-level with graceful fallback: `try: from mpi4py import MPI; RANK = MPI.COMM_WORLD.Get_rank() except ImportError: RANK = 0`. Single-process / no-MPI runs are unchanged.
- **D-05:** `HyperparamOptimizer(config).run()` is a collective MPI operation — every rank must call it. This is the only call that must NOT be gated.
- **D-06:** All other orchestration (file writes, `_write_run_config`, mkdir, `_already_done` check, `_read_json`, `_selfcal_best_params`, `EvaluationRunner(...).run()`, logging the "done" line) gates behind `if RANK == 0:`.
- **D-07:** Eval-only phases (baseline_no_hpo, baseline_with_selfcal): the entire `run_eval_only` call (including `EvaluationRunner`) skips on non-rank-0 ranks — there is no collective operation inside these.

### Cluster Configs Content (PARA-03 / BUDG-01)
- **D-08:** The 5 HPO configs in `configs_horeka/` change only `search_strategy: sobol` → `search_strategy: auto`. Everything else (tier, n_trials, search_space, default_params, subsampling) is copied verbatim from `configs/`.
- **D-09:** `max_points_per_frame: 1000` and `step: 8` are preserved as-is in cluster configs (BUDG-01 — safe starting point, not full density).
- **D-10:** The 2 eval-only configs in `configs_horeka/` are exact copies of their `configs/` originals — no changes needed, included only for completeness of the `--configs-dir` replacement.

### BUDG-04 Validation Artifacts
- **D-11:** `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` — `--nodes=2`, `--ntasks-per-node=2` (4 ranks total, small enough to schedule quickly), `--time=00:30:00`. Uses `--configs-dir baseline_experiments/configs_horeka` and `--phase selfcal` to run just one optimize config (kobitski_ew06_alignment) at low n_trials.
- **D-12:** `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml` — copy of `configs/smoke/kobitski_ew06_alignment.yaml` with `search_strategy: auto` replacing `sobol`. `n_trials: 2` retained. This is the config the multi-rank test job targets.
- **D-13:** Passing criteria (for documentation in the sbatch / README): SLURM log shows 4 distinct rank IDs; `best_params.json` appears exactly once in the output directory (no duplicate writes); no Python tracebacks.

### Claude's Discretion
- Exact error message / log line when a non-rank-0 process skips a phase (e.g., `"[rank %d] skipping orchestration"`)
- Whether the RANK guard wraps the entire `run_phase()` call in `main()` or is pushed into each helper — either achieves D-06/D-07; inside `main()` is simpler
- `[VERIFY ON HOREKA]` placeholder count and wording in `launch_horeka_multirank_test.sbatch` (follow same convention as Phase 51 scripts)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### HoreKa Plan (primary source)
- `baseline_experiments/HOREKA_PLAN.md` §"Phase 1 — Multi-trial parallelism" — MUST READ: concrete rank-gating pattern for `run_all.py` (§1.3), cluster config strategy (§1.4), job script template (§1.5), and validation sequence (§1.7). This is the authoritative spec for PARA-01/02/03.

### Orchestrator to modify
- `baseline_experiments/scripts/run_all.py` — the file receiving D-01 (--configs-dir arg) and D-04–D-07 (RANK guard). Read in full before planning — understand `run_optimize_then_eval`, `run_eval_only`, `run_phase`, and `main()` signatures.

### Propulate backend (read-only — do not modify)
- `eval/search_strategies.py` lines 371–490 — `PropulateSearch.search()` already handles MPI internally (non-rank-0 returns empty list, rank-0 gathers). No changes to this file.
- `eval/runners/optimizer.py` lines 575–615 — `_detect_backend()` resolves `search_strategy: auto` → `propulate` when `SLURM_JOB_ID` is set or `world_size > 1`. Confirms why `auto` is correct for cluster configs.

### Existing scripts (adaptation bases)
- `baseline_experiments/scripts/launch_horeka.sbatch` — adaptation base for `launch_horeka_multirank_test.sbatch`; reduce nodes/ntasks, point at smoke config, set short wall-clock
- `baseline_experiments/configs/smoke/kobitski_ew06_alignment.yaml` — copy source for `configs_horeka/smoke/kobitski_ew06_alignment.yaml`; only `search_strategy` changes

### Requirements
- `.planning/REQUIREMENTS.md` §"Multi-Trial Parallelism" — PARA-01, PARA-02, PARA-03
- `.planning/REQUIREMENTS.md` §"Budget-Bounded Execution" — BUDG-01, BUDG-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `baseline_experiments/configs/selfcal/kobitski_ew06_alignment.yaml` — copy source for all 5 HPO cluster configs; only `search_strategy: sobol` → `search_strategy: auto` changes
- `baseline_experiments/scripts/launch_horeka.sbatch` (Phase 51) — direct adaptation base for the multi-rank test sbatch; reduce `--nodes=4 --ntasks-per-node=4` → `--nodes=2 --ntasks-per-node=2`, change `--time` to `00:30:00`, swap `--phase all` → `--phase selfcal`, add `--configs-dir`

### Established Patterns
- `[VERIFY ON HOREKA]` inline comment convention — already used in Phase 51 SLURM scripts (11 markers in `launch_horeka.sbatch`, 9 in `launch_horeka_smoke.sbatch`); multi-rank test sbatch must follow the same convention
- `os.chdir(REPO_ROOT)` at start of `main()` — `--configs-dir` path must be resolved to absolute *before* this call (D-02)
- `_already_done()` check + `--force` pattern in `run_all.py` — idempotent re-invocation; the RANK-0 guard preserves this (only rank-0 checks and writes)

### Integration Points
- `run_all.py:main()` argparse block — where `--configs-dir` arg is added; `CONFIGS` variable is reassigned from the arg value
- `run_all.py:run_phase()` / `run_optimize_then_eval()` / `run_eval_only()` — where RANK guards are inserted per D-06/D-07
- `baseline_experiments/scripts/launch_horeka.sbatch` (Phase 51 artifact) — the multi-rank test sbatch is a reduced-scope variant of this; it passes `--configs-dir baseline_experiments/configs_horeka`

</code_context>

<specifics>
## Specific Ideas

- `--configs-dir` default in argparse: `default=str(SUITE_ROOT / "configs")` — keeps backward compat with no-arg invocations
- Multi-rank test sbatch node count: 2 nodes × 2 ranks (4 total) — small enough to get scheduled in minutes on HoreKa dev/test queue; large enough to exercise multi-rank coordination
- Test config `n_trials: 2` with `search_strategy: auto` — 2 trials across 4 ranks is enough to observe rank-ID diversity in logs; propulate handles the collective internally
- Validation log check: `grep "rank" <slurm-output>` should show ranks 0, 1, 2, 3; `ls <output_dir>/best_params.json | wc -l` should be `1` (not 4)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 52-multi-rank-parallelism-validation*
*Context gathered: 2026-07-15*
