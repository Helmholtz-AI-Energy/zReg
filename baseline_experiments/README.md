# baseline_experiments

Self-contained experiment suite layered on top of the zReg evaluation
framework (`eval/`, `run_eval.py`). Lives at the repo root, next to
`.planning/`, deliberately separate from `configs/` and `experiments/`
(which hold ad-hoc/dev scenario configs and their run outputs) so this
suite's configs, orchestration, and results can be reasoned about as one
unit.

## Layout

```
baseline_experiments/
  configs/
    selfcal/                 3 configs — self-registration HPO calibration
    baseline_no_hpo/         4 configs on disk, only ew06_vs_shah run by default — see
                              "Runtime" below for why the other 3 are scoped out
    ground_truth/            2 configs — self-registration HPO, random transform
    baseline_with_selfcal/   4 configs on disk, only ew06_vs_shah run by default
  scripts/
    run_all.py               orchestrator — imports EvalConfig/HyperparamOptimizer/
                              EvaluationRunner directly (not via subprocess/run_eval.py,
                              see "Why not run_eval.py" below)
    merge_params.py          merges the 3 selfcal best_params.json into 1 dict
    aggregate_results.py     walks experiments/, builds summary.csv + summary.md
    aggregate_cost.py        walks experiments/, builds summary/compute_cost.md
  experiments/                output root (gitignored — large per-run artifacts)
    selfcal/<name>/
    baseline_no_hpo/<name>/
    ground_truth/<name>/
    baseline_with_selfcal/<name>/
    summary/summary.{csv,md}
    summary/compute_cost.md
```

## What each phase does

Real data available: 4 Kobitski embryo trajectories (`ew_06`, `ew_08`,
`ew_11`, `ew_12`; tracklets format) and 1 Shah sample (`sample-1`; CSV).
Kobitski ew_06 averages ~16.6k points/frame over 370 frames; Shah sample-1
averages ~4.1k points/frame over 420 frames — Kobitski is the expensive one.

1. **selfcal** — self-registration hyperparameter calibration. Each
   trajectory is registered against a *fixed*, known rigid perturbation of
   itself (`transform_spec` in the config), so the metrics measure recovery
   error directly against ground truth. Three runs:
   - `kobitski_ew06_alignment` — alignment stage only
   - `shah_alignment` — alignment stage only
   - `shah_label_transfer` — alignment + label transfer (Shah CSV carries a
     per-cell `id` column, giving LabelTransferStage real labels; Kobitski
     selfcal never calibrates label-transfer params)

   Mode: `optimize` then `eval` (HyperparamOptimizer, then EvaluationRunner
   with the merged best_params). Writes `best_params.json`,
   `search_history.json`, and `eval_report.json`.

2. **baseline_no_hpo** — full pipeline (alignment + label transfer), a
   Kobitski embryo (source) registered against Shah sample-1 (target).
   `run_all.py` only runs `ew06_vs_shah` by default (see "Runtime" below);
   configs for `ew08_vs_shah`, `ew11_vs_shah`, `ew12_vs_shah` still exist
   under `configs/baseline_no_hpo/` and can be run manually (see
   "Running"). Uses each config's `default_params` directly — no optimizer
   call, no best_params.json anywhere in the run. This is the "no HPO, no
   selfcal knowledge" reference point.

3. **ground_truth** — same self-registration structure as selfcal (source
   trajectory vs. a synthetically transformed copy of itself, with HPO), but
   the perturbation is a **random** rigid transform instead of the fixed one
   used for calibration, so the search is validated against a
   non-cherry-picked perturbation:
   - `kobitski_ew06`: alignment only, same as its selfcal counterpart —
     Kobitski has no usable label for label transfer (`label` is
     continuous RGB-like noise, `id` is a unique per-point tracking
     identity — 6930/6930 unique in frame 0 of ew_06 — verified directly
     against the real tracklets file, not a semantic class). Transform:
     `rotation_deg=11.31, axis=[-0.826,-0.186,-0.785], sigma=0.1111` —
     generated once via `random.Random(123)`.
   - `shah_sample1`: both stages — Shah's `label` field carries real
     small-integer classes, so it's the one config in this suite that
     exercises label transfer against a genuine ground truth. Transform:
     `rotation_deg=28.71, axis=[0.933,-0.13,0.589], sigma=0.0883` —
     generated once via `random.Random(456)`.

   These are baked as static values into the YAML (not regenerated at
   runtime) for reproducibility — see the generation snippet at the top of
   each `configs/ground_truth/*.yaml`.

4. **baseline_with_selfcal** — identical pipeline to `baseline_no_hpo`
   (same `ew06_vs_shah` pair by default), but `run_all.py` merges the three
   selfcal `best_params.json` files (see "Param merge rule" below) and
   passes the merged dict directly to `EvaluationRunner`, instead of using
   config defaults. Lets you compare "no calibration" vs.
   "selfcal-calibrated" on the same pair.

Note on scope: the original ask mentioned "five different trajectory
pairs" for the baseline runs; per follow-up clarification, only Kobitski
embryos vs. Shah are in scope (4 possible pairs, since Shah has just the
one sample) — and after real-data runtime testing (see "Runtime" below),
scoped down further to **1** pair (`ew06_vs_shah`) run by default. The
other 3 pair configs remain on disk for manual runs.

## Param merge rule (`merge_params.py`)

Every `baseline_with_selfcal` pair uses a Kobitski embryo as source and
Shah as target, so its alignment params should reflect calibration from
*both* real datasources, and its label-transfer params from Shah's
calibration (the only one that has any).

- Key in both dicts being merged, both numeric → arithmetic mean (rounded
  to int if both inputs were int).
- Key in both, non-numeric, equal → keep it.
- Key in both, non-numeric, conflicting → fall back to the pair config's
  `default_params[key]` (logged as a warning) — neither calibration run is
  arbitrarily preferred for categorical choices like `dtw_dist_fn`.
- Key in only one dict → passthrough, nothing to merge against.
- `defaults` is only consulted for conflict resolution and to fill any key
  still missing after combining all three real sources — never unioned
  into intermediate merge steps (a bug caught and fixed via unit test: an
  earlier version leaked defaults into the first merge stage, which then
  made Shah's real calibrated label-transfer values look like "conflicts"
  against those defaults and silently discarded them).

Order: `merge_two(kobitski_alignment, shah_alignment)` → average alignment
params from both datasources, then `merge_two(that, shah_label_transfer)` →
folds in Shah's label-transfer params (passthrough, nothing to average
against) and lets its re-calibrated alignment values contribute to the
average as a third source.

## Cost tracking (`aggregate_cost.py`)

Produces `experiments/summary/compute_cost.md` — per-run wall-clock
duration, status (pending / in progress / done), trial count and average
time/trial for optimize runs, plus dataset scale (frames x mean
points/frame — the dominant cost driver per the smoke-test finding above).

Since `run_all.py` runs every experiment in one long-lived process with no
per-run subprocess boundary, there's no per-run CPU-seconds counter to
read. Timing is instead derived from filesystem mtimes:
`run_config.yaml` (written at the start of each run) ->
`best_params.json` (written when the optimizer finishes, optimize-mode
runs only) -> `eval_report.json` (written at the very end). This works
retroactively on runs already in progress or finished, with no
instrumentation needed in the running process, and safely reports partial
elapsed time (`"in progress"`) for whatever run is currently active.

## Why not shell out to run_eval.py

`run_eval.py` appends a `datetime.now()` timestamp subdirectory to
`output_dir` on every invocation — convenient for ad-hoc runs, but it makes
it impossible for `baseline_with_selfcal` to know in advance which path
`selfcal`'s `best_params.json` will land in. `run_all.py` instead imports
`EvalConfig` / `HyperparamOptimizer` / `EvaluationRunner` directly (same
imports `run_eval.py` uses internally) and controls `output_dir` itself —
one fixed, predictable path per run, matching the layout above.

## Runtime — read before launching

A single alignment pass (no HPO, no label transfer) on the full-density
Kobitski ew_06 trajectory did not finish in 32 minutes during smoke
testing. Every optimize-mode config here is set to `tier: dev` (not
`full`) because of this: `full` would add another 50 real-data trials on
top of dev's 20, and full-tier was ruled out as impractical on this
machine.

**A first attempt at the full suite (all 5 optimize runs + 4 pairs each
for baseline_no_hpo/baseline_with_selfcal) ran for 25 hours and never
finished its first optimize run.** Per-iteration cost climbed from ~1.3
min to 77 min over 140 iterations, and system swap filled to 95% — see
investigation below for what that was and wasn't caused by.

**Confirmed root causes (investigated by running real Kobitski data
directly against `zreg.pairwise_distance_matrix.create_pairwise_distance_matrix`,
outside the full pipeline, to isolate cause from symptom):**

1. **`cpd_penalty: "nonrigid"` is broken, unrelated to the slowdown.**
   `NonRigidCPD.registration()` never sets `self.transformation` (a known
   quirk — `eval/stages/alignment.py` already has a documented workaround
   for this exact issue, "CR-01: NonRigidCPD does not set
   self.transformation... use registration() return value instead" — but
   `create_pairwise_distance_matrix` never applies it). Every trial that
   samples `cpd_penalty="nonrigid"` crashes with `AttributeError` within
   seconds, every time, reproduced directly against real Kobitski data.
   This means **nonrigid could never have been what ran for 25 hours** —
   it would've failed in seconds, not hours. `"nonrigid"` has been removed
   from every `search_space.cpd_penalty` list in this suite's configs so
   HPO trials don't waste budget on a guaranteed-0.0-score option.
2. `create_pairwise_distance_matrix`'s `stored_transforms` dict retains a
   full CPD transform per `(i, j)` pair for an entire windowed sweep,
   unbounded, until the function returns. For nonrigid CPD specifically
   this would have been catastrophic (`NonRigidTransformation.g` is a
   dense `(n_points, n_points)` kernel matrix — ~1.1-1.9 GB per transform
   at Kobitski's point density). **Fixed** in
   `src/zreg/pairwise_distance_matrix.py` (skip storing nonrigid
   transforms; the existing, tested fallback path in
   `_build_aligned_cloud` — D-10 — recomputes fresh CPD for the bounded
   number of frames that actually need it). This fix is real and worth
   keeping, but per (1) it **could not have been the cause of the observed
   25-hour slowdown**, since nonrigid never successfully ran in the first
   place.
3. **The actual cause of the 25-hour slowdown remains unconfirmed.**
   Direct testing of `cpd_type="rigid"` on real Kobitski data (25 frames,
   ~209 pairs, same ~7,000-9,500 points/frame scale, 26 minutes of
   continuous execution) showed **no memory growth** — RSS fluctuated
   between 390 MB and 2.3 GB with no runaway trend. The most likely
   remaining explanation is that CPD registration (rigid and affine
   included, not just nonrigid) has an inherent O(n²)-per-EM-iteration
   cost, and at Kobitski's real point density (up to ~22,000 points/frame)
   combined with `window_size=20` (41 pairs/row x 370 rows), that's simply
   very expensive by nature — compounded by real memory pressure building
   up over many hours that a 26-minute test isn't long enough to
   reproduce. This was not root-caused further; **treat multi-hour or
   multi-day single-run times as a real possibility**, not a bug you
   should expect me to have silently fixed.

**Scope reduction in response:** `baseline_no_hpo` and
`baseline_with_selfcal` are scoped down to a single pair
(`ew06_vs_shah`) instead of 4, cutting the suite from 13 runs to 7. This
does not reduce the cost of any individual run — it only reduces how many
of them execute.

**Environment fix:** the machine has 16 GB RAM total and was seeing
multi-hour stalls between individual CPD iterations (not just slow compute
— the whole process periodically froze) caused by system-wide memory
pressure from other running apps (multiple concurrent Claude Code CLI
sessions, Electron apps, browser tabs, etc.). Closing those apps and
launching under `caffeinate -s` (prevents the machine from sleeping
mid-run — see "Running" below) cut per-iteration time from 1-3 *hours*
down to ~9 *minutes* for a CPD-heavy trial. This fixed the stalling but
not the underlying per-trial cost.

**Temporal downsampling (the actual per-trial cost fix):** `step`
(`eval/stages/alignment.py:252,255`) applies a stride to the source/target
frame sequence *before* the pairwise distance matrix is built, directly
cutting the number of CPD calls a trial makes — unlike `max_points_per_frame`,
which shrinks the cost of each call but not how many there are. It was
being searched over `[1, 2]` (a coin-flip between full resolution and a 2x
cut), so half of all trials got no benefit.

History: before Phase 63, `HyperparamOptimizer` ignored
`config.default_params` and merged its own hardcoded defaults (`step: 1`)
into every trial, so `default_params.step: 8` alone did not take effect
during HPO (a wasted 15.5-hour run). The workaround was `step: [8]` as a
single-value list in `search_space` of all 5 optimize configs. Every trial
now uses ~46 of Kobitski's 370 frames / ~53 of Shah's 420.

**PROVISIONAL (pending user confirmation) — warm start and `default_params` (Phase 63 HPC-01, D-09):**

- Since Phase 63 the optimizer uses each config's `default_params` for every
  key it does not search (builtin fallbacks only fill keys the config
  omits). This changes HPO behaviour and results for about 19 configs that
  fix keys such as `step`, `cpd_penalty` or `window_size` in
  `default_params`; HPO results produced before Phase 63 are not directly
  comparable with new ones. The single-value `step: [8]` search-space
  entries are now redundant but harmless.
- `baseline_with_combined` evaluates the merged selfcal/ground_truth
  calibration as its first HPO trial (warm start), in addition to using it
  as `default_params`. The merged params and every upstream
  `best_params.json` are validated through `EvalConfig.model_validate`
  before any trial runs.
- Under Propulate (`search_strategy: propulate`, or `auto` on HoreKa) the
  warm-start seeds of each tier are evaluated on rank 0 as ordinary trials
  before the Propulate search starts, and one WARNING per tier says that
  Propulate's population itself is not seeded. The seed's score competes
  for `best_params.json`, but the evolutionary search does not start from it.
- `ZREG_CLEAR_CHECKPOINTS=1` / `--clear-checkpoints` only removes Propulate
  checkpoint files; it does not delete best_params.json. After a
  search-space change such as Phase 63 D-08 (`cosine` removed from
  `dtw_dist_fn`), re-run the `selfcal` and `ground_truth` phases with
  `--force`: `baseline_with_combined` rejects a stale `best_params.json`
  with an error that names the artifact.

The `step` fix alone was still not enough: closing apps + `caffeinate` helped for
a while, but system swap crept back up (7.2GB -> 9.2GB -> 11.3GB total
over one day) and per-row iteration cost degraded again (~5min/row ->
~30min/row, with a 94-minute single-row spike), even for `rigid`/`affine`
CPD at the reduced (step=8) frame count. A single dev-tier trial that
should have taken ~2.7 min by formula took 9h12m to reach 53% completion
before being killed. This also invalidated an earlier claim in this file
("a 26-minute rigid-CPD test showed no memory leak") — that test covered
only 209 comparisons over 26 minutes; the real degradation only showed up
at a scale and duration that test never reached. Point density (points/frame),
not just frame count, was still the full ~16,572 (Kobitski) / ~4,113 (Shah)
throughout all of this — `step` alone was never going to be enough.

### Spatial subsampling (`max_points_per_frame`) — what actually fit the suite into hours, not weeks

Calibrated empirically, not estimated, after two earlier estimates (6.5-9
days, then revised to 2-4+ weeks mid-run) both turned out wrong once real
degradation showed up. Measured directly against real Kobitski ew_06 data
by calling `create_pairwise_distance_matrix` outside the optimizer (no
memory-pressure confound from a long-running process):

| max_points_per_frame | worst case (window=20, rigid CPD) |
| --- | --- |
| 400 | 71.1s |
| 1000 | 268.0s (4.47 min) |
| 1500 | 488.6s (8.14 min) |

At `max_points_per_frame=1000, step=8` (47 strided frames), window=5/10/20
rigid-CPD trials measured 130.5s / 176.5s / 278.6s; a no-CPD (`cpd_penalty:
null`) trial measured ~4s regardless of window. This calibration is now
baked into `aggregate_cost.py` (`CPD_TRIAL_SECONDS`, `NO_CPD_TRIAL_SECONDS`)
and used to project total suite runtime in `compute_cost.md`'s "Estimated
total runtime" section — **projected grand total: ~4h12m (best case) to
~4h16m (worst case)**, landing in the 3-5h target. If `max_points_per_frame`
or `step` are changed, that calibration must be re-measured or the
projection will silently drift from reality.

Applied to **all 7 configs** now (not just the 5 optimize ones):
`max_points_per_frame: 1000` added to every config, `default_params.step`
changed from `1` to `8` in `baseline_no_hpo`/`baseline_with_selfcal`
(reversing the earlier "leave eval-only runs at full resolution" decision
— that only made sense under a per-run cost budget, not a whole-suite
3-5h budget), and `default_params.window_size` reduced from `60` to `10`
in the same two configs (at 47 strided frames, `window=60` would have
degenerated to an unwindowed full sweep — `window` only meaningfully
truncates the pairwise matrix when it's smaller than the frame count).

Run it detached from any session that might be interrupted (`nohup`,
`tmux`, `screen`, or as a backgrounded process you don't depend on staying
attached to), and under `caffeinate -s` so the machine doesn't sleep
mid-run.

## Running

```bash
# from the repo root
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS libomp workaround, see project-synthetic-datasets memory

# see the execution plan without running anything
python baseline_experiments/scripts/run_all.py --phase all --dry-run

# run everything, in order (selfcal -> baseline_no_hpo -> ground_truth -> baseline_with_selfcal)
python baseline_experiments/scripts/run_all.py --phase all

# or one phase at a time
python baseline_experiments/scripts/run_all.py --phase selfcal
python baseline_experiments/scripts/run_all.py --phase baseline_no_hpo
python baseline_experiments/scripts/run_all.py --phase ground_truth
python baseline_experiments/scripts/run_all.py --phase baseline_with_selfcal   # needs selfcal done first

# re-run something that already completed (idempotent by default — skips
# any run whose output_dir already has eval_report.json)
python baseline_experiments/scripts/run_all.py --phase selfcal --force

# aggregate whatever has completed so far into a summary table
python baseline_experiments/scripts/aggregate_results.py

# aggregate time & compute cost per run (safe to run anytime, including
# while the suite is still in progress — picks up partial elapsed time
# for whichever run is currently active)
python baseline_experiments/scripts/aggregate_cost.py
```

`run_all.py` is idempotent per-run (checks for `eval_report.json` in each
run's `output_dir`), so it's safe to re-invoke after a crash, interruption,
or partial run — already-completed runs are skipped unless `--force` is
passed.

## Data dependency

This worktree does not have `data/external/` checked out (it's gitignored
and only present in the main zReg checkout). A local symlink was added at
`data/external/sample -> <main-checkout>/data/external/sample` so this
suite can run from within the worktree; it is not tracked by git (the
`data/external/.gitignore` ignores everything except `.gitignore` itself).
If working from a different checkout, recreate that symlink or copy the
real data into place first.
