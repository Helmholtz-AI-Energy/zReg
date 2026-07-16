"""Aggregate wall-clock time and compute-scale cost per experiment.

`run_all.py` runs everything in a single long-lived process (no per-run
subprocess boundary — see README.md "Why not run_eval.py"), so there is no
per-run CPU-seconds counter to read. Timing here is instead derived
retroactively from filesystem mtimes, which works for runs that are already
in progress or finished without needing any instrumentation added to the
running process:

- start_time  = mtime of run_config.yaml (written first, before the
  optimizer/runner call — see run_all.py _write_run_config)
- optimize_done_time = mtime of best_params.json (optimize-mode runs only;
  written at the very end of HyperparamOptimizer.run(), before
  EvaluationRunner starts)
- end_time = mtime of eval_report.json (written last, at the very end of
  EvaluationRunner.run())

This gives an exact wall-clock split between the HPO search and the final
eval pass for optimize-mode runs, and an exact total for eval-only runs.
Runs that have started (run_config.yaml exists) but not finished
(eval_report.json missing) are reported as "in progress" using the current
time as a running total. Runs that haven't started yet are "pending".

"Compute" beyond wall-clock is approximated by dataset scale (frames x
mean points/frame — the dominant cost driver observed during smoke testing,
see README.md) and, for optimize runs, the number of trials actually
executed (from search_history.json) with an average time/trial derived
from the optimize-phase wall-clock.

Estimated total runtime
------------------------
In addition to observed per-run timing, this script projects a total suite
runtime using a calibration measured directly against real Kobitski ew_06
data (2026-07-15): with `max_points_per_frame=1000, step=8` (47 strided
frames), a single rigid-CPD alignment trial (`create_pairwise_distance_matrix`
called directly, bypassing the optimizer) took 130.5s/176.5s/278.6s at
window_size=5/10/20 respectively; a no-CPD (`cpd_penalty: null`) trial took
~4s regardless of window. These are read from each config's own
`search_space`/`default_params` (not hardcoded per-run), so the estimate
adapts if window_size/cpd_penalty choices change, but will silently drift
if `max_points_per_frame` or `step` are changed without re-measuring
CALIBRATION below. For runs with real observed data (done or in-progress),
the actual measurement is used instead of the projection.

Usage
-----
    python baseline_experiments/scripts/aggregate_cost.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml

SUITE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = SUITE_ROOT / "experiments"
CONFIGS_ROOT = SUITE_ROOT / "configs"

# (phase, name) in the same fixed order as run_all.py's PHASES/PHASE_ORDER.
# Scoped to a single pair (ew06_vs_shah) for baseline_no_hpo/baseline_with_selfcal
# — see run_all.py's BASELINE_NO_HPO/BASELINE_WITH_SELFCAL comment for why.
RUNS = [
    ("selfcal", "kobitski_ew06_alignment"),
    ("selfcal", "shah_alignment"),
    ("selfcal", "shah_label_transfer"),
    ("baseline_no_hpo", "ew06_vs_shah"),
    ("ground_truth", "kobitski_ew06"),
    ("ground_truth", "shah_sample1"),
    ("baseline_with_selfcal", "ew06_vs_shah"),
]

# Optimize-mode runs (mode="full": HyperparamOptimizer then EvaluationRunner).
# All others are eval-only (single EvaluationRunner pass, config default_params
# or selfcal-merged params, no search).
OPTIMIZE_RUNS = {
    ("selfcal", "kobitski_ew06_alignment"),
    ("selfcal", "shah_alignment"),
    ("selfcal", "shah_label_transfer"),
    ("ground_truth", "kobitski_ew06"),
    ("ground_truth", "shah_sample1"),
}

# Static dataset scale — the dominant cost driver (see README.md smoke-test
# finding: Kobitski's ~4x point density vs. Shah is why it's the slow one).
# Measured directly via zreg.dataset loaders, not estimates.
DATASET_STATS = {
    ("selfcal", "kobitski_ew06_alignment"): ("kobitski_ew06", 370, 16572),
    ("selfcal", "shah_alignment"): ("shah_sample1", 420, 4113),
    ("selfcal", "shah_label_transfer"): ("shah_sample1", 420, 4113),
    ("baseline_no_hpo", "ew06_vs_shah"): ("kobitski_ew06 + shah_sample1", 370, 16572),
    ("ground_truth", "kobitski_ew06"): ("kobitski_ew06", 370, 16572),
    ("ground_truth", "shah_sample1"): ("shah_sample1", 420, 4113),
    ("baseline_with_selfcal", "ew06_vs_shah"): ("kobitski_ew06 + shah_sample1", 370, 16572),
}


# (phase, name) -> config path, used to read search_space/default_params for estimation.
CONFIG_PATHS = {
    ("selfcal", "kobitski_ew06_alignment"): CONFIGS_ROOT / "selfcal" / "kobitski_ew06_alignment.yaml",
    ("selfcal", "shah_alignment"): CONFIGS_ROOT / "selfcal" / "shah_alignment.yaml",
    ("selfcal", "shah_label_transfer"): CONFIGS_ROOT / "selfcal" / "shah_label_transfer.yaml",
    ("baseline_no_hpo", "ew06_vs_shah"): CONFIGS_ROOT / "baseline_no_hpo" / "ew06_vs_shah.yaml",
    ("ground_truth", "kobitski_ew06"): CONFIGS_ROOT / "ground_truth" / "kobitski_ew06.yaml",
    ("ground_truth", "shah_sample1"): CONFIGS_ROOT / "ground_truth" / "shah_sample1.yaml",
    ("baseline_with_selfcal", "ew06_vs_shah"): CONFIGS_ROOT / "baseline_with_selfcal" / "ew06_vs_shah.yaml",
}

# Calibration measured 2026-07-15 against real Kobitski ew_06 data at
# max_points_per_frame=1000, step=8 (47 strided frames) — see module
# docstring "Estimated total runtime". Seconds per single alignment trial
# (one create_pairwise_distance_matrix call), by window_size.
CPD_TRIAL_SECONDS = {5: 130.5, 10: 176.5, 20: 278.6}
NO_CPD_TRIAL_SECONDS = 4.0
# Not directly measured — a rough placeholder for LabelTransferStage's added
# cost per trial when run_label_transfer=True. KNN-based, expected cheap
# relative to CPD, but unverified at time of writing.
LABEL_TRANSFER_OVERHEAD_SECONDS = 30.0
# eval/runners/optimizer.py module constants (duplicated here to avoid
# importing optimizer.py's heavy torch/optuna dependency chain into this
# lightweight aggregation script).
SANITY_N_TRIALS = 5
DEV_N_TRIALS = 20
SANITY_TRIAL_SECONDS = 1.0  # tiny synthetic data (3 frames, 50 pts) - negligible


def _nearest_cpd_cost(window_size: int) -> float:
    """CPD_TRIAL_SECONDS lookup with nearest-key fallback for uncalibrated window sizes."""
    if window_size in CPD_TRIAL_SECONDS:
        return CPD_TRIAL_SECONDS[window_size]
    nearest = min(CPD_TRIAL_SECONDS, key=lambda w: abs(w - window_size))
    return CPD_TRIAL_SECONDS[nearest]


def _avg_trial_seconds(cpd_penalty_choices: list, window_choices: list, label_transfer: bool) -> float:
    """Average per-trial cost across a search_space's cpd_penalty x window_size grid."""
    non_null = [c for c in cpd_penalty_choices if c is not None]
    frac_cpd = len(non_null) / len(cpd_penalty_choices) if cpd_penalty_choices else 0.0
    avg_cpd = sum(_nearest_cpd_cost(w) for w in window_choices) / len(window_choices) if window_choices else 0.0
    cost = frac_cpd * avg_cpd + (1 - frac_cpd) * NO_CPD_TRIAL_SECONDS
    if label_transfer:
        cost += LABEL_TRANSFER_OVERHEAD_SECONDS
    return cost


def _estimate_optimize_run(config: dict) -> float:
    search_space = config.get("search_space", {})
    cpd_choices = search_space.get("cpd_penalty", [None])
    window_choices = search_space.get("window_size", [10])
    label_transfer = config.get("run_label_transfer", True)
    avg_trial = _avg_trial_seconds(cpd_choices, window_choices, label_transfer)
    sanity = SANITY_N_TRIALS * SANITY_TRIAL_SECONDS
    dev = DEV_N_TRIALS * avg_trial
    final_eval = avg_trial  # one more trial-equivalent pass with the winning params
    return sanity + dev + final_eval


# Only this run's params are runtime-dependent (merged from selfcal best_params.json
# at execution time — see run_all.py / merge_params.py); its config default_params
# are a fallback only, not what actually runs. Every other eval-only run's params
# are fixed by its own config, so best case == worst case for those.
RUNTIME_DEPENDENT_PARAMS = {("baseline_with_selfcal", "ew06_vs_shah")}


def _estimate_eval_only_run(config: dict, runtime_dependent: bool) -> tuple[float, float]:
    default_params = config.get("default_params", {})
    window = default_params.get("window_size", 10)
    label_transfer = config.get("run_label_transfer", True)
    best = _avg_trial_seconds([default_params.get("cpd_penalty")], [window], label_transfer)
    if not runtime_dependent:
        return best, best
    worst = _avg_trial_seconds(["rigid"], [window], label_transfer)
    return best, worst


def estimate_run_seconds(phase: str, name: str) -> tuple[float, float]:
    """Returns (best_case_seconds, worst_case_seconds) — equal unless the run's
    actual params are runtime-dependent (see RUNTIME_DEPENDENT_PARAMS)."""
    path = CONFIG_PATHS[(phase, name)]
    with open(path) as f:
        config = yaml.safe_load(f)
    if (phase, name) in OPTIMIZE_RUNS:
        est = _estimate_optimize_run(config)
        return est, est
    return _estimate_eval_only_run(config, runtime_dependent=(phase, name) in RUNTIME_DEPENDENT_PARAMS)


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def _mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


def _run_cost(phase: str, name: str) -> dict:
    run_dir = EXPERIMENTS_ROOT / phase / name
    is_optimize = (phase, name) in OPTIMIZE_RUNS
    dataset_label, frames, mean_pts = DATASET_STATS[(phase, name)]
    est_best, est_worst = estimate_run_seconds(phase, name)

    row = {
        "phase": phase,
        "name": name,
        "dataset": dataset_label,
        "frames": frames,
        "mean_pts_per_frame": mean_pts,
        "mode": "optimize+eval" if is_optimize else "eval-only",
        "status": "pending",
        "start_time": "",
        "end_time": "",
        "duration": "",
        "n_trials_run": "",
        "avg_time_per_trial": "",
        # Projected seconds if this run hasn't finished yet (see estimate_run_seconds);
        # overwritten with the real measured duration below once one exists.
        "projected_seconds": (est_best + est_worst) / 2,
        "projected_range": (est_best, est_worst),
        "actual_seconds": None,
    }

    start = _mtime(run_dir / "run_config.yaml")
    if start is None:
        return row
    row["start_time"] = start.isoformat(timespec="seconds")

    end = _mtime(run_dir / "eval_report.json")

    if end is None:
        elapsed = datetime.now() - start
        row["status"] = "in progress"
        row["duration"] = f"{_fmt_duration(elapsed.total_seconds())} so far"
    else:
        row["status"] = "done"
        row["end_time"] = end.isoformat(timespec="seconds")
        row["duration"] = _fmt_duration((end - start).total_seconds())
        row["actual_seconds"] = (end - start).total_seconds()

    if is_optimize:
        best_params_path = run_dir / "best_params.json"
        optimize_done = _mtime(best_params_path)
        if optimize_done is not None:
            optimize_seconds = (optimize_done - start).total_seconds()
            history_path = run_dir / "search_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    n_trials = len(json.load(f))
                row["n_trials_run"] = n_trials
                if n_trials:
                    row["avg_time_per_trial"] = _fmt_duration(optimize_seconds / n_trials)

    return row


def build_rows() -> list[dict]:
    return [_run_cost(phase, name) for phase, name in RUNS]


def write_markdown(rows: list[dict], path: Path) -> None:
    done = [r for r in rows if r["status"] == "done"]
    in_progress = [r for r in rows if r["status"] == "in progress"]
    pending = [r for r in rows if r["status"] == "pending"]

    lines = [
        "# baseline_experiments — time & compute cost",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}.",
        "",
        f"{len(done)}/{len(rows)} runs finished, {len(in_progress)} in progress, {len(pending)} pending.",
        "",
        "Timing is derived from filesystem mtimes (run_config.yaml -> "
        "best_params.json -> eval_report.json), not process instrumentation "
        "— see aggregate_cost.py docstring. \"Compute\" is approximated by "
        "dataset scale (frames x mean points/frame) since all runs share one "
        "long-lived process with no per-run CPU-seconds counter.",
        "",
        "| phase | name | dataset | frames | mean pts/frame | mode | status | duration | n_trials_run | avg time/trial |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join(
                str(r[k]) for k in (
                    "phase", "name", "dataset", "frames", "mean_pts_per_frame",
                    "mode", "status", "duration", "n_trials_run", "avg_time_per_trial",
                )
            ) + " |"
        )

    if done:
        total_seconds = sum(
            (datetime.fromisoformat(r["end_time"]) - datetime.fromisoformat(r["start_time"])).total_seconds()
            for r in done
        )
        lines += [
            "",
            f"**Total wall-clock time across {len(done)} finished run(s): {_fmt_duration(total_seconds)}.**",
        ]

    # --- Estimated total runtime ---
    # Calibrated projection (see module docstring "Estimated total runtime")
    # for pending/in-progress runs; real measured duration for done runs.
    lines += [
        "",
        "## Estimated total runtime",
        "",
        "Calibrated against real Kobitski ew_06 data at max_points_per_frame=1000, "
        "step=8 (see module docstring). Done runs use their real measured duration; "
        "pending/in-progress runs use the calibrated projection (best case = worst "
        "case except baseline_with_selfcal, whose actual params depend on what "
        "selfcal converges to at runtime — see estimate_run_seconds).",
        "",
        "| phase | name | status | basis | best case | worst case |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    total_best = 0.0
    total_worst = 0.0
    for r in rows:
        if r["actual_seconds"] is not None:
            basis = "actual"
            best = worst = r["actual_seconds"]
        else:
            basis = "projected"
            best, worst = r["projected_range"]
        total_best += best
        total_worst += worst
        lines.append(
            f"| {r['phase']} | {r['name']} | {r['status']} | {basis} "
            f"| {_fmt_duration(best)} | {_fmt_duration(worst)} |"
        )
    lines += [
        "",
        f"**Estimated grand total: {_fmt_duration(total_best)} (best case) "
        f"— {_fmt_duration(total_worst)} (worst case).**",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    rows = build_rows()
    out_path = EXPERIMENTS_ROOT / "summary" / "compute_cost.md"
    write_markdown(rows, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
