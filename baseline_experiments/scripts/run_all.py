"""Orchestrates the full baseline_experiments suite.

Runs four phases, in order, against the zReg evaluation framework
(``eval.config.EvalConfig`` / ``eval.runners.HyperparamOptimizer`` /
``eval.runners.EvaluationRunner``) imported directly rather than shelled out
through run_eval.py — this avoids run_eval.py's per-invocation timestamped
output_dir (it would make it impossible to predict the path where selfcal
writes best_params.json before baseline_with_selfcal needs to read it).

Phases (see baseline_experiments/README.md for the full design rationale):

1. selfcal            — optimize+eval (mode="full") self-registration HPO;
                         Kobitski ew_06 alignment-only, Shah alignment-only,
                         Shah alignment+label-transfer.
2. baseline_no_hpo     — eval-only, config default_params, 4 Kobitski-vs-Shah
                         pairs.
3. ground_truth        — optimize+eval (mode="full") self-registration HPO
                         with a random (not fixed) transform_spec; Kobitski
                         ew_06 alone, Shah sample-1 alone; both stages.
4. baseline_with_selfcal — eval-only, same 4 pairs as phase 2, but params are
                         merged from phase 1's three best_params.json files
                         (see merge_params.py) instead of config defaults.

Each run is idempotent: if ``eval_report.json`` already exists in a run's
output_dir, it is skipped unless ``--force`` is passed. This lets the full
suite be safely re-invoked after a crash or interruption.

**Runtime warning:** all optimize-mode configs (selfcal x3, ground_truth x2)
are set to ``tier: dev``, not ``full`` — smoke-testing showed a single
alignment pass on the full-density real Kobitski ew_06 trajectory (370
frames, ~16.6k points/frame) did not finish in 32 minutes. At tier=dev, each
optimize run still does 5 sanity trials (fast, tiny synthetic data) + 20 dev
trials (full, unsampled real data, ~30-90+ min/trial for Kobitski-involving
runs). With 5 optimize runs total this suite is expected to take on the
order of DAYS of continuous sequential compute — run it with nohup/tmux or
similar, not attached to an interactive session that might be interrupted.

Usage
-----
    python baseline_experiments/scripts/run_all.py --phase all
    python baseline_experiments/scripts/run_all.py --phase selfcal
    python baseline_experiments/scripts/run_all.py --phase all --dry-run
    python baseline_experiments/scripts/run_all.py --phase all --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path(__file__).resolve().parents[1]

# eval/ is a namespace dir (no __init__.py) and zreg lives under src/ —
# mirrors the sys.path injection in run_eval.py.
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval.config import EvalConfig, EvalConfigError  # noqa: E402
from eval.runners import EvaluationRunner, HyperparamOptimizer  # noqa: E402

from merge_params import merge_selfcal_params  # noqa: E402

log = logging.getLogger("run_all")

CONFIGS = SUITE_ROOT / "configs"

# (phase, name, config_path) — order within a phase is execution order.
SELFCAL = [
    ("selfcal", "kobitski_ew06_alignment", CONFIGS / "selfcal" / "kobitski_ew06_alignment.yaml"),
    ("selfcal", "shah_alignment", CONFIGS / "selfcal" / "shah_alignment.yaml"),
    ("selfcal", "shah_label_transfer", CONFIGS / "selfcal" / "shah_label_transfer.yaml"),
]
# Scoped down to a single pair (ew06_vs_shah — the embryo already used for
# selfcal calibration) instead of all 4 Kobitski embryos, after real-data
# testing showed even a single eval-only pass on full-resolution Kobitski
# data can take 30-90+ min; running all 4 pairs here on top of the 5
# optimize-mode runs was not worth the added wall-clock. configs/baseline_
# no_hpo/{ew08,ew11,ew12}_vs_shah.yaml and configs/baseline_with_selfcal/
# {ew08,ew11,ew12}_vs_shah.yaml still exist on disk if you want to run any
# of them manually later (see README.md "Running").
BASELINE_NO_HPO = [
    ("baseline_no_hpo", name, CONFIGS / "baseline_no_hpo" / f"{name}.yaml")
    for name in ("ew06_vs_shah",)
]
GROUND_TRUTH = [
    ("ground_truth", "kobitski_ew06", CONFIGS / "ground_truth" / "kobitski_ew06.yaml"),
    ("ground_truth", "shah_sample1", CONFIGS / "ground_truth" / "shah_sample1.yaml"),
]
BASELINE_WITH_SELFCAL = [
    ("baseline_with_selfcal", name, CONFIGS / "baseline_with_selfcal" / f"{name}.yaml")
    for name in ("ew06_vs_shah",)
]

PHASES = {
    "selfcal": SELFCAL,
    "baseline_no_hpo": BASELINE_NO_HPO,
    "ground_truth": GROUND_TRUTH,
    "baseline_with_selfcal": BASELINE_WITH_SELFCAL,
}
PHASE_ORDER = ["selfcal", "baseline_no_hpo", "ground_truth", "baseline_with_selfcal"]


def _load_config(config_path: Path) -> EvalConfig:
    config = EvalConfig.from_yaml(config_path)
    # Resolve output_dir against REPO_ROOT so the suite works regardless of
    # the cwd it's invoked from (config YAMLs use paths relative to REPO_ROOT).
    out = Path(config.output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    return config.model_copy(update={"output_dir": str(out)})


def _write_run_config(config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "run_config.yaml")


def _already_done(output_dir: Path) -> bool:
    return (output_dir / "eval_report.json").exists()


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_optimize_then_eval(name: str, config_path: Path, force: bool, dry_run: bool) -> None:
    config = _load_config(config_path)
    output_dir = Path(config.output_dir)

    if not force and _already_done(output_dir):
        log.info("[%s] SKIP (eval_report.json already exists at %s)", name, output_dir)
        return
    log.info("[%s] optimize+eval -> %s (tier=%s n_trials=%s)", name, output_dir, config.tier, config.n_trials)
    if dry_run:
        return

    _write_run_config(config_path, output_dir)
    HyperparamOptimizer(config).run()

    best_params_path = output_dir / "best_params.json"
    optimized = _read_json(best_params_path) if best_params_path.exists() else {}
    params = {**config.default_params, **optimized}

    EvaluationRunner(config, params).run()
    log.info("[%s] done", name)


def run_eval_only(name: str, config_path: Path, params: dict | None, force: bool, dry_run: bool) -> None:
    config = _load_config(config_path)
    output_dir = Path(config.output_dir)

    if not force and _already_done(output_dir):
        log.info("[%s] SKIP (eval_report.json already exists at %s)", name, output_dir)
        return
    resolved_params = config.default_params if params is None else params
    log.info("[%s] eval-only -> %s", name, output_dir)
    if dry_run:
        return

    _write_run_config(config_path, output_dir)
    EvaluationRunner(config, resolved_params).run()
    log.info("[%s] done", name)


def _selfcal_best_params() -> tuple[dict, dict, dict, dict]:
    """Load the three selfcal best_params.json files plus a defaults dict.

    Returns
    -------
    tuple
        ``(kobitski_alignment, shah_alignment, shah_label_transfer, defaults)``.

    Raises
    ------
    FileNotFoundError
        If a selfcal run hasn't produced best_params.json yet — run the
        ``selfcal`` phase first.
    """
    paths = {
        "kobitski_alignment": CONFIGS / "selfcal" / "kobitski_ew06_alignment.yaml",
        "shah_alignment": CONFIGS / "selfcal" / "shah_alignment.yaml",
        "shah_label_transfer": CONFIGS / "selfcal" / "shah_label_transfer.yaml",
    }
    results = {}
    for key, cfg_path in paths.items():
        config = _load_config(cfg_path)
        bp_path = Path(config.output_dir) / "best_params.json"
        if not bp_path.exists():
            raise FileNotFoundError(
                f"{bp_path} not found — run the 'selfcal' phase before 'baseline_with_selfcal'."
            )
        results[key] = _read_json(bp_path)

    defaults_config = _load_config(CONFIGS / "baseline_with_selfcal" / "ew06_vs_shah.yaml")
    return (
        results["kobitski_alignment"],
        results["shah_alignment"],
        results["shah_label_transfer"],
        defaults_config.default_params,
    )


def run_phase(phase: str, force: bool, dry_run: bool) -> None:
    runs = PHASES[phase]

    if phase in ("selfcal", "ground_truth"):
        for _, name, cfg_path in runs:
            run_optimize_then_eval(name, cfg_path, force, dry_run)

    elif phase == "baseline_no_hpo":
        for _, name, cfg_path in runs:
            run_eval_only(name, cfg_path, params=None, force=force, dry_run=dry_run)

    elif phase == "baseline_with_selfcal":
        if dry_run and not all(
            (Path(_load_config(sc[2]).output_dir) / "best_params.json").exists() for sc in SELFCAL
        ):
            log.info("[baseline_with_selfcal] DRY-RUN: selfcal best_params.json not yet available — would merge at real run time")
            for _, name, cfg_path in runs:
                run_eval_only(name, cfg_path, params=None, force=force, dry_run=True)
            return

        kobitski_align, shah_align, shah_lt, defaults = _selfcal_best_params()
        merged = merge_selfcal_params(kobitski_align, shah_align, shah_lt, defaults)
        log.info("[baseline_with_selfcal] merged params: %s", merged)
        for _, name, cfg_path in runs:
            run_eval_only(name, cfg_path, params=merged, force=force, dry_run=dry_run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the baseline_experiments suite")
    parser.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_report.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan without running any pipeline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

    # Config YAMLs use paths relative to REPO_ROOT (e.g. data_path,
    # output_dir) — make that resolution robust regardless of invocation cwd.
    os.chdir(REPO_ROOT)

    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    try:
        for phase in phases:
            log.info("=== phase: %s ===", phase)
            run_phase(phase, force=args.force, dry_run=args.dry_run)
    except (EvalConfigError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
