"""Orchestrates the full baseline_experiments suite.

Runs four phases, in order, against the zReg evaluation framework
(``eval.config.EvalConfig`` / ``eval.runners.HyperparamOptimizer`` /
``eval.runners.EvaluationRunner``) imported directly rather than shelled out
through run_eval.py — this avoids run_eval.py's per-invocation timestamped
output_dir (it would make it impossible to predict the path where selfcal
writes best_params.json before baseline_with_selfcal needs to read it).

Phases (see baseline_experiments/README.md for the full design rationale):

1. selfcal            — optimize+eval self-registration HPO; Kobitski ew_06
                         alignment-only, Shah alignment-only, Shah alignment+
                         label-transfer. Produces best_params.json consumed by
                         phase 4.
2. baseline_no_hpo     — eval-only, config default_params, ew06_vs_shah pair.
3. ground_truth        — optimize+eval self-registration HPO with a random
                         (not fixed) transform_spec; Kobitski ew_06 alone,
                         Shah sample-1 alone; both stages. Produces
                         best_params.json consumed by phase 5.
4. baseline_with_selfcal — eval-only, ew06_vs_shah, params merged from phase
                         1's three best_params.json (see merge_params.py).
5. baseline_with_groundtruth — eval-only, ew06_vs_shah, params merged from
                         phase 3's two best_params.json. Validates whether
                         HPO calibrated against a random unseen perturbation
                         generalises better than selfcal to the real cross-
                         embryo task.

Each run is idempotent: if ``eval_report.json`` already exists in a run's
output_dir, it is skipped unless ``--force`` is passed. This lets the full
suite be safely re-invoked after a crash or interruption.

**Runtime:** tier=dev → 5 sanity trials (tiny synthetic data, fast) + 20 dev
trials (real data, step=8 temporal subsampling, max_points=1000 spatial
subsampling). On HoreKa with 16 MPI ranks and GPU acceleration the 5 HPO
runs complete in 43–122 min (see launch_horeka.sbatch for per-phase
breakdown). Locally without GPU/MPI the same runs can take hours per
experiment — use nohup/tmux or the cluster.

Usage
-----
    python baseline_experiments/scripts/run_all.py --phase all
    python baseline_experiments/scripts/run_all.py --phase selfcal
    python baseline_experiments/scripts/run_all.py --phase all --dry-run
    python baseline_experiments/scripts/run_all.py --phase all --force
    python baseline_experiments/scripts/run_all.py --phase all --configs-dir baseline_experiments/configs_horeka
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

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError:
    COMM = None
    RANK = 0  # single-process / no-MPI — unchanged local behaviour

from eval.config import EvalConfig, EvalConfigError  # noqa: E402
from eval.runners import EvaluationRunner, HyperparamOptimizer  # noqa: E402

from merge_params import merge_groundtruth_params, merge_selfcal_params  # noqa: E402

log = logging.getLogger("run_all")

# PHASE_ORDER does not reference CONFIGS and stays as a module-level constant.
PHASE_ORDER = ["selfcal", "baseline_no_hpo", "ground_truth", "baseline_with_selfcal", "baseline_with_groundtruth"]


def _build_phase_lists(configs_dir: Path) -> dict[str, list]:
    """Build phase run lists from the given configs directory.

    Parameters
    ----------
    configs_dir:
        Root configs directory (e.g. ``SUITE_ROOT / "configs"`` locally or
        ``SUITE_ROOT / "configs_horeka"`` on the cluster).

    Returns
    -------
    dict
        Keys: ``"selfcal"``, ``"baseline_no_hpo"``, ``"ground_truth"``,
        ``"baseline_with_selfcal"``.  Values: lists of
        ``(phase, name, config_path)`` tuples.
    """
    # (phase, name, config_path) — order within a phase is execution order.
    selfcal = [
        ("selfcal", "kobitski_ew06_alignment", configs_dir / "selfcal" / "kobitski_ew06_alignment.yaml"),
        ("selfcal", "shah_alignment", configs_dir / "selfcal" / "shah_alignment.yaml"),
        ("selfcal", "shah_label_transfer", configs_dir / "selfcal" / "shah_label_transfer.yaml"),
    ]
    # Scoped down to a single pair (ew06_vs_shah — the embryo already used for
    # selfcal calibration) instead of all 4 Kobitski embryos, after real-data
    # testing showed even a single eval-only pass on full-resolution Kobitski
    # data can take 30-90+ min; running all 4 pairs here on top of the 5
    # optimize-mode runs was not worth the added wall-clock. configs/baseline_
    # no_hpo/{ew08,ew11,ew12}_vs_shah.yaml and configs/baseline_with_selfcal/
    # {ew08,ew11,ew12}_vs_shah.yaml still exist on disk if you want to run any
    # of them manually later (see README.md "Running").
    baseline_no_hpo = [
        ("baseline_no_hpo", name, configs_dir / "baseline_no_hpo" / f"{name}.yaml")
        for name in ("ew06_vs_shah",)
    ]
    ground_truth = [
        ("ground_truth", "kobitski_ew06", configs_dir / "ground_truth" / "kobitski_ew06.yaml"),
        ("ground_truth", "shah_sample1", configs_dir / "ground_truth" / "shah_sample1.yaml"),
    ]
    baseline_with_selfcal = [
        ("baseline_with_selfcal", name, configs_dir / "baseline_with_selfcal" / f"{name}.yaml")
        for name in ("ew06_vs_shah",)
    ]
    baseline_with_groundtruth = [
        ("baseline_with_groundtruth", name, configs_dir / "baseline_with_groundtruth" / f"{name}.yaml")
        for name in ("ew06_vs_shah",)
    ]
    return {
        "selfcal": selfcal,
        "baseline_no_hpo": baseline_no_hpo,
        "ground_truth": ground_truth,
        "baseline_with_selfcal": baseline_with_selfcal,
        "baseline_with_groundtruth": baseline_with_groundtruth,
    }


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


def _clear_propulate_checkpoints(output_dir: Path) -> None:
    """Delete propulate checkpoint .pkl files from output_dir (rank-0 only).

    Propulate writes checkpoint files to checkpoint_path (= output_dir in our
    setup). Stale checkpoints from a pre-fix run carry the old float-index
    categorical encoding and will corrupt param decoding when the job resumes.
    This removes only *.pkl files; eval_report.json and other outputs are kept.
    """
    if not output_dir.exists():
        return
    removed = list(output_dir.glob("*.pkl"))
    for f in removed:
        f.unlink()
    if removed:
        log.info("[clear-checkpoints] removed %d .pkl file(s) from %s", len(removed), output_dir)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_optimize_then_eval(name: str, config_path: Path, force: bool, dry_run: bool, clear_checkpoints: bool = False) -> None:
    config = _load_config(config_path)
    output_dir = Path(config.output_dir)

    skip = False
    if RANK == 0:
        if not force and _already_done(output_dir):
            log.info("[%s] SKIP (eval_report.json already exists at %s)", name, output_dir)
            skip = True
        else:
            log.info("[%s] optimize+eval -> %s (tier=%s n_trials=%s)", name, output_dir, config.tier, config.n_trials)
            if dry_run:
                skip = True
            else:
                if clear_checkpoints:
                    _clear_propulate_checkpoints(output_dir)
                _write_run_config(config_path, output_dir)

    # Broadcast skip decision so every rank agrees before the collective call.
    # Without this, rank-0 returning early leaves other ranks deadlocked on
    # the internal comm.Barrier() inside HyperparamOptimizer.run().
    if COMM is not None:
        skip = COMM.bcast(skip, root=0)
    if skip:
        return

    # HyperparamOptimizer.run() is a collective MPI operation — every rank
    # must call this (propulate needs all ranks to participate; gating on
    # rank-0 only would deadlock on the internal comm.Barrier()).
    HyperparamOptimizer(config).run()

    if RANK == 0:
        best_params_path = output_dir / "best_params.json"
        optimized = _read_json(best_params_path) if best_params_path.exists() else {}
        params = {**config.default_params, **optimized}

        EvaluationRunner(config, params).run()
        log.info("[%s] done", name)
    else:
        log.debug("[rank %d] non-rank-0: participated in collective HPO, skipping orchestration", RANK)


def run_eval_only(name: str, config_path: Path, params: dict | None, force: bool, dry_run: bool) -> None:
    # Eval-only phases have no collective MPI operation — skip entirely on
    # non-rank-0 to avoid duplicate file writes and EvaluationRunner calls.
    if RANK != 0:
        return

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


def _selfcal_best_params(configs_dir: Path) -> tuple[dict, dict, dict, dict]:
    """Load the three selfcal best_params.json files plus a defaults dict.

    Parameters
    ----------
    configs_dir:
        Root configs directory used for this run (resolved absolute path).

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
        "kobitski_alignment": configs_dir / "selfcal" / "kobitski_ew06_alignment.yaml",
        "shah_alignment": configs_dir / "selfcal" / "shah_alignment.yaml",
        "shah_label_transfer": configs_dir / "selfcal" / "shah_label_transfer.yaml",
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

    defaults_config = _load_config(configs_dir / "baseline_with_selfcal" / "ew06_vs_shah.yaml")
    return (
        results["kobitski_alignment"],
        results["shah_alignment"],
        results["shah_label_transfer"],
        defaults_config.default_params,
    )


def _groundtruth_best_params(configs_dir: Path) -> tuple[dict, dict, dict]:
    """Load ground_truth best_params.json files plus a defaults dict.

    Returns
    -------
    tuple
        ``(kobitski_alignment, shah_both, defaults)``.
        ``shah_both`` is from ground_truth/shah_sample1 which runs both alignment
        and label transfer, so it covers all param keys needed for the merge.

    Raises
    ------
    FileNotFoundError
        If a ground_truth run hasn't produced best_params.json yet — run the
        ``ground_truth`` phase first.
    """
    paths = {
        "kobitski": configs_dir / "ground_truth" / "kobitski_ew06.yaml",
        "shah": configs_dir / "ground_truth" / "shah_sample1.yaml",
    }
    results = {}
    for key, cfg_path in paths.items():
        config = _load_config(cfg_path)
        bp_path = Path(config.output_dir) / "best_params.json"
        if not bp_path.exists():
            raise FileNotFoundError(
                f"{bp_path} not found — run the 'ground_truth' phase before 'baseline_with_groundtruth'."
            )
        results[key] = _read_json(bp_path)

    defaults_config = _load_config(configs_dir / "baseline_with_groundtruth" / "ew06_vs_shah.yaml")
    return (
        results["kobitski"],
        results["shah"],
        defaults_config.default_params,
    )


def run_phase(phase: str, phases_map: dict[str, list], configs_dir: Path, force: bool, dry_run: bool, clear_checkpoints: bool = False) -> None:
    runs = phases_map[phase]

    if phase in ("selfcal", "ground_truth"):
        for _, name, cfg_path in runs:
            run_optimize_then_eval(name, cfg_path, force, dry_run, clear_checkpoints=clear_checkpoints)

    elif phase == "baseline_no_hpo":
        for _, name, cfg_path in runs:
            run_eval_only(name, cfg_path, params=None, force=force, dry_run=dry_run)

    elif phase == "baseline_with_selfcal":
        selfcal_runs = phases_map["selfcal"]
        if dry_run and not all(
            (Path(_load_config(sc[2]).output_dir) / "best_params.json").exists() for sc in selfcal_runs
        ):
            log.info("[baseline_with_selfcal] DRY-RUN: selfcal best_params.json not yet available — would merge at real run time")
            for _, name, cfg_path in runs:
                run_eval_only(name, cfg_path, params=None, force=force, dry_run=True)
            return

        kobitski_align, shah_align, shah_lt, defaults = _selfcal_best_params(configs_dir)
        merged = merge_selfcal_params(kobitski_align, shah_align, shah_lt, defaults)
        log.info("[baseline_with_selfcal] merged params: %s", merged)
        for _, name, cfg_path in runs:
            run_eval_only(name, cfg_path, params=merged, force=force, dry_run=dry_run)

    elif phase == "baseline_with_groundtruth":
        gt_runs = phases_map["ground_truth"]
        if dry_run and not all(
            (Path(_load_config(gt[2]).output_dir) / "best_params.json").exists() for gt in gt_runs
        ):
            log.info("[baseline_with_groundtruth] DRY-RUN: ground_truth best_params.json not yet available — would merge at real run time")
            for _, name, cfg_path in runs:
                run_eval_only(name, cfg_path, params=None, force=force, dry_run=True)
            return

        kobitski_gt, shah_gt, defaults = _groundtruth_best_params(configs_dir)
        # shah_gt produced both alignment and label-transfer params (run_label_transfer=true);
        # merge_groundtruth_params passes it as both shah_alignment and shah_label_transfer.
        merged = merge_groundtruth_params(kobitski_gt, shah_gt, defaults)
        log.info("[baseline_with_groundtruth] merged params: %s", merged)
        for _, name, cfg_path in runs:
            run_eval_only(name, cfg_path, params=merged, force=force, dry_run=dry_run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the baseline_experiments suite")
    parser.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-run even if eval_report.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan without running any pipeline")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--configs-dir",
        default=str(SUITE_ROOT / "configs"),
        help="Root configs directory (default: baseline_experiments/configs/). Pass baseline_experiments/configs_horeka for cluster runs.",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help=(
            "Delete stale propulate *.pkl checkpoint files from each optimize-mode output directory "
            "before starting HPO. Preserves eval_report.json. Use after a crashed run so propulate "
            "does not resume from a checkpoint that has the old float-index categorical encoding."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve --configs-dir to absolute BEFORE os.chdir(REPO_ROOT) — a
    # relative path passed on the CLI would be silently broken by the chdir
    # (D-02).
    configs_dir = Path(args.configs_dir).resolve()

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

    # Config YAMLs use paths relative to REPO_ROOT (e.g. data_path,
    # output_dir) — make that resolution robust regardless of invocation cwd.
    os.chdir(REPO_ROOT)

    phases_map = _build_phase_lists(configs_dir)

    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    try:
        for phase in phases:
            log.info("=== phase: %s ===", phase)
            run_phase(phase, phases_map, configs_dir, force=args.force, dry_run=args.dry_run, clear_checkpoints=args.clear_checkpoints)
    except (EvalConfigError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
