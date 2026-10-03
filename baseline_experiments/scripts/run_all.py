"""Orchestrates the full baseline_experiments suite.

Runs four phases, in order, against the zReg evaluation framework
(``eval.config.EvalConfig`` / ``eval.runners.HyperparamOptimizer`` /
``eval.runners.EvaluationRunner``) imported directly rather than shelled out
through run_eval.py — this avoids run_eval.py's per-invocation timestamped
output_dir (it would make it impossible to predict the path where selfcal
writes best_params.json before baseline_with_combined needs to read it).

Phases (see baseline_experiments/README.md for the full design rationale):

1. selfcal            — optimize+eval self-registration HPO; Kobitski ew_06
                         alignment-only, Shah alignment-only, Shah alignment+
                         label-transfer. Produces best_params.json consumed by
                         phase 4.
2. baseline_no_hpo     — eval-only, config default_params, ew06_vs_shah pair.
3. ground_truth        — optimize+eval self-registration HPO with a random
                         (not fixed) transform_spec; Kobitski ew_06 alone,
                         Shah sample-1 alone; both stages. Produces
                         best_params.json consumed by phase 4.
4. baseline_with_combined — optimize+eval, ew06_vs_shah (real cross-embryo task).
                         Merged params from phases 1 + 3 (merge_combined_params,
                         see merge_params.py) are snapped onto the combined
                         config's search space (nearest choice per key) and
                         injected as default_params and as the warm start
                         (first HPO trial). Every
                         upstream best_params.json and the merged params are
                         validated through EvalConfig.model_validate first
                         (Phase 63 HPC-01). Uses same search space as the
                         selfcal/ground_truth full-pipeline runs.

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

from merge_params import merge_combined_params  # noqa: E402

log = logging.getLogger("run_all")

# PHASE_ORDER does not reference CONFIGS and stays as a module-level constant.
PHASE_ORDER = ["selfcal", "baseline_no_hpo", "ground_truth", "baseline_with_combined"]


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
        ``"baseline_with_combined"``.  Values: lists of
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
    # no_hpo/{ew08,ew11,ew12}_vs_shah.yaml still exist on disk if you want to run
    # any of them manually later (see README.md "Running").
    baseline_no_hpo = [
        ("baseline_no_hpo", name, configs_dir / "baseline_no_hpo" / f"{name}.yaml")
        for name in ("ew06_vs_shah",)
    ]
    ground_truth = [
        ("ground_truth", "kobitski_ew06", configs_dir / "ground_truth" / "kobitski_ew06.yaml"),
        ("ground_truth", "shah_sample1", configs_dir / "ground_truth" / "shah_sample1.yaml"),
    ]
    baseline_with_combined = [
        ("baseline_with_combined", name, configs_dir / "baseline_with_combined" / f"{name}.yaml")
        for name in ("ew06_vs_shah",)
    ]
    return {
        "selfcal": selfcal,
        "baseline_no_hpo": baseline_no_hpo,
        "ground_truth": ground_truth,
        "baseline_with_combined": baseline_with_combined,
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


def _propulate_checkpoint_files(output_dir: Path) -> list[Path]:
    """Propulate checkpoint files in ``output_dir`` (``*.pkl``, ``*.pickle``, ``*.bkp``)."""
    if not output_dir.exists():
        return []
    return [f for pattern in ("*.pkl", "*.pickle", "*.bkp") for f in output_dir.glob(pattern)]


# Records the search space the Propulate checkpoints in an output_dir were
# written under (63-REVIEW WR-03).
SEARCH_SPACE_FINGERPRINT = "search_space_fingerprint.json"

_CLEAR_HINT = (
    "to discard them and restart this HPO run, re-submit with ZREG_CLEAR_CHECKPOINTS=1 "
    "(or pass --clear-checkpoints to run_all.py)"
)


def _search_space_fingerprint(search_space: dict) -> str:
    return json.dumps(search_space, sort_keys=True, default=str)


def _checkpoint_search_space_error(output_dir: Path, search_space: dict) -> str | None:
    """Compare the checkpoints' recorded search space with the current one (rank 0).

    Propulate checkpoints written under a different search space break the
    resumed run inside ``_decode_param`` (e.g. a ``dtw_dist_fn: cosine``
    individual after Phase 63 D-08). Checkpoints are never deleted here; the
    operator decides (63-REVIEW WR-03).

    - No checkpoints: the current search space is recorded in
      ``SEARCH_SPACE_FINGERPRINT`` (a fresh or freshly cleared run).
    - Checkpoints and a matching record: ``None`` (normal resume).
    - Checkpoints and a different record: an error message that names
      ``ZREG_CLEAR_CHECKPOINTS=1`` / ``--clear-checkpoints``.
    - Checkpoints without a record (written before this check existed): a
      WARNING with the same hint, and the resume proceeds. No record is
      written, so it never vouches for those checkpoints.

    Returns
    -------
    str or None
        The error message, or ``None`` when the run may proceed.
    """
    record = output_dir / SEARCH_SPACE_FINGERPRINT
    current = _search_space_fingerprint(search_space)
    if not _propulate_checkpoint_files(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(current)
        return None
    if not record.exists():
        log.warning(
            "[checkpoints] %s holds Propulate checkpoints without a recorded search space; "
            "resuming from them. If the search space changed since they were written (e.g. "
            "Phase 63 D-08), the resume fails while decoding old individuals; %s.",
            output_dir,
            _CLEAR_HINT,
        )
        return None
    recorded = record.read_text()
    if recorded != current:
        return (
            f"{output_dir}: the Propulate checkpoints were written under a different search space "
            f"({recorded}) than the current config ({current}), so resuming from them would fail "
            f"while decoding old individuals; {_CLEAR_HINT}."
        )
    return None


def _clear_propulate_checkpoints(output_dir: Path) -> None:
    """Delete propulate checkpoint files from output_dir (rank-0 only).

    Propulate 1.2.x writes island_<N>_ckpt.pickle (not *.pkl — earlier versions
    used *.pkl). Both patterns are matched to stay robust across versions. Stale
    checkpoints cause ValueError in _decode_param when the search space changes
    between runs (e.g. a param that allowed None is later fixed to a single value).
    This removes only checkpoint files; eval_report.json and other outputs are kept.
    """
    removed = _propulate_checkpoint_files(output_dir)
    for f in removed:
        f.unlink()
    if removed:
        log.info("[clear-checkpoints] removed %d checkpoint file(s) from %s", len(removed), output_dir)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _with_params_validated(config: EvalConfig, extra_params: dict, *, source: str | Path, phase: str | None = None) -> EvalConfig:
    """Return ``config`` with ``extra_params`` merged into ``default_params``, re-validated.

    ``model_copy(update=...)`` skips every pydantic validator, so a merged
    value such as ``dtw_dist_fn: cosine`` (unsupported since Phase 63 D-08)
    would pass unnoticed and only fail inside each HPO trial. The merged
    config is therefore rebuilt with ``EvalConfig.model_validate``, which runs
    all field and model validators (Phase 63-07, Review cycle 1 MEDIUM-3).

    Parameters
    ----------
    config:
        Loaded config whose ``default_params`` the extra params override.
    extra_params:
        Params to merge over ``config.default_params``.
    source:
        Where ``extra_params`` came from (an artifact path or a description);
        named in the error message.
    phase:
        Suite phase that produced the artifact (``"selfcal"`` /
        ``"ground_truth"``). When given, the error message says how to
        regenerate a stale artifact.

    Returns
    -------
    EvalConfig
        A new, fully validated config.

    Raises
    ------
    ValueError
        If the merged config fails validation. The message names ``source``,
        the config's output_dir and the first validation error.
    """
    from pydantic import ValidationError

    try:
        return EvalConfig.model_validate(
            {**config.model_dump(), "default_params": {**config.default_params, **extra_params}}
        )
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        detail = f"{loc + ': ' if loc else ''}{first['msg']}"
        msg = f"{source}: invalid params for config with output_dir {config.output_dir}: {detail}."
        if phase is not None:
            # 63-REVIEW WR-04: name a recovery the HoreKa launchers can perform.
            report = Path(source).parent / "eval_report.json"
            msg += (
                f" The artifact predates a config change such as Phase 63 D-08. Regenerate it: "
                f"delete {report} and re-run the '{phase}' phase (only this run is redone), or "
                f"re-run the '{phase}' phase with --force (HoreKa launchers: ZREG_FORCE=1 sbatch "
                "<launcher>; redoes every run of the phase). Add ZREG_CLEAR_CHECKPOINTS=1 / "
                "--clear-checkpoints if the run's Propulate checkpoints predate the change too; "
                "that only removes checkpoint files, it does not delete best_params.json."
            )
        raise ValueError(msg) from e


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _project_to_search_space(params: dict, search_space: dict) -> tuple[dict, dict]:
    """Snap every searched key of ``params`` onto one of its search-space choices.

    ``merge_params.merge_two`` averages numeric keys, so a merged warm start
    can hold values that are not choices of the combined config's search space
    (e.g. ``k_neighbours`` 3/10 -> 6). Such a seed makes ``BayesianSearch``
    raise before any trial, and under Propulate/Sobol it is evaluated with
    values the config forbids (Phase 63 review CR-01 / WR-02).

    Rules, per key that is in both ``params`` and ``search_space``:

    - value already a choice: kept;
    - numeric value with numeric choices: nearest numeric choice (ties go to
      the choice listed first);
    - anything else (e.g. ``dtw_dist_fn: euclidean`` vs ``[cpd]``): the first
      choice.

    Keys outside ``search_space`` and keys with an empty choice list are
    returned unchanged.

    Returns
    -------
    tuple[dict, dict]
        ``(projected, changed)`` where ``changed`` maps each replaced key to
        ``(old_value, new_value)``.
    """
    out = dict(params)
    changed: dict = {}
    for key, choices in search_space.items():
        if key not in out or not choices:
            continue
        value = out[key]
        if value in choices:
            continue
        numeric = [c for c in choices if _is_number(c)]
        if _is_number(value) and numeric:
            new = min(numeric, key=lambda c: abs(c - value))
        else:
            new = choices[0]
        out[key] = new
        changed[key] = (value, new)
    return out, changed


def run_optimize_then_eval(name: str, config_path: Path, force: bool, dry_run: bool, clear_checkpoints: bool = False, warmstart_params: dict | None = None) -> None:
    config = _load_config(config_path)
    if warmstart_params is not None:
        # Validated merge (never model_copy for params): runs on every rank
        # before the skip bcast below, so every rank raises identically and
        # none blocks in a collective.
        source = "baseline_with_combined merged warm start"
        # Validate the raw merge first so an unsupported value (e.g. a stale
        # dtw_dist_fn) fails loudly instead of being silently snapped away.
        _with_params_validated(config, warmstart_params, source=source)
        # CR-01 / WR-02: snap averaged / foreign values onto this config's
        # search space so the seed is a legal trial for every strategy.
        warmstart_params, changed = _project_to_search_space(warmstart_params, config.search_space)
        if changed and RANK == 0:
            log.warning(
                "[%s] warm start projected onto the search space: %s",
                name,
                ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changed.items()),
            )
        config = _with_params_validated(config, warmstart_params, source=source)
    output_dir = Path(config.output_dir)

    skip = False
    error: str | None = None
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
                error = _checkpoint_search_space_error(output_dir, config.search_space)
                if error is None:
                    _write_run_config(config_path, output_dir)

    # Broadcast the skip decision (and a stale-checkpoint error) so every rank
    # agrees before the collective call. Without this, rank-0 returning early
    # leaves other ranks deadlocked on the internal comm.Barrier() inside
    # HyperparamOptimizer.run().
    if COMM is not None:
        skip, error = COMM.bcast((skip, error), root=0)
    if error is not None:
        raise RuntimeError(f"[{name}] {error}")
    if skip:
        return

    # HyperparamOptimizer.run() is a collective MPI operation — every rank
    # must call this (propulate needs all ranks to participate; gating on
    # rank-0 only would deadlock on the internal comm.Barrier()).
    # Phase 63 HPC-01: the injected params are also the first HPO trial (warm
    # start), not just the fallback for non-searched keys.
    HyperparamOptimizer(config, warm_start=[warmstart_params] if warmstart_params else None).run()

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


def _combined_best_params(configs_dir: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Load all five HPO best_params.json files plus a defaults dict.

    Returns
    -------
    tuple
        ``(kobitski_sc_alignment, shah_sc_alignment, shah_sc_label_transfer,
        kobitski_gt_alignment, shah_gt_both, defaults)``.

    Raises
    ------
    FileNotFoundError
        If any selfcal or ground_truth run hasn't produced best_params.json yet.
    ValueError
        If an artifact fails validation against its own config (e.g. a stale
        ``dtw_dist_fn: cosine`` from before Phase 63 D-08). The message names
        the artifact and says how to regenerate it.
    """
    selfcal_paths = {
        "kobitski_sc_alignment": configs_dir / "selfcal" / "kobitski_ew06_alignment.yaml",
        "shah_sc_alignment": configs_dir / "selfcal" / "shah_alignment.yaml",
        "shah_sc_label_transfer": configs_dir / "selfcal" / "shah_label_transfer.yaml",
    }
    gt_paths = {
        "kobitski_gt_alignment": configs_dir / "ground_truth" / "kobitski_ew06.yaml",
        "shah_gt_both": configs_dir / "ground_truth" / "shah_sample1.yaml",
    }
    results = {}
    sources = [("selfcal", selfcal_paths), ("ground_truth", gt_paths)]
    for phase, paths in sources:
        for key, cfg_path in paths.items():
            config = _load_config(cfg_path)
            bp_path = Path(config.output_dir) / "best_params.json"
            if not bp_path.exists():
                raise FileNotFoundError(
                    f"{bp_path} not found — run the 'selfcal' and 'ground_truth' phases before 'baseline_with_combined'."
                )
            params = _read_json(bp_path)
            # Phase 63-07 (MEDIUM-3): reject a stale artifact loudly instead of
            # merging it; the validated config is discarded, the dict is kept.
            _with_params_validated(config, params, source=bp_path, phase=phase)
            results[key] = params

    defaults_config = _load_config(configs_dir / "baseline_with_combined" / "ew06_vs_shah.yaml")
    return (
        results["kobitski_sc_alignment"],
        results["shah_sc_alignment"],
        results["shah_sc_label_transfer"],
        results["kobitski_gt_alignment"],
        results["shah_gt_both"],
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

    elif phase == "baseline_with_combined":
        upstream_runs = phases_map["selfcal"] + phases_map["ground_truth"]
        if dry_run and not all(
            (Path(_load_config(r[2]).output_dir) / "best_params.json").exists() for r in upstream_runs
        ):
            log.info("[baseline_with_combined] DRY-RUN: best_params.json not yet available — would merge at real run time")
            for _, name, cfg_path in runs:
                run_optimize_then_eval(name, cfg_path, force=force, dry_run=True, clear_checkpoints=clear_checkpoints)
            return

        kobitski_sc, shah_sc_align, shah_sc_lt, kobitski_gt, shah_gt, defaults = _combined_best_params(configs_dir)
        merged = merge_combined_params(kobitski_sc, shah_sc_align, shah_sc_lt, kobitski_gt, shah_gt, defaults)
        log.info("[baseline_with_combined] warmstart params (default_params override): %s", merged)
        for _, name, cfg_path in runs:
            run_optimize_then_eval(name, cfg_path, force=force, dry_run=dry_run, clear_checkpoints=clear_checkpoints, warmstart_params=merged)


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
