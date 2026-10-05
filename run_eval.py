"""CLI entrypoint for the zReg evaluation framework (FRAME-11).

Usage::

    python run_eval.py --config configs/alignment_sanity.yaml --mode optimize
    python run_eval.py --config configs/combined_full.yaml --mode full --output-dir runs/exp1
    python run_eval.py --config configs/alignment_dev.yaml --mode eval --verbose

Flags
-----
--config          (required) Path to a YAML configuration file loadable by
                  ``EvalConfig.from_yaml``.
--mode            (required) Run mode: ``optimize``, ``eval``, or ``full``.
--output-dir      (optional) Override ``config.output_dir`` at runtime.
--warm-start-from (optional) Path to a ``best_params.json`` file or a directory
                  containing one (searches the most-recent timestamped subdir).
                  In ``optimize``/``full`` mode the loaded params seed the first
                  tier's warm-start.  In ``eval`` mode they replace missing local
                  ``best_params.json`` (cross-job param forwarding, EXT-04).
--verbose         (optional) Set ``config.verbose = True`` at runtime.

Mode behaviour
--------------
optimize
    Run ``HyperparamOptimizer(config).run()``.  Writes ``best_params.json``
    and ``search_history.json`` to ``output_dir``.

eval
    Load ``{output_dir}/best_params.json`` if it exists (D-01) and pass the
    params to ``EvaluationRunner(config, params).run()``.  Falls back to an
    empty dict (EvalConfig defaults apply) when the file is absent.

full
    Chain optimizer then runner: ``HyperparamOptimizer(config).run()`` is
    called first (it writes ``best_params.json``), then the file is read and
    passed to ``EvaluationRunner`` (Pitfall 5 — read AFTER write).

Reproducibility
---------------
``run_config.yaml`` is copied to ``output_dir`` at the start of every run,
*before* the pipeline executes (D-10 / Pitfall 6).  Failed runs still leave
the config for post-hoc debugging.

MPI (Phase 63 HPC-04)
---------------------
Under MPI, rank 0 chooses the run stamp and broadcasts it, so all ranks share
one timestamped ``output_dir``; only rank 0 creates that directory, writes
``run_config.yaml`` and runs ``EvaluationRunner``.  Rank 0 broadcasts the
outcome of the directory setup together with the stamp; if it failed, every
other rank raises ``RuntimeError`` instead of waiting on a collective that
rank 0 never reaches.  ``HyperparamOptimizer`` stays collective on all ranks;
non-root ranks return 0 right after the setup in ``eval`` mode and after the
collective HPO in ``full`` mode.

Error handling
--------------
Only ``EvalConfigError`` is caught; it prints a one-line message to stderr
and returns exit code 1 (D-09).  All other exceptions propagate (they indicate
bugs, not user error).

Decisions implemented: D-01 through D-10.
macOS-ARM import order: eval.* imported after sys.path injection.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# sys.path injection — run_eval.py is at repo root so _repo_root == repo root.
# eval/ is a namespace directory (no __init__.py); this insertion makes
# `from eval.X import ...` work without a package install (D-07 / Pitfall 2).
_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_repo_root))

# src/ insertion — makes `from zreg.<module> import ...` work when zreg is not
# installed as a package (src layout convention, mirrors conftest.py).
_src_root = _repo_root / "src"
if str(_src_root) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_src_root))

from eval.config import EvalConfig, EvalConfigError
from eval.runners import EvaluationRunner, HyperparamOptimizer
from eval.runners.optimizer import _mpi_world_comm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (D-05 / D-06)."""
    p = argparse.ArgumentParser(description="zReg evaluation framework CLI")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--mode",
        required=True,
        choices=["optimize", "eval", "full"],
        help="Run mode",
    )
    p.add_argument("--output-dir", default=None, help="Override config.output_dir")
    p.add_argument(
        "--warm-start-from",
        default=None,
        metavar="PATH",
        help=(
            "Path to a best_params.json or a directory containing one "
            "(picks the most-recent timestamped subdir). Seeds HPO warm-start "
            "in optimize/full mode; used as fallback params in eval mode."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def _write_run_config(config_path: str, output_dir: str) -> None:
    """Copy input YAML to output_dir/run_config.yaml before pipeline runs (D-10).

    Uses ``shutil.copy`` for an exact byte-for-byte copy — no YAML defaults
    expansion.  Called *before* any optimizer or runner invocation so that
    even a failed run leaves a reproducibility record (Pitfall 6).
    """
    dst = Path(output_dir) / "run_config.yaml"
    shutil.copy(config_path, dst)


def _load_best_params(output_dir: str, fallback: dict | None = None) -> dict:
    """Load best_params.json from output_dir if it exists and is non-empty (D-01).

    Returns ``{**fallback, **optimized}`` so config defaults fill any keys the
    optimizer did not set.  Falls back to ``fallback`` alone (or ``{}``) when
    the file is absent or empty.  Logs which source was used at INFO level (D-02).
    """
    base = dict(fallback) if fallback else {}
    path = Path(output_dir) / "best_params.json"
    if path.exists():
        with open(path) as f:
            optimized = json.load(f)
        if optimized:
            log.info("Loaded optimized params from %s", path)
            return {**base, **optimized}
    log.info("No best_params.json found — using config defaults")
    return base


def _load_warm_start(path: str) -> list[dict] | None:
    """Return a single-element warm-start list from a prior run's best_params.json.

    Accepts either a direct path to ``best_params.json`` or a base output
    directory.  When given a directory, checks for ``best_params.json`` at the
    top level first, then searches timestamped subdirectories and picks the most
    recent one (lexicographic sort on dirname).  Returns ``None`` when no
    readable params are found.
    """
    p = Path(path)
    if p.is_file():
        target = p
    elif p.is_dir():
        direct = p / "best_params.json"
        if direct.exists():
            target = direct
        else:
            candidates = sorted(
                d / "best_params.json"
                for d in p.iterdir()
                if d.is_dir() and (d / "best_params.json").exists()
            )
            if not candidates:
                log.warning("--warm-start-from: no best_params.json found under %s", p)
                return None
            target = candidates[-1]
    else:
        log.warning("--warm-start-from path not found: %s", p)
        return None

    with open(target) as f:
        params = json.load(f)
    if not params:
        return None
    log.info("Loaded warm-start params from %s", target)
    return [params]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Parse arguments, load config, and dispatch to the requested run mode.

    Parameters
    ----------
    argv : list[str] or None
        Argument list.  ``None`` reads from ``sys.argv[1:]`` (production);
        an explicit list is used by tests (Pattern 1 / D-08).

    Returns
    -------
    int
        0 on success, 1 on ``EvalConfigError``.

    Notes
    -----
    Under MPI, rank 0 chooses the run stamp and broadcasts it; only rank 0
    writes run_config.yaml and runs EvaluationRunner (Phase 63 HPC-04).  A
    rank-0 failure while creating the run directory is broadcast with the
    stamp, and the other ranks raise ``RuntimeError`` naming it.
    """
    args = _build_parser().parse_args(argv)

    # Load and validate config — catch only EvalConfigError (D-09 / Pitfall 8)
    try:
        config = EvalConfig.from_yaml(args.config)
    except EvalConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Apply runtime overrides via model_copy (Pitfall 1 — idiomatic pydantic v2)
    if args.output_dir is not None:
        config = config.model_copy(update={"output_dir": args.output_dir})
    if args.verbose:
        config = config.model_copy(update={"verbose": True})

    # Configure logging when verbose is active
    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    # HPC-04: under MPI every rank must share ONE run directory.  All ranks
    # reach this point identically (config errors returned above on every
    # rank), so no rank can skip the collective below.
    comm = _mpi_world_comm()
    rank = comm.Get_rank() if comm is not None else 0

    # Rank 0 alone picks the stamp, creates the directory and writes
    # run_config.yaml.  T-23-01: resolve output_dir to prevent path traversal
    # before any write.
    run_stamp = None
    setup_error: Exception | None = None
    if rank == 0:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            run_dir = (Path(config.output_dir) / run_stamp).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            # D-10 / Pitfall 6: write run_config.yaml BEFORE pipeline executes
            _write_run_config(args.config, str(run_dir))
        except Exception as exc:
            setup_error = exc

    if comm is not None:
        # One bcast carries the stamp and the setup outcome, so a rank-0
        # failure reaches every rank before any further collective (no hang).
        err_msg = None if setup_error is None else f"{type(setup_error).__name__}: {setup_error}"
        run_stamp, err_msg = comm.bcast((run_stamp, err_msg), root=0)
        if setup_error is None and err_msg is not None:
            raise RuntimeError(
                f"[rank {rank}] rank 0 failed to set up the run directory: {err_msg}"
            )
    if setup_error is not None:
        raise setup_error
    if comm is not None:
        comm.Barrier()  # run directory and run_config.yaml exist before any rank uses them

    resolved_output_dir = str((Path(config.output_dir) / run_stamp).resolve())
    config = config.model_copy(update={"output_dir": resolved_output_dir})

    # Resolve cross-job warm-start seed (EXT-04).  None when flag not provided.
    warm_start = _load_warm_start(args.warm_start_from) if args.warm_start_from else None

    # Mode dispatch
    if args.mode == "optimize":
        HyperparamOptimizer(config, warm_start=warm_start).run()
    elif args.mode == "eval":
        if rank != 0:
            # HPC-04: a single writer for eval_report.json and plots.
            log.debug("[rank %d] evaluation runs on rank 0 only", rank)
            return 0
        # Prefer local best_params.json (written by a prior optimize/full run in
        # the same output_dir); fall back to the forwarded warm-start params so
        # that cross-tier forwarding (synthetic → real) works without a local file.
        params = _load_best_params(config.output_dir, fallback=None)
        if not params and warm_start:
            params = warm_start[0]
        params = {**dict(config.default_params), **params} if params else dict(config.default_params)
        EvaluationRunner(config, params).run()
    else:  # args.mode == "full"
        # Pitfall 5: optimizer writes best_params.json; read it AFTER
        HyperparamOptimizer(config, warm_start=warm_start).run()
        if rank != 0:
            # HPC-04: HPO is collective, evaluation has a single writer.
            log.debug("[rank %d] participated in collective HPO; evaluation runs on rank 0", rank)
            return 0
        params = _load_best_params(config.output_dir, fallback=config.default_params)
        EvaluationRunner(config, params).run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
