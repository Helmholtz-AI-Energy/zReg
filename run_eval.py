"""CLI entrypoint for the zReg evaluation framework (FRAME-11).

Usage::

    python run_eval.py --config configs/alignment_sanity.yaml --mode optimize
    python run_eval.py --config configs/combined_full.yaml --mode full --output-dir runs/exp1
    python run_eval.py --config configs/alignment_dev.yaml --mode eval --verbose

Flags
-----
--config      (required) Path to a YAML configuration file loadable by
              ``EvalConfig.from_yaml``.
--mode        (required) Run mode: ``optimize``, ``eval``, or ``full``.
--output-dir  (optional) Override ``config.output_dir`` at runtime.
--verbose     (optional) Set ``config.verbose = True`` at runtime.

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
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# src/ insertion — makes `from zreg.X import ...` work when zreg is not
# installed as a package (src layout convention, mirrors conftest.py).
_src_root = _repo_root / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from eval.config import EvalConfig, EvalConfigError
from eval.runners import EvaluationRunner, HyperparamOptimizer

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

    # T-23-01: resolve output_dir to prevent path traversal before any write
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_output_dir = str((Path(config.output_dir) / run_stamp).resolve())
    config = config.model_copy(update={"output_dir": resolved_output_dir})

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # D-10 / Pitfall 6: write run_config.yaml BEFORE pipeline executes
    _write_run_config(args.config, config.output_dir)

    # Mode dispatch
    if args.mode == "optimize":
        HyperparamOptimizer(config).run()
    elif args.mode == "eval":
        params = _load_best_params(config.output_dir, fallback=config.default_params)
        EvaluationRunner(config, params).run()
    elif args.mode == "full":
        # Pitfall 5: optimizer writes best_params.json; read it AFTER
        HyperparamOptimizer(config).run()
        params = _load_best_params(config.output_dir, fallback=config.default_params)
        EvaluationRunner(config, params).run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
