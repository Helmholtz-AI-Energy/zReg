"""CLI entrypoint for the Phase 49 label-transfer benchmark (D-01).

Usage::

    python benchmark_label_transfer.py --config configs/benchmark_smoke.yaml \\
        --held-out-seeds 0:4 --n-classes 4

    python benchmark_label_transfer.py --config configs/benchmark_real.yaml \\
        --held-out-seeds 500:550 --n-classes 6 --methods knn_voting,egnn --verbose

Flags
-----
--config           (required) Path to a YAML configuration file loadable by
                   ``EvalConfig.from_yaml``.
--output-dir       (optional) Override ``config.output_dir`` at runtime.
--held-out-seeds   (optional, default ``"0:8"``) ``"start:stop"`` string
                   parsed into ``range(start, stop)`` and forwarded verbatim
                   to ``LabelTransferBenchmark.run_leakage_guard`` (Pattern 3
                   / Assumption A2, 49-RESEARCH.md). This value is NEVER
                   hardcoded internally — checkpoints produced by
                   ``train_label_transfer.py`` carry no seed-provenance
                   metadata, so the caller is solely responsible for
                   supplying a range disjoint from whatever seeds the
                   checkpoint being evaluated was actually trained on. For
                   D-01's own freshly-initialized smoke-test checkpoint
                   (trained on zero seeds), any range is trivially "held
                   out"; for a real trained checkpoint, choose a range past
                   the seeds used in training (e.g. if trained on
                   ``range(0, 500)``, use ``--held-out-seeds 500:550``).
--n-classes        (optional, default 4) Number of label classes for the
                   held-out synthetic dimension's ``DataFactory.
                   generate_training_set`` call. Must match the checkpoint's
                   trained ``n_classes`` hyperparameter or the learned
                   methods' internal ``one_hot`` call raises.
--methods          (optional, default all four) Comma-separated subset of
                   ``knn_voting,cpd_weighted,pointnet2,egnn``. Defaults to
                   ``LabelTransferStage.VALID_METHODS`` (all four) when
                   omitted.
--verbose          (optional) Enable INFO-level logging.

Dimensions run
--------------
1. **Synthetic held-out leakage guard** (always run): compares all requested
   methods against a held-out synthetic seed range via
   ``LabelTransferBenchmark.run_leakage_guard`` — the always-available
   dimension, since it requires no real data files.
2. **Real-data qualitative dimension** (optional, best-effort): only
   attempted when ``config.data_path`` is set to something other than the
   ``"unused"`` sentinel. Loads real data via ``DataFactory.load_real()``
   and runs ``LabelTransferBenchmark.compare_methods`` with
   ``has_ground_truth=False`` (real ``label`` fields are not the learned
   models' training vocabulary — 49-RESEARCH.md Pitfall 1). A
   ``FileNotFoundError``/``ValueError`` here (missing data files, unmounted
   external dataset, class-count mismatch) is caught, recorded as a note on
   the merged report, and does NOT abort the run (D-03 — document, don't
   crash).

Output
------
A single merged ``benchmark_report.json`` is written to a timestamped
subdirectory of ``config.output_dir`` via
``LabelTransferBenchmark.save_report`` (mirrors ``EvaluationRunner``'s fixed
filename convention).

Error handling
--------------
Only ``EvalConfigError`` is caught at config-load time; it prints a one-line
message to stderr and returns exit code 1, mirroring ``run_eval.py``'s
``EvalConfigError`` handling exactly. All other exceptions propagate (they
indicate bugs, not user error).

D-01's promise
--------------
This entry point wraps Plan 02's ``LabelTransferBenchmark`` unmodified.
Running it now against freshly-initialized smoke-test checkpoints proves the
machinery is user-runnable end-to-end; running it later against real trained
checkpoints (and, optionally, real mounted data) requires zero code changes
— only different ``--config``/``--held-out-seeds`` values.

No timeout, point-count guard, or subsampling is added here (D-03 —
scope-expanding mitigations are explicitly out of scope for this phase;
``EvalConfig.max_points_per_frame`` already exists if the user needs
subsampling).

macOS-ARM import order: eval.* imported after sys.path injection, mirroring
run_eval.py/train_label_transfer.py's convention.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# sys.path injection — benchmark_label_transfer.py is at repo root (mirrors
# run_eval.py's convention exactly).
_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# src/ insertion — makes `from zreg.X import ...` work when zreg is not
# installed as a package (src layout convention, mirrors conftest.py).
_src_root = _repo_root / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from eval.config import EvalConfig, EvalConfigError
from eval.data_factory import DataFactory
from eval.runners import LabelTransferBenchmark
from eval.stages import LabelTransferStage
from eval.types import BenchmarkReport

log = logging.getLogger(__name__)

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    p = argparse.ArgumentParser(description="Phase 49 label-transfer benchmark CLI")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output-dir", default=None, help="Override config.output_dir")
    p.add_argument(
        "--held-out-seeds",
        default="0:8",
        help=(
            "'start:stop' string parsed into range(start, stop) and forwarded "
            "to run_leakage_guard. Caller-supplied — never hardcoded (Pattern 3 / A2). "
            "Choose a range disjoint from the checkpoint's training seeds."
        ),
    )
    p.add_argument(
        "--n-classes",
        type=int,
        default=4,
        help="Number of label classes for the held-out synthetic dimension (default 4)",
    )
    p.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated subset of knn_voting,cpd_weighted,pointnet2,egnn. "
            "Defaults to all four when omitted."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def _parse_held_out_seeds(spec: str) -> range:
    """Parse a caller-supplied 'start:stop' string into a range object.

    Parameters
    ----------
    spec : str
        ``"start:stop"`` string, e.g. ``"0:4"`` or ``"500:550"``.

    Returns
    -------
    range
        ``range(int(start), int(stop))``.
    """
    start, stop = spec.split(":")
    return range(int(start), int(stop))


def _parse_methods(spec: str | None) -> tuple[str, ...]:
    """Parse a caller-supplied comma-separated methods string, or default to all four.

    Parameters
    ----------
    spec : str or None
        Comma-separated subset of ``LabelTransferStage.VALID_METHODS``, or
        ``None`` to use all four.

    Returns
    -------
    tuple[str, ...]
        The parsed methods tuple.
    """
    if spec is None:
        return LabelTransferStage.VALID_METHODS
    return tuple(m.strip() for m in spec.split(","))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Parse arguments, run the benchmark, and write a merged report.

    Parameters
    ----------
    argv : list[str] or None
        Argument list. ``None`` reads from ``sys.argv[1:]`` (production); an
        explicit list is used by tests.

    Returns
    -------
    int
        0 on success, 1 on ``EvalConfigError``.
    """
    args = _build_parser().parse_args(argv)

    # Load and validate config — catch only EvalConfigError, mirroring
    # run_eval.py's EvalConfigError handling exactly.
    try:
        config = EvalConfig.from_yaml(args.config)
    except EvalConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        config = config.model_copy(update={"output_dir": args.output_dir})
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # T-49-06: resolve output_dir to a timestamped subdirectory before any write.
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resolved_output_dir = str((Path(config.output_dir) / run_stamp).resolve())
    config = config.model_copy(update={"output_dir": resolved_output_dir})
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    held_out_seeds = _parse_held_out_seeds(args.held_out_seeds)
    methods = _parse_methods(args.methods)
    params = config.default_params

    benchmark = LabelTransferBenchmark(config, methods=methods)

    # Dimension 1 (always available): held-out synthetic leakage guard.
    syn_report = benchmark.run_leakage_guard(held_out_seeds, params, n_classes=args.n_classes)
    results = list(syn_report.results)
    notes = list(syn_report.notes)

    # Dimension 2 (optional, best-effort): real-data qualitative check.
    # T-49-07: never let a real-data load failure abort the whole run (D-03).
    if config.data_path != "unused":
        try:
            real = DataFactory(config).load_real()
            real_report = benchmark.compare_methods(
                real, real, params, dataset_source="real_qualitative", has_ground_truth=False
            )
            results.extend(real_report.results)
        except (FileNotFoundError, ValueError) as e:
            notes.append(f"real-data dimension failed: {e}")

    report = BenchmarkReport(params=dict(params), results=results, notes=notes)
    benchmark.save_report(report, config.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
