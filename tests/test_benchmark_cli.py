"""CLI smoke tests for benchmark_label_transfer.py (49-03-PLAN.md Task 2).

Proves D-01's own promise end-to-end: ``benchmark_label_transfer.main(argv)``
runs against freshly-initialized smoke checkpoints (mirroring Phase 47/48's
D-01 pattern) and writes a valid ``benchmark_report.json`` with one
``MethodBenchmarkResult`` per method — the machinery is user-runnable without
editing test code.

Does NOT depend on tests/test_benchmark_runner.py's fixtures — pytest
fixtures are not shared across test modules without a shared conftest.py, so
the smoke checkpoint construction is reproduced inline here (same pattern:
direct ``model_cls(**hyperparams)`` + ``save_checkpoint``, no subprocess).
"""

# zreg (and scipy) must be imported before torch/torch_geometric on macOS ARM
# to avoid duplicate libomp initialisation (SIGABRT) — mirrors
# tests/conftest.py, tests/test_train_label_transfer.py,
# tests/test_benchmark_runner.py.
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import json
import sys
from pathlib import Path

# Defensive repo-root sys.path insertion mirroring tests/test_train_label_transfer.py
# — this file is self-contained even if conftest's own insertion were ever skipped.
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml

from train_label_transfer import save_checkpoint
from zreg.models import EGNNLabelTransfer, PointNet2LabelTransfer

import benchmark_label_transfer as cli

N_CLASSES = 4


def _write_smoke_config(tmp_path: Path) -> Path:
    """Write a tmp YAML config wired to inline smoke checkpoints.

    ``default_params`` contains BOTH LabelTransferStage's REQUIRED_PARAMS
    AND AlignmentStage's REQUIRED_PARAMS so the cpd_weighted internal
    AlignmentStage call has everything it needs (mirrors
    tests/test_benchmark_runner.py's BENCHMARK_PARAMS exactly).
    """
    pointnet2_hyperparams = {"n_classes": N_CLASSES, "hidden_dim": 32}
    egnn_hyperparams = {"n_classes": N_CLASSES, "hidden_dim": 32, "n_layers": 4}

    pointnet2_path = tmp_path / "pointnet2.pt"
    egnn_path = tmp_path / "egnn.pt"

    save_checkpoint(
        PointNet2LabelTransfer(**pointnet2_hyperparams),
        "pointnet2",
        pointnet2_hyperparams,
        epoch=0,
        path=str(pointnet2_path),
    )
    save_checkpoint(
        EGNNLabelTransfer(**egnn_hyperparams),
        "egnn",
        egnn_hyperparams,
        epoch=0,
        path=str(egnn_path),
    )

    config_dict = {
        "data_path": "unused",
        "pointnet2_checkpoint_path": str(pointnet2_path),
        "egnn_checkpoint_path": str(egnn_path),
        "output_dir": str(tmp_path / "out"),
        "default_params": {
            # LabelTransferStage.REQUIRED_PARAMS
            "k_neighbours": 5,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
            # AlignmentStage.REQUIRED_PARAMS
            "window_size": 10,
            "step": 1,
            "dtw_dist_fn": "euclidean",
            "cpd_penalty": "rigid",
            "n_breakpoints": 5,
        },
    }

    cfg_path = tmp_path / "smoke_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config_dict, f)
    return cfg_path


def _find_report_json(out_root: Path) -> Path:
    """Locate benchmark_report.json under out_root's timestamped subdir."""
    matches = list(out_root.rglob("benchmark_report.json"))
    assert len(matches) == 1, f"expected exactly one report, found {matches}"
    return matches[0]


class TestCliSmoke:
    """D-01: the CLI runs end-to-end against inline smoke checkpoints."""

    def test_cli_smoke_writes_report(self, tmp_path) -> None:
        cfg_path = _write_smoke_config(tmp_path)

        rc = cli.main(
            [
                "--config",
                str(cfg_path),
                "--held-out-seeds",
                "0:4",
                "--n-classes",
                str(N_CLASSES),
            ]
        )
        assert rc == 0

        report_path = _find_report_json(tmp_path / "out")
        with open(report_path) as f:
            data = json.load(f)

        assert len(data["results"]) == 4
        for entry in data["results"]:
            for key in ("method", "dataset_source", "knn_consistency", "latency_seconds", "error"):
                assert key in entry


class TestCliInvalidConfig:
    """Invalid config returns 1 (EvalConfigError handled, no traceback)."""

    def test_cli_invalid_config_returns_1(self, tmp_path) -> None:
        missing_path = tmp_path / "does_not_exist.yaml"
        rc = cli.main(
            ["--config", str(missing_path), "--held-out-seeds", "0:4"]
        )
        assert rc == 1


class TestCliHeldOutSeedsParsed:
    """--held-out-seeds is caller-supplied, not hardcoded (Pattern 3 / A2)."""

    def test_cli_held_out_seeds_parsed(self) -> None:
        parser = cli._build_parser()
        ns = parser.parse_args(["--config", "x.yaml", "--held-out-seeds", "3:7"])
        assert ns.held_out_seeds == "3:7"
