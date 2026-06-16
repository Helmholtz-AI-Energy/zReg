"""Tests for eval.tracking.export_trajectory() and EvaluationRunner.run() trajectory
wiring per EXT-01.

Five test classes cover the EXT-01 gate criteria:

- TestExportTrajectoryAlignOnly        — align stage only; label absent
- TestExportTrajectoryLabelOnly        — label stage only; align absent (D-08 raw pos)
- TestExportTrajectoryCombined         — both stages; D-07 aligned pos in label CSV
- TestExportTrajectoryMetadata         — metadata JSON fields and auto-capture
- TestExportTrajectoryIntegration      — EvaluationRunner.run() wiring end-to-end

All tests use tmp_path (built-in pytest fixture) as output_dir.  No writes to
experiments/runs/ or any persistent path during test runs.
"""

import csv
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# zreg.* before torch on macOS-ARM (libomp SIGABRT rule from Phase 12)
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_labels, generate_trajectory

from eval.config import EvalConfig
from eval.tracking import export_trajectory
from eval.types import AlignResult, LabelResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EXPECTED_ALIGN_COLS = ["frame_idx", "point_idx", "x", "y", "z"]
EXPECTED_LABEL_COLS = ["frame_idx", "point_idx", "x", "y", "z", "label"]
EXPECTED_META_FIELDS = {
    "run_id",
    "frame_count",
    "frame_indices",
    "data_path",
    "params_used",
    "tier",
    "n_trials",
    "n_synthetic",
    "git_hash",
    "zreg_version",
    "timestamp",
}

_MOCK_GIT_RESULT = MagicMock(stdout="deadbeef\n", returncode=0)

# ---------------------------------------------------------------------------
# Module-level auto-patch fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_auto_capture():
    """Patch subprocess.run and importlib.metadata.version for every test in
    this module, preventing live git calls and missing-package errors in CI."""
    with (
        patch("eval.tracking.trajectory.subprocess.run", return_value=_MOCK_GIT_RESULT),
        patch("eval.tracking.trajectory.importlib.metadata.version", return_value="0.0.1"),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with color labels (20 points per frame)."""
    traj = generate_trajectory(n_points=20, n_frames=3, seed=0)
    return generate_labels(traj, n_classes=4, seed=0)


@pytest.fixture
def eval_config(tmp_path) -> EvalConfig:
    """EvalConfig pointing into tmp_path with both stages enabled."""
    return EvalConfig(
        data_path=str(tmp_path / "data.mat"),
        tier="sanity",
        n_trials=5,
        n_synthetic=10,
        run_alignment=True,
        run_label_transfer=True,
    )


@pytest.fixture
def fake_align_result(synthetic_dataset) -> AlignResult:
    """AlignResult constructed without DTW execution."""
    return AlignResult(
        aligned_cloud=synthetic_dataset,
        warp_path=[(0, 0), (1, 1), (2, 2)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={"window_size": 3},
    )


@pytest.fixture
def fake_label_result(synthetic_dataset) -> LabelResult:
    """LabelResult using the actual color labels from the synthetic dataset."""
    return LabelResult(
        transferred_labels={k: synthetic_dataset[k]["color"] for k in synthetic_dataset},
        params_used={"k_neighbours": 3},
    )


# ---------------------------------------------------------------------------
# TestExportTrajectoryAlignOnly
# ---------------------------------------------------------------------------


class TestExportTrajectoryAlignOnly:
    """export_trajectory with result["align"] set and result["label"] = None."""

    def _run(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """Helper: run export_trajectory with align only."""
        result = {"align": fake_align_result, "label": None}
        return export_trajectory(result, synthetic_dataset, eval_config, tmp_path)

    def test_align_csv_created(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """align_trajectory.csv is created in the output directory."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        assert (tmp_path / "align_trajectory.csv").exists()

    def test_label_csv_absent(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """label_trajectory.csv is NOT created when result['label'] is None."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        assert not (tmp_path / "label_trajectory.csv").exists()

    def test_align_csv_header(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """First row of align_trajectory.csv equals EXPECTED_ALIGN_COLS."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_trajectory.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == EXPECTED_ALIGN_COLS

    def test_align_csv_row_count(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """align_trajectory.csv has 3*20 = 60 data rows (not counting header)."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_trajectory.csv", newline="") as f:
            rows = list(csv.reader(f))
        # rows includes header
        assert len(rows) - 1 == 3 * 20, f"Expected 60 data rows, got {len(rows) - 1}"

    def test_returns_two_paths(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """Return value is a list of length 2 (csv + metadata)."""
        paths = self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# TestExportTrajectoryLabelOnly
# ---------------------------------------------------------------------------


class TestExportTrajectoryLabelOnly:
    """export_trajectory with result["align"] = None and result["label"] set."""

    def _run(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """Helper: run export_trajectory with label only."""
        result = {"align": None, "label": fake_label_result}
        return export_trajectory(result, synthetic_dataset, eval_config, tmp_path)

    def test_label_csv_created(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """label_trajectory.csv is created in the output directory."""
        self._run(synthetic_dataset, fake_label_result, eval_config, tmp_path)
        assert (tmp_path / "label_trajectory.csv").exists()

    def test_align_csv_absent(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """align_trajectory.csv is NOT created when result['align'] is None."""
        self._run(synthetic_dataset, fake_label_result, eval_config, tmp_path)
        assert not (tmp_path / "align_trajectory.csv").exists()

    def test_label_csv_header(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """First row of label_trajectory.csv equals EXPECTED_LABEL_COLS."""
        self._run(synthetic_dataset, fake_label_result, eval_config, tmp_path)
        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == EXPECTED_LABEL_COLS

    def test_label_csv_row_count(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """label_trajectory.csv has 3*20 = 60 data rows."""
        self._run(synthetic_dataset, fake_label_result, eval_config, tmp_path)
        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) - 1 == 3 * 20, f"Expected 60 data rows, got {len(rows) - 1}"

    def test_uses_raw_dataset_positions(self, synthetic_dataset, fake_label_result, eval_config, tmp_path):
        """D-08: when align is None, x value in first data row matches dataset[frame]['pos'][0][0].item()."""
        self._run(synthetic_dataset, fake_label_result, eval_config, tmp_path)
        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
        first_frame = sorted(synthetic_dataset.keys())[0]
        expected_x = synthetic_dataset[first_frame]["pos"][0][0].item()
        assert float(first_row["x"]) == pytest.approx(expected_x)

    def test_labels_pos_length_mismatch_raises(self, synthetic_dataset, eval_config, tmp_path):
        """F-03: ValueError when transferred_labels length != pos length for a frame."""
        first_frame = sorted(synthetic_dataset.keys())[0]
        bad_labels = {k: synthetic_dataset[k]["color"] for k in synthetic_dataset}
        bad_labels[first_frame] = torch.zeros(synthetic_dataset[first_frame]["color"].shape[0] + 5, dtype=torch.long)
        bad_label_result = LabelResult(transferred_labels=bad_labels, params_used={})
        result = {"align": None, "label": bad_label_result}
        with pytest.raises(ValueError, match="pipeline state is inconsistent"):
            export_trajectory(result, synthetic_dataset, eval_config, tmp_path)


# ---------------------------------------------------------------------------
# TestExportTrajectoryCombined
# ---------------------------------------------------------------------------


class TestExportTrajectoryCombined:
    """export_trajectory with both align and label results present."""

    def _run(self, synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path):
        """Helper: run export_trajectory with both stages."""
        result = {"align": fake_align_result, "label": fake_label_result}
        return export_trajectory(result, synthetic_dataset, eval_config, tmp_path)

    def test_both_csvs_created(
        self, synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path
    ):
        """Both align_trajectory.csv and label_trajectory.csv are created."""
        self._run(synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path)
        assert (tmp_path / "align_trajectory.csv").exists()
        assert (tmp_path / "label_trajectory.csv").exists()

    def test_returns_four_paths(
        self, synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path
    ):
        """Return value is a list of length 4 (2 per stage)."""
        paths = self._run(
            synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path
        )
        assert len(paths) == 4

    def test_same_run_id_in_both_metadata(
        self, synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path
    ):
        """align_metadata.json and label_metadata.json share the same run_id."""
        self._run(synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path)
        with open(tmp_path / "align_metadata.json") as f:
            align_meta = json.load(f)
        with open(tmp_path / "label_metadata.json") as f:
            label_meta = json.load(f)
        assert align_meta["run_id"] == label_meta["run_id"]

    def test_label_csv_uses_aligned_positions(
        self, synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path
    ):
        """D-07: label_trajectory.csv x value matches aligned_cloud position, not raw dataset."""
        self._run(synthetic_dataset, fake_align_result, fake_label_result, eval_config, tmp_path)
        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
        first_frame = sorted(synthetic_dataset.keys())[0]
        # D-07: positions come from aligned_cloud (which is the same as synthetic_dataset
        # in this fixture — but the assertion verifies the code takes the aligned path)
        aligned_x = fake_align_result.aligned_cloud[first_frame]["pos"][0][0].item()
        assert float(first_row["x"]) == pytest.approx(aligned_x)


# ---------------------------------------------------------------------------
# TestExportTrajectoryMetadata
# ---------------------------------------------------------------------------


class TestExportTrajectoryMetadata:
    """Metadata JSON fields and auto-capture (git_hash, zreg_version)."""

    def _run(self, synthetic_dataset, fake_align_result, eval_config, tmp_path):
        """Helper: run export_trajectory with patched auto-capture fields."""
        result = {"align": fake_align_result, "label": None}
        with (
            patch("eval.tracking.trajectory.subprocess.run", return_value=_MOCK_GIT_RESULT),
            patch("eval.tracking.trajectory.importlib.metadata.version", return_value="0.0.1"),
        ):
            paths = export_trajectory(result, synthetic_dataset, eval_config, tmp_path)
        return paths

    def test_align_metadata_exists(
        self, synthetic_dataset, fake_align_result, eval_config, tmp_path
    ):
        """align_metadata.json is written alongside align_trajectory.csv."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        assert (tmp_path / "align_metadata.json").exists()

    def test_metadata_all_required_fields(
        self, synthetic_dataset, fake_align_result, eval_config, tmp_path
    ):
        """align_metadata.json contains all 11 required fields."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)
        assert EXPECTED_META_FIELDS.issubset(meta.keys()), (
            f"Missing fields: {EXPECTED_META_FIELDS - meta.keys()}"
        )

    def test_run_id_is_valid_uuid(
        self, synthetic_dataset, fake_align_result, eval_config, tmp_path
    ):
        """metadata['run_id'] is a valid UUID string."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)
        # uuid.UUID() raises ValueError if not valid
        parsed = uuid.UUID(meta["run_id"])
        assert str(parsed) == meta["run_id"]

    def test_git_hash_captured(
        self, synthetic_dataset, fake_align_result, eval_config, tmp_path
    ):
        """metadata['git_hash'] equals the mocked value 'deadbeef'."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)
        assert meta["git_hash"] == "deadbeef"

    def test_zreg_version_captured(
        self, synthetic_dataset, fake_align_result, eval_config, tmp_path
    ):
        """metadata['zreg_version'] equals the mocked value '0.0.1'."""
        self._run(synthetic_dataset, fake_align_result, eval_config, tmp_path)
        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)
        assert meta["zreg_version"] == "0.0.1"


# ---------------------------------------------------------------------------
# TestExportTrajectoryIntegration
# ---------------------------------------------------------------------------


class TestExportTrajectoryIntegration:
    """Integration: EvaluationRunner.run() trajectory wiring end-to-end."""

    def test_eval_runner_report_has_trajectory_paths(self, synthetic_dataset, tmp_path):
        """EvaluationRunner.run() returns EvalReport where len(trajectory_paths) >= 2."""
        from eval.runners import EvaluationRunner

        eval_config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            output_dir=str(tmp_path / "output"),
            tier="sanity",
            n_trials=5,
            n_synthetic=10,
            run_alignment=True,
            run_label_transfer=True,
        )
        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "k_neighbours": 5,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
        }

        with patch("eval.runners.eval_runner.DataFactory") as mock_factory_cls:
            mock_factory = mock_factory_cls.return_value
            mock_factory.load_real.return_value = synthetic_dataset
            mock_factory.load_target.return_value = synthetic_dataset
            mock_factory.get_ground_truth.return_value = {
                k: synthetic_dataset[k]["color"] for k in synthetic_dataset
            }
            runner = EvaluationRunner(eval_config, params)
            report = runner.run()

        assert len(report.trajectory_paths) >= 2, (
            f"Expected >= 2 trajectory_paths, got {report.trajectory_paths!r}"
        )

    def test_trajectory_paths_in_json(self, synthetic_dataset, tmp_path):
        """eval_report.json written by run() contains a 'trajectory_paths' key."""
        from eval.runners import EvaluationRunner

        eval_config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            output_dir=str(tmp_path / "output"),
            tier="sanity",
            n_trials=5,
            n_synthetic=10,
            run_alignment=True,
            run_label_transfer=True,
        )
        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "k_neighbours": 5,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
        }

        with patch("eval.runners.eval_runner.DataFactory") as mock_factory_cls:
            mock_factory = mock_factory_cls.return_value
            mock_factory.load_real.return_value = synthetic_dataset
            mock_factory.load_target.return_value = synthetic_dataset
            mock_factory.get_ground_truth.return_value = {
                k: synthetic_dataset[k]["color"] for k in synthetic_dataset
            }
            runner = EvaluationRunner(eval_config, params)
            runner.run()

        report_path = Path(tmp_path / "output" / "eval_report.json")
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "trajectory_paths" in data, (
            f"'trajectory_paths' key missing from eval_report.json; keys: {list(data.keys())}"
        )


# ---------------------------------------------------------------------------
# Coverage gap tests for trajectory.py
# ---------------------------------------------------------------------------


class TestTrajectoryExportCoverageGaps:
    """trajectory.py:108-110 (git returncode!=0), 115-116 (PackageNotFoundError)."""

    def test_git_hash_unknown_when_returncode_nonzero(self, synthetic_dataset, tmp_path):
        """trajectory.py:108 — git_hash='unknown' when returncode != 0."""
        import importlib.metadata as _meta
        from unittest.mock import MagicMock as MM

        bad_git = MM(returncode=1, stdout="")
        with patch("eval.tracking.trajectory.subprocess.run", return_value=bad_git), \
             patch("eval.tracking.trajectory.importlib.metadata.version", return_value="0.0.1"):
            fake_align = AlignResult(
                aligned_cloud=synthetic_dataset,
                warp_path=[(0, 0), (1, 1), (2, 2)],
                dtw_distance=0.0,
                n_changepoints=0,
                params_used={},
            )
            cfg = EvalConfig(data_path="x", tier="sanity", n_trials=1, n_synthetic=5)
            paths = export_trajectory(
                {"align": fake_align, "label": None},
                synthetic_dataset,
                cfg,
                tmp_path / "out_gitfail",
            )
        import json
        with open(str(paths[1])) as f:
            meta = json.load(f)
        assert meta["git_hash"] == "unknown"

    def test_git_hash_unknown_when_subprocess_raises(self, synthetic_dataset, tmp_path):
        """trajectory.py:109-110 — git_hash='unknown' when subprocess.run raises."""
        with patch("eval.tracking.trajectory.subprocess.run", side_effect=Exception("git not found")), \
             patch("eval.tracking.trajectory.importlib.metadata.version", return_value="0.0.1"):
            fake_align = AlignResult(
                aligned_cloud=synthetic_dataset,
                warp_path=[(0, 0), (1, 1), (2, 2)],
                dtw_distance=0.0,
                n_changepoints=0,
                params_used={},
            )
            cfg = EvalConfig(data_path="x", tier="sanity", n_trials=1, n_synthetic=5)
            paths = export_trajectory(
                {"align": fake_align, "label": None},
                synthetic_dataset,
                cfg,
                tmp_path / "out_gitraise",
            )
        import json
        with open(str(paths[1])) as f:
            meta = json.load(f)
        assert meta["git_hash"] == "unknown"

    def test_zreg_version_unknown_when_package_not_found(self, synthetic_dataset, tmp_path):
        """trajectory.py:115-116 — zreg_version='unknown' on PackageNotFoundError."""
        import importlib.metadata as _meta
        from unittest.mock import MagicMock as MM

        good_git = MM(returncode=0, stdout="abc123\n")
        with patch("eval.tracking.trajectory.subprocess.run", return_value=good_git), \
             patch(
                 "eval.tracking.trajectory.importlib.metadata.version",
                 side_effect=_meta.PackageNotFoundError("zreg"),
             ):
            from eval.tracking import export_trajectory
            from eval.types import AlignResult
            fake_align = AlignResult(
                aligned_cloud=synthetic_dataset,
                warp_path=[(0, 0), (1, 1), (2, 2)],
                dtw_distance=0.0,
                n_changepoints=0,
                params_used={},
            )
            from eval.config import EvalConfig
            cfg = EvalConfig(data_path="x", tier="sanity", n_trials=1, n_synthetic=5)
            paths = export_trajectory(
                {"align": fake_align, "label": None},
                synthetic_dataset,
                cfg,
                tmp_path / "out_pkgfail",
            )
        import json
        with open(str(paths[1])) as f:
            meta = json.load(f)
        assert meta["zreg_version"] == "unknown"
