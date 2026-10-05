"""Failing tests for eval.tracking.trajectory.export_trajectory (TDD RED phase).

Tests cover all acceptance criteria from 24-01-PLAN.md:
- importability from eval.tracking.trajectory
- align-only result: 2 paths returned, align_trajectory.csv + align_metadata.json written
- label-only result: 2 paths returned, label_trajectory.csv + label_metadata.json written
- both stages: 4 paths, same run_id UUID in both metadata files
- CSV headers exactly frame_idx,point_idx,x,y,z and frame_idx,point_idx,x,y,z,label
- row count equals sum of point counts across frames
- x/y/z are floats; label is int
- align_metadata.json has all 11 required keys
- output_dir is created automatically (parents=True, exist_ok=True)
"""

import csv
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_MOCK_GIT_RESULT = MagicMock(stdout="deadbeef\n", returncode=0)
_MOCK_VERSION = "0.1.0"

REQUIRED_META_KEYS = {
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


def _make_config():
    """Return a minimal EvalConfig-like object with required fields."""
    from eval.config import EvalConfig

    return EvalConfig(
        data_path="/data/test.mat",
        tier="sanity",
        n_trials=5,
        n_synthetic=10,
    )


def _make_zreg_pc(n_points: int, seed: int = 0):
    """Return a zRegPointCloud instance with a 'pos' tensor."""
    from zreg.core.dataset import zRegPointCloud

    torch.manual_seed(seed)
    pos = torch.randn(n_points, 3)
    pc = zRegPointCloud()
    pc["pos"] = pos
    return pc


def _make_align_result(dataset: dict):
    """Return an AlignResult whose aligned_cloud mirrors the dataset."""
    from eval.types import AlignResult

    # Use the dataset directly as aligned_cloud (same zRegPointCloud instances)
    aligned_cloud = dict(dataset)
    return AlignResult(
        aligned_cloud=aligned_cloud,
        warp_path=[(0, 0), (1, 1)],
        dtw_distance=0.5,
        n_changepoints=0,
        params_used={"method": "dtw"},
    )


def _make_label_result(dataset: dict):
    """Return a LabelResult with one 1-D label tensor per frame."""
    from eval.types import LabelResult

    transferred_labels = {}
    for frame_idx, pc in dataset.items():
        n = pc["pos"].shape[0]
        transferred_labels[frame_idx] = torch.zeros(n, dtype=torch.long)
    return LabelResult(
        transferred_labels=transferred_labels,
        params_used={"k": 3},
    )


# ---------------------------------------------------------------------------
# Patch context manager: mock git + version inside trajectory module
# ---------------------------------------------------------------------------

def _patched():
    """Context manager that patches subprocess and importlib inside trajectory."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch(
            "eval.tracking.trajectory.subprocess.run",
            return_value=_MOCK_GIT_RESULT,
        ), patch(
            "eval.tracking.trajectory.importlib.metadata.version",
            return_value=_MOCK_VERSION,
        ):
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestExportTrajectoryImportable:
    """export_trajectory is importable from eval.tracking.trajectory."""

    def test_direct_import(self):
        from eval.tracking.trajectory import export_trajectory  # noqa: F401

        assert callable(export_trajectory)


class TestExportTrajectoryAlignOnly:
    """When result['label'] is None, only align files are written."""

    def test_returns_two_paths(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5), 1: _make_zreg_pc(4)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert len(paths) == 2

    def test_align_csv_exists(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        assert (tmp_path / "align_trajectory.csv").exists()

    def test_align_meta_exists(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        assert (tmp_path / "align_metadata.json").exists()

    def test_no_label_files_written(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        assert not (tmp_path / "label_trajectory.csv").exists()
        assert not (tmp_path / "label_metadata.json").exists()

    def test_align_csv_header(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(3)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_trajectory.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["frame_idx", "point_idx", "x", "y", "z"]

    def test_align_csv_row_count(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5), 1: _make_zreg_pc(3)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_trajectory.csv", newline="") as f:
            rows = list(csv.reader(f))

        # Subtract header row
        data_rows = rows[1:]
        expected = sum(pc["pos"].shape[0] for pc in dataset.values())
        assert len(data_rows) == expected

    def test_align_csv_xyz_are_floats(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_trajectory.csv", newline="") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)

        # x, y, z must be parseable as float (not "1" masquerading as int)
        assert isinstance(float(first_row["x"]), float)

    def test_align_metadata_all_11_keys(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)

        assert REQUIRED_META_KEYS.issubset(set(meta.keys()))


class TestExportTrajectoryLabelOnly:
    """When result['align'] is None, only label files are written."""

    def test_returns_two_paths(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5), 1: _make_zreg_pc(4)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert len(paths) == 2

    def test_label_csv_exists(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        assert (tmp_path / "label_trajectory.csv").exists()

    def test_no_align_files_written(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        assert not (tmp_path / "align_trajectory.csv").exists()
        assert not (tmp_path / "align_metadata.json").exists()

    def test_label_csv_header(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(3)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["frame_idx", "point_idx", "x", "y", "z", "label"]

    def test_label_csv_label_column_is_int(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(3)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "label_trajectory.csv", newline="") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)

        label_val = first_row["label"]
        # Must be parseable as int, and must NOT contain a decimal point
        assert "." not in label_val
        assert isinstance(int(label_val), int)

    def test_label_metadata_all_11_keys(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "label_metadata.json") as f:
            meta = json.load(f)

        assert REQUIRED_META_KEYS.issubset(set(meta.keys()))


class TestExportTrajectoryBothStages:
    """When both align and label results are present, all 4 files are written."""

    def test_returns_four_paths(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(5), 1: _make_zreg_pc(4)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        label_result = _make_label_result(dataset)
        result = {"align": align_result, "label": label_result}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert len(paths) == 4

    def test_same_run_id_in_both_metadata(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(3)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        label_result = _make_label_result(dataset)
        result = {"align": align_result, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_metadata.json") as f:
            align_meta = json.load(f)
        with open(tmp_path / "label_metadata.json") as f:
            label_meta = json.load(f)

        assert align_meta["run_id"] == label_meta["run_id"]

    def test_run_id_is_uuid_format(self, tmp_path):
        """run_id must look like a UUID (8-4-4-4-12 hex)."""
        import re
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        label_result = _make_label_result(dataset)
        result = {"align": align_result, "label": label_result}

        with _patched():
            export_trajectory(result, dataset, config, tmp_path)

        with open(tmp_path / "align_metadata.json") as f:
            meta = json.load(f)

        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_re.match(meta["run_id"])


class TestExportTrajectoryOutputDirCreation:
    """output_dir is created automatically."""

    def test_nested_output_dir_created(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            export_trajectory(result, dataset, config, nested)

        assert nested.exists()


class TestExportTrajectoryReturnOrder:
    """Returned paths follow the documented order: align_csv, align_meta, label_csv, label_meta."""

    def test_align_only_path_order(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        result = {"align": align_result, "label": None}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert paths[0].endswith("align_trajectory.csv")
        assert paths[1].endswith("align_metadata.json")

    def test_label_only_path_order(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        label_result = _make_label_result(dataset)
        result = {"align": None, "label": label_result}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert paths[0].endswith("label_trajectory.csv")
        assert paths[1].endswith("label_metadata.json")

    def test_both_stages_path_order(self, tmp_path):
        from eval.tracking.trajectory import export_trajectory

        dataset = {0: _make_zreg_pc(2)}
        config = _make_config()
        align_result = _make_align_result(dataset)
        label_result = _make_label_result(dataset)
        result = {"align": align_result, "label": label_result}

        with _patched():
            paths = export_trajectory(result, dataset, config, tmp_path)

        assert paths[0].endswith("align_trajectory.csv")
        assert paths[1].endswith("align_metadata.json")
        assert paths[2].endswith("label_trajectory.csv")
        assert paths[3].endswith("label_metadata.json")
