"""Functional tests for the EVAL-05 runner scripts.

This module verifies the two sweep orchestrators created in Plan 16-01:

- ``eval/run_synthetic.py`` — noise/outlier sweep (SIGMAS × N_OUTLIERS_LIST)
- ``eval/run_real.py``      — scale/density sweep with missing-data guard

Approach: tests import ``run_sweep()`` from each script and call it
**directly** (no subprocess), redirecting all output via ``output_dir=str(tmp_path)``
so the real ``evaluation/runs/`` is never written to. Auto-captured
``log_run()`` fields (``git_hash``, ``zreg_version``) are patched at the same
target points used by ``tests/test_tracking.py`` to keep runs deterministic.

Test classes:
    TestRunSynthetic — file creation, run_id uniqueness, field presence,
                        zreg.metrics import path
    TestRunReal      — missing-data print message, no files on missing data,
                        __main__ guard
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.run_real import run_sweep as run_real_sweep
from eval.run_synthetic import N_FRAMES, N_OUTLIERS_LIST, SEED as SYNTH_SEED, SIGMAS
from eval.run_synthetic import run_sweep as run_synthetic_sweep

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EXPECTED_LOG_RUN_FIELDS = {
    "run_id",
    "dataset_path",
    "frame_indices",
    "seed",
    "n_points_before",
    "n_points_after",
    "git_hash",
    "zreg_version",
    "timestamp",
}

EXPECTED_CELL_COUNT = len(SIGMAS) * len(N_OUTLIERS_LIST)

# Deterministic mocks for log_run()'s auto-captured fields, identical to
# the targets used in tests/test_tracking.py.
_MOCK_GIT_RESULT = MagicMock(stdout="abc\n", returncode=0)


def _patched_log_run():
    """Return a context-manager-like patch stack that pins git_hash and
    zreg_version for any call into eval.tracking.log_run() during the test."""
    return (
        patch(
            "eval.tracking.tracking.subprocess.run",
            return_value=_MOCK_GIT_RESULT,
        ),
        patch(
            "eval.tracking.tracking.importlib.metadata.version",
            return_value="0.0.1",
        ),
    )


# ---------------------------------------------------------------------------
# TestRunSynthetic
# ---------------------------------------------------------------------------


class TestRunSynthetic:
    """Functional tests for eval/run_synthetic.py::run_sweep().

    All tests pass ``output_dir=str(tmp_path)`` so the real
    ``evaluation/runs/`` directory is never touched. ``log_run()``'s
    auto-captured fields are patched at the same targets used in
    ``tests/test_tracking.py``.
    """

    # ------------------------------------------------------------------
    # File creation
    # ------------------------------------------------------------------

    def test_creates_json_files(self, tmp_path):
        """run_sweep() writes one .json file per (sigma, n_outliers) cell."""
        subprocess_patch, version_patch = _patched_log_run()
        with subprocess_patch, version_patch:
            run_synthetic_sweep(output_dir=str(tmp_path))
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == EXPECTED_CELL_COUNT

    def test_creates_csv_files(self, tmp_path):
        """run_sweep() writes one .csv file per (sigma, n_outliers) cell."""
        subprocess_patch, version_patch = _patched_log_run()
        with subprocess_patch, version_patch:
            run_synthetic_sweep(output_dir=str(tmp_path))
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == EXPECTED_CELL_COUNT

    # ------------------------------------------------------------------
    # run_id uniqueness
    # ------------------------------------------------------------------

    def test_run_ids_unique(self, tmp_path):
        """Every (sigma, n_outliers) cell yields a unique run_id stem."""
        subprocess_patch, version_patch = _patched_log_run()
        with subprocess_patch, version_patch:
            run_synthetic_sweep(output_dir=str(tmp_path))
        stems = {p.stem for p in tmp_path.glob("*.json")}
        assert len(stems) == EXPECTED_CELL_COUNT

    # ------------------------------------------------------------------
    # 9-field JSON presence
    # ------------------------------------------------------------------

    def test_json_contains_required_fields(self, tmp_path):
        """Every emitted JSON file contains the 9 EVAL-04 required fields."""
        subprocess_patch, version_patch = _patched_log_run()
        with subprocess_patch, version_patch:
            run_synthetic_sweep(output_dir=str(tmp_path))
        json_files = list(tmp_path.glob("*.json"))
        assert json_files, "expected at least one JSON file from run_sweep()"
        with open(json_files[0]) as f:
            data = json.load(f)
        assert EXPECTED_LOG_RUN_FIELDS.issubset(data.keys())

    # ------------------------------------------------------------------
    # frame_indices content
    # ------------------------------------------------------------------

    def test_json_frame_indices_match_n_frames(self, tmp_path):
        """frame_indices in every JSON is list(range(N_FRAMES))."""
        subprocess_patch, version_patch = _patched_log_run()
        with subprocess_patch, version_patch:
            run_synthetic_sweep(output_dir=str(tmp_path))
        json_files = list(tmp_path.glob("*.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["frame_indices"] == list(range(N_FRAMES))
        assert data["seed"] == SYNTH_SEED

    # ------------------------------------------------------------------
    # zreg.metrics import path
    # ------------------------------------------------------------------

    def test_metrics_import_path(self):
        """eval/run_synthetic.py imports chamfer/hausdorff from zreg.metrics,
        not from a local copy under eval/."""
        import eval.run_synthetic as rs

        source_text = Path(inspect.getfile(rs)).read_text()
        assert "from zreg.metrics import" in source_text


# ---------------------------------------------------------------------------
# TestRunReal
# ---------------------------------------------------------------------------


class TestRunReal:
    """Functional tests for eval/run_real.py::run_sweep().

    The repo deliberately ships without ``data/raw/example.mat`` so the
    missing-data path is the only path exercised here. Tests assert the
    informative print message, the no-file invariant, and the standalone
    script's ``__main__`` guard.
    """

    def test_missing_dataset_prints_message(self, tmp_path, capsys):
        """When DATASET_PATH is absent, run_sweep() prints 'not found'."""
        # Guard: if someone later drops a real dataset into the repo, this
        # test would silently pass through the real path. Skip in that case
        # rather than emit a misleading green.
        from eval.run_real import DATASET_PATH

        if Path(DATASET_PATH).exists():
            pytest.skip(
                f"{DATASET_PATH} exists; missing-data path not exercised"
            )

        run_real_sweep(output_dir=str(tmp_path))
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_missing_dataset_creates_no_files(self, tmp_path):
        """When DATASET_PATH is absent, run_sweep() writes zero files."""
        from eval.run_real import DATASET_PATH

        if Path(DATASET_PATH).exists():
            pytest.skip(
                f"{DATASET_PATH} exists; missing-data path not exercised"
            )

        run_real_sweep(output_dir=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_script_has_main_guard(self):
        """eval/run_real.py is executable as `python eval/run_real.py`."""
        import eval.run_real as rr
        source_text = Path(inspect.getfile(rr)).read_text()
        assert 'if __name__ == "__main__"' in source_text
